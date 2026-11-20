"""
TF-IDF 多分类基线实验。

模型：
  1) Word n-gram TF-IDF (1~2 gram) + LogisticRegression / LinearSVC
  2) Char n-gram TF-IDF (3~5 gram) + LogisticRegression / LinearSVC
  3) Word + Char 拼接后再训练（简单融合）

指标：
  - 每类 Precision / Recall / F1 / Support
  - Macro F1、Weighted F1、Accuracy
  - 混淆矩阵（归一化 + 原始计数两张图）
  - 错误样本 Top 20（最置信的误判）

输出目录：output/tfidf_cls_{时间戳}/
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score)
from sklearn.pipeline import FeatureUnion, Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
# 让 tfidf/ 子目录脚本能 import 项目根目录的 common 包
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from common.label import CLS_ID2LABEL, CLS_NUM_LABELS, assert_labels  # noqa: E402

SPLIT_DIR = BASE_DIR / "data" / "dataset" / "split"
OUT_ROOT = BASE_DIR / "output" / "tfidf"

LABELS = CLS_ID2LABEL  # 向后兼容：老代码里用到的 LABELS[i]
assert_labels()

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- 数据


def load_split(path: Path) -> tuple[list[str], list[int], list[str]]:
    texts, ys, srcs = [], [], []
    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        texts.append(rec["text"])
        ys.append(rec["cls_label"])
        srcs.append(rec["source"])
    return texts, ys, srcs


# ---------------------------------------------------------------- 分词


def _ensure_jieba() -> None:
    """延迟 import jieba，避免 char-only 模式下也需要它。"""
    import jieba  # noqa: WPS433
    jieba.setLogLevel(60)


def _jieba_tokenize(text: str) -> list[str]:
    _ensure_jieba()
    import jieba
    # 用全模式，对公告的关键词（基金、分红、定期报告）不会漏切
    tokens = [t.strip() for t in jieba.lcut(text, cut_all=False)
              if t.strip() and len(t.strip()) >= 2]
    return tokens


# ---------------------------------------------------------------- 模型


def word_vectorizer(max_features: int, min_df: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=_jieba_tokenize, ngram_range=(1, 2),
        max_features=max_features, min_df=min_df, max_df=0.95,
        sublinear_tf=True, lowercase=False,
    )


def char_vectorizer(max_features: int, min_df: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5),
        max_features=max_features, min_df=min_df, max_df=0.95,
        sublinear_tf=True, lowercase=False,
    )


def build_pipeline(model: str, max_features: int, min_df: int, fusion: bool) -> Pipeline:
    if fusion:
        vec = FeatureUnion([
            ("word", word_vectorizer(max_features // 2, min_df)),
            ("char", char_vectorizer(max_features // 2, min_df)),
        ])
    else:
        vec = word_vectorizer(max_features, min_df) if model.endswith("_word") \
            else char_vectorizer(max_features, min_df)

    if model.startswith("logreg"):
        clf = LogisticRegression(
            C=4.0, max_iter=2000, class_weight="balanced",
            solver="lbfgs", n_jobs=None,
        )
    else:  # linear_svc / linear_svc_word / linear_svc_char
        from sklearn.svm import LinearSVC
        clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
    return Pipeline([("tfidf", vec), ("clf", clf)])


# ---------------------------------------------------------------- 评估


def evaluate(y_true: list[int], y_pred: np.ndarray,
             y_prob: np.ndarray | None, texts: list[str],
             srcs: list[str], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    label_ids = list(range(CLS_NUM_LABELS))
    label_names = [LABELS[i] for i in label_ids]
    report = classification_report(
        y_true, y_pred, labels=label_ids,
        target_names=label_names, output_dict=True,
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    acc = float((np.asarray(y_true) == y_pred).mean())

    # --- 分类报告 JSON ---
    metrics = {
        "accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "per_class": report,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- 分类报告文本 ---
    txt = classification_report(
        y_true, y_pred, labels=label_ids, target_names=label_names)
    (out_dir / "classification_report.txt").write_text(txt, encoding="utf-8")

    # --- 混淆矩阵 ---
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    _plot_cm(cm, label_names, out_dir / "confusion_matrix.png")
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    _plot_cm(cm_norm, label_names,
             out_dir / "confusion_matrix_normalized.png", fmt=".2f")

    # --- Top 20 最置信误判 ---
    if y_prob is not None:
        wrong = np.where(np.asarray(y_true) != y_pred)[0]
        if len(wrong) > 0:
            top = wrong[np.argsort(-y_prob[wrong].max(axis=1))[:20]]
            with (out_dir / "top_errors.jsonl").open("w", encoding="utf-8") as f:
                for idx in top:
                    rec = {
                        "source": srcs[idx],
                        "text": texts[idx][:120],
                        "true": LABELS[y_true[idx]],
                        "pred": LABELS[int(y_pred[idx])],
                        "pred_conf": float(y_prob[idx].max()),
                        "true_conf": float(y_prob[idx, y_true[idx]]),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return metrics


def _plot_cm(cm: np.ndarray, labels: list[str], out: Path, fmt: str = "d") -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           xlabel="预测类别", ylabel="真实类别")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = format(cm[i, j], fmt)
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 主流程


def run(args: argparse.Namespace) -> None:
    out_dir = OUT_ROOT / f"tfidf_cls_{dt.datetime.now():%Y%m%d_%H%M%S}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    tr_X, tr_y, tr_s = load_split(SPLIT_DIR / "cls_train.jsonl")
    va_X, va_y, va_s = load_split(SPLIT_DIR / "cls_val.jsonl")
    te_X, te_y, te_s = load_split(SPLIT_DIR / "cls_test.jsonl")
    print(f"[data] train={len(tr_X)} val={len(va_X)} test={len(te_X)}")

    pipeline = build_pipeline(args.model, args.max_features, args.min_df, args.fusion)
    print(f"[model] {args.model}  max_features={args.max_features}  fusion={args.fusion}")

    # 训练
    t0 = time.time()
    pipeline.fit(tr_X, tr_y)
    fit_t = time.time() - t0
    print(f"[fit] 用时 {fit_t:.1f}s")

    # 保存实验参数（带上 fit 时间）
    cfg = vars(args).copy()
    cfg["fit_time_sec"] = round(fit_t, 3)
    cfg["out_dir"] = str(out_dir)
    (out_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    va_pred = pipeline.predict(va_X).tolist()
    va_prob = None
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        va_prob = pipeline.predict_proba(va_X)

    va_metrics = evaluate(va_y, np.asarray(va_pred), va_prob, va_X, va_s,
                          out_dir / "val")
    print(f"[val]  macro_F1={va_metrics['macro_f1']:.4f}  "
          f"weighted_F1={va_metrics['weighted_f1']:.4f}  "
          f"acc={va_metrics['accuracy']:.4f}")

    # 测试集评估
    te_pred = pipeline.predict(te_X).tolist()
    te_prob = None
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        te_prob = pipeline.predict_proba(te_X)

    te_metrics = evaluate(te_y, np.asarray(te_pred), te_prob, te_X, te_s,
                          out_dir / "test")
    print(f"[test] macro_F1={te_metrics['macro_f1']:.4f}  "
          f"weighted_F1={te_metrics['weighted_f1']:.4f}  "
          f"acc={te_metrics['accuracy']:.4f}")

    # 保存模型
    with (out_dir / "pipeline.pkl").open("wb") as f:
        pickle.dump(pipeline, f)

    # 汇总对比
    summary = {
        "model": args.model, "fusion": args.fusion,
        "fit_time_sec": fit_t,
        "val_macro_f1": va_metrics["macro_f1"],
        "val_weighted_f1": va_metrics["weighted_f1"],
        "test_macro_f1": te_metrics["macro_f1"],
        "test_weighted_f1": te_metrics["weighted_f1"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="logreg_fusion",
                   choices=["logreg_fusion", "logreg_word", "logreg_char",
                            "linear_svc_fusion", "linear_svc_word", "linear_svc_char"])
    p.add_argument("--max-features", type=int, default=80_000)
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--fusion", action="store_true",
                   help="同时启用 word 和 char 两路 TF-IDF 拼接")
    args = p.parse_args()
    # 便捷：model 名自带 fusion 后缀时自动开启
    args.fusion = args.fusion or "_fusion" in args.model
    run(args)


if __name__ == "__main__":
    main()
