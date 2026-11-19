"""
离线批推理：公告切片 -> 公告类别。

双模型策略：
  - 主模型：RoBERTa-4L + CE （校准良好，ECE~0.002，置信度可靠）
  - 兜底模型：TF-IDF + LinearSVC （RoBERTa 加载失败 / 单批推理异常 /
    预测置信度低于阈值时自动回退到 TF-IDF）

输入：jsonl，每行 {"text": "...", "slice_id": "..."}  （slice_id 可选，会回传）
输出：jsonl，每行对应一条输入：
  {
    "announcement_content": "公告文本",
    "announcement_category": "定期报告",
    "model": "roberta-4l-ce-v1" | "tfidf-linearsvc-v1",
    "confidence": 0.998,
    "slice_id": "..."              // 输入有则回传
  }

用法：
  python service/fund_announcement_classify.py \
      --input  data/dataset/split/cls_test.jsonl \
      --output output/service/cls_test_pred.jsonl
  # 只用 TF-IDF：
  python service/fund_announcement_classify.py --input in.jsonl --output out.jsonl --primary tfidf
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from common.label import CLS_ID2LABEL, CLS_NUM_LABELS  # noqa: E402

# 模型标识（写入输出的 model 字段）
ROBERTA_TAG = "roberta-4l-ce-v1"
TFIDF_TAG = "tfidf-linearsvc-v1"

# 默认路径
DEFAULT_TFIDF_PKL = BASE_DIR / "output" / "tfidf" / \
    "tfidf_cls_20261118_155427_linear_svc_word" / "pipeline.pkl"
DEFAULT_ROBERTA_CKPT = BASE_DIR / "output" / "roberta" / \
    "roberta_best_s42_ce_cls4_e3.pt"
DEFAULT_ROBERTA_DIR = BASE_DIR / "pretrain_models" / "chinese-roberta-wwm-ext"

# ---------------------------------------------------------------------------
# TF-IDF 反序列化依赖：训练时 pipeline.pkl 把 _jieba_tokenize 绑定在 __main__
# 上，joblib.load 反序列化时需要当前主模块存在同名符号。此处重新定义，
# 实现与 tfidf/train_cls.py 完全一致，保证分词行为相同、特征对齐。
# ---------------------------------------------------------------------------
def _jieba_tokenize(text: str) -> list[str]:
    import jieba
    jieba.setLogLevel(60)
    return [t.strip() for t in jieba.lcut(text, cut_all=False)
            if t.strip() and len(t.strip()) >= 2]


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def load_tfidf(pkl_path: Path):
    import joblib
    if not pkl_path.exists():
        raise FileNotFoundError(f"TF-IDF pipeline 不存在: {pkl_path}")
    return joblib.load(pkl_path)


def load_roberta(ckpt_path: Path, model_dir: Path, max_length: int):
    """加载 RoBERTa 分类器；失败返回 None（由调用方决定是否回退 TF-IDF）。"""
    import torch
    from transformers import AutoTokenizer
    from roberta.train_roberta import BertClassifier
    if not ckpt_path.exists():
        print(f"[warn] RoBERTa ckpt 不存在: {ckpt_path}", file=sys.stderr)
        return None, None
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = BertClassifier(model_dir, num_labels=CLS_NUM_LABELS,
                           pool="cls", dropout=0.1)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        print(f"[warn] RoBERTa load_state_dict missing={missing} "
              f"unexpected={unexpected}", file=sys.stderr)
    model.eval()
    return tok, model


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
def predict_tfidf(pipe, texts: list[str]) -> list[tuple[str, float]]:
    """返回 [(类别名, 置信度), ...]。LinearSVC 无 predict_proba，
    用 softmax(decision_function) 近似置信度（仅供参考，非校准概率）。"""
    import numpy as np
    df = pipe.decision_function(texts)            # (n, k)
    df = np.asarray(df, dtype=float)
    if df.ndim == 1:
        df = df.reshape(1, -1)
    e = np.exp(df - df.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    return [(CLS_ID2LABEL[int(p)], float(probs[i, p])) for i, p in enumerate(preds)]


def predict_roberta(tok, model, texts: list[str], batch_size: int,
                    max_length: int) -> list[tuple[str, float] | None]:
    """返回 [(类别名, 置信度) or None, ...]；某批推理异常时该批返回 None，
    由调用方回退 TF-IDF。"""
    import torch
    out: list = [None] * len(texts)
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            batch = texts[s:s + batch_size]
            try:
                enc = tok(batch, truncation=True, max_length=max_length,
                          padding="max_length", return_tensors="pt")
                logits = model(enc["input_ids"], enc["attention_mask"])
                probs = torch.softmax(logits, dim=-1)
                conf, idx = probs.max(dim=-1)
                for j, (c, i_) in enumerate(zip(conf.tolist(), idx.tolist())):
                    out[s + j] = (CLS_ID2LABEL[int(i_)], float(c))
            except Exception as e:  # noqa: BLE001
                print(f"[warn] RoBERTa 批次 {s}-{s + len(batch)} 推理失败: {e} "
                      f"-> 该批回退 TF-IDF", file=sys.stderr)
                # 该批保持 None，触发回退
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def read_input(path: Path) -> tuple[list[str], list[str | None]]:
    texts: list[str] = []
    slice_ids: list[str | None] = []
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] 第 {ln} 行 JSON 解析失败，已跳过", file=sys.stderr)
                continue
            text = rec.get("text")
            if not text or not str(text).strip():
                print(f"[warn] 第 {ln} 行缺少 text 字段，已跳过", file=sys.stderr)
                continue
            texts.append(str(text))
            slice_ids.append(rec.get("slice_id"))
    return texts, slice_ids


def main() -> None:
    ap = argparse.ArgumentParser(description="公告类别离线批推理（RoBERTa 主 + TF-IDF 兜底）")
    ap.add_argument("--input", required=True, help="输入 jsonl 路径")
    ap.add_argument("--output", required=True, help="输出 jsonl 路径")
    ap.add_argument("--primary", default="roberta", choices=["roberta", "tfidf"],
                    help="主模型；roberta 时 TF-IDF 作为兜底")
    ap.add_argument("--tfidf_pkl", default=str(DEFAULT_TFIDF_PKL))
    ap.add_argument("--roberta_ckpt", default=str(DEFAULT_ROBERTA_CKPT))
    ap.add_argument("--roberta_dir", default=str(DEFAULT_ROBERTA_DIR))
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--roberta_min_conf", type=float, default=0.0,
                    help="RoBERTa 置信度低于此值时回退 TF-IDF；0 表示不启用")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    texts, slice_ids = read_input(in_path)
    if not texts:
        print("[error] 输入为空", file=sys.stderr)
        sys.exit(1)
    print(f"[data] 读取 {len(texts)} 条，主模型={args.primary}")

    t0 = time.time()
    pipe = load_tfidf(Path(args.tfidf_pkl))
    print(f"[load] TF-IDF 就绪 ({time.time() - t0:.1f}s)")

    tok, model = (None, None)
    roberta_ok = False
    if args.primary == "roberta":
        t1 = time.time()
        try:
            tok, model = load_roberta(Path(args.roberta_ckpt),
                                      Path(args.roberta_dir), args.max_length)
            roberta_ok = tok is not None and model is not None
            if roberta_ok:
                print(f"[load] RoBERTa 就绪 ({time.time() - t1:.1f}s)")
            else:
                print("[warn] RoBERTa 不可用，全部回退 TF-IDF", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] RoBERTa 加载失败: {e} -> 全部回退 TF-IDF",
                  file=sys.stderr)

    # 推理
    cats: list[str] = [""] * len(texts)
    confs: list[float] = [0.0] * len(texts)
    models: list[str] = [""] * len(texts)

    if roberta_ok:
        t2 = time.time()
        rb = predict_roberta(tok, model, texts, args.batch_size, args.max_length)
        print(f"[infer] RoBERTa 完成 ({time.time() - t2:.1f}s)")
        fallback_idx = [i for i, r in enumerate(rb)
                        if r is None or
                        (args.roberta_min_conf > 0 and r[1] < args.roberta_min_conf)]
        for i, r in enumerate(rb):
            if r is not None and i not in set(fallback_idx):
                cats[i], confs[i], models[i] = r[0], r[1], ROBERTA_TAG
        if fallback_idx:
            fb_texts = [texts[i] for i in fallback_idx]
            tf = predict_tfidf(pipe, fb_texts)
            for j, i in enumerate(fallback_idx):
                cats[i], confs[i], models[i] = tf[j][0], tf[j][1], TFIDF_TAG
            print(f"[fallback] {len(fallback_idx)} 条回退 TF-IDF")
    else:
        t3 = time.time()
        tf = predict_tfidf(pipe, texts)
        print(f"[infer] TF-IDF 完成 ({time.time() - t3:.1f}s)")
        for i, (c, p) in enumerate(tf):
            cats[i], confs[i], models[i] = c, p, TFIDF_TAG

    # 写出
    n_rob = sum(1 for m in models if m == ROBERTA_TAG)
    n_tf = len(models) - n_rob
    with out_path.open("w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            rec = {
                "announcement_content": text,
                "announcement_category": cats[i],
                "model": models[i],
                "confidence": round(confs[i], 6),
            }
            if slice_ids[i] is not None:
                rec["slice_id"] = slice_ids[i]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[done] 写出 {len(texts)} 条 -> {out_path}  "
          f"(RoBERTa={n_rob}, TF-IDF={n_tf}, 总耗时 {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
