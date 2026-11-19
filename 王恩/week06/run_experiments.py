"""统一实验入口：同一份数据、同一套指标，结果可逐条追溯。"""
import argparse
import json
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from baselines import build_baseline
from data import DEFAULT_DATA, ROOT, load_data

BASELINES = ("rule", "tfidf_nb", "tfidf_lr", "tfidf_svm")
NEURAL = ("char_cnn", "encoder_frozen", "encoder_finetune")


def metrics(truth, predictions, labels):
    if len(truth) != len(predictions) or not truth:
        raise ValueError("预测数量必须与非空测试集一致")
    if set(predictions) - set(labels):
        raise ValueError("模型返回了标签集合以外的预测")
    return {"accuracy": accuracy_score(truth, predictions),
            "macro_f1": f1_score(truth, predictions, labels=labels, average="macro", zero_division=0),
            "per_class": classification_report(truth, predictions, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(truth, predictions, labels=labels).tolist()}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(output_dir, reports, meta):
    lines = ["# 12 语种分类对比实验（本次实际运行）", "",
             f"数据指纹：`{meta['fingerprint']}`", "",
             "| 方法 | 状态 | Test Accuracy | Test Macro-F1 | 训练/选择(s) | 测试推理(s) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for method, report in reports.items():
        if report["status"] == "ok":
            lines.append(f"| {method} | 完成 | {report['accuracy']:.4f} | {report['macro_f1']:.4f} | "
                         f"{report['fit_seconds']:.3f} | {report['predict_seconds']:.3f} |")
        else:
            lines.append(f"| {method} | 失败，详见 results.json | — | — | — | — |")
    lines.extend(["", "CNN/Encoder 仅按 Dev Macro-F1 选 epoch；测试集只评测选中的权重。",
                  "训练/选择耗时包含验证和最佳权重复制，不含预训练模型加载；首次推理无预热。",
                  "未运行的方法不生成分数。多随机种子、数据量变化的结果请保存到不同输出目录。", ""])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--methods", nargs="+", choices=BASELINES + NEURAL,
                        default=[*BASELINES, "char_cnn"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-class", type=int, default=0, help="0 表示全部；其他值为每类上限")
    parser.add_argument("--eval-per-class", type=int, default=0, help="同时限制 dev/test 每类上限；仅供快速实验")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--lr", type=float, default=0.001, help="CNN/冻结 encoder 的分类头学习率")
    parser.add_argument("--encoder-lr", type=float, default=2e-5, help="微调 encoder 学习率，分类头为其 5 倍")
    parser.add_argument("--encoder", default="distilbert-base-multilingual-cased", help="HF 模型名或本地目录")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--save-models", action="store_true", help="保存拟合后的模型，默认只保留报告")
    args = parser.parse_args()
    for key in ("epochs", "batch_size", "max_length", "threads", "lr", "encoder_lr"):
        if getattr(args, key) <= 0:
            parser.error(f"{key} 必须为正数")
    if args.train_per_class < 0 or args.eval_per_class < 0:
        parser.error("每类上限不能为负数")
    args.methods = list(dict.fromkeys(args.methods))
    return args


def main():
    args = parse_args()
    splits, labels, meta = load_data(args.data_dir, args.seed, args.train_per_class, args.eval_per_class)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = {"python": sys.version, "platform": platform.platform()}
    for package in ("numpy", "scikit-learn", "torch", "transformers"):
        try:
            environment[package] = version(package)
        except PackageNotFoundError:
            environment[package] = "not installed"
    manifest = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "data": meta, "environment": environment,
                "split_ids": {s: [r["id"] for r in rows] for s, rows in splits.items()}}
    save_json(args.output_dir / "manifest.json", manifest)
    print("数据：", {s: len(rows) for s, rows in splits.items()}, "去重：", meta["removed_duplicates"], flush=True)
    truth, reports = [r["label"] for r in splits["test"]], {}
    for method in args.methods:
        print(f"开始 {method}", flush=True)
        try:
            if method in BASELINES:
                model = build_baseline(method, args.seed)
                started = time.perf_counter()
                model.fit([r["text"] for r in splits["train"]], [r["label"] for r in splits["train"]])
                fit_seconds = time.perf_counter() - started
                started = time.perf_counter()
                predictions = list(model.predict([r["text"] for r in splits["test"]]))
                extra = {"fit_seconds": fit_seconds, "predict_seconds": time.perf_counter() - started}
                if args.save_models:
                    import joblib
                    joblib.dump(model, args.output_dir / f"{method}.joblib")
            else:
                from neural import train_neural
                predictions, extra = train_neural(method, splits, labels, args, args.output_dir)
            reports[method] = {"status": "ok", **metrics(truth, predictions, labels), **extra}
            save_json(args.output_dir / f"{method}_predictions.json", {
                "fingerprint": meta["fingerprint"], "labels": labels,
                "predictions": [{**row, "pred": pred} for row, pred in zip(splits["test"], predictions)]})
            print(f"  test acc={reports[method]['accuracy']:.4f} f1={reports[method]['macro_f1']:.4f}", flush=True)
        except Exception as exc:
            reports[method] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            print(f"  失败：{reports[method]['error']}", file=sys.stderr, flush=True)
        save_json(args.output_dir / "results.json", {"data": meta, "reports": reports})
        write_summary(args.output_dir, reports, meta)
    print(f"结果：{args.output_dir.resolve() / 'summary.md'}")
    return int(any(report["status"] != "ok" for report in reports.values()))


if __name__ == "__main__":
    raise SystemExit(main())
