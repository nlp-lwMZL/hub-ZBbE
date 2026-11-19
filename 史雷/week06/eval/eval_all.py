"""
统一评测汇总：把各路方法的结果拉到一张表上

教学重点：
  1. 四路方法必须跑在**同一批样本**上（common/data.py 的 clean_val + 固定 seed），
     否则数字不可比。这个脚本先检查每份结果的 meta（n_eval / seed），不一致就报警。
  2. 只报 Accuracy 会被头部类骗，所以同时给 Macro-F1 和「预测覆盖类别数」——
     后者能一眼看出 zero-shot 的塌缩。
  3. 汇总结果写到 outputs/summary.md，可直接粘进 ARCHITECTURE.md。

用法：
  python eval/eval_all.py                 # 汇总已有结果（先跑各路脚本）
  python eval/eval_all.py --run-light     # 先跑规则 + TF-IDF，再汇总
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

OUT = CODE_ROOT / "outputs"

# 方法名 → 结果文件名（跑过哪一路就读哪一份）
SOURCES = [
    ("规则 / 关键词", "rule_pred.json"),
    ("TF-IDF + SVM", "tfidf_pred.json"),
    ("BERT 微调", "bert_pred.json"),
    ("Qwen3 zero-shot", "zero_shot_pred.json"),
    ("Qwen3 LoRA SFT", "lora_pred.json"),
    ("Qwen3 全量微调", "full_ft_pred.json"),
    ("API zero-shot（deepseek-v4-flash）", "api_zero_shot_pred.json"),
]

# bert-large-chinese 对比实验（笔记本 MPS：训练集 1 万条 × 2 epoch）
SOURCES_LARGE = [
    ("规则 / 关键词", "rule_pred.json"),
    ("TF-IDF + SVM（全量 26.6 万训练）", "tfidf_pred_full.json"),
    ("TF-IDF + SVM（1 万训练，与 BERT 对齐）", "tfidf_pred_10k.json"),
    ("BERT-large 微调 · cls 池化", "bert_pred_large_cls.json"),
    ("BERT-large 微调 · mean 池化", "bert_pred_large_mean.json"),
    ("BERT-large 微调 · max 池化", "bert_pred_large_max.json"),
    ("BERT-large 微调 · cls · 不加类别权重", "bert_pred_large_cls_nocw.json"),
]


def load(name):
    p = OUT / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] {name} 解析失败：{e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-light", action="store_true", help="先跑规则与 TF-IDF（秒级）")
    ap.add_argument("--large", action="store_true", help="汇总 bert-large 对比实验，写入 summary_large.md")
    args = ap.parse_args()
    sources = SOURCES_LARGE if args.large else SOURCES

    if args.run_light:
        for script in ("rule/classify_rule.py", "tfidf/classify_tfidf.py"):
            print(f"→ 运行 {script}")
            subprocess.run([sys.executable, str(CODE_ROOT / script), "--n", "500"], check=False, cwd=CODE_ROOT)

    rows, metas = [], {}
    for label, fname in sources:
        data = load(fname)
        if not data:
            rows.append({"method": label, "acc": "-", "macro_f1": "-", "n": "-", "note": "未运行"})
            continue
        r = data["report"]
        # 老版本结果文件可能没有 acc_incl_unparsed，这里用逐条预测重算（解析失败算答错）
        if "acc_incl_unparsed" not in r and data.get("preds"):
            ps = data["preds"]
            r["acc_incl_unparsed"] = round(sum(1 for x in ps if x["pred"] == x["true"]) / len(ps), 4)
        m = data.get("meta", {})
        metas[label] = m
        dist = r.get("pred_distribution") or {}
        rows.append({
            "method": label,
            "acc": r.get("acc_incl_unparsed", r.get("acc", "-")),
            "macro_f1": r.get("macro_f1", "-"),
            "n": m.get("n_eval", r.get("n")),
            "used_classes": dist.get("used_classes", "-"),
            "unparsed": r.get("unparsed", 0),
            "note": "",
        })

    # 口径检查：样本量/seed 不一致就要提醒，这是「数字不可比」最常见的原因
    seeds = {v.get("seed") for v in metas.values() if v.get("seed") is not None}
    sizes = {v.get("n_eval") for v in metas.values() if v.get("n_eval") is not None}
    if len(sizes) > 1:
        print(f"[warn] 各路样本量不一致：{sizes}，结果不可直接比较")
    if len(seeds) > 1:
        print(f"[warn] 各路 seed 不一致：{seeds}，抽样集合可能不同")

    header = "| 方法 | 准确率(含失败) | Macro-F1 | 样本数 | 覆盖类别 | 解析失败 |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(f"| {r['method']} | {r['acc']} | {r['macro_f1']} | {r['n']} | "
                     f"{r.get('used_classes', '-')} | {r.get('unparsed', '-')} |")
    table = "\n".join(lines)

    print("\n" + table + "\n")
    note = []
    note.append("口径：全部分方法在同一批「去重后的干净验证样本」上评测（seed=42）。")
    if not args.large:
        note.append("参考：全量 val（含 11% 重复）上 TF-IDF 0.541 / BERT 0.551，去重后分别降到 0.509 / 0.534。")
    (OUT / ("summary_large.md" if args.large else "summary.md")).write_text(table + "\n\n" + "\n".join(note) + "\n", encoding="utf-8")
    print(f"→ 汇总写入 {OUT / ('summary_large.md' if args.large else 'summary.md')}")


if __name__ == "__main__":
    main()


