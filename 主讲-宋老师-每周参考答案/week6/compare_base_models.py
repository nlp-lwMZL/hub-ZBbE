# -*- coding: utf-8 -*-
"""
换基座结果对比：把 train_base_model.py 跑出的两份结果放在一起，回答三个问题

  1. 谁更好？（Macro-F1 / Accuracy / 每类 F1）
  2. 差得算不算真差距？（配对 bootstrap 置信区间 + 翻转样本数）
  3. 差距从哪来？（哪些类别提升、哪些类别反而掉了）

为什么必须做「配对」比较：
  两个模型是在**同一批 8,901 条验证样本**上评的，样本顺序也一致。
  所以可以逐条对比——比只比两个汇总数字严谨得多：
  汇总差 0.02 可能只由 200 条样本的翻转造成，那很可能就是随机波动；
  如果 bootstrap 出来的 95% 区间不跨 0，才敢说「换基座真的有用」。

用法：
  python compare_base_models.py                                  # 自动读 results/ 下的 bert / roberta
  python compare_base_models.py --a bert-base-chinese --b chinese-roberta-wwm-ext
  python compare_base_models.py --results_dir outputs/base_models --n_boot 2000

产物：results/compare_base_models.md + results/compare_base_models.csv（并在终端打印）
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

# 默认读「与本脚本同目录的 outputs/base_models」——把脚本放到 code/ 下运行时正好对上
RESULTS_DIR = Path(__file__).resolve().parent / "outputs" / "base_models"

# 类别顺序与 common/labels.py 的 LABELS 严格一致（只用于打印每类名字，不参与计算）
LABELS = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育",
          "科技", "军事", "旅游", "国际", "证券", "农业", "电竞"]


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro", labels=list(range(len(LABELS))), zero_division=0))


def load_pair(results_dir: Path, tag_a: str, tag_b: str):
    def load(tag):
        log_p = results_dir / f"{tag}_log.json"
        pred_p = results_dir / f"{tag}_pred.json"
        if not log_p.exists() or not pred_p.exists():
            raise SystemExit(f"缺少结果文件：{log_p} 或 {pred_p}\n先跑 train_base_model.py --model {tag}")
        return json.loads(log_p.read_text(encoding="utf-8")), json.loads(pred_p.read_text(encoding="utf-8"))
    return load(tag_a), load(tag_b)


def paired_bootstrap(y, pa, pb, n_boot=2000, seed=42):
    """
    配对 bootstrap：每次从 8,901 条里**有放回地**抽同样多条，
    在抽到的样本上重算两个模型的 Macro-F1，得到两者差值的分布。

    为什么不直接用「两个模型的 F1 差」下结论：那个差是**一次抽样**的产物，
    换个验证集可能就是另一个方向。这里给出「差值的 95% 区间」——
    区间整体在 0 之上，才算显著优于。
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = [y[j] for j in idx]
        diffs[i] = macro_f1(yy, [pb[j] for j in idx]) - macro_f1(yy, [pa[j] for j in idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi), float(diffs.mean())


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=str(RESULTS_DIR))
    ap.add_argument("--a", default="bert-base-chinese", help="基线（第一个基座）的产物前缀")
    ap.add_argument("--b", default="chinese-roberta-wwm-ext", help="对比基座的产物前缀")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out_dir", default=str(RESULTS_DIR))
    args = ap.parse_args()

    results_dir, out_dir = Path(args.results_dir), Path(args.out_dir)
    (la, pa), (lb, pb) = load_pair(results_dir, args.a, args.b)

    y, y2 = pa["y"], pb["y"]
    if y != y2:
        raise SystemExit("两份预测的标签序列不一致：不是同一批评测样本，不能配对比较")

    fa, fb = la["best_macro_f1"], lb["best_macro_f1"]
    delta = fb - fa
    lo, hi, mean_diff = paired_bootstrap(y, pa["pred"], pb["pred"], args.n_boot)

    # 逐条对比：两模型都对的样本不用管，真正影响结论的是「你错我对」的那部分
    same = sum(1 for s, u in zip(pa["pred"], pb["pred"]) if s == u)
    only_a = sum(1 for s, u, t in zip(pa["pred"], pb["pred"], y) if s == t and u != t)
    only_b = sum(1 for s, u, t in zip(pa["pred"], pb["pred"], y) if u == t and s != t)

    per_a = la["per_class_f1"]
    per_b = lb["per_class_f1"]
    rows = []
    for name in LABELS:
        rows.append({"类别": name, "A_F1": per_a[name], "B_F1": per_b[name],
                     "差值": round(per_b[name] - per_a[name], 4)})
    rows.sort(key=lambda r: -r["差值"])

    md = []
    md.append("# 换基座对比：BERT vs 中文 RoBERTa\n")
    md.append(f"A = `{la['meta']['base']}`　B = `{lb['meta']['base']}`\n")
    md.append("## 口径（完全一致，只换基座）\n")
    m = la["meta"]
    md.append(f"- 训练集 {m['n_train']:,} 条；评测集 clean val {m['n_clean_val']:,} 条（去掉与训练集重复的样本）")
    md.append(f"- {m['epochs']} epoch，batch {m['batch_size']}，max_length {m['max_length']}，"
              f"{m['pool']} 池化，基座 lr {m['lr']}、分类头 ×{m['head_lr_mult']:g}，seed {m['seed']}")
    md.append("- 两个基座参数量："
              f"{m['params_M']}M vs {lb['meta']['params_M']}M（结构一致，推理成本一样）")
    md.append(f"- 取 3 个 epoch 里 Macro-F1 最好的一轮：A = epoch {la['best_epoch']}，B = epoch {lb['best_epoch']}\n")

    md.append("## 主结果\n")
    md.append("| 基座 | 参数量 | 最优 epoch | Accuracy | Macro-F1 | 训练耗时 | 相对 A |")
    md.append("|---|---|---|---|---|---|---|")
    for tag, l in ((args.a, la), (args.b, lb)):
        tot = sum(e["elapsed_s"] for e in l["epochs_log"])
        md.append(f"| {tag} | {l['meta']['params_M']}M | {l['best_epoch']} | {fmt(l['best_acc'])} | "
                  f"**{fmt(l['best_macro_f1'])}** | {tot:.0f}s（{tot/60:.1f} min） | "
                  f"{'—' if tag == args.a else f'{delta:+.4f}'} |")

    md.append("\n## 逐 epoch（看收敛快慢，不只看最后一轮）\n")
    md.append("| epoch | A Macro-F1 | A 耗时 | B Macro-F1 | B 耗时 |")
    md.append("|---|---|---|---|---|")
    for ea, eb in zip(la["epochs_log"], lb["epochs_log"]):
        md.append(f"| {ea['epoch']} | {fmt(ea['val_macro_f1'])} | {ea['elapsed_s']:.0f}s | "
                  f"{fmt(eb['val_macro_f1'])} | {eb['elapsed_s']:.0f}s |")

    md.append("\n## 这个差距算不算真的？\n")
    verdict = ("差异显著（区间不跨 0）" if lo > 0 or hi < 0 else "差异不显著（区间跨 0）")
    md.append(f"- 配对 bootstrap（{args.n_boot} 次重采样）：ΔMacro-F1 = {delta:+.4f}，"
              f"95% 区间 [{lo:+.4f}, {hi:+.4f}] → **{verdict}**")
    md.append(f"- 逐条对比：两模型预测相同的样本 {same:,}/{len(y):,}（{same/len(y):.1%}）；"
              f"只有 A 对 {only_a:,} 条，只有 B 对 {only_b:,} 条（净变化 {only_b-only_a:+,}）")

    md.append("\n## 哪些类别变了（按 Δ 排序，正= B 更好）\n")
    md.append("| 类别 | A F1 | B F1 | Δ |")
    md.append("|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['类别']} | {fmt(r['A_F1'])} | {fmt(r['B_F1'])} | {r['差值']:+.4f} |")

    text = "\n".join(md) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare_base_models.md").write_text(text, encoding="utf-8")
    with open(out_dir / "compare_base_models.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["基座", "参数量M", "最优epoch", "Accuracy", "Macro-F1",
                                          "训练耗时s", "相对A"])
        w.writeheader()
        for tag, l in ((args.a, la), (args.b, lb)):
            tot = sum(e["elapsed_s"] for e in l["epochs_log"])
            w.writerow({"基座": tag, "参数量M": l["meta"]["params_M"], "最优epoch": l["best_epoch"],
                        "Accuracy": l["best_acc"], "Macro-F1": l["best_macro_f1"],
                        "训练耗时s": round(tot, 1), "相对A": "" if tag == args.a else round(delta, 4)})

    print(text)
    print(f"已写入 → {out_dir / 'compare_base_models.md'} / .csv")


if __name__ == "__main__":
    main()
