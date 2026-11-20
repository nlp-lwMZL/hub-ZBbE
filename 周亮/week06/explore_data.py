"""
CNVD 漏洞描述 severity 三分类 — 探索性数据分析（EDA）

仿照课程 src/explore_data.py，适配本作业的 jsonl 数据（高/中/低 → 0/1/2），
并新增「数据量消融模拟」一节，为后续不同训练规模的消融实验做准备。

教学重点：
  1. 训练前为什么要做 EDA：
     类别不均衡（中:低 ≈ 6.2x）→ 类别权重 / 采样策略消融的依据；
     长度分布 → max_length 截断策略的依据（字符数 ≠ token 数）
  2. 跨集合一致性：train/val/test 三份共享自然分布（≈55.3/35.8/8.9）
     是数据方案的设计承诺，分析时要「验证」而不是「假设」
  3. 数据量消融的嵌套采样：同一 seed 对 train_pool 做一次随机置换后取前缀，
     train_N = train_5k + 前缀(N-5000)。前缀性质保证任意两个规模严格嵌套
     （train_5k ⊂ train_10k ⊂ train_20k …），且跨运行可复现；
     这样「5k vs 10k vs 20k」的差异才纯粹来自数据量，不含抽样噪声
  4. 低类是瓶颈：仅占 8.9%，训练规模 N 直接决定低类绝对样本数，
     进而决定模型能否学会该类（低类 recall 是本任务的观测重点）

使用方式：
  python explore_data.py                          # 纯分析（不写任何数据文件）
  python explore_data.py --export                 # 分析 + 导出嵌套消融子集到 data/ablation/
  python explore_data.py --ablation_sizes 15000 30000 --export
  python explore_data.py --skip_token             # 跳过 token 分析（无需 tokenizer）

依赖：
  pip install matplotlib seaborn transformers
  token 分析默认探测本地 bert-base-chinese（词表 vocab_size=21128，与训练用
  ModelScope iic/nlp_roberta_backbone_base_std 一致，已对照其 config.json）；
  本地找不到时经 huggingface（或 hf-mirror 镜像）下载。
"""

import os
import json
import random
import hashlib
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

import urllib.request

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 只落盘不弹窗，Colab/无头环境同样可用
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

matplotlib.rcParams["axes.unicode_minus"] = False


# ── 字体：优先用系统中文字体，找不到就用英文 ─────────────────────────────────
def _find_chinese_font():
    # matplotlib 认的是字体文件「内部」的家族名（family name），不是文件名。
    # 例如 macOS 的 PingFang.ttc 内部含 SC/HK/TC/MO 多个地区版本，
    # matplotlib 扫描 .ttc 时只注册其中一个，实测注册名可能是 "PingFang HK"。
    candidates = [
        # macOS
        "PingFang SC", "PingFang HK", "PingFang TC",
        "STHeiti", "Heiti TC", "Songti SC", "Kaiti SC",
        "Hiragino Sans GB", "Yuanti SC", "Lantinghei SC",
        # Windows
        "Microsoft YaHei", "SimHei",
        # Linux
        "Noto Sans CJK SC", "WenQuanYi Micro Hei",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


_CN_FONT = _find_chinese_font()
if _CN_FONT:
    plt.rcParams["font.family"] = _CN_FONT
    print(f"[信息] 图表使用中文字体: {_CN_FONT}")
else:
    print("[警告] 未找到中文字体，图表标签将显示为英文或方块字")
    print('        排查可用: python -c "import matplotlib.font_manager as fm; '
          'print(sorted({f.name for f in fm.fontManager.ttflist}))"')


HERE = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_lines(path: Path) -> int:
    """只数行数、不解析（用于 all.jsonl 这类大存档文件的快速校验）"""
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_data(data_dir: Path):
    """加载三划分 + pool + 元信息；pool 缺失时返回 None（消融模拟自动跳过）"""
    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        name2id = json.load(f)                       # {"高": 0, "中": 1, "低": 2}
    id2name = {v: k for k, v in name2id.items()}     # {0: "高", 1: "中", 2: "低"}

    manifest = None
    if (data_dir / "manifest.json").exists():
        with open(data_dir / "manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)

    splits = {
        "train": load_jsonl(data_dir / "train_5k.jsonl"),
        "val":   load_jsonl(data_dir / "val_2k.jsonl"),
        "test":  load_jsonl(data_dir / "test_3k.jsonl"),
    }
    pool = None
    if (data_dir / "train_pool.jsonl").exists():
        pool = load_jsonl(data_dir / "train_pool.jsonl")
    return splits, pool, name2id, id2name, manifest


# ─────────────────────────────────────────────────────────────────────────────
# 1. 跨集合一致性总览（本数据方案的设计承诺：三份共享自然分布）
# ─────────────────────────────────────────────────────────────────────────────

def cross_split_overview(splits, pool, id2name, manifest, data_dir):
    print(f"\n{'='*74}")
    print("【总览】各集合规模 / 类别分布 / 文本长度（跨集合一致性检查）")
    print(f"{'='*74}")

    header = f"  {'集合':<8}{'条数':>8}  " \
             + "".join(f"{id2name[i]:>12}" for i in sorted(id2name)) \
             + f"  {'低类占比':>8}{'平均字长':>9}{'中位字长':>9}"
    print(header)

    table = {}
    sets = dict(splits)
    if pool is not None:
        sets["pool"] = pool
    for name, rows in sets.items():
        dist = Counter(r["label_id"] for r in rows)
        lens = np.array([len(r["text"]) for r in rows])
        low_pct = dist.get(2, 0) / len(rows) * 100
        print(f"  {name:<8}{len(rows):>8,}  "
              + "".join(f"{dist.get(i, 0):>12,}" for i in sorted(id2name))
              + f"  {low_pct:>7.2f}%{lens.mean():>9.1f}{np.median(lens):>9.1f}")
        table[name] = {
            "rows": len(rows),
            "dist": {id2name[i]: int(dist.get(i, 0)) for i in sorted(id2name)},
            "low_pct": round(low_pct, 4),
            "char_len_mean": round(float(lens.mean()), 2),
            "char_len_median": round(float(np.median(lens)), 2),
        }

    # all.jsonl 只做行数校验（52MB 存档，不重复解析）
    all_path = data_dir / "all.jsonl"
    if all_path.exists():
        n = count_lines(all_path)
        expected = sum(t["rows"] for t in table.values())
        ok = "✓" if n == expected else "✗ 行数不符！"
        print(f"\n  all.jsonl 行数 {n:,} vs 三划分+pool 合计 {expected:,}  {ok}")
        table["all_rows"] = n

    if manifest:
        ratios = manifest.get("exact_ratios", {})
        ratio_str = " / ".join(
            f"{k} {v*100:.2f}%" for k, v in sorted(ratios.items(), key=lambda x: -x[1]))
        print(f"  manifest 精确分层比例: {ratio_str}")
        print("  校验: 上表三份低类占比应全部贴近 8.90%（分层抽样无漂移）")
    return table


# ─────────────────────────────────────────────────────────────────────────────
# 2. 类别分布分析（三份合一图，直接对比一致性）
# ─────────────────────────────────────────────────────────────────────────────

def analyze_label_distribution(splits, id2name, output_dir: Path):
    lids = sorted(id2name)
    names = [id2name[i] for i in lids]

    summary = {}
    for split_name, data in splits.items():
        counts = Counter(r["label_id"] for r in data)
        max_cnt = max(counts.values())
        print(f"\n{'='*50}")
        print(f"【{split_name}】类别分布（共 {len(data)} 条）")
        print(f"{'='*50}")
        for lid in lids:
            cnt = counts.get(lid, 0)
            bar = "█" * (cnt * 30 // max_cnt)
            print(f"  {lid:2d} {id2name[lid]:<2s} | {bar:<30s} {cnt:6d} ({cnt/len(data)*100:.1f}%)")
        lo, hi = min(counts.values()), max(counts.values())
        print(f"\n  不均衡比 (max/min): {hi/lo:.1f}x"
              f"（最多类 {hi} 条 vs 最少类 {lo} 条 → 类别权重消融的动机）")
        summary[split_name] = {id2name[l]: int(counts.get(l, 0)) for l in lids}

    # 一张图三个子图：train / val / test 分布对比（占比刻度，看"是否同形"）
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = sns.color_palette("husl", len(names))
    for ax, (split_name, data) in zip(axes, splits.items()):
        counts = Counter(r["label_id"] for r in data)
        cnts = [counts.get(l, 0) for l in lids]
        total = sum(cnts)
        pcts = [c / total * 100 for c in cnts]
        bars = ax.bar(names, pcts, color=colors, edgecolor="white", linewidth=0.5)
        for bar, c, p in zip(bars, cnts, pcts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{c}\n({p:.1f}%)", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{split_name}（N={total}）", fontsize=12)
        ax.set_xlabel("severity")
        ax.set_ylabel("占比 (%)")
        ax.set_ylim(0, max(pcts) * 1.35)
    fig.suptitle("类别分布 — train / val / test 共享自然分布（低类 ≈ 8.9%）", fontsize=13)
    plt.tight_layout()
    save_path = output_dir / "label_dist_splits.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  图表已保存 → {save_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 3. 文本长度分析（字符级）
# ─────────────────────────────────────────────────────────────────────────────

def analyze_char_length(data, split_name, output_dir: Path):
    lengths = np.array([len(r["text"]) for r in data])

    print(f"\n{'='*50}")
    print(f"【{split_name}】文本长度统计（字符数）")
    print(f"{'='*50}")
    stats = {
        "均值":   np.mean(lengths),
        "中位数": np.median(lengths),
        "最短":   np.min(lengths),
        "最长":   np.max(lengths),
        "P75":    np.percentile(lengths, 75),
        "P90":    np.percentile(lengths, 90),
        "P95":    np.percentile(lengths, 95),
        "P99":    np.percentile(lengths, 99),
    }
    for name, val in stats.items():
        print(f"  {name:6s}: {val:.1f}")

    # 截断阈值覆盖情况（192 为本作业主跑 max_length，256 为消融档）
    print("\n  截断损失评估（长度 > 阈值的样本会被截断）：")
    for threshold in [64, 128, 192, 256]:
        pct = (lengths > threshold).mean() * 100
        tag = " ← 老师代码默认值，对本数据严重不足" if threshold == 64 else (
              " ← 本作业主跑配置" if threshold == 192 else
              " ← 本作业消融配置" if threshold == 256 else "")
        print(f"    > {threshold:3d}: {pct:6.2f}%  {tag}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(lengths, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.4)
    for t, ls in [(64, ":"), (128, "--"), (192, "-"), (256, "-.")]:
        axes[0].axvline(t, color="red", linestyle=ls, alpha=0.7, label=f"len={t}")
    axes[0].set_title(f"{split_name} — 字符长度分布")
    axes[0].set_xlabel("字符数")
    axes[0].set_ylabel("样本数")
    axes[0].legend(fontsize=8)

    thresholds = np.arange(1, min(400, int(lengths.max()) + 1))
    coverage = [(lengths <= t).mean() * 100 for t in thresholds]
    axes[1].plot(thresholds, coverage, color="#C44E52")
    axes[1].axhline(95, color="gray", linestyle="--", label="95% 覆盖率")
    axes[1].axhline(99, color="gray", linestyle=":", label="99% 覆盖率")
    for target_pct in [95, 99]:
        idx = next((i for i, c in enumerate(coverage) if c >= target_pct), None)
        if idx is not None:
            axes[1].axvline(thresholds[idx], color="#C44E52", linestyle="--", alpha=0.5)
            axes[1].text(thresholds[idx] + 3, target_pct - 4,
                         f"{thresholds[idx]} 字\n覆盖 {target_pct}%", fontsize=8)
    axes[1].set_title(f"{split_name} — 截断长度 vs 覆盖率")
    axes[1].set_xlabel("截断长度（字符数）")
    axes[1].set_ylabel("样本覆盖率 (%)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    save_path = output_dir / f"char_length_{split_name}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  图表已保存 → {save_path}")

    return {"mean": float(np.mean(lengths)), "median": float(np.median(lengths)),
            "p95": float(np.percentile(lengths, 95))}


# ─────────────────────────────────────────────────────────────────────────────
# 4. 各类别长度对比（不同 severity 的描述长度是否系统性不同）
# ─────────────────────────────────────────────────────────────────────────────

def analyze_length_by_label(data, id2name, split_name, output_dir: Path):
    label_lengths = {}
    for r in data:
        label_lengths.setdefault(r["label_id"], []).append(len(r["text"]))

    lids = sorted(label_lengths)
    names = [id2name[l] for l in lids]
    all_lengths = [label_lengths[l] for l in lids]
    means = [np.mean(l) for l in all_lengths]

    print(f"\n{'='*50}")
    print(f"【{split_name}】各类别平均字符长度")
    print(f"{'='*50}")
    for lid, name, mean_l, ls in zip(lids, names, means, all_lengths):
        print(f"  {lid:2d} {name:<2s}: 均值 {mean_l:6.1f} 字，中位 {np.median(ls):6.1f} 字")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].boxplot(all_lengths, tick_labels=names, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#AEC6CF", alpha=0.7),
                    showfliers=False)
    axes[0].set_title(f"{split_name} — 各类别长度箱线图（不画离群点）")
    axes[0].set_ylabel("字符数")

    colors = sns.color_palette("pastel", len(names))
    axes[1].bar(names, means, color=colors, edgecolor="gray", linewidth=0.5)
    axes[1].set_title(f"{split_name} — 各类别平均长度")
    axes[1].set_ylabel("平均字符数")

    plt.tight_layout()
    save_path = output_dir / f"length_by_label_{split_name}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  图表已保存 → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Token 长度分析（BERT WordPiece：字符数 ≠ token 数）
# ─────────────────────────────────────────────────────────────────────────────

def analyze_token_length(data, tokenizer, split_name, output_dir: Path):
    """
    实际训练时 max_length 截断的是 token 数（[CLS]/[SEP] 各占 1），
    中文 BERT 基本一字一 token，但英文产品名/版本号会被拆成多个 subword。
    词表说明：vocab_size=21128，与 ModelScope iic/nlp_roberta_backbone_base_std
    相同（已对照其 config.json），因此本统计对实际训练模型有效。
    """
    texts = [r["text"] for r in data]
    encoded = tokenizer(texts, add_special_tokens=False, truncation=False, padding=False)
    token_lengths = np.array([len(ids) for ids in encoded["input_ids"]])

    print(f"\n{'='*50}")
    print(f"【{split_name}】Token 长度统计（BERT WordPiece，不含特殊符号）")
    print(f"{'='*50}")
    for name, val in {"均值": np.mean(token_lengths), "中位数": np.median(token_lengths),
                      "最长": np.max(token_lengths), "P90": np.percentile(token_lengths, 90),
                      "P95": np.percentile(token_lengths, 95),
                      "P99": np.percentile(token_lengths, 99)}.items():
        print(f"  {name:6s}: {val:.1f}")

    print("\n  含 [CLS]/[SEP] 后的截断比例（+2 token）：")
    trunc = {}
    for max_len in [64, 128, 192, 256]:
        exceeded = ((token_lengths + 2) > max_len).mean() * 100
        trunc[max_len] = round(exceeded, 2)
        note = "（本作业主跑）" if max_len == 192 else ""
        print(f"    max_length={max_len:3d}: {exceeded:6.2f}%  {note}")

    char_lengths = np.array([len(t) for t in texts])
    ratio = token_lengths / np.maximum(char_lengths, 1)
    print(f"\n  Token/字符 比值: 均值={ratio.mean():.3f}，"
          f"P5={np.percentile(ratio, 5):.3f}，P95={np.percentile(ratio, 95):.3f}")
    print("  （>1 的部分来自英文产品名/版本号的 subword 拆分）")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(token_lengths, bins=60, color="#55A868", edgecolor="white", linewidth=0.4,
            label="Token 长度")
    ax.hist(char_lengths, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.4,
            alpha=0.5, label="字符长度")
    for t in [64, 128, 192, 256]:
        ax.axvline(t, color="red", linestyle="--", alpha=0.7, label=f"max_len={t}")
    ax.set_title(f"{split_name} — 字符数 vs Token 数分布对比")
    ax.set_xlabel("长度")
    ax.set_ylabel("样本数")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_path = output_dir / f"token_length_{split_name}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  图表已保存 → {save_path}")

    return {"mean": float(np.mean(token_lengths)),
            "median": float(np.median(token_lengths)),
            "p95": float(np.percentile(token_lengths, 95)),
            "truncation_pct": trunc}


def resolve_tokenizer(explicit):
    """tokenizer 解析顺序：显式路径 → 本地 bert-base-chinese → HF/镜像下载。
    HF_ENDPOINT 必须在 transformers 首次 import 之前设置才生效，故先探测网络。"""
    if explicit:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(explicit)
            print(f"[信息] Token 分析使用指定 tokenizer: {explicit}")
            return tok
        except Exception as e:
            print(f"[跳过 Token 分析] 加载 {explicit} 失败: {e}")
            return None

    # 0) 先探测网络（在 transformers 首次 import 之前决定 endpoint）
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
    except Exception:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    # 1) 本地探测：作业目录 + nlp 根下最多两层（老师项目的 pretrain_models）
    nlp_root = HERE.parent.parent   # homework/week6 → homework → nlp 根
    candidates = [HERE / "pretrain_models" / "bert-base-chinese"]
    candidates += sorted(nlp_root.glob("*/pretrain_models/bert-base-chinese"))
    candidates += sorted(nlp_root.glob("*/*/pretrain_models/bert-base-chinese"))
    for c in candidates:
        if (c / "vocab.txt").exists():
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(str(c))
                print(f"[信息] Token 分析使用本地 tokenizer: {c}")
                return tok
            except Exception:
                continue

    # 2) 在线下载 bert-base-chinese（endpoint 已在上面的探测中决定）
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bert-base-chinese")
        src = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        print(f"[信息] Token 分析经 {src} 下载 bert-base-chinese tokenizer"
              "（词表 21128，与训练用 ModelScope roberta backbone 一致）")
        return tok
    except Exception as e:
        print(f"[跳过 Token 分析] 未找到可用 tokenizer（可用 --tokenizer 指定路径）: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. 各类别样本展示
# ─────────────────────────────────────────────────────────────────────────────

def show_samples(data, id2name, n_per_class=2, max_chars=120):
    print(f"\n{'='*50}")
    print(f"各类别样本示例（每类 {n_per_class} 条，超 {max_chars} 字截断）")
    print(f"{'='*50}")
    by_label = {}
    for r in data:
        by_label.setdefault(r["label_id"], []).append(r)
    for lid in sorted(by_label):
        print(f"\n  [{lid}] {id2name[lid]}")
        for r in by_label[lid][:n_per_class]:
            t = r["text"] if len(r["text"]) <= max_chars else r["text"][:max_chars] + "…"
            print(f"       • {t}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. 数据量消融模拟（本脚本相对老师代码的核心增量）
# ─────────────────────────────────────────────────────────────────────────────

def build_nested_sets(train_base, pool, sizes, seed):
    """
    嵌套采样：train_N = train_base + pool_perm[:N-5000]
    同一 seed 对 pool 做一次随机置换。前缀性质保证：
      N1 < N2 ⇒ train_N1 ⊂ train_N2（严格嵌套，且跨运行可复现）。
    「5k vs 10k vs 20k」的组间差异因此只含数据量这一个变量。
    """
    perm = list(pool)
    random.Random(seed).shuffle(perm)
    nested = {}
    for n in sizes:
        need = n - len(train_base)
        if need > len(perm):
            raise ValueError(f"规模 {n} 超出可用池（最大 {len(train_base) + len(perm)}）")
        nested[n] = list(train_base) + perm[:need]
    return nested


def analyze_ablation_sizes(train_base, pool, sizes, seed, id2name, output_dir: Path):
    lids = sorted(id2name)
    names = [id2name[i] for i in lids]
    base_n = len(train_base)

    pool_dist = Counter(r["label_id"] for r in pool)
    pool_low_pct = pool_dist.get(2, 0) / len(pool) * 100

    print(f"\n{'='*78}")
    print(f"【数据量消融模拟】嵌套采样：train_N = train_5k + pool 置换前缀 (N-5000)，seed={seed}")
    print(f"{'='*78}")
    print(f"  池分布（{len(pool):,} 条）：" + "，".join(
        f"{id2name[i]} {pool_dist.get(i, 0):,}（{pool_dist.get(i, 0)/len(pool)*100:.2f}%）"
        for i in lids))
    print(f"  期望：各规模的低类占比围绕池占比 {pool_low_pct:.2f}% 小幅波动（随机抽样噪声）\n")

    # 基准 5k 也放进分析序列，作为曲线起点
    seq = [base_n] + sorted(s for s in sizes if s != base_n)
    nested = build_nested_sets(train_base, pool, seq[1:], seed)
    sets = {base_n: list(train_base), **nested}

    header = (f"  {'规模':>8} " + "".join(f"{n:>12}" for n in names)
              + f" {'低类占比':>10} {'vs池偏差':>10} {'平均字长':>9} {'嵌套':>6}")
    print(header)

    records = []
    prev_ids = None
    for n in seq:
        rows = sets[n]
        dist = Counter(r["label_id"] for r in rows)
        low_pct = dist.get(2, 0) / n * 100
        mean_len = np.mean([len(r["text"]) for r in rows])
        dev = low_pct - pool_low_pct
        cur_ids = {r["id"] for r in rows}
        if prev_ids is None:
            nested_ok = "基准"
            nested_bool = True
        else:
            nested_bool = prev_ids <= cur_ids
            nested_ok = "✓" if nested_bool else "✗"
        prev_ids = cur_ids
        print(f"  {n:>8,} "
              + "".join(f"{dist.get(i, 0):>12,}" for i in lids)
              + f" {low_pct:>9.2f}% {dev:>+9.2f}p {mean_len:>9.1f} {nested_ok:>6}")
        records.append({
            "N": n,
            "dist": {id2name[i]: int(dist.get(i, 0)) for i in lids},
            "low_pct": round(low_pct, 4),
            "dev_vs_pool_pp": round(dev, 4),
            "low_count": int(dist.get(2, 0)),
            "char_len_mean": round(float(mean_len), 2),
            "nested": nested_bool,
        })

    max_dev = max(abs(r["dev_vs_pool_pp"]) for r in records)
    print("\n  解读：")
    print(f"  · 低类占比偏差最大 {max_dev:.2f}p（相对池分布）——同 seed 下这是固定属性，"
          "±1p 内的构成差异远小于数据量本身的影响，对比依然有效")
    print(f"  · 低类绝对数从 5k 的 {records[0]['low_count']} 条随规模近似线性增长"
          " → 数据量 ×10 ≈ 低类样本 ×10，这是低类 recall 提升的主要来源")
    print("  · 平均字长各规模应稳定（同一池子的随机前缀，分布不变）")

    # 双面板图：占比稳定性 + 各类计数增长
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xs = [r["N"] for r in records]
    colors = sns.color_palette("husl", len(names))

    for i, name in enumerate(names):
        ys = [r["dist"][name] / r["N"] * 100 for r in records]
        axes[0].plot(xs, ys, "o-", color=colors[i], label=f"{name} 占比")
        axes[0].axhline(pool_dist.get(lids[i], 0) / len(pool) * 100,
                        color=colors[i], linestyle=":", alpha=0.6)
        yc = [r["dist"][name] for r in records]
        axes[1].plot(xs, yc, "o-", color=colors[i], label=f"{name} 条数")
    low_color = colors[names.index("低")]
    for i, r in enumerate(records):
        # 首/尾点向内对齐，避免标注被坐标轴边缘裁切
        ha = "left" if i == 0 else "right" if i == len(records) - 1 else "center"
        axes[1].annotate(f"低 {r['low_count']}", (r["N"], r["dist"]["低"]),
                         textcoords="offset points",
                         xytext=(6 if ha == "left" else -6 if ha == "right" else 0, 9),
                         ha=ha, fontsize=8, color=low_color)
    for ax, title in [(axes[0], "各规模类别占比 vs 池分布（虚线为池占比）"),
                      (axes[1], "各规模类别条数（低类是瓶颈）")]:
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x//1000}k" for x in xs])
        ax.set_xlabel("训练规模（嵌套增长）")
        ax.set_title(title)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("占比 (%)")
    axes[1].set_ylabel("条数")
    plt.tight_layout()
    save_path = output_dir / "ablation_sizes_comparison.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  图表已保存 → {save_path}")

    return {"rule": "train_N = train_5k + shuffle(train_pool, seed)[:N-5000]",
            "seed": seed,
            "pool_dist": {id2name[i]: int(pool_dist.get(i, 0)) for i in lids},
            "pool_low_pct": round(pool_low_pct, 4),
            "sizes": records}


# ─────────────────────────────────────────────────────────────────────────────
# 8. 导出消融子集（可选：--export）
# ─────────────────────────────────────────────────────────────────────────────

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_ablation_sets(nested, out_dir: Path, base_file: Path, val_ids, test_ids,
                         name2id, seed):
    """写出 data/ablation/train_{N}k.jsonl + manifest；写完读回校验。
    嵌套链含基准：train_5k ⊂ train_10k ⊂ …（嵌套是构造出来的，但必须读回验证）"""
    out_dir.mkdir(parents=True, exist_ok=True)
    base_rows = load_jsonl(base_file)
    base_ids = {r["id"] for r in base_rows}

    exported = []
    for n, rows in sorted(nested.items()):
        fname = f"train_{n//1000}k.jsonl" if n % 1000 == 0 else f"train_{n}.jsonl"
        path = out_dir / fname
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                o = dict(r)
                o["split"] = "train"           # 导出后身份变为训练集
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        exported.append({"N": n, "file": str(path), "sha256": sha256_of(path)})
        print(f"  导出 {fname}（{n:,} 行）")

    # ── 读回校验（不信构造、只信读回）──
    chain_ids = {len(base_rows): base_ids}
    checks = {}
    ok_all = True
    for item in exported:
        n, path = item["N"], Path(item["file"])
        back = load_jsonl(path)
        ids = {r["id"] for r in back}
        c = {
            "rows": len(back) == n,
            "ids_unique": len(ids) == len(back),
            "no_overlap_val": not (ids & val_ids),
            "no_overlap_test": not (ids & test_ids),
            "label_id_ok": all(r["label_id"] == name2id[r["label"]] for r in back),
        }
        chain_ids[n] = ids
        checks[str(n)] = c
        ok_all &= all(c.values())
    chain = sorted(chain_ids)
    nested_ok = all(chain_ids[chain[i]] <= chain_ids[chain[i+1]] for i in range(len(chain)-1))
    checks["nested_chain_incl_base"] = nested_ok
    ok_all &= nested_ok

    manifest = {
        "seed": seed,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rule": "train_N = train_5k + shuffle(train_pool, seed)前缀 (N-5000) 条；"
                "同 seed 下任意规模的导出互相嵌套、跨运行逐字节一致",
        "base": "data/train_5k.jsonl（原样作为 N=5000 点，不重复导出）",
        "files": exported,
        "checks": checks,
        "all_passed": ok_all,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    status = "全部通过 ✓" if ok_all else "存在失败项 ✗"
    print(f"\n  读回校验（行数/ID唯一/与val,test无交集/label_id/嵌套链）: {status}")
    print(f"  清单 → {out_dir / 'manifest.json'}")
    return {"manifest": manifest, "all_passed": ok_all}


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="CNVD severity 数据探索性分析（含数据量消融模拟与可选导出）")
    parser.add_argument("--data_dir", default=HERE / "data", type=Path)
    parser.add_argument("--output_dir", default=HERE / "outputs" / "figures", type=Path)
    parser.add_argument("--tokenizer", default=None,
                        help="token 分析用 tokenizer（本地路径或 hub 名）；默认自动探测/下载")
    parser.add_argument("--skip_token", action="store_true",
                        help="跳过 token 长度分析（无需 tokenizer）")
    parser.add_argument("--ablation_sizes", nargs="+", type=int,
                        default=[10000, 20000, 50000],
                        help="数据量消融规模（自动并入基准 5000 作起点）；"
                             "上限 = 5000 + train_pool 总量")
    parser.add_argument("--seed", type=int, default=42,
                        help="嵌套采样 seed（须与 data/manifest.json 的 seed 一致）")
    parser.add_argument("--export", action="store_true",
                        help="把消融子集导出为 data/ablation/train_Nk.jsonl（默认只分析不写数据文件）")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits, pool, name2id, id2name, manifest = load_data(args.data_dir)
    pool_desc = f"{len(pool):,}" if pool is not None else "缺失"
    print(f"数据加载完成：train={len(splits['train'])}, val={len(splits['val'])}, "
          f"test={len(splits['test'])}, pool={pool_desc}")

    summary = {"generated_at": datetime.now().isoformat(timespec="seconds"),
               "data_dir": str(args.data_dir), "seed": args.seed}

    # 1. 跨集合一致性总览
    summary["overview"] = cross_split_overview(splits, pool, id2name, manifest, args.data_dir)

    # 2. 类别分布（三份合一图）
    summary["label_dist"] = analyze_label_distribution(splits, id2name, args.output_dir)

    # 3. 字符长度（train 为准；其余集合长度已在总览表）
    summary["char_length_train"] = analyze_char_length(splits["train"], "train", args.output_dir)

    # 4. 各类别长度对比
    analyze_length_by_label(splits["train"], id2name, "train", args.output_dir)

    # 5. Token 长度（需要 tokenizer；加载失败自动跳过）
    if not args.skip_token:
        tok = resolve_tokenizer(args.tokenizer)
        if tok is not None:
            summary["token_length_train"] = analyze_token_length(
                splits["train"], tok, "train", args.output_dir)

    # 6. 样本展示
    show_samples(splits["train"], id2name, n_per_class=2)

    # 7. 数据量消融模拟 + 8. 可选导出（pool 缺失则跳过）
    if pool is None:
        print("\n[跳过消融模拟] 未找到 train_pool.jsonl")
    else:
        base_n = len(splits["train"])
        max_n = base_n + len(pool)
        oversize = [s for s in args.ablation_sizes if s > max_n]
        if oversize:
            print(f"\n[错误] 规模 {oversize} 超出上限 {max_n}（train_5k + 全部 pool），已退出")
            return
        sizes = sorted({s for s in args.ablation_sizes if s > base_n})
        ignored = [s for s in args.ablation_sizes if s <= base_n]
        if ignored:
            print(f"\n[提示] 规模 {ignored} 不大于基准 {base_n}，已忽略（基准自动作为起点）")

        summary["ablation"] = analyze_ablation_sizes(
            splits["train"], pool, sizes, args.seed, id2name, args.output_dir)

        if args.export:
            if not sizes:
                print("\n[跳过导出] 无大于基准的规模")
            else:
                print(f"\n{'='*78}")
                print(f"【导出消融子集】→ {args.data_dir / 'ablation'}")
                print(f"{'='*78}")
                nested = build_nested_sets(splits["train"], pool, sizes, args.seed)
                val_ids = {r["id"] for r in splits["val"]}
                test_ids = {r["id"] for r in splits["test"]}
                result = export_ablation_sets(
                    nested, args.data_dir / "ablation",
                    args.data_dir / "train_5k.jsonl", val_ids, test_ids,
                    name2id, args.seed)
                summary["export"] = {
                    "files": [f["file"] for f in result["manifest"]["files"]],
                    "all_passed": result["all_passed"],
                }

    # 机器可读汇总（写报告直接引用）
    summary_path = args.output_dir.parent / "explore_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n所有图表已保存至 {args.output_dir}")
    print(f"机器可读汇总 → {summary_path}")


if __name__ == "__main__":
    main()
