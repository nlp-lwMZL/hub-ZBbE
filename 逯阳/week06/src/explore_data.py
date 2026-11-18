"""
TNEWS 数据集探索性分析

教学重点：
  1. 为什么在训练前要做数据分析（了解类别不均衡、长度分布，指导截断策略）
  2. 字符长度 vs Token 长度的差异（BERT tokenizer 会把字拆成 subword）
  3. 类别不均衡的量化（最多类 vs 最少类的比例）

使用方式：
  python explore_data.py
  python explore_data.py --data_dir ../data --output_dir ../outputs/figures

依赖：
  pip install matplotlib seaborn transformers
"""

import json
import argparse
from pathlib import Path
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np

# ── 路径锚点 ────────────────────────────────────────────────────────────────
# 命令行给的相对路径一律按「脚本所在目录 src/」解析，不跟着当前工作目录跑。
# 这样无论从哪个目录调用本脚本，默认值都能找对地方。
SRC_DIR   = Path(__file__).parent
ROOT      = SRC_DIR.parent          # 项目根：text_classification项目\text_classification项目
DATA_DIR  = ROOT / "data"
FIG_DIR   = ROOT / "outputs" / "figures"
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"


def _resolve(p: Path) -> Path:
    """绝对路径原样返回；相对路径锚定到 src/ 目录。"""
    return p if p.is_absolute() else (SRC_DIR / p).resolve()


matplotlib.rcParams["axes.unicode_minus"] = False

# ── 字体：优先用系统中文字体，找不到就用英文 ──────────────────────────────────
def _find_chinese_font():
    candidates = [
        "SimHei", "Microsoft YaHei", "PingFang SC",
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
else:
    print("[警告] 未找到中文字体，图表标签将显示为英文或方块字")


def load_data(data_dir: Path):
    with open(data_dir / "train.json", encoding="utf-8") as f:
        train = json.load(f)
    with open(data_dir / "val.json", encoding="utf-8") as f:
        val = json.load(f)
    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    # label_map 的 key 来自 JSON，都是字符串；统一转为 int key
    id2name = {int(k): v for k, v in label_map["id2name"].items()}
    return train, val, id2name


# ─────────────────────────────────────────────────────────────────────────────
# 1. 类别分布分析
# ─────────────────────────────────────────────────────────────────────────────

def analyze_label_distribution(data, id2name, split_name, output_dir: Path):
    labels = [item["label"] for item in data]
    counts = Counter(labels)
    sorted_items = sorted(counts.items(), key=lambda x: x[0])

    names = [id2name[lid] for lid, _ in sorted_items]
    cnts  = [cnt for _, cnt in sorted_items]

    print(f"\n{'='*50}")
    print(f"【{split_name}】类别分布（共 {len(data)} 条）")
    print(f"{'='*50}")
    max_cnt, min_cnt = max(cnts), min(cnts)
    for (lid, cnt), name in zip(sorted_items, names):
        bar = "█" * (cnt * 30 // max_cnt)
        print(f"  {lid:2d} {name:4s} | {bar:<30s} {cnt:6d} ({cnt/len(data)*100:.1f}%)")
    print(f"\n  最多类: {id2name[max(counts, key=counts.get)]} ({max_cnt} 条)")
    print(f"  最少类: {id2name[min(counts, key=counts.get)]} ({min_cnt} 条)")
    print(f"  不均衡比 (max/min): {max_cnt/min_cnt:.1f}x")

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = sns.color_palette("husl", len(names))
    bars = ax.bar(names, cnts, color=colors, edgecolor="white", linewidth=0.5)
    for bar, cnt in zip(bars, cnts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(cnt), ha="center", va="bottom", fontsize=8)
    ax.set_title(f"TNEWS {split_name} — 类别分布", fontsize=13)
    ax.set_xlabel("类别")
    ax.set_ylabel("样本数")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    save_path = output_dir / f"label_dist_{split_name}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  图表已保存 → {save_path}")

    return counts


# ─────────────────────────────────────────────────────────────────────────────
# 2. 文本长度分析（字符级）
# ─────────────────────────────────────────────────────────────────────────────

def analyze_char_length(data, id2name, split_name, output_dir: Path):
    lengths = [len(item["sentence"]) for item in data]
    lengths = np.array(lengths)

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

    # 长度 > 128 的比例（BERT 常用截断长度）
    for threshold in [64, 128, 256]:
        pct = (lengths > threshold).sum() / len(lengths) * 100
        print(f"  长度 > {threshold:3d} 的占比: {pct:.2f}%  "
              f"({'截断会影响这部分样本' if pct > 0 else '无截断损失'})")

    # 直方图
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(lengths, bins=50, color="#4C72B0", edgecolor="white", linewidth=0.4)
    for t, ls in [(64, "--"), (128, "-"), (256, ":")]:
        axes[0].axvline(t, color="red", linestyle=ls, alpha=0.7, label=f"len={t}")
    axes[0].set_title(f"{split_name} — 字符长度分布")
    axes[0].set_xlabel("字符数")
    axes[0].set_ylabel("样本数")
    axes[0].legend(fontsize=8)

    # 截断后覆盖率曲线
    thresholds = np.arange(1, min(300, lengths.max() + 1))
    coverage = [(lengths <= t).sum() / len(lengths) * 100 for t in thresholds]
    axes[1].plot(thresholds, coverage, color="#C44E52")
    axes[1].axhline(95, color="gray", linestyle="--", label="95% 覆盖率")
    axes[1].axhline(99, color="gray", linestyle=":", label="99% 覆盖率")
    # 标出达到 95% / 99% 的截断长度
    for target_pct in [95, 99]:
        idx = next((i for i, c in enumerate(coverage) if c >= target_pct), None)
        if idx is not None:
            axes[1].axvline(thresholds[idx], color="#C44E52", linestyle="--", alpha=0.5)
            axes[1].text(thresholds[idx] + 2, target_pct - 3,
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

    return lengths


# ─────────────────────────────────────────────────────────────────────────────
# 3. 各类别长度对比
# ─────────────────────────────────────────────────────────────────────────────

def analyze_length_by_label(data, id2name, split_name, output_dir: Path):
    label_lengths = {}
    for item in data:
        lid = item["label"]
        label_lengths.setdefault(lid, []).append(len(item["sentence"]))

    sorted_lids = sorted(label_lengths.keys())
    names = [id2name[lid] for lid in sorted_lids]
    all_lengths = [label_lengths[lid] for lid in sorted_lids]
    means = [np.mean(l) for l in all_lengths]

    print(f"\n{'='*50}")
    print(f"【{split_name}】各类别平均字符长度")
    print(f"{'='*50}")
    for lid, name, mean_l in zip(sorted_lids, names, means):
        print(f"  {lid:2d} {name:4s}: {mean_l:.1f} 字")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 箱线图
    axes[0].boxplot(all_lengths, tick_labels=names, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#AEC6CF", alpha=0.7))
    axes[0].set_title(f"{split_name} — 各类别长度箱线图")
    axes[0].set_ylabel("字符数")
    axes[0].tick_params(axis="x", rotation=40)

    # 均值柱状图
    colors = sns.color_palette("pastel", len(names))
    axes[1].bar(names, means, color=colors, edgecolor="gray", linewidth=0.5)
    axes[1].set_title(f"{split_name} — 各类别平均长度")
    axes[1].set_ylabel("平均字符数")
    axes[1].tick_params(axis="x", rotation=40)

    plt.tight_layout()
    save_path = output_dir / f"length_by_label_{split_name}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  图表已保存 → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Token 长度分析（BERT tokenizer）
# ─────────────────────────────────────────────────────────────────────────────

def analyze_token_length(data, tokenizer, split_name, output_dir: Path):
    """
    字符数 ≠ Token 数：BERT 的 WordPiece 分词会把长词拆成多个 subword。
    实际训练时截断 max_length 指的是 Token 数，这里量化两者差距。
    """
    print(f"\n{'='*50}")
    print(f"【{split_name}】Token 长度统计（BERT tokenizer，不含特殊符号）")
    print(f"{'='*50}")

    sentences = [item["sentence"] for item in data]
    # batch_encode_plus 比逐条快很多
    encoded = tokenizer(sentences, add_special_tokens=False,
                        truncation=False, padding=False)
    token_lengths = np.array([len(ids) for ids in encoded["input_ids"]])

    stats = {
        "均值":   np.mean(token_lengths),
        "中位数": np.median(token_lengths),
        "最短":   np.min(token_lengths),
        "最长":   np.max(token_lengths),
        "P90":    np.percentile(token_lengths, 90),
        "P95":    np.percentile(token_lengths, 95),
        "P99":    np.percentile(token_lengths, 99),
    }
    for name, val in stats.items():
        print(f"  {name:6s}: {val:.1f}")

    # 加上 [CLS] 和 [SEP] 后超出 128 的比例
    for max_len in [64, 128]:
        exceeded = ((token_lengths + 2) > max_len).sum()  # +2 for [CLS][SEP]
        print(f"  max_length={max_len} 时截断比例: {exceeded/len(token_lengths)*100:.2f}%")

    # Token 长度 vs 字符长度对比
    char_lengths = np.array([len(item["sentence"]) for item in data])
    ratio = token_lengths / np.maximum(char_lengths, 1)
    print(f"\n  Token/字符 比值: 均值={ratio.mean():.3f}, "
          f"P5={np.percentile(ratio,5):.3f}, P95={np.percentile(ratio,95):.3f}")
    print("  （中文 BERT 基本 1 字 = 1 token，比值接近 1.0）")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(token_lengths, bins=50, color="#55A868", edgecolor="white", linewidth=0.4,
            label="Token 长度")
    ax.hist(char_lengths, bins=50, color="#4C72B0", edgecolor="white", linewidth=0.4,
            alpha=0.5, label="字符长度")
    for t in [64, 128]:
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


# ─────────────────────────────────────────────────────────────────────────────
# 5. 各类别样本展示
# ─────────────────────────────────────────────────────────────────────────────

def show_samples(data, id2name, n_per_class=3):
    print(f"\n{'='*50}")
    print("各类别样本示例")
    print(f"{'='*50}")
    by_label = {}
    for item in data:
        by_label.setdefault(item["label"], []).append(item["sentence"])

    for lid in sorted(by_label.keys()):
        samples = by_label[lid][:n_per_class]
        print(f"\n  [{lid:2d}] {id2name[lid]}")
        for s in samples:
            print(f"       • {s}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default=str(DATA_DIR), type=Path)
    parser.add_argument("--output_dir", default=str(FIG_DIR),  type=Path)
    parser.add_argument("--model_path",
                        default=str(BERT_PATH),
                        type=Path,
                        help="用于统计 Token 长度的 BERT tokenizer 路径")
    parser.add_argument("--skip_token", action="store_true",
                        help="跳过 Token 长度分析（无需加载 tokenizer）")
    return parser.parse_args()


def main():
    args = parse_args()
    # 先把三个路径统一解析成绝对路径，后面所有用到它们的地方都不会再受 CWD 影响
    args.data_dir   = _resolve(args.data_dir)
    args.output_dir = _resolve(args.output_dir)
    args.model_path = _resolve(args.model_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"数据目录  : {args.data_dir}")
    print(f"输出目录  : {args.output_dir}")

    train, val, id2name = load_data(args.data_dir)
    print(f"数据加载完成：train={len(train)}, val={len(val)}")

    # 类别分布
    analyze_label_distribution(train, id2name, "train", args.output_dir)
    analyze_label_distribution(val,   id2name, "val",   args.output_dir)

    # 字符长度
    analyze_char_length(train, id2name, "train", args.output_dir)

    # 各类别长度
    analyze_length_by_label(train, id2name, "train", args.output_dir)

    # Token 长度（需要 tokenizer）
    if not args.skip_token:
        try:
            from transformers import BertTokenizer
            print(f"\n加载 Tokenizer: {args.model_path}")
            tokenizer = BertTokenizer.from_pretrained(str(args.model_path))
            analyze_token_length(train, tokenizer, "train", args.output_dir)
        except Exception as e:
            print(f"[跳过 Token 分析] {e}")
            print("可用 --skip_token 参数明确跳过，或检查 --model_path 路径")

    # 样本展示
    show_samples(train, id2name, n_per_class=3)

    print(f"\n所有图表已保存至 {args.output_dir}")


if __name__ == "__main__":
    main()
