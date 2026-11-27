"""
图表生成：把 outputs/metrics/disc/*.json 的实验结果画成 PPT 可直接用的图

PPT 用图规范（本脚本全部遵守）：
  - 白底、150 dpi、字号 ≥ 10、标题不贴边
  - 中文字体（微软雅黑/黑体），负号正常显示
  - 每张图只表达一个结论，柱子上直接标数值（PPT 投影时看得清）

六张图（对应课件里的六个教学点）：
  1. method_comparison.png     各方法 trigger/argument/event 三级 F1 对比（主图）
  2. illegal_bio_comparison.png linear vs CRF 的非法 BIO 序列数（为什么需要 CRF）
  3. per_role_f1.png           各角色论元的 F1（哪类论元最难抽 → 长尾/自由文本）
  4. training_curves.png       BERT 训练曲线（loss 与内层验证 span F1）
  5. type_confusion.png        事件类型混淆矩阵（哪两类最容易混）
  6. latency_cost.png          延迟/参数量对照表（工程视角：不是越准越好）

使用方式：
  python eval/make_figures.py                 # 全部重画
  python eval/make_figures.py --only method_comparison per_role_f1
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")   # 必须在 pyplot 之前设置：无界面后端，服务器/脚本里不会卡住
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

METRIC_DIR = ROOT / "outputs" / "metrics" / "disc"
LOG_DIR = ROOT / "outputs" / "logs"
FIG_DIR = ROOT / "outputs" / "figures"
DATA_DIR = ROOT / "data" / "processed"
PRED_DIR = ROOT / "outputs" / "disc"

# ── 中文字体：Windows 上优先微软雅黑，找不到再退化到黑体/思源 ──
CJK_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
                  "WenQuanYi Zen Hei", "Arial Unicode MS"]

def setup_style() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    picked = [f for f in CJK_CANDIDATES if f in available]
    plt.rcParams["font.sans-serif"] = picked + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False   # 中文字体下负号会变方块，关掉
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    if not picked:
        print("[warn] 没找到中文字体，中文可能显示成方块；请安装微软雅黑/黑体")

def load_metric(name: str) -> dict | None:
    p = METRIC_DIR / f"{name}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save(fig, filename: str) -> None:
    """统一保存：白底 + 紧边界（不贴边）+ 150dpi。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / filename
    fig.savefig(out, bbox_inches="tight", pad_inches=0.18, facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out.relative_to(ROOT)}")

# ───────────────────── 1. 方法对比（主图）─────────────────────
def fig_method_comparison(main_methods: list[tuple[str, str]]) -> None:
    rows = []
    for key, label in main_methods:
        r = load_metric(key)
        if not r:
            continue
        rows.append((label, r["trigger"]["f1"], r["argument"]["f1"],
                     r["event_exact"]["f1"]))
    if not rows:
        print("  [skip] method_comparison：没有可用的指标文件")
        return

    labels = [r[0] for r in rows]
    x = np.arange(len(labels))
    width = 0.26
    series = [("触发词 F1", [r[1] for r in rows], "#4C78A8"),
              ("论元 F1", [r[2] for r in rows], "#F58518"),
              ("事件级 F1", [r[3] for r in rows], "#54A24B")]

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    for i, (name, vals, color) in enumerate(series):
        bars = ax.bar(x + (i - 1) * width, vals, width, label=name, color=color,
                      edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("F1 分数")
    ax.set_ylim(0, 1.0)
    ax.set_title("事件抽取方法对比（dev_extract 930 句，统一评测口径）", pad=14)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_method_compare.png")

# ───────────────────── 2. 非法 BIO 三联对比（P25）─────────────────────
def fig_illegal_bio(families: list[tuple[str, str, str]], hard_key: str = "") -> None:
    """
    families: [(族名, linear指标key, crf指标key)]，例如 ("BERT(12层)", "bert_linear_256", "bert_crf_256")

    三根柱子讲清因果链（这是 P25 的正确表述）：
      金标自身   —— 重叠论元投影到 BIO 的固有代价（不可消除的地板值）
      +Linear    —— 逐 token 独立预测，没有标签间约束 → 非法率暴涨
      +CRF       —— 转移矩阵学到约束 → 非法序列大幅下降（软约束，不是归零）
      +CRF硬约束 —— 非法转移分数直接置 -inf → 严格 0
    如果只画 linear vs crf、不画金标，读者会把「非法」全归咎于模型，归因就错了。
    """
    items = []   # (标签, 非法句数, 总句数, 颜色)
    base = None
    baseline_path = METRIC_DIR / "bio_gold_baseline.json"
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            base = (json.load(f).get("splits") or {}).get("dev")
    if base and base.get("illegal"):
        st = base["illegal"]
        items.append(("金标自身\n(重叠论元投影)", st["illegal_seqs"], st["total_seqs"], "#9E9E9E"))

    for fam, lin_key, crf_key in families:
        for key, tag, color in ((lin_key, f"{fam}\n+Linear", "#D9534F"),
                                (crf_key, f"{fam}\n+CRF(软约束)", "#4C78A8")):
            r = load_metric(key)
            if not r:
                continue
            st = (r.get("meta") or {}).get("illegal_dev") or {}
            if not st:
                continue
            items.append((tag, st.get("illegal_seqs", 0), st.get("total_seqs", 1), color))

    if hard_key:
        r = load_metric(hard_key)
        if r:
            st = (r.get("meta") or {}).get("illegal_dev") or {}
            if st:
                # 硬约束这一根要特别标注口径：token 级严格 0，字符级仍 69 句
                # （因为空格不产生 token，跨空格的 span 在字符级看起来非法，见 NOTES 踩坑 #11）。
                # 图上同时给两个数，避免读者以为"硬约束没生效"。
                items.append(("BERT+CRF 硬约束\n(token级 0 / 字符级)",
                              st.get("illegal_seqs", 0), st.get("total_seqs", 1), "#54A24B"))

    if not items:
        print("  [skip] illegal_bio_comparison：缺少指标文件")
        return

    labels = [it[0] for it in items]
    counts = [it[1] for it in items]
    totals = [max(it[2], 1) for it in items]
    colors = [it[3] for it in items]
    x = np.arange(len(items))

    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    bars = ax.bar(x, counts, color=colors, edgecolor="white", linewidth=0.8, width=0.62)
    for b, c, t in zip(bars, counts, totals):
        ax.text(b.get_x() + b.get_width() / 2, c + max(counts) * 0.015,
                f"{c} 条\n({c / t:.1%})", ha="center", va="bottom", fontsize=10)
    # 金标地板线：便于看出「CRF 降到了哪里、硬约束是否到 0」
    if base and base.get("illegal"):
        ax.axhline(base["illegal"]["illegal_seqs"], color="#9E9E9E", linestyle="--",
                   linewidth=1.2, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.8)
    ax.set_ylabel("含非法 BIO 的句子数（dev 930 句）")
    ax.set_title("非法 BIO 序列：金标地板 → Linear 暴涨 → CRF 显著下降 → 硬约束归零", pad=14)
    ax.set_ylim(0, max(counts) * 1.24)
    fig.tight_layout()
    save(fig, "fig_illegal_bio.png")

# ───────────────────── 3. 各角色 F1 ─────────────────────
def fig_per_role(method_key: str, method_label: str, min_support: int = 20) -> None:
    r = load_metric(method_key)
    if not r or not r.get("per_role"):
        print(f"  [skip] per_role_f1：缺少 {method_key} 的 per_role")
        return
    items = [(role, v["f1"], v["tp"] + v["fn"]) for role, v in r["per_role"].items()]
    items = [it for it in items if it[2] >= min_support]
    items.sort(key=lambda x: x[1])
    top = items[:18]      # 只画最难的 18 个角色（且 dev 支持数 ≥20）：PPT 上一张图能看清
    labels = [f"{lbl}（{n}）" for lbl, _, n in top]
    vals = [v for _, v, _ in top]

    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    colors = ["#D9534F" if v < 0.2 else ("#F0AD4E" if v < 0.5 else "#5CB85C")
              for v in vals]
    bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9.5)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("论元 F1（括号内为该角色在 dev 的金标出现次数）")
    ax.set_xlim(0, 1.0)
    ax.set_title(f"各角色论元 F1（最难的 18 个）— {method_label}", pad=14)
    fig.tight_layout()
    save(fig, "fig_per_role_f1.png")

# ───────────────────── 4. 训练曲线 ─────────────────────
def _load_train_log(run: str) -> list[dict] | None:
    """
    读某个方法的训练日志，返回「每 epoch 的记录列表」。

    为什么单独抽个函数：训练日志在 2026-11 重构时换了**文件名和结构**两件事，
    两处都要兼容，否则图会静默变成空白（本项目真实踩过：日志明明有 6 个，
    图却打印「没有训练日志」跳过）：
      - 旧：`logs/train_<run>.json`，内容就是 `[{epoch, train_loss, inner_val_span_f1, ...}]`
      - 新：`logs/<method>_train.json`，内容是 `{"epochs": [{epoch, train_loss, val: {...}}], ...}`
    新格式是主路径；旧格式保留兼容（读到旧文件也照样能画）。
    """
    for p in (LOG_DIR / f"{run}_train.json",        # 新命名（主路径）
              LOG_DIR / f"train_{run}.json"):       # 旧命名（兼容）
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # 新格式：dict，epoch 列表在 "epochs" 里；旧格式：直接就是 list
        epochs = raw.get("epochs", []) if isinstance(raw, dict) else raw
        if epochs:
            return epochs
    return None


def _epoch_val_f1(rec: dict) -> float | None:
    """从一条 epoch 记录里取「内层验证 span F1」，兼容新旧两种字段位置。"""
    if "inner_val_span_f1" in rec:            # 旧格式：平铺
        return rec["inner_val_span_f1"]
    val = rec.get("val") or {}                # 新格式：嵌在 val 里
    return val.get("inner_span_f1")


def fig_training_curves(runs: list[tuple[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    plotted = 0
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    for i, (run, label) in enumerate(runs):
        log = _load_train_log(run)
        if not log:
            continue
        epochs = [x["epoch"] for x in log]
        axes[0].plot(epochs, [x["train_loss"] for x in log], marker="o",
                     color=colors[i % len(colors)], label=label)
        vals = [_epoch_val_f1(x) for x in log]
        if any(v is not None for v in vals):
            axes[1].plot(epochs, [v if v is not None else float("nan") for v in vals],
                         marker="o", color=colors[i % len(colors)], label=label)
        plotted += 1
    if not plotted:
        print("  [skip] training_curves：没有训练日志"
              f"（找的是 {LOG_DIR}/<method>_train.json）")
        plt.close(fig)
        return
    axes[0].set_title("训练损失", pad=12)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[1].set_title("内层验证集 span F1（用于选 epoch，dev 不参与选点）", pad=12)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("span F1")
    xticks = sorted({int(e) for run, _ in runs for e in _epochs_of(run)})
    for ax in axes:
        if xticks:
            ax.set_xticks(xticks)
        ax.legend(frameon=False)
    fig.suptitle("BERT 微调训练曲线", fontsize=13.5, y=1.03)
    fig.tight_layout()
    save(fig, "fig_training_curves.png")

def _epochs_of(run: str) -> list[int]:
    log = _load_train_log(run)
    return [x["epoch"] for x in log] if log else []

# ───────────────────── 5. 类型混淆矩阵 ─────────────────────
def fig_type_confusion(method_key: str, method_label: str) -> None:
    """
    从预测文件与金标现算混淆矩阵（不重跑模型）。行=金标类型，列=预测类型。
    """
    gold_p = DATA_DIR / "dev_extract.jsonl"
    pred_p = PRED_DIR / f"pred_{method_key}.jsonl"
    if not gold_p.exists() or not pred_p.exists():
        print(f"  [skip] type_confusion：缺 {pred_p.name}")
        return

    def load_jsonl(path):
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    gold = load_jsonl(gold_p)
    pred = {(r.get("id"), r.get("sent_id")): r for r in load_jsonl(pred_p)}
    types = sorted({(g.get("event") or {}).get("type", "") for g in gold} - {""})
    tid = {t: i for i, t in enumerate(types)}
    mat = np.zeros((len(types), len(types)), dtype=int)
    for g in gold:
        ge = g.get("event") or {}
        gt = ge.get("type", "")
        if gt not in tid:
            continue
        p = pred.get((g.get("id"), g.get("sent_id")), {})
        pe = (p.get("events") or [{}])
        pt = (pe[0].get("type") if pe else "") or ""
        col = tid.get(pt, -1)
        if col >= 0:
            mat[tid[gt], col] += 1

    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    # 行归一化成召回率，避免热门类型（股份回购 979 条）压掉长尾类型
    norm = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types, fontsize=10)
    for i in range(len(types)):
        for j in range(len(types)):
            if mat[i, j] > 0:
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=8.5,
                        color="white" if norm[i, j] > 0.55 else "#222222")
    ax.set_xlabel("预测类型")
    ax.set_ylabel("金标类型")
    ax.set_title(f"事件类型混淆矩阵（行归一化=召回率）— {method_label}", pad=14)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="行内占比")
    fig.tight_layout()
    save(fig, "fig_type_confusion.png")

# ───────────────────── 6. 延迟 / 成本表 ─────────────────────
def fig_latency_cost(methods: list[tuple[str, str]]) -> None:
    rows = []
    for key, label in methods:
        r = load_metric(key)
        if not r:
            continue
        m = r.get("meta") or {}
        rows.append([label,
                     f"{m.get('ms_per_sentence', '-')}",
                     f"{m.get('params_m', '-')}",
                     f"{m.get('train_s', '-')}",
                     f"{r['trigger']['f1']:.3f}",
                     f"{r['argument']['f1']:.3f}"])
    if not rows:
        print("  [skip] latency_cost：没有可用指标")
        return

    fig, ax = plt.subplots(figsize=(11.0, 0.62 * len(rows) + 1.9))
    ax.axis("off")
    cols = ["方法", "延迟(ms/句)", "参数量(M)", "训练(s)", "触发词F1", "论元F1"]
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor("#4C78A8")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F4F7FB")
    ax.set_title("工程视角：精度 / 延迟 / 参数量的三角权衡", pad=18, fontsize=13.5)
    fig.tight_layout()
    save(fig, "fig_latency_cost.png")

# ───────────────────── 主流程 ─────────────────────
MAIN_METHODS = [
    ("rule", "L0 规则"),
    ("bilstm_crf", "BiLSTM-CRF"),
    ("bert_linear_256", "BERT+Linear"),
    ("bert_crf_256", "BERT+CRF"),
    ("pipeline", "两段式流水线"),
]

ABLATIONS = [
    ("bert_linear_128", "BERT+Linear\n(max_len=128)"),
    # 4 层消融跑的是 **linear** 解码器（与主线 bert_linear_256 对照才干净），文件名是
    # bert4L_linear_256。写成 bert4L_crf_256 会让这根柱子**静默消失**（和 compare_methods 同一个坑）。
    ("bert4L_linear_256", "BERT+Linear\n(前 4 层)"),
    ("bert_crf_256_hard", "BERT+CRF\n(非法转移硬约束)"),
]

def main():
    ap = argparse.ArgumentParser(description="生成 PPT 用图")
    ap.add_argument("--only", nargs="*", default=None,
                    help="只画这几张（method_comparison / illegal_bio_comparison / "
                         "per_role_f1 / training_curves / type_confusion / latency_cost）")
    ap.add_argument("--role_method", default="bert_crf_256")
    ap.add_argument("--curve_runs", nargs="*", default=None)
    ap.add_argument("--confusion_method", default="bert_crf_256")
    args = ap.parse_args()
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    want = set(args.only) if args.only else {
        "method_comparison", "illegal_bio_comparison", "per_role_f1",
        "training_curves", "type_confusion", "latency_cost"}

    if "method_comparison" in want:
        fig_method_comparison(MAIN_METHODS + ABLATIONS)
    if "illegal_bio_comparison" in want:
        fig_illegal_bio([("BERT(12层)", "bert_linear_256", "bert_crf_256"),
                         ("BiLSTM", "bilstm_linear", "bilstm_crf")],
                        hard_key="bert_crf_256_hard")  # 只有复评成功（metrics 存在）才画这一根
    if "per_role_f1" in want:
        label = next((lb for k, lb in MAIN_METHODS if k == args.role_method), args.role_method)
        fig_per_role(args.role_method, label)
    if "training_curves" in want:
        runs = args.curve_runs or [("bert_linear_256", "BERT+Linear"),
                                   ("bert_crf_256", "BERT+CRF")]
        fig_training_curves([(r, lb) for r, lb in runs])
    if "type_confusion" in want:
        label = next((lb for k, lb in MAIN_METHODS if k == args.confusion_method),
                     args.confusion_method)
        fig_type_confusion(args.confusion_method, label)
    if "latency_cost" in want:
        fig_latency_cost(MAIN_METHODS)
    print("完成。")

if __name__ == "__main__":
    main()
