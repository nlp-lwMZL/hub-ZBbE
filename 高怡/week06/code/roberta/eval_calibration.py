"""
分类模型的置信度校准评估：看模型是否「过度自信」。

指标
----
- Acc / Macro-F1：基础效果
- mean_conf：平均最大预测概率
- overconf_gap = mean_conf - acc：>0 即过度自信
- ECE（等宽分箱 / 等频分箱）：期望校准误差，越小越准
- NLL / Brier：概率质量的整体度量
- high_conf_err：置信度 ≥ 0.99 却判错的样本数

图
--
- 可靠性图（reliability diagram）：置信度 vs 实际准确率，落在对角线下方 = 过度自信
- 置信度分布直方图

用法
----
# 评估 output/roberta 下全部 ckpt 的 val + test
python roberta/eval_calibration.py

# 指定 ckpt / 集合 / 分箱数
python roberta/eval_calibration.py --ckpt output/roberta/roberta_best_s42_ce_ls01_cls6.pt \
    --splits train,val,test --bins 15

输出
----
output/roberta/calibration/{ckpt名}_{split}.json / .png
output/roberta/calibration/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "roberta"))

from common.label import CLS_ID2LABEL, CLS_NUM_LABELS  # noqa: E402
from train_roberta import (AnnDataset, BertClassifier, PRETRAIN_DIR,  # noqa: E402
                           load_split)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = BASE_DIR / "output" / "roberta" / "calibration"


# ---------------------------------------------------------------- 推理


def infer_layers(state_dict: dict) -> int:
    """从 state_dict 的 encoder 层号推断实际层数（4 层与 6 层 ckpt 混放时必需）。"""
    idx = [int(k.split(".")[3]) for k in state_dict if k.startswith("bert.encoder.layer.")]
    return max(idx) + 1 if idx else 6


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        logits_all.append(logits.float().cpu())
        y_all.append(batch["labels"])
    return torch.cat(logits_all).numpy(), torch.cat(y_all).numpy()


# ---------------------------------------------------------------- 校准指标


def _groups_equal_width(conf: np.ndarray, n_bins: int) -> list[np.ndarray]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    return [np.where(which == b)[0] for b in range(n_bins)]


def _groups_adaptive(conf: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """等频分箱：置信度高度集中时（如全在 0.999）等宽分箱会全挤进一格，等频才有区分度。"""
    order = np.argsort(conf, kind="mergesort")
    return [g for g in np.array_split(order, n_bins) if len(g) > 0]


def calibration_rows(conf: np.ndarray, correct: np.ndarray,
                     groups: list[np.ndarray]) -> tuple[float, list[dict]]:
    total, ece, rows = len(conf), 0.0, []
    for g in groups:
        if len(g) == 0:
            continue
        c, a = float(conf[g].mean()), float(correct[g].mean())
        ece += len(g) / total * abs(c - a)
        rows.append({"n": int(len(g)), "conf": round(c, 6),
                     "acc": round(a, 6), "gap": round(a - c, 6)})
    return float(ece), rows


def plot_reliability(rows: list[dict], ece: float, title: str, out_path: Path) -> None:
    conf = np.array([r["conf"] for r in rows])
    acc = np.array([r["acc"] for r in rows])
    cnt = np.array([r["n"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    lo = float(min(conf.min(), acc.min()))
    hi = 1.0
    pad = (hi - lo) * 0.25 + 2e-4
    ax.plot([lo - pad, hi], [lo - pad, hi], "--", color="gray", lw=1, label="完美校准")
    ax.plot(conf, acc, "o-", color="#E8862E", ms=5, lw=1.5, label="实际")
    ax.set_xlim(max(0.0, lo - pad), min(1.0, hi + pad * 0.2))
    ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad * 0.2))
    ax.set_xlabel("平均置信度")
    ax.set_ylabel("实际准确率")
    ax.set_title(f"{title}\nECE={ece:.4f}（点在对角线下方=过度自信）")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    ax = axes[1]
    pos = np.arange(len(rows))
    ax.bar(pos, cnt, color="#2E9BD6")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{c:.3f}" for c in conf], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("分箱平均置信度")
    ax.set_ylabel("样本数")
    ax.set_title("置信度分布（等频分箱）")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_split(model, records, tok, split: str, args, device) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    ds = AnnDataset(records, tok, args.max_length)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=0)
    logits, y = predict(model, dl, device)

    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = (pred == y).astype(np.float64)

    acc = float(accuracy_score(y, pred))
    macro_f1 = float(f1_score(y, pred, average="macro"))
    mean_conf = float(conf.mean())
    nll = float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean())
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    brier = float(((probs - onehot) ** 2).sum(1).mean())

    ece_ew, rows_ew = calibration_rows(conf, correct, _groups_equal_width(conf, args.bins))
    ece_ad, rows_ad = calibration_rows(conf, correct, _groups_adaptive(conf, args.bins))

    per_class = {}
    for cid in range(CLS_NUM_LABELS):
        m = y == cid
        if m.sum() == 0:
            continue
        per_class[CLS_ID2LABEL[cid]] = {
            "n": int(m.sum()),
            "acc": round(float(correct[m].mean()), 6),
            "mean_conf": round(float(conf[m].mean()), 6),
            "gap": round(float(correct[m].mean() - conf[m].mean()), 6),
        }

    res = {
        "split": split, "n": int(len(y)),
        "acc": round(acc, 6), "macro_f1": round(macro_f1, 6),
        "mean_conf": round(mean_conf, 6),
        "overconf_gap": round(mean_conf - acc, 6),
        "nll": round(nll, 6), "brier": round(brier, 6),
        "ece_equal_width": round(ece_ew, 6),
        "ece_adaptive": round(ece_ad, 6),
        "high_conf_err": int(((conf >= 0.99) & (correct == 0)).sum()),
        "bins_equal_width": rows_ew,
        "bins_adaptive": rows_ad,
        "per_class": per_class,
    }
    return res


# ---------------------------------------------------------------- 主流程


def main() -> None:
    ap = argparse.ArgumentParser(description="分类模型置信度校准评估")
    ap.add_argument("--ckpt", nargs="+", default=None,
                    help="ckpt 路径，可多个；缺省为 output/roberta/roberta_best_*.pt")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = args.ckpt or sorted(str(p) for p in
                                (BASE_DIR / "output" / "roberta").glob("roberta_best_*.pt"))
    if not ckpts:
        raise SystemExit("没有找到 ckpt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(PRETRAIN_DIR))
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    data = {s: load_split(s) for s in splits}

    summary_rows = []
    for ckpt_path in ckpts:
        ckpt_path = Path(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        n_layers = infer_layers(ckpt["state_dict"])
        model = BertClassifier(PRETRAIN_DIR, num_labels=ckpt.get("num_labels", CLS_NUM_LABELS),
                               pool=ckpt.get("pool", "cls"), n_layers=n_layers).to(device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"\n[ckpt] {ckpt_path.name}  layers={n_layers}  "
              f"pool={ckpt.get('pool', 'cls')}", flush=True)

        for split in splits:
            res = evaluate_split(model, data[split], tok, split, args, device)
            name = ckpt_path.stem.replace("roberta_best_", "")
            (out_dir / f"{name}_{split}.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            plot_reliability(res["bins_adaptive"], res["ece_adaptive"],
                             f"{name} / {split}", out_dir / f"{name}_{split}.png")

            print(f"  [{split}] n={res['n']} acc={res['acc']:.4f} "
                  f"macro_f1={res['macro_f1']:.4f} conf={res['mean_conf']:.4f} "
                  f"gap={res['overconf_gap']:+.4f} "
                  f"ECE(等宽)={res['ece_equal_width']:.4f} "
                  f"ECE(等频)={res['ece_adaptive']:.4f} "
                  f"NLL={res['nll']:.4f} Brier={res['brier']:.4f} "
                  f"高置信误判={res['high_conf_err']}", flush=True)

            summary_rows.append({
                "ckpt": name, "layers": n_layers, "split": split, "n": res["n"],
                "acc": res["acc"], "macro_f1": res["macro_f1"],
                "mean_conf": res["mean_conf"], "overconf_gap": res["overconf_gap"],
                "ece_equal_width": res["ece_equal_width"],
                "ece_adaptive": res["ece_adaptive"],
                "nll": res["nll"], "brier": res["brier"],
                "high_conf_err": res["high_conf_err"],
            })
        del model

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n[汇总] {csv_path}")


if __name__ == "__main__":
    main()
