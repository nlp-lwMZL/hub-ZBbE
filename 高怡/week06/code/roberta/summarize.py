"""
汇总 roberta 多次实验结果，生成对比表 + 折线图。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output" / "roberta"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def collect() -> pd.DataFrame:
    rows = []
    for log_file in sorted(OUT_DIR.glob("roberta_train_log_*.json")):
        log = json.loads(log_file.read_text(encoding="utf-8"))
        cfg = log.get("config", {})
        best = max(log["epochs"], key=lambda e: e["val_macro_f1"])
        tag = log_file.stem.replace("roberta_train_log_", "")
        rows.append({
            "tag": tag,
            "pool": cfg.get("pool", "cls"),
            "loss": cfg.get("loss_desc", "CE"),
            "class_weight": "on" if cfg.get("class_weight") else "off",
            "fgm": "on" if cfg.get("fgm") else "off",
            "epochs": len(log["epochs"]),
            "best_val_macro_f1": best["val_macro_f1"],
            "best_val_weighted_f1": best.get("val_weighted_f1", 0),
            "best_val_acc": best["val_acc"],
            "best_epoch": best["epoch"],
        })
    return pd.DataFrame(rows).sort_values("best_val_macro_f1", ascending=False, ignore_index=True)


def main() -> None:
    df = collect()
    if df.empty:
        print("没有找到任何 roberta_train_log_*.json")
        return
    summary_path = OUT_DIR / "roberta_summary.csv"
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[summary] {summary_path}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
