"""
汇总三类方法的指标到一张对比表（可直接贴报告）。

数据口径纪律（老师 PPT 反复强调）：
  1. 所有数字从 outputs/eval/*.json 的**真实产物**读，绝不填估算值；
  2. 没跑的方法显式标「未跑」，不藏不猜；
  3. 每格带 run_name 溯源，图注和实际跑的参数不会打架；
  4. 三类方法都用 common/metrics.evaluate 算的实体级 F1，同口径可比；
     SFT 臂额外带工程指标（JSON 合法率等），单独一列。

用法：python eval/compare.py [--split test] [--out summary.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from common import paths


def find_metrics(split: str) -> list[dict]:
    """扫描eval目录下所有 <run>_<split>.json，识别方法类型。"""
    out = []
    for p in sorted(paths.eval_dir().glob(f"*_{split}.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                   # noqa: BLE001
            continue
        method = data.get("method", "")
        if "SFT" in method:
            family = "L4 Qwen3-SFT"
        elif "BiLSTM" in method:
            family = "L1 BiLSTM"
        elif "RoBERTa" in method:
            family = "L2 RoBERTa"
        else:
            family = "其他"
        m = data.get("metrics", {})
        out.append({
            "file": p.name, "family": family, "method": method,
            "run": data.get("run_name", p.stem.replace(f"_{split}", "")),
            "precision": m.get("precision"), "recall": m.get("recall"),
            "f1": m.get("f1"), "type_macro_f1": m.get("type_macro_f1"),
            "n": data.get("num_samples"),
            "eng": data.get("engineering"),
            "split": data.get("split", split),
        })
    return out


def fmt(v, width=7) -> str:
    return f"{v:>{width}.4f}" if isinstance(v, (int, float)) else f"{'未跑':>{width}}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "validation", "train"])
    ap.add_argument("--out", default=None, help="输出 md 路径；默认 eval 目录下 summary_<split>.md")
    a = ap.parse_args()

    rows = find_metrics(a.split)
    lines = [
        f"# week7 NER 三类方法对比（split={a.split}）",
        "",
        "> 口径：所有数字由 `common/metrics.evaluate` 在 cdtier/clean 同一份数据上计算；"
        "实体级 P/R/F1（三元组完全匹配）。未跑的方法显式标「未跑」。",
        "",
        "| 层级 | 方法 | run_name | N | P | R | F1 | type_macro_F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['family']} | {r['method']} | `{r['run']}` | {r['n']} | "
            f"{fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | "
            f"{fmt(r['type_macro_f1'])} |"
        )
    if not rows:
        lines.append("| — | 暂无指标产物 | — | — | — | — | — | — |")

    lines += ["", "## SFT 臂工程指标（生成式独有成本）", ""]
    sft_rows = [r for r in rows if r["eng"]]
    if sft_rows:
        lines += ["| run | JSON合法 | 修复 | 类型幻觉 | 文本未找到 | 平均延迟(s/条) |",
                  "|---|---|---|---|---|---|"]
        for r in sft_rows:
            e = r["eng"]
            lines.append(f"| `{r['run']}` | {e.get('json_valid')} | {e.get('repaired')} | "
                         f"{e.get('illegal_type')} | {e.get('text_not_found')} | "
                         f"{e.get('avg_latency_s')} |")
    else:
        lines.append("（SFT 臂未跑或产物里无工程指标）")

    lines += [
        "",
        "## 读表提醒",
        "",
        "- **L1 vs L2** 的差 = 预训练值多少点（同标签、同 CRF、同评测，唯一差别是编码器）；",
        "- **L2 vs L4** 的差 = 判别式 vs 生成式微调（SFT 召回强、边界可能糙，看 type_macro 辅助判断）；",
        "- loss 绝对值不能跨方法比（CE vs CRF NLL vs LM loss 量纲不同），只看实体级 F1；",
        "- SFT 的 type_macro 若异常低，检查是否有 schema 外的幻觉类型（工程指标里的「类型幻觉」）。",
        "",
    ]
    text = "\n".join(lines)
    out = Path(a.out) if a.out else paths.eval_dir() / f"summary_{a.split}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
