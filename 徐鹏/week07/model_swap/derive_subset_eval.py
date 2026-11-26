# -*- coding: utf-8 -*-
"""
从 n=200 的预测文件里切出 n=60 前缀子集，单独出一份指标。

为什么不在 n=60 上重新推理：
  canonical_order 保证 n=60 是 n=200 的**前缀子集**（同一 seed、同一排序）。
  切预测 + 用 gold_n60 重新评测，等价于重新跑一遍，但省掉 3 倍 GPU 时间。

用法：
  python model_swap/derive_subset_eval.py \
      --pred outputs/llm/pred_qwen25_zero_n200.jsonl \
      --gold outputs/llm/gold_n60.jsonl \
      --out outputs/metrics/llm/qwen25_zero_n60.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

import evaluate_events as EV  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", default="outputs/llm/gold_n60.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default="outputs/model_swap/_pred_prefix.jsonl")
    args = ap.parse_args()

    gold_path = ROOT / args.gold
    gold_rows = EV.load_jsonl(gold_path)
    keys = [EV.row_key(g) for g in gold_rows]
    keyset = set(keys)

    pred_rows = EV.load_jsonl(ROOT / args.pred)
    pmap = {EV.row_key(r): r for r in pred_rows}
    missing = [k for k in keys if k not in pmap]
    if missing:
        print(f"[warn] 预测里缺 {len(missing)} 条（按全漏处理）")

    tmp = ROOT / args.tmp
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        for k in keys:
            f.write(json.dumps(pmap.get(k, {"uid": k, "events": []}), ensure_ascii=False) + "\n")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "common" / "evaluate_events.py"),
                    "--gold", str(gold_path), "--pred", str(tmp),
                    "--json", str(out), "--quiet"], check=True, cwd=str(ROOT))
    tmp.unlink(missing_ok=True)

    m = json.loads(out.read_text(encoding="utf-8"))
    print(f"{out}: n={m['num_samples']} trigger={m['trigger']['f1']:.4f} "
          f"argument={m['argument']['f1']:.4f} event={m['event_exact']['f1']:.4f}")


if __name__ == "__main__":
    main()
