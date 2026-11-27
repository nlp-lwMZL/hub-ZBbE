"""
作业编排：训练 4 个模型（linear/crf × baseline/roberta）+ 对两个 CRF ckpt 做硬约束复评。

全部复用课程原脚本（bert/train.py、bert/evaluate_hard_constraint.py），只通过
--bert_path / --run_name 传参，课程代码零修改 —— 评测口径因此和原项目完全一致。

用法：
  python homework/run_compare.py                  # 全量：4 训练 + 2 硬约束复评
  python homework/run_compare.py --skip-existing  # 已有产物则跳过（断点续跑）
  python homework/run_compare.py --dry-run        # 只打印命令，不执行
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from l2a_common import (COURSE_CKPT, COURSE_METRIC, HW_METRIC, HARD_PAIRS,
                       MODELS, ROOT, RUNS, model_path, rsync_hw7_from_course)

PY = sys.executable
ENV = dict(os.environ)
ENV.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _metrics_exists(run_name: str) -> bool:
    # 两处都查：课程脚本的中转站 & 我们自己的 homework 目录
    return (COURSE_METRIC / f"{run_name}.json").exists() or \
           (HW_METRIC / f"{run_name}.json").exists()


def run(cmd, label, skip_ok=False):
    print(f"\n{'='*70}\n>>> {label}\n    {' '.join(str(c) for c in cmd)}\n{'='*70}", flush=True)
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=ROOT, env=ENV)
    dt = time.time() - t0
    if ret.returncode != 0:
        raise SystemExit(f"[FAILED] {label} (exit {ret.returncode}) after {dt:.0f}s")
    print(f"[done] {label} in {dt:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-existing", action="store_true", help="已有指标文件则跳过该步")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="只跑指定 run_name（如 hw7_crf_roberta），默认全部")
    args = ap.parse_args()

    # 1) 训练 4 个模型
    for decoder, run_name, model_key, need_hard in RUNS:
        if args.only and run_name not in args.only:
            continue
        if args.skip_existing and _metrics_exists(run_name):
            print(f"[skip] {run_name} 已有指标，跳过训练")
            continue
        cmd = [PY, "bert/train.py", "--decoder", decoder,
               "--bert_path", str(model_path(model_key)), "--run_name", run_name]
        if args.dry_run:
            print("DRY:", " ".join(cmd)); continue
        run(cmd, f"train {run_name} ({MODELS[model_key]['name']}, {decoder})")

    # 2) 对两个 CRF ckpt 做解码期硬约束复评（不重训）
    for base, hard in HARD_PAIRS:
        if args.only and base not in args.only and hard not in args.only:
            continue
        if args.skip_existing and _metrics_exists(hard):
            print(f"[skip] {hard} 已有指标，跳过硬约束复评")
            continue
        ckpt = COURSE_CKPT / f"{base}.pt"
        if not ckpt.exists():
            print(f"[warn] 找不到 ckpt {ckpt}，跳过 {hard}（先跑完训练）")
            continue
        cmd = [PY, "bert/evaluate_hard_constraint.py",
               "--ckpt", str(ckpt), "--run_name", hard, "--soft_json", base]
        if args.dry_run:
            print("DRY:", " ".join(cmd)); continue
        run(cmd, f"hard-constraint decode {hard}")

    # 全部课程脚本跑完之后，把 hw7_* 整体同步到 homework/ 子目录
    if not args.dry_run:
        rsync_hw7_from_course()

    print("\n全部完成。下一步运行：python homework/analysis.py")


if __name__ == "__main__":
    main()
