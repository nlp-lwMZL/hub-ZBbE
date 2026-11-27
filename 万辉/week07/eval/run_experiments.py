"""
一键复现判别式链路的全部实验（按顺序跑，每个任务独立进程 + 独立日志）

为什么要有这个脚本：
  本项目要产出「同一口径下 6 种方法的对比表」。如果靠人手一条条敲命令，
  很容易出现「某条用了别的 max_len / 别的 epoch」这种不可比的问题，
  而且中途一个失败就没人知道表里哪一行是旧数字。这里把每条实验的
  完整命令、耗时、退出码都记下来（outputs/logs/run_*.log + run_summary.json），
  PPT 上的每个数字都能追溯到一条具体命令。

使用方式：
  python eval/run_experiments.py --group all          # 全部（约 80-100 分钟，GPU）
  python eval/run_experiments.py --group main         # 只跑两条主线 BERT（先出核心数字）
  python eval/run_experiments.py --group pipeline     # 只跑两段式
  python eval/run_experiments.py --group ablation     # 只跑消融
  python eval/run_experiments.py --group bilstm       # 只跑 BiLSTM
  python eval/run_experiments.py --only bert_crf_256  # 跑指定的一条
  python eval/run_experiments.py --group main --dry_run   # 只打印命令不执行

注意：GPU 只有一张（RTX 4060 8GB），本脚本**串行**执行，绝不并行；
跑前会自动检查显存占用，>1GB 就提醒你先等别的任务结束。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows: torch 与 numpy 各带 libiomp5md，不设会 OMP Error #15

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

PY = sys.executable
LOG_DIR = ROOT / "outputs" / "logs"

# (名字, 组别, 解释, 参数列表)
EXPERIMENTS = [
    # ── 主线：BERT 两种解码器（PPT 主表的两行）──
    ("bert_linear_256", "main", "BERT(12层)+Linear，max_len=256，3 epoch",
     ["bert/train.py", "--decoder", "linear", "--max_len", "256",
      "--epochs", "3", "--run_name", "bert_linear_256"]),
    ("bert_crf_256", "main", "BERT(12层)+CRF，max_len=256，3 epoch（主线最佳）",
     ["bert/train.py", "--decoder", "crf", "--max_len", "256",
      "--epochs", "3", "--run_name", "bert_crf_256"]),

    # ── 核心教学对比：联合 vs 两段式 ──
    ("pipeline", "pipeline", "两段式：检测器 + 13分类 + 13个类型条件论元专家",
     ["bert_pipeline/train.py"]),

    # ── L1 基线：无预训练 ──
    ("bilstm_crf", "bilstm", "字符级 BiLSTM(256)-CRF，3 epoch（无预训练；时间预算内取下限）",
     ["bilstm_crf/train.py", "--decoder", "crf", "--epochs", "3",
      "--run_name", "bilstm_crf"]),
    ("bilstm_linear", "bilstm", "字符级 BiLSTM(256)+Linear（非法 BIO 对照）",
     ["bilstm_crf/train.py", "--decoder", "linear", "--epochs", "5",
      "--run_name", "bilstm_linear"]),

    # ── 消融 ──
    ("bert_linear_128", "ablation", "消融：max_len=128（截断更多，看截断代价）",
     ["bert/train.py", "--decoder", "linear", "--max_len", "128",
      "--epochs", "3", "--run_name", "bert_linear_128"]),
    ("bert4L_linear_256", "ablation", "消融：BERT 只保留前 4 层（省磁盘/加速，与 bert_linear_256 可比）",
     ["bert/train.py", "--decoder", "linear", "--max_len", "256",
      "--bert_layers", "4", "--epochs", "3", "--run_name", "bert4L_linear_256"]),
    # 硬约束消融不重训：直接对已训好的 CRF 做解码期约束（省 40 分钟 GPU，见该脚本 docstring）
    ("bert_crf_256_hard", "ablation", "消融：CRF 硬约束（解码期，复用 bert_crf_256 权重）",
     ["bert/evaluate_hard_constraint.py"]),
]

def gpu_used_mb() -> int:
    """读当前显存占用（MiB）；查不到返回 -1（不阻塞流程）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1

def run_one(name: str, desc: str, argv: list[str], dry_run: bool) -> dict:
    cmd = [PY] + argv
    print("\n" + "=" * 78)
    print(f"▶ {name}：{desc}")
    print("  " + " ".join(["python"] + argv))
    if dry_run:
        return {"name": name, "desc": desc, "cmd": cmd, "status": "dry_run"}

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"run_{name}.log"
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        tail = []
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            tail.append(line.rstrip())
            if len(tail) > 400:
                tail.pop(0)
            # 只回显关键行，避免刷屏（完整输出在 log 里）
            s = line.strip()
            if any(k in s for k in ("epoch ", "指标已写入", "检测器", "诊断", "Traceback",
                                    "Error", "训练耗时", "★", "n200")):
                print("  " + s[:150], flush=True)
        proc.wait()
    elapsed = time.perf_counter() - t0
    status = "ok" if proc.returncode == 0 else f"FAILED(exit={proc.returncode})"
    print(f"  ⏱ {elapsed:.0f}s  → {status}  (日志 {log_path.relative_to(ROOT)})")
    return {"name": name, "desc": desc, "cmd": cmd, "status": status,
            "elapsed_s": round(elapsed, 1), "log": str(log_path.relative_to(ROOT))}

def main():
    ap = argparse.ArgumentParser(description="一键复现全部判别式实验（串行，单卡）")
    ap.add_argument("--group", choices=["all", "main", "pipeline", "bilstm", "ablation"],
                    default="all")
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定名字的实验")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--skip_gpu_check", action="store_true")
    args = ap.parse_args()

    todo = EXPERIMENTS
    if args.only:
        todo = [e for e in EXPERIMENTS if e[0] in args.only]
    elif args.group != "all":
        todo = [e for e in EXPERIMENTS if e[1] == args.group]
    if not todo:
        print("没有匹配的实验")
        return

    if not args.dry_run and not args.skip_gpu_check:
        used = gpu_used_mb()
        if used > 1024:
            print(f"⚠ 当前显存已占用 {used} MiB（>1GB）。本项目约定同一时刻只跑一个 GPU 任务，"
                  f"请先等其它任务结束（或加 --skip_gpu_check 强制继续）。")
            return
        print(f"GPU 检查通过：当前占用 {used} MiB")

    print(f"将串行执行 {len(todo)} 个实验：")
    for n, g, d, _ in todo:
        print(f"  - [{g}] {n}: {d}")

    results = []
    t_all = time.perf_counter()
    for name, group, desc, argv in todo:
        results.append(run_one(name, desc, argv, args.dry_run))

    total = time.perf_counter() - t_all
    print("\n" + "=" * 78)
    print(f"全部完成，总耗时 {total / 60:.1f} 分钟")
    for r in results:
        print(f"  {r['name']:<22} {r['status']:<22} "
              f"{r.get('elapsed_s', '')}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / "run_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"total_s": round(total, 1), "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"汇总已写入 → {out.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
