"""
一键串行跑三类方法的正式实验 + 断点续跑 + fail-fast。

设计（week6 的成熟经验）：
  - 每个实验先算目标指标文件路径，**有结果就跳过**（断线重连重跑同一命令即可续）；
  - fail-fast：每步检查退出码，非 0 立即停下并打印 [FAIL]，不掩盖故障；
  - 只跑「训练」，评测单独用各臂的 evaluate.py / run_eval.sh，避免训练跑一半去评测。

实验清单（可通过 --only / --skip 筛选，逗号分隔 run_name）：
  1. bilstm_crf   L1 字符级 BiLSTM+CRF（5 epoch，CPU/GPU 都可）
  2. bilstm_linear L1 对照：去掉 CRF
  3. roberta_linear L2 RoBERTa+Linear
  4. roberta_crf    L2 RoBERTa+CRF（主线）
  5. qwen_lora_sft  L4 Qwen3-0.6B LoRA SFT（只训旁路）
  6. qwen_full_sft  L4 Qwen3-0.6B 全量 SFT（分层学习率，消融「训多少参数」）

用法：
  python eval/run_all.py --list                # 看会跑什么
  python eval/run_all.py                       # 跑全部 tier1
  python eval/run_all.py --only roberta_crf    # 只补跑某组
  python eval/run_all.py --force               # 忽略已完成强制重跑
  python eval/run_all.py --tokenizer /本机/bert # 本机冒烟用本地 bert 代替 roberta
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from common import paths

PY = sys.executable

EXPERIMENTS = [
    {"name": "bilstm_crf", "family": "L1", "cmd": [PY, "bilstm_crf/train.py", "--decoder", "crf",
        "--run_name", "bilstm_crf", "--epochs", "5"]},
    {"name": "bilstm_linear", "family": "L1", "cmd": [PY, "bilstm_crf/train.py", "--decoder", "linear",
        "--run_name", "bilstm_linear", "--epochs", "5"]},
    {"name": "roberta_linear", "family": "L2", "cmd": [PY, "bert/train.py", "--decoder", "linear",
        "--run_name", "roberta_linear", "--epochs", "3"]},
    {"name": "roberta_crf", "family": "L2", "cmd": [PY, "bert/train.py", "--decoder", "crf",
        "--run_name", "roberta_crf", "--epochs", "3"]},
    {"name": "qwen_lora_sft", "family": "L4", "cmd": [PY, "llm_lora/train_sft.py",
        "--mode", "lora", "--run_name", "qwen_lora_sft", "--epochs", "1", "--num_train", "2000"]},
    {"name": "qwen_full_sft", "family": "L4", "cmd": [PY, "llm_lora/train_sft.py",
        "--mode", "full", "--run_name", "qwen_full_sft", "--epochs", "1", "--num_train", "2000",
        "--grad_ckpt", "--optim", "adamw8bit", "--dtype", "fp32", "--lr", "2e-5"]},
]


def done_marker(name: str) -> Path:
    return paths.log_dir() / f"summary_{name}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None, help="只跑这些 run_name（逗号分隔）")
    ap.add_argument("--skip", default=None, help="跳过这些 run_name（逗号分隔）")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tokenizer", default=None, help="传给 bert 臂的本地 tokenizer（本机冒烟用）")
    ap.add_argument("--bert_path", default=None, help="传给 bert 臂的本地底座")
    a = ap.parse_args()

    only = set(a.only.split(",")) if a.only else None
    skip = set(a.skip.split(",")) if a.skip else set()

    print("实验清单：")
    for e in EXPERIMENTS:
        mark = "✓已完成" if done_marker(e["name"]).exists() and not a.force else "待跑"
        print(f"  [{e['family']}] {e['name']:<16} {mark}")
    if a.list:
        return

    ran, skipped, failed = 0, 0, 0
    for e in EXPERIMENTS:
        name = e["name"]
        if only is not None and name not in only:
            continue
        if name in skip:
            continue
        marker = done_marker(name)
        if marker.exists() and not a.force:
            print(f"\n[SKIP] {name} 已有结果 {marker}")
            skipped += 1
            continue
        cmd = list(e["cmd"])
        # 本机冒烟：把本地 tokenizer/bert 追加到 bert 臂命令
        if name.startswith("roberta"):
            if a.tokenizer:
                cmd += ["--tokenizer", a.tokenizer]
            if a.bert_path:
                cmd += ["--bert_path", a.bert_path]
        print(f"\n[RUN] {name}: {' '.join(cmd)}")
        cp = subprocess.run(cmd, cwd=str(CODE_DIR))
        if cp.returncode != 0:
            print(f"[FAIL] {name} 退出码 {cp.returncode}，停止后续实验")
            failed += 1
            break
        ran += 1
        print(f"[OK] {name} 完成")

    print(f"\n==== 汇总：跑 {ran} / 跳 {skipped} / 失败 {failed} ====")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
