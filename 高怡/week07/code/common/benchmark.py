"""推理延迟 / 吞吐基准测试（与训练解耦，不影响训练）。

测量两个口径：
  - 在线延迟：batch=1，单条公告端到端耗时（含滑窗多 chunk），报 mean/p50/p95；
  - 批量吞吐：batch=N，每秒处理文档数 / 字符数。

按公告字符数分桶（短 <=300、中 301-600、长 >600），便于观察长度对延迟的影响。

示例
----
# 自动使用最新一次训练的 best checkpoint
python -m common.benchmark

# 指定 checkpoint
python -m common.benchmark --ckpt extract/output/bilstm/<run>/bilstm_crf_best.pt

# 快速验证（只取 50 条）
python -m common.benchmark --limit 50

结果会打印到控制台，并写入 checkpoint 同目录的 latency.json。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.data import (  # noqa: E402
    EventBIODataset,
    Vocab,
    build_dataloader,
    load_records,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="事件抽取推理延迟/吞吐基准测试")
    p.add_argument("--ckpt", default="", help="checkpoint 路径；默认取最新 run 的 best")
    p.add_argument("--data", default=str(PROJECT_ROOT / "data/dataset/split/val.jsonl"))
    p.add_argument("--arch", default="bilstm", choices=["bilstm"])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--stride", type=int, default=384)
    p.add_argument("--warmup", type=int, default=10, help="预热样本数（不计入统计）")
    p.add_argument("--batch-size", type=int, default=32, help="吞吐测试的 batch 大小")
    p.add_argument("--limit", type=int, default=0, help="只取前 N 条（快速验证）")
    p.add_argument("--output", default="", help="结果 json 路径；默认 ckpt 同目录 latency.json")
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def find_latest_ckpt() -> Path:
    candidates = list((PROJECT_ROOT / "extract/output").glob("*/*/*_best.pt"))
    if not candidates:
        raise FileNotFoundError("未找到 checkpoint，请用 --ckpt 指定")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_model(ckpt: dict, arch: str) -> torch.nn.Module:
    cfg = ckpt["model_config"]
    if arch == "bilstm":
        from extract.bilstm_crf.model import BiLSTMCRF

        return BiLSTMCRF(**cfg)
    raise ValueError(f"不支持的 arch: {arch}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _stats(values: list[float]) -> dict:
    """values 单位为秒，返回毫秒统计。"""
    if not values:
        return {"n": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    s = sorted(values)
    n = len(s)
    return {
        "n": n,
        "mean_ms": round(statistics.mean(s) * 1000, 3),
        "p50_ms": round(s[n // 2] * 1000, 3),
        "p95_ms": round(s[min(n - 1, int(n * 0.95))] * 1000, 3),
    }


def _bucket(n_chars: int) -> str:
    if n_chars <= 300:
        return "short"
    if n_chars <= 600:
        return "medium"
    return "long"


@torch.no_grad()
def measure(
    model: torch.nn.Module,
    records: list[dict],
    vocab: Vocab,
    device: torch.device,
    max_len: int,
    stride: int,
    warmup: int,
    batch_size: int,
) -> dict:
    model.eval()
    ds = EventBIODataset(records, vocab, max_len=max_len, stride=stride)
    doc_len = {str(r.get("source", "")): len(r.get("text") or "") for r in records}

    # ---------- 在线延迟：batch=1 ----------
    loader1 = build_dataloader(ds, batch_size=1, shuffle=False, pad_id=vocab.pad_id)
    per_doc: dict[str, float] = defaultdict(float)
    for i, batch in enumerate(loader1):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        if i < warmup:
            model(ids, mask)
            continue
        _sync(device)
        t0 = time.perf_counter()
        model(ids, mask)
        _sync(device)
        per_doc[batch["meta"][0]["source"]] += time.perf_counter() - t0

    times = list(per_doc.values())
    total_chars = sum(doc_len.get(s, 0) for s in per_doc)
    latency = {
        "docs": len(times),
        "per_doc": _stats(times),
        "ms_per_kchar": round(sum(times) * 1000 / max(total_chars / 1000, 1e-9), 3),
        "by_length": {},
    }
    bucket_times: dict[str, list[float]] = defaultdict(list)
    for src, t in per_doc.items():
        bucket_times[_bucket(doc_len.get(src, 0))].append(t)
    for name in ("short", "medium", "long"):
        if bucket_times.get(name):
            latency["by_length"][name] = _stats(bucket_times[name])

    # ---------- 批量吞吐：batch=N ----------
    loaderN = build_dataloader(ds, batch_size=batch_size, shuffle=False, pad_id=vocab.pad_id)
    n_chunks = 0
    model_time = 0.0
    for i, batch in enumerate(loaderN):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        if i < 1:  # 预热一个 batch
            model(ids, mask)
            continue
        _sync(device)
        t0 = time.perf_counter()
        model(ids, mask)
        _sync(device)
        model_time += time.perf_counter() - t0
        n_chunks += ids.shape[0]

    throughput = {
        "batch_size": batch_size,
        "chunks": n_chunks,
        "chunks_per_sec": round(n_chunks / max(model_time, 1e-9), 2),
        "chars_per_sec": round(total_chars / max(model_time, 1e-9), 1),
        "total_model_sec": round(model_time, 2),
    }
    return {"latency": latency, "throughput": throughput}


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    ckpt_path = Path(args.ckpt) if args.ckpt else find_latest_ckpt()

    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = Vocab.load(ckpt_path.parent / "vocab.json")
    model = build_model(ckpt, args.arch).to(device)
    model.load_state_dict(ckpt["model_state"])

    records = load_records(args.data, limit=args.limit)
    print(f"checkpoint: {ckpt_path}")
    print(f"device: {device} | 文档数: {len(records)} | warmup: {args.warmup}")

    result = {
        "ckpt": str(ckpt_path),
        "device": str(device),
        "arch": args.arch,
        "max_len": args.max_len,
        "stride": args.stride,
        **measure(
            model, records, vocab, device,
            args.max_len, args.stride, args.warmup, args.batch_size,
        ),
    }

    lat = result["latency"]
    print("\n[在线延迟 batch=1]")
    print(
        f"  每条文档: mean={lat['per_doc']['mean_ms']}ms "
        f"p50={lat['per_doc']['p50_ms']}ms p95={lat['per_doc']['p95_ms']}ms "
        f"(n={lat['per_doc']['n']})"
    )
    print(f"  每千字: {lat['ms_per_kchar']}ms")
    for name, st in lat["by_length"].items():
        print(
            f"  [{name}] mean={st['mean_ms']}ms p50={st['p50_ms']}ms "
            f"p95={st['p95_ms']}ms (n={st['n']})"
        )

    tp = result["throughput"]
    print(f"\n[批量吞吐 batch={tp['batch_size']}]")
    print(
        f"  {tp['chunks_per_sec']} chunk/s | {tp['chars_per_sec']} char/s "
        f"（模型耗时 {tp['total_model_sec']}s）"
    )

    out = Path(args.output) if args.output else ckpt_path.parent / "latency.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
