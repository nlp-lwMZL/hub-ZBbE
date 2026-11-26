#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data.py —— CNVD 漏洞严重级别分类 · 数据下载与三划分（定稿方案）

流程：
  1. 下载/校验原始 parquet（官方 train 116,871 + test 12,986，共 129,857 行）
  2. 合并成单一数据池 —— 官方 split 按 CNVD ID 划分，存在 15.7% 的描述级泄漏，
     因此官方 train/test 边界彻底废弃，一律从合并池重新划分
  3. 按 description.strip() 分组去重：同一描述对应多个 severity 的冲突组【整组丢弃】
     （不投票、不取多数，保证可复现；丢弃数量记入 manifest）
  4. 按 label 分层抽样：精确比例 + 最大余数法分配整数配额，固定 seed=42，
     一次性切出互斥四份（先切 test，再 val，再 train，剩余全部进 train_pool）
  5. 输出 7 个文件 + 全量校验，任何一项校验失败立即以非零码退出

产出（data/ 目录）：
  train_5k.jsonl / val_2k.jsonl / test_3k.jsonl   主实验三件套
  train_pool.jsonl                                 数据量消融专用（不含上面三份）
  all.jsonl                                        全量存档，每行带 split 字段
  label_map.json                                   唯一的标签映射来源
  manifest.json                                    生成参数 + 统计 + 校验结果

为什么 train_pool 不包含 train_5k / val / test：
  数据量消融（5k vs 10k vs 20k...）只允许从 train_pool 抽训练样本，
  否则会把评测样本抽进训练集 —— 那是数据泄漏，指标会虚高。
  做数据量消融时建议嵌套采样：N_k 组 = train_5k + 从 train_pool 再抽 (N-5)k，
  这样 5k vs 20k 才是纯粹的「数据量」变量，不混入抽样噪声。

为什么 val/test 的数字也来自最大余数法而不是四舍五入：
  精确比例约为 中 55.34% / 高 35.76% / 低 8.90%，四舍五入会让三个集合
  的加总不等于目标行数、且各集合边际分布有微小出入。

用法：
  python prepare_data.py           # 原始文件齐全时约 1 分钟内跑完
  python prepare_data.py --force  # 强制重新下载原始 parquet
  HF_ENDPOINT=https://hf-mirror.com python prepare_data.py   # 指定下载镜像
"""

import argparse
import json
import math
import random
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("缺少依赖：请先在 py312-tc 环境安装 pandas 和 pyarrow")

# ---------------------------------------------------------------- 常量（唯一事实来源）

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

DATASET_ID = "CIRCL/Vulnerability-CNVD"
TRAIN_FILE = "train-00000-of-00001.parquet"
TEST_FILE = "test-00000-of-00001.parquet"

# 下载源：依次尝试。huggingface.co 不通时自动退到 hf-mirror.com。
DEFAULT_SOURCES = ["https://huggingface.co", "https://hf-mirror.com"]
MIN_BYTES = {TRAIN_FILE: 10_000_000, TEST_FILE: 1_000_000}  # 完整性下限（实际 22.6MB / 2.6MB）

SEED = 42
SIZES = {"test_3k": 3000, "val_2k": 2000, "train_5k": 5000}  # 抽取顺序即字典顺序：先 test
LABEL_MAP = {"高": 0, "中": 1, "低": 2}  # 唯一映射来源，BERT/LLM 两侧代码都必须读 label_map.json


# ---------------------------------------------------------------- 下载

def ensure_raw(force: bool) -> dict:
    """确保两个原始 parquet 存在且完整，返回 {文件名: 路径}。"""
    import os

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sources = DEFAULT_SOURCES
    if os.environ.get("HF_ENDPOINT"):  # 允许用环境变量强制指定下载源
        sources = [os.environ["HF_ENDPOINT"].rstrip("/")]

    out = {}
    for fname in (TRAIN_FILE, TEST_FILE):
        dst = RAW_DIR / fname
        if force:
            dst.unlink(missing_ok=True)
        if dst.exists() and dst.stat().st_size >= MIN_BYTES[fname]:
            print(f"[skip] {fname} 已存在（{dst.stat().st_size:,} B）")
            out[fname] = dst
            continue

        ok = False
        for src in sources:
            url = f"{src}/datasets/{DATASET_ID}/resolve/main/data/{fname}"
            print(f"[down] {url}")
            tmp = dst.with_suffix(".tmp")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                if tmp.stat().st_size < MIN_BYTES[fname]:
                    raise IOError(f"下载不完整：仅 {tmp.stat().st_size:,} B")
                tmp.rename(dst)  # 先写临时文件再改名，避免半截文件被误认完整
                print(f"[done] {fname}（{dst.stat().st_size:,} B）")
                ok = True
                break
            except Exception as e:  # noqa: BLE001 —— 下载失败要换源重试，而不是崩溃
                print(f"[fail] {type(e).__name__}: {e}")
                tmp.unlink(missing_ok=True)
        if not ok:
            sys.exit(f"错误：{fname} 从所有源下载均失败，请检查网络或手动放置到 {RAW_DIR}")
        out[fname] = dst
    return out


# ---------------------------------------------------------------- 合并与去重

def load_and_clean(paths: dict):
    """合并官方 train+test，按 description 去重、丢弃标签冲突组。返回 (干净池 records, 统计 meta)。"""
    tr = pd.read_parquet(paths[TRAIN_FILE])
    te = pd.read_parquet(paths[TEST_FILE])
    raw_counts = {"官方train": len(tr), "官方test": len(te)}

    df = pd.concat([tr, te], ignore_index=True)
    df = df[["id", "title", "description", "severity"]]
    df = df.rename(columns={"severity": "label"})
    df["text"] = df["description"].astype(str).str.strip()  # 去重键 = description 去首尾空白
    df["title"] = df["title"].astype(str).str.strip()
    n_merged = len(df)

    df = df[df["text"].str.len() > 0]  # 防御：空描述直接丢弃
    n_empty = n_merged - len(df)

    # 标签合法性检查：数据集卡片声明只有 高/中/低 三类
    unknown = set(df["label"].unique()) - set(LABEL_MAP)
    if unknown:
        sys.exit(f"错误：出现 LABEL_MAP 之外的标签 {unknown}，请先人工检查数据")

    n_unique_texts = df["text"].nunique()

    # 冲突组：同一 description 对应多个 severity → 整组丢弃（不用投票，保证可复现）
    per_text_labels = df.groupby("text")["label"].nunique()
    conflict_keys = per_text_labels[per_text_labels > 1].index
    n_conflict_groups = int(len(conflict_keys))
    n_conflict_rows = int(df["text"].isin(conflict_keys).sum())

    clean = df[~df["text"].isin(conflict_keys)]
    # 组内多行且标签一致 → 压成一行。按 id 排序后取第一条，结果与运行环境无关
    clean = clean.sort_values("id").drop_duplicates("text", keep="first")

    records = clean[["id", "title", "text", "label"]].to_dict("records")
    meta = {
        "raw_rows": raw_counts,
        "merged_rows": n_merged,
        "empty_text_dropped": int(n_empty),
        "unique_texts": int(n_unique_texts),
        "conflict_groups_dropped": n_conflict_groups,
        "conflict_rows_dropped": n_conflict_rows,
        "clean_pool_size": len(records),
    }
    return records, meta


# ---------------------------------------------------------------- 分层抽样

def apportion(total: int, counts: dict, pool: int) -> dict:
    """最大余数法：把 total 个名额按 counts 的比例精确分到各类，加总严格等于 total。"""
    raw = {k: total * c / pool for k, c in counts.items()}
    base = {k: int(math.floor(v)) for k, v in raw.items()}
    rest = total - sum(base.values())
    # 余数从大到小补 1；余数相同按标签名排序，保证跨环境结果一致
    order = sorted(raw, key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:rest]:
        base[k] += 1
    return base


def stratified_split(records: list):
    """按类分层、固定 seed 一次性切出 test/val/train/train_pool 四份（互斥）。"""
    counts = {}
    for r in records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    pool = len(records)
    quotas = {name: apportion(n, counts, pool) for name, n in SIZES.items()}
    for name, n in SIZES.items():
        assert sum(quotas[name].values()) == n, f"{name} 配额加总 {sum(quotas[name].values())} != {n}"

    by_label = {lab: [] for lab in LABEL_MAP}
    for r in records:
        by_label[r["label"]].append(r)

    rng = random.Random(SEED)  # 同一个 rng 贯穿所有类别，配合固定的遍历顺序 → 结果可复现
    splits = {name: [] for name in SIZES}
    splits["train_pool"] = []
    for lab in LABEL_MAP:  # 固定遍历顺序 高→中→低
        bucket = by_label[lab]
        rng.shuffle(bucket)
        pos = 0
        for name in SIZES:  # 先切 test（最重要的集合最先拿走完整配额）
            take = quotas[name][lab]
            splits[name].extend(bucket[pos:pos + take])
            pos += take
        splits["train_pool"].extend(bucket[pos:])  # 剩余全部留给数据量消融
    return splits, quotas, counts, pool


# ---------------------------------------------------------------- 写文件

def with_label_id(r: dict, split_tag: str | None = None) -> dict:
    o = {
        "id": r["id"],
        "title": r["title"],  # 主实验不用 title，但存进文件：title+description 消融组不必重跑抽样
        "text": r["text"],
        "label": r["label"],
        "label_id": LABEL_MAP[r["label"]],  # 与 label 双存，防止两侧代码映射错位
    }
    if split_tag:
        o["split"] = split_tag
    elif "split" in r:  # all.jsonl 的行在 main 里已打好 split 标签，写出时保留
        o["split"] = r["split"]
    return o


def write_jsonl(path: Path, rows: list, split_tag: str | None = None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(with_label_id(r, split_tag), ensure_ascii=False) + "\n")
    print(f"[out] {path.name:<18} {len(rows):>7,} 行")


def write_label_map() -> None:
    path = DATA_DIR / "label_map.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, ensure_ascii=False, indent=2)
    print(f"[out] {path.name:<18} {json.dumps(LABEL_MAP, ensure_ascii=False)}")


# ---------------------------------------------------------------- 校验

def validate(files: dict, quotas: dict, meta: dict) -> dict:
    """读回全部产出文件做机器校验，任何一项失败直接退出非零码。"""
    rows = {}
    for name, path in files.items():
        with open(path, encoding="utf-8") as f:
            rows[name] = [json.loads(line) for line in f]

    checks = {}

    # 1. 行数
    expect = {"train_5k": 5000, "val_2k": 2000, "test_3k": 3000}
    expect["train_pool"] = meta["clean_pool_size"] - sum(expect.values())
    expect["all"] = meta["clean_pool_size"]
    checks["row_counts"] = all(len(rows[k]) == v for k, v in expect.items())
    if not checks["row_counts"]:
        bad = {k: (len(rows[k]), v) for k, v in expect.items() if len(rows[k]) != v}
        sys.exit(f"校验失败 · 行数不符: {bad}")

    # 2. 三件套 + pool 互斥（按 text），并集覆盖全池
    texts = {k: {r["text"] for r in rows[k]} for k in ("train_5k", "val_2k", "test_3k", "train_pool")}
    n_union = len(texts["train_5k"] | texts["val_2k"] | texts["test_3k"] | texts["train_pool"])
    n_sum = sum(len(v) for v in texts.values())
    checks["mutually_exclusive"] = (n_union == n_sum == meta["clean_pool_size"])
    if not checks["mutually_exclusive"]:
        sys.exit(f"校验失败 · 集合不互斥或覆盖不全: union={n_union} sum={n_sum} pool={meta['clean_pool_size']}")

    # 3. all.jsonl 的 split 字段与四个文件一一对应
    from collections import Counter
    split_counter = Counter(r["split"] for r in rows["all"])
    checks["all_split_tags"] = (
        split_counter["train"] == 5000 and split_counter["val"] == 2000
        and split_counter["test"] == 3000 and split_counter["pool"] == expect["train_pool"]
    )
    if not checks["all_split_tags"]:
        sys.exit(f"校验失败 · all.jsonl 的 split 统计异常: {dict(split_counter)}")

    # 4. 每个文件的分布与配额一致，且 label_id 映射正确
    for name in ("train_5k", "val_2k", "test_3k"):
        dist = Counter(r["label"] for r in rows[name])
        checks[f"dist_{name}"] = all(dist[k] == v for k, v in quotas[name].items())
        if not checks[f"dist_{name}"]:
            sys.exit(f"校验失败 · {name} 分布与配额不符: {dict(dist)} vs {quotas[name]}")
    checks["label_id_consistent"] = all(
        r["label_id"] == LABEL_MAP[r["label"]] for name in rows for r in rows[name]
    )
    if not checks["label_id_consistent"]:
        sys.exit("校验失败 · label_id 与 label_map 不一致")

    # 5. id 全局唯一（防上游脏数据混入重复行）
    ids = [r["id"] for name in ("train_5k", "val_2k", "test_3k", "train_pool") for r in rows[name]]
    checks["ids_unique"] = len(ids) == len(set(ids))
    if not checks["ids_unique"]:
        sys.exit("校验失败 · id 出现重复，请检查上游数据")

    checks["all_passed"] = True
    return checks


# ---------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser(description="CNVD 数据下载与三划分（定稿方案）")
    ap.add_argument("--force", action="store_true", help="强制重新下载原始 parquet")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = ensure_raw(args.force)

    records, meta = load_and_clean(paths)
    print(
        f"[clean] 合并 {meta['merged_rows']:,} 行 → 唯一描述 {meta['unique_texts']:,} 组；"
        f"丢弃冲突组 {meta['conflict_groups_dropped']}（涉及 {meta['conflict_rows_dropped']:,} 行）"
        f" → 干净池 {meta['clean_pool_size']:,}"
    )

    splits, quotas, counts, pool = stratified_split(records)
    ratios = {k: round(c / pool, 4) for k, c in counts.items()}
    print(f"[split] 池内精确比例 {ratios}，seed={SEED}")

    files = {
        "train_5k": DATA_DIR / "train_5k.jsonl",
        "val_2k": DATA_DIR / "val_2k.jsonl",
        "test_3k": DATA_DIR / "test_3k.jsonl",
        "train_pool": DATA_DIR / "train_pool.jsonl",
        "all": DATA_DIR / "all.jsonl",
    }
    write_jsonl(files["train_5k"], splits["train_5k"], "train")
    write_jsonl(files["val_2k"], splits["val_2k"], "val")
    write_jsonl(files["test_3k"], splits["test_3k"], "test")
    write_jsonl(files["train_pool"], splits["train_pool"], "pool")
    all_rows = ([dict(r, split="train") for r in splits["train_5k"]]
                + [dict(r, split="val") for r in splits["val_2k"]]
                + [dict(r, split="test") for r in splits["test_3k"]]
                + [dict(r, split="pool") for r in splits["train_pool"]])
    write_jsonl(files["all"], all_rows)
    write_label_map()

    checks = validate(files, quotas, meta)

    manifest = {
        "seed": SEED,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "dataset": DATASET_ID,
            "files": {f: {"bytes": paths[f].stat().st_size} for f in paths},
            "note": "官方 train/test 按 CNVD ID 划分，存在 15.7% 描述级泄漏，已合并废弃",
        },
        "dedup": {
            "key": "description.strip()",
            **meta,
        },
        "label_map": LABEL_MAP,
        "exact_ratios": ratios,
        "quotas": {k: {lab: quotas[k][lab] for lab in LABEL_MAP} for k in SIZES},
        "outputs": {
            k: {"rows": len(splits[k]),
                "dist": {lab: sum(1 for r in splits[k] if r["label"] == lab) for lab in LABEL_MAP}}
            for k in ("train_5k", "val_2k", "test_3k", "train_pool")
        },
        "usage_rules": [
            "val 只用于选 epoch / 最佳 ckpt；所有消融组的最终指标一律在 test 上报",
            "数据量消融只从 train_pool.jsonl 抽样，建议嵌套：N_k = train_5k + 从 pool 抽 (N-5)k",
            "all.jsonl 仅作存档与追溯，禁止直接当训练集",
        ],
        "checks": checks,
    }
    with open(DATA_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("[out] manifest.json")

    print("\n===== 摘要 =====")
    print(f"干净池 {meta['clean_pool_size']:,}（中 {counts['中']:,} / 高 {counts['高']:,} / 低 {counts['低']:,}）")
    for k in ("train_5k", "val_2k", "test_3k", "train_pool"):
        d = manifest["outputs"][k]["dist"]
        print(f"{k:<11} {d['中']:>6,} 中 / {d['高']:>6,} 高 / {d['低']:>5,} 低   共 {manifest['outputs'][k]['rows']:>7,} 行")
    print("全部校验通过 ✓")


if __name__ == "__main__":
    main()
