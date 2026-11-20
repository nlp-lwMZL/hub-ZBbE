"""
数据读取与「干净子集」构建（全项目共用）

教学重点：
  1. 这份 TNEWS 切分里，验证集有 1099/10000 条与训练集**完全重复**（11%）。
     不做处理的话，靠词面匹配的 TF-IDF 会「背答案」，指标被显著虚高
     （实测：全量 val 上 F1 0.541，去重后只有 0.509）。
  2. 所以本项目所有方法统一在 clean 子集上评测——**同一批样本**是横向比较的前提。
  3. 重复样本不是这份数据独有的问题：真实业务里同一条公告被多家媒体转载、
     同一个问句被不同用户重复提交，都会造成「训练/验证泄漏」。

用法：
  from common.data import load_splits, clean_val, sample_subset
"""

import json
import random
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = CODE_ROOT / "data"


def _load(name: str):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def load_splits():
    """返回 (train, val)，每条是 {"idx", "sentence", "label"}"""
    return _load("train.json"), _load("val.json")


def clean_val(train=None, val=None):
    """
    去掉与训练集句子完全重复的验证样本，返回 (clean_list, n_dup)。

    为什么按「整句完全相同」判定：
      这里要暴露的是最朴素的泄漏形态——同一条数据被切到了两边。
      更隐蔽的近似重复（改标题、换标点）需要 SimHash / 向量去重，属于第 8 周的内容。
    """
    if train is None or val is None:
        train, val = load_splits()
    seen = {d["sentence"] for d in train}
    clean = [d for d in val if d["sentence"] not in seen]
    return clean, len(val) - len(clean)


def sample_subset(items, n, seed=42):
    """
    固定 seed 抽样：四次实验（规则/TF-IDF/BERT/LLM）必须用同一批样本，
    否则「差 2 个点」可能只是抽样波动。n 超过总数时直接返回全部。
    """
    if n >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, n)


def load_clean_val(n=None, seed=42):
    """最常用入口：读数据 → 去重 →（可选）抽样，返回 (samples, meta)"""
    train, val = load_splits()
    clean, n_dup = clean_val(train, val)
    samples = sample_subset(clean, n if n else len(clean), seed)
    meta = {
        "n_train": len(train),
        "n_val": len(val),
        "n_dup": n_dup,
        "dup_rate": round(n_dup / len(val), 4) if val else 0.0,
        "n_clean": len(clean),
        "n_eval": len(samples),
        "seed": seed,
    }
    return samples, meta
