# -*- coding: utf-8 -*-
"""
事件数据读写 + 固定 seed 采样 + 统一目标 JSON 构造（LLM 链路的公共底座）

教学重点 / 为什么单独抽这个模块：
  1. **同一批样本**：llm_zero_shot/（本目录 LLM）、bert/ 等判别式方法、service/（系统组合）要横向比 F1，
     如果各自随机抽样本，"谁更高"就毫无意义 —— 采样规则只在这里实现一次，全员复用。
     （真实踩坑：两个脚本都写 seed=42，但一个 random.sample、一个分层采样，
       实测 100 条里只重合 1 条，gold 论元数 142 vs 319，难度分布都不同。）
  2. **样本唯一键**：DuEE-Fin 的 `id` 是「文档 id」不是「句子 id」——
     一篇公告切出多句带事件的句子，它们共用同一个 id（dev 930 句只有 823 个 id）。
     所以唯一键必须是 `id|sent_id`，预测文件也必须把 sent_id 一起带出来；
     只按 id 对齐会让后面的句子覆盖前面的，指标莫名其妙掉几个点。
  3. **目标 JSON 三处必须逐字一致**：训练（train_sft.py）、推理（llm_extract.py / evaluate_sft.py）、
     评测（common/evaluate_events.py）。格式只要差一个字段名（args vs arguments），
     模型学到的就是另一种格式，评测直接判全错。所以 to_target_json 只写一份。

使用方式：
  from event_data import load_schema, load_split, sample_rows, write_eval_ids
  schema = load_schema()
  rows   = sample_rows(load_split("dev"), n=200)          # 固定 seed=42
  target = to_target_json(row["event"])                   # 训练目标 JSON 字符串
"""

import json
import random
from pathlib import Path

# 项目根 = llm_zero_shot/ 的上一级（母版/副本都成立）
ROOT           = Path(__file__).resolve().parent.parent
DATA_DIR       = ROOT / "data" / "processed"
OUTPUT_LLM_DIR = ROOT / "outputs" / "llm"

SCHEMA_PATH       = DATA_DIR / "event_schema.json"
TRAIN_PATH        = DATA_DIR / "train_extract.jsonl"
DEV_PATH          = DATA_DIR / "dev_extract.jsonl"

SEED = 42  # 全项目统一随机种子（采样、mock、训练初始化都用它）


# ══════════════════════════════════════════════════════════════════════════════
# 读写
# ══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path) -> list[dict]:
    """逐行读 JSONL（空行跳过）。大文件也安全：一行一条，不整体进内存的 JSON 数组。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_schema(path=None) -> dict:
    """
    读事件类型 → 论元角色 的定义。

    event_schema.json 的结构：
      {"质押": ["质押方", "质权方", ...], ..., "__enums__": {"公司上市.环节": [...]}}
    注意 `__enums__` 不是事件类型（它放的是枚举型角色的候选值），
    直接 for 遍历 schema 会把 "__enums__" 当成第 14 个事件类型 —— 这是常见坑。
    """
    path = Path(path) if path else SCHEMA_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    schema = {k: list(v) for k, v in raw.items() if not k.startswith("__")}
    enums = raw.get("__enums__", {})
    return {"event_types": schema, "enums": enums}


def load_split(split: str = "dev") -> list[dict]:
    """读 train / dev 的句子级单事件数据（一行一个句子一个事件）。"""
    path = TRAIN_PATH if split == "train" else DEV_PATH
    return load_jsonl(path)


def uid(row: dict) -> str:
    """样本唯一键：id|sent_id（详见模块头部的坑 2）。"""
    sent_id = row.get("sent_id")
    return f'{row.get("id", "")}|{sent_id}' if sent_id else str(row.get("id", ""))


# ══════════════════════════════════════════════════════════════════════════════
# 固定 seed 采样（全项目唯一口径）
# ══════════════════════════════════════════════════════════════════════════════

def canonical_order(rows: list[dict], seed: int = SEED) -> list[dict]:
    """
    把 dev 打成一个**确定性**的随机顺序，供各方案按"取前 n 条"的方式抽样。

    为什么先按 (id, sent_id) 排序再 shuffle，而不是直接 random.sample？
      random.sample 的结果依赖「输入列表的顺序」。dev_extract.jsonl 以后只要重新生成
      （比如换了一版清洗脚本、行序变了），同一个 seed 抽出来的样本就全变了，
      历史数字立刻不可比。先排序再 shuffle 就把顺序依赖去掉：
      只要样本集合一样，谁跑、哪天跑、文件行序怎么变，抽到的都是同一批。

    为什么返回"顺序"而不是"抽样结果"：
      这样 n=100/150/200 的评测集是**嵌套子集**（前 150 = 前 200 的前 150），
      小样本实验结果可以直接和大样本结果并排看，不会出现"换了样本所以数字不可比"。

    复现方式（其它同学/判别式链路照抄这三行即可）：
      rows = sorted(rows, key=lambda r: (r["id"], r["sent_id"]))
      random.Random(42).shuffle(rows)
      rows[:n]
    """
    ordered = sorted(rows, key=lambda r: (str(r.get("id", "")), str(r.get("sent_id", ""))))
    random.Random(seed).shuffle(ordered)
    return ordered


def sample_rows(rows: list[dict], n: int, seed: int = SEED) -> list[dict]:
    """取 canonical_order 的前 n 条（n 超过总条数时全取）。"""
    return canonical_order(rows, seed)[: min(n, len(rows))]


def write_eval_ids(path, rows_sampled: list[dict], seed: int = SEED) -> None:
    """
    记录本次评测实际用了哪些样本（id + sent_id），供 leader 与判别式结果对比。

    文件结构：
      {"seed": 42,
       "rule": "sort by (id,sent_id) then random.Random(42).shuffle, take prefix",
       "ids": {"n200": [{"id":..., "sent_id":...}, ...], ...}}
    同一个文件里可以并存多个 n（n150 是 n200 的前缀），用 merge 写入。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    data["seed"] = seed
    data["rule"] = "sort by (id,sent_id) then random.Random(seed).shuffle, take prefix"
    data.setdefault("ids", {})[f"n{len(rows_sampled)}"] = [
        {"id": r.get("id"), "sent_id": r.get("sent_id")} for r in rows_sampled
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_gold_subset(path, rows: list[dict]) -> None:
    """
    把 sampled 的金标单独写一份，供 common/evaluate_events.py 当 --gold。

    为什么不能直接拿整个 dev_extract 当 --gold：
      评测脚本遇到"预测文件里没有的样本"会按「全漏」处理（这是对的，防止删分母）。
      但如果你只跑了 200 条却把 930 条 dev 全当 gold，剩下 730 条就全算漏抽，
      F1 会被压低到只剩真实值的 1/5 —— 这不是模型差，是命令用错了。

    保留原行（含 event 字段）不改造：评测脚本自己会把 {"event": ...} 包成 events 列表。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 目标 JSON / 预测 JSON 构造
# ══════════════════════════════════════════════════════════════════════════════

def to_target_json(event: dict) -> str:
    """
    金标事件 → 训练目标 JSON 字符串（**训练与推理共用同一套字段名**）。

    为什么字段名用 arguments / argument，而不是 gold 里的 args / text：
      - gold 的 `args` 里带 start/end（我们不想让模型学 offset，见 event_prompt.py 的说明）；
      - 换一套"干净"的名字（arguments/argument）能让学生一眼看出
        「发给模型的是文本 span，不是 offset」，避免误把 gold 原样塞进 prompt。
      评测脚本两种名字都认（args/arguments、text/argument），所以不影响统一评测。

    输出形状（单事件句就一个元素）：
      {"events":[{"type":"质押",
                  "trigger":{"text":"质押"},
                  "arguments":[{"role":"质押方","argument":"傅宇晨"}]}]}
    """
    trig = event.get("trigger") or {}
    trig_text = trig.get("text", "") if isinstance(trig, dict) else str(trig)
    arguments = []
    for a in event.get("args") or []:
        text = a.get("text", "") or ""
        role = a.get("role", "") or ""
        if text and role:
            arguments.append({"role": role, "argument": text})
    return json.dumps(
        {"events": [{"type": event.get("type", ""),
                     "trigger": {"text": trig_text},
                     "arguments": arguments}]},
        ensure_ascii=False,
    )


def make_pred_row(row: dict, events: list[dict]) -> dict:
    """
    组装预测行：**必须带 id + sent_id**（评测按 id|sent_id 对齐，见模块头部坑 2）。

    events 为空列表时也要写 —— 解析失败/空回复的行必须保留在预测文件里。
    删掉这些行会让分母变小，F1 虚高（这是学生最容易犯的指标作弊错误）。
    """
    return {"id": row.get("id"), "sent_id": row.get("sent_id"), "text": row.get("text", ""),
            "events": events}


def gold_events_of(row: dict) -> list[dict]:
    """取出金标事件列表（数据是句子级单事件，所以最多 1 个）。"""
    if isinstance(row.get("event"), dict):
        return [row["event"]]
    return row.get("events") or []
