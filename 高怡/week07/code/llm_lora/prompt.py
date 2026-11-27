"""提示词与目标 JSON 构造（LLM 事件抽取）。

设计要点
--------
1. **不让模型输出 start/end**：大模型不擅长精确数数，位置由 ``align.py`` 在解码后
   用「子串回填」的方式在原文本中定位。
2. **模型只输出助手轮次**：system/user 部分在 ``data.py`` 里统一置 ``label=-100``，
   不参与 loss。
3. **无 think**：Qwen3 在 ``enable_thinking=False`` 时仍会注入一个空的 think 块，
   这里把它作为 prompt 的一部分（见 train.py），于是生成从 JSON 直接开始。
"""

from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.schema import COMMON_ROLES, EVENT_SCHEMA  # noqa: E402

NO_EVENT = "无"


@lru_cache(maxsize=1)
def build_system_prompt() -> str:
    lines = [
        "你是基金公司人事变动公告的事件抽取器。请从用户给出的公告文本中抽取一个事件，"
        "并且**只输出一个 JSON 对象**。",
        "",
        "输出格式：",
        '{"event_type": "<事件类型>", "trigger": "<触发词原文>", '
        '"arguments": [{"role": "<槽位名>", "text": "<原文片段>"}]}',
        "",
        "规则：",
        "1. event_type 只能取以下之一：" + "、".join(EVENT_SCHEMA.keys()) + f"、{NO_EVENT}。",
        "2. trigger 必须是原文中表示该事件的词，且与原文**逐字一致**。",
        "3. arguments 的 role 只能取「公共槽位」或该事件类型的「专属槽位」；"
        "text 必须与原文**逐字一致**。",
        "4. 原文中不存在的槽位直接省略，不要输出 null，不要编造。",
        "5. 不要输出 start / end / 位置 / 偏移等任何位置信息。",
        "6. 不要输出 markdown 代码块、解释、思考过程，只输出 JSON 本身。",
        f'7. 若文本中没有符合的事件，输出 {{"event_type": "{NO_EVENT}", "trigger": "", "arguments": []}}。',
        "",
        "公共槽位（所有事件类型通用）：" + "、".join(COMMON_ROLES),
        "各事件类型专属槽位：",
    ]
    common = set(COMMON_ROLES)
    for ev_type, spec in EVENT_SCHEMA.items():
        specific = [r for r in spec["roles"] if r not in common]
        lines.append(f"- {ev_type}: " + ("、".join(specific) if specific else "（无）"))
    return "\n".join(lines)


def build_user_prompt(text: str) -> str:
    return f"公告文本：\n{text}\n\n请输出该公告对应的事件 JSON。"


def _role_order(ev_type: str) -> dict[str, int]:
    roles = EVENT_SCHEMA.get(ev_type, {}).get("roles", [])
    return {r: i for i, r in enumerate(roles)}


def build_target(event: dict) -> str:
    """把标注事件转成训练目标 JSON 字符串（去掉 start/end）。"""
    if not event:
        return json.dumps({"event_type": NO_EVENT, "trigger": "", "arguments": []},
                          ensure_ascii=False, separators=(",", ":"))
    ev_type = str(event.get("type") or NO_EVENT)
    trigger = (event.get("trigger") or {}).get("text", "") or ""
    args = []
    for a in event.get("args") or []:
        role = (a.get("role") or "").strip()
        txt = (a.get("text") or "").strip()
        if role and txt:
            args.append({"role": role, "text": txt})
    order = _role_order(ev_type)
    args.sort(key=lambda a: (order.get(a["role"], 999), a["text"]))
    obj = {"event_type": ev_type, "trigger": trigger, "arguments": args}
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_model_output(raw: str) -> dict | None:
    """从生成文本里抽取第一个 JSON 对象；失败返回 None。"""
    if not raw:
        return None
    text = raw.strip()
    # 去掉可能出现的代码块围栏
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # 先尝试整体解析，再尝试截取最外层花括号
    candidates = [text]
    m = _JSON_OBJ_RE.search(text)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict) and "event_type" in obj:
            return obj
    return None


def normalize_prediction(obj: dict) -> dict:
    """把模型输出规整为 {"type", "trigger", "arguments":[{role,text}]}。

    非法事件类型统一视为「无事件」；槽位不在该类型 schema 内的丢弃。
    """
    ev_type = str(obj.get("event_type") or NO_EVENT).strip()
    if ev_type not in EVENT_SCHEMA:
        ev_type = NO_EVENT
    trigger = obj.get("trigger")
    trigger = trigger.strip() if isinstance(trigger, str) else ""
    allowed = set(EVENT_SCHEMA.get(ev_type, {}).get("roles", []))
    args = []
    raw_args = obj.get("arguments")
    if isinstance(raw_args, list):
        for a in raw_args:
            if not isinstance(a, dict):
                continue
            role = str(a.get("role") or "").strip()
            txt = str(a.get("text") or "").strip()
            if role in allowed and txt:
                args.append({"role": role, "text": txt})
    return {"type": ev_type, "trigger": trigger, "arguments": args}
