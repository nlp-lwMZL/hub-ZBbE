"""推理后处理：用规则补齐 4 类「说明」槽位。

这 4 类槽位（聘任说明 / 变动说明 / 履职说明 / 变动内容）会覆盖整句、与其它槽位
重叠，无法做字符级 BIO 标注，因此不进入模型标签空间，改由本模块在推理后按规则补。

规则与 build_event_annotation_dataset.py 的生成口径保持一致：
  - clause_span：优先取「上述调整自公告之日起生效」等固定短语；否则从 trigger
    起点向右扩展到句号/分号或最多 60 字；
  - 槽位名由事件类型决定：助理聘任→聘任说明；代为/恢复履职→履职说明；其余→变动说明；
  - 变动内容：仅「其他人事变动」使用，取公告标题（找不到则用 trigger）；
  - 说明类仅在论元不足 min_args（默认 4）时才补。
"""

from __future__ import annotations

import bisect
import re

from .label import EXCLUDED_ROLES, TRIGGER_LABEL
from .schema import infer_event_type

EXPLANATION_PHRASES: tuple[str, ...] = (
    "上述调整自公告之日起生效",
    "自公告之日起生效",
    "上述事项自公告之日起生效",
)

# 事件类型 -> 兜底说明槽位名
EXPLANATION_ROLE: dict[str, str] = {
    "基金经理助理聘任": "聘任说明",
    "代为履职": "履职说明",
    "恢复履职": "履职说明",
}
DEFAULT_EXPLANATION_ROLE = "变动说明"


class Compact:
    """去空白串 + 到原文 offset 的映射（与生成脚本等价的最小实现）。"""

    def __init__(self, text: str):
        chars: list[str] = []
        offsets: list[int] = []
        for i, ch in enumerate(text):
            if not ch.isspace():
                chars.append(ch)
                offsets.append(i)
        self.text = text
        self.s = "".join(chars)
        self.offsets = offsets

    def span(self, a: int, b: int) -> dict | None:
        if a < 0 or b <= a or b > len(self.offsets):
            return None
        start, end = self.offsets[a], self.offsets[b - 1] + 1
        return {"text": self.text[start:end], "start": start, "end": end}


def clause_span(text: str, trigger_start: int, max_len: int = 60) -> dict | None:
    """取说明句：优先固定短语，否则 trigger 所在句（到 。；; 或 max_len 截止）。"""
    c = Compact(text)
    for phrase in EXPLANATION_PHRASES:
        p = c.s.find(phrase)
        if p >= 0:
            return c.span(p, p + len(phrase))
    a = bisect.bisect_left(c.offsets, trigger_start)
    b = a
    while b < len(c.s) and b - a < max_len and c.s[b] not in "。；;":
        b += 1
    return c.span(a, b) if b > a else None


def _add(out: list[dict], role: str, span: dict | None) -> None:
    if not span or not span["text"].strip():
        return
    out.append(
        {"role": role, "text": span["text"], "start": span["start"], "end": span["end"]}
    )


def fill_explanation_roles(
    event_type: str,
    text: str,
    trigger: dict,
    args: list[dict],
    title: str | None = None,
    min_args: int = 4,
    max_len: int = 60,
) -> list[dict]:
    """在模型抽出的干净槽位基础上，按规则补齐说明类槽位。

    event_type : 由槽位推断的事件类型（见 common.schema.infer_event_type）。
    trigger    : {"text","start","end"}，模型抽到的触发词。
    args       : 模型抽出的槽位 [{"role","text","start","end"}]（不含说明类）。
    title      : 公告标题；「其他人事变动」用于生成「变动内容」。
    返回补齐后的新列表（不修改入参）。
    """
    out = [dict(a) for a in args]
    existing_roles = {a.get("role") for a in out}

    # 其他人事变动：变动内容 = 标题（找不到则 trigger），无条件补
    if event_type == "其他人事变动" and "变动内容" not in existing_roles:
        span = None
        if title:
            c = Compact(text)
            t = re.sub(r"\s+", "", title)
            p = c.s.find(t)
            if p >= 0:
                span = c.span(p, p + len(t))
        _add(out, "变动内容", span or trigger)

    # 论元不足时补兜底说明
    if len(out) < min_args:
        role = EXPLANATION_ROLE.get(event_type, DEFAULT_EXPLANATION_ROLE)
        if role not in {a.get("role") for a in out}:
            _add(out, role, clause_span(text, int(trigger["start"]), max_len))
    return out


def assemble_event(
    text: str,
    doc_spans: list[tuple[int, int, str]],
    title: str | None = None,
) -> dict | None:
    """把文档级预测 span 组装成最终事件结构。

    doc_spans : [(start, end, base_label)]，绝对 offset，含 TRIGGER。
    返回 {"type","trigger","args"}；无 TRIGGER 时返回 None。

    流程：取 trigger → 推断事件类型 → 干净槽位转 args → 规则补说明类槽位。
    """
    trigger = None
    args: list[dict] = []
    for start, end, base in sorted(doc_spans):
        if base == TRIGGER_LABEL:
            if trigger is None:
                trigger = {"text": text[start:end], "start": start, "end": end}
            continue
        if base in EXCLUDED_ROLES:
            continue
        args.append(
            {"role": base, "text": text[start:end], "start": start, "end": end}
        )
    if trigger is None:
        return None
    event_type = infer_event_type({b for _, _, b in doc_spans})
    filled = fill_explanation_roles(event_type, text, trigger, args, title=title)
    return {"type": event_type, "trigger": trigger, "args": filled}
