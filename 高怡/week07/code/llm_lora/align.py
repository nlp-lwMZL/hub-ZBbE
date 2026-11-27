"""把模型输出的 event JSON 回填到原文，计算 start / end。

策略
----
模型只给「角色 + 原文片段」，位置在解码后由本模块确定：

1. 先给**较长**的片段分配位置（长串更具体，先占位可减少短串误抢）；
2. 每个片段在原文中可能有多次出现，**按角色与事件类型选锚点**（规则见下）；
3. 若所有出现都与已占位区间重叠，则该片段视为无法定位而丢弃（计入 unmatched）；
4. trigger 最后分配，优先选不与任何槽位重叠的出现位置。

锚点规则（依据全量 35178 条统计，括号内为该角色多次出现时规则命中 gold 的比例）
------------------------------------------------------------------------------
- `基金主代码`：语料 99.99% 标注为首次出现（33819/33824），直接用首次，并充当锚点。
- 主类事件（增聘 / 离任 / 变更）里：
  - `基金名称`：取离基金主代码最近的那次（命中 99.2%，取首次仅 3.7%）；
  - `基金管理人`：同样取离基金主代码最近的那次（97.9%，取首次仅 59.5%）。
- `任职日期` / `离任日期`：取离触发词最近的那次（87.7% / 97.9%，取首次仅 26.3% / 18.8%）。
- 稀有类型（恢复履职、代为履职、助理聘任/离任、其他人事变动）的金标恰恰是**首次出现**
  （恢复履职的管理人 100%、代为履职 97.8%、代为履职的名称 89.4%），这些类型保持首次，
  不套用锚点。
- `生效日期` 两种规则命中率都只有 70%，属标注真歧义，不特殊处理。

名称去重
--------
模型有时会同时输出简称与全称（如「鹏华丰颐债券」与「鹏华丰颐债券型证券投资基金」），
这是同一实体的两个都真实存在的字符串。语料只标一个，故这里只保留一个：
优先最短，其次离主代码最近。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.label import TRIGGER_LABEL  # noqa: E402
from common.schema import EVENT_SCHEMA  # noqa: E402


def _find_all(text: str, s: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if not s:
        return out
    i = text.find(s)
    while i != -1:
        out.append((i, i + len(s)))
        i = text.find(s, i + 1)
    return out


def _overlaps(span: tuple[int, int], used: list[tuple[int, int]]) -> bool:
    s, e = span
    return any(not (e <= us or s >= ue) for us, ue in used)


MAIN_TYPES = {"基金经理增聘", "基金经理离任", "基金经理变更"}
CODE_ANCHOR_ROLES = {"基金名称", "基金管理人"}
DATE_ANCHOR_ROLES = {"任职日期", "离任日期"}


def _dedupe_names(items: list[tuple[str, str]], text: str,
                  code_span, use_code_anchor: bool) -> list[tuple[str, str]]:
    """模型常同时输出简称与全称，语料只标一个：保留最短的，同长取离主代码最近的。"""
    names = [kv for kv in items if kv[0] == "基金名称"]
    if len(names) <= 1:
        return items
    scored = []
    for _, txt in names:
        occ = _find_all(text, txt)
        if not occ:
            continue
        if use_code_anchor and code_span is not None:
            d = min(abs(s - code_span[0]) for s, _ in occ)
        else:
            d = occ[0][0]
        scored.append(((len(txt), d), txt))
    if not scored:
        return items
    keep = min(scored)[1]
    return [kv for kv in items if kv[0] != "基金名称"] + [("基金名称", keep)]


def attach_spans(pred: dict, text: str) -> dict:
    """pred: {"type","trigger","arguments":[{role,text}]} -> 带 start/end 的事件。"""
    ev_type = pred.get("type", "")
    result = {
        "type": ev_type,
        "trigger": None,
        "args": [],
        "unmatched": [],
    }
    if ev_type not in EVENT_SCHEMA:
        return result

    text = text or ""
    allowed = set(EVENT_SCHEMA[ev_type]["roles"])
    use_code_anchor = ev_type in MAIN_TYPES  # 稀有类型的金标是首次出现，不套锚点

    # 去重
    items: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for a in pred.get("arguments") or []:
        role = (a.get("role") or "").strip()
        txt = (a.get("text") or "").strip()
        if role not in allowed or not txt:
            continue
        key = (role, txt)
        if key in seen:
            continue
        seen.add(key)
        items.append(key)

    # trigger 先定位：既是日期锚点，也避免被长槽位抢占
    trigger = (pred.get("trigger") or "").strip()
    trigger_span = None
    if trigger:
        occ = _find_all(text, trigger)
        if occ:
            trigger_span = occ[0]

    # 基金主代码：语料 99.99% 标注为首次出现，作为名称/管理人的锚点
    code_span = None
    for role, txt in items:
        if role == "基金主代码":
            occ = _find_all(text, txt)
            if occ:
                code_span = occ[0]
            break

    items = _dedupe_names(items, text, code_span, use_code_anchor)
    items.sort(key=lambda kv: -len(kv[1]))  # 长者优先占位

    used: list[tuple[int, int]] = []
    for role, txt in items:
        cands = _find_all(text, txt)
        if not cands:
            result["unmatched"].append(txt)
            continue
        if use_code_anchor and code_span is not None and role in CODE_ANCHOR_ROLES:
            # 取离基金主代码最近的那次（基本信息表同一行）
            cands = sorted(cands, key=lambda sp: abs(sp[0] - code_span[0]))
        elif trigger_span is not None and role in DATE_ANCHOR_ROLES:
            # 日期锚到触发词（"增聘日期"/"离任日期"总在触发词附近）
            cands = sorted(cands, key=lambda sp: abs(sp[0] - trigger_span[0]))
        free = [sp for sp in cands if not _overlaps(sp, used)]
        if not free:
            result["unmatched"].append(txt)
            continue
        s, e = free[0]
        used.append((s, e))
        result["args"].append({"role": role, "text": txt, "start": s, "end": e})

    if trigger:
        cands = _find_all(text, trigger)
        if trigger_span is not None and not _overlaps(trigger_span, used):
            chosen = trigger_span
        else:
            non_overlap = [sp for sp in cands if not _overlaps(sp, used)]
            chosen = (non_overlap or cands or [None])[0]
        if chosen is None:
            result["unmatched"].append(trigger)
        else:
            s, e = chosen
            used.append((s, e))
            result["trigger"] = {"text": trigger, "start": s, "end": e}

    return result


def to_metric_event(spanned: dict) -> dict:
    """转成 common.metrics.compute_event_exact_match 需要的事件结构。"""
    trigger = None
    if spanned.get("trigger"):
        t = spanned["trigger"]
        trigger = (t["start"], t["end"], TRIGGER_LABEL)
    args = frozenset(
        (a["start"], a["end"], a["role"]) for a in spanned.get("args", [])
    )
    return {"type": spanned.get("type", ""), "trigger": trigger, "args": args}


def spans_of(spanned: dict) -> list[tuple[int, int, str]]:
    """展平成 (start, end, base) 列表，供 label 级 F1 使用。"""
    spans: list[tuple[int, int, str]] = []
    if spanned.get("trigger"):
        t = spanned["trigger"]
        spans.append((t["start"], t["end"], TRIGGER_LABEL))
    for a in spanned.get("args", []):
        spans.append((a["start"], a["end"], a["role"]))
    return sorted(spans)


def gold_event_to_metric(event: dict) -> dict:
    """标注事件（含 start/end）-> 指标结构。"""
    trigger = None
    t = (event or {}).get("trigger") or {}
    if t.get("text"):
        trigger = (int(t["start"]), int(t["end"]), TRIGGER_LABEL)
    args = frozenset(
        (int(a["start"]), int(a["end"]), a["role"])
        for a in (event or {}).get("args") or []
        if a.get("text")
    )
    return {"type": (event or {}).get("type", ""), "trigger": trigger, "args": args}


def gold_spans(event: dict) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    t = (event or {}).get("trigger") or {}
    if t.get("text"):
        spans.append((int(t["start"]), int(t["end"]), TRIGGER_LABEL))
    for a in (event or {}).get("args") or []:
        if a.get("text"):
            spans.append((int(a["start"]), int(a["end"]), a["role"]))
    return sorted(spans)
