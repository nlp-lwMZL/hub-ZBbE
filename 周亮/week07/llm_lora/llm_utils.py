"""
生成式抽取的容错工具：JSON 解析兜底 + 类型白名单 + text.find 对齐 + 重复文本处理。

老师 PPT 里反复强调的生成式三大坑，这里逐一落地：
  1. JSON 畸形（模型爱包 ```json 围栏、前后夹带解释、被 max_new_tokens 截断）
     -> 三层修复：剥围栏 -> 括号配平扫第一个完整对象 -> 截断补全；
  2. 类型幻觉（模型输出 schema 外的类型名，type_macro 分母被撑大）
     -> 类型白名单校验，非法类型丢弃并计数；
  3. offset 不可信（生成模型算字符下标极不可靠）
     -> 只信文本片段，用 text.find() 对齐回字符 span；不信任模型给的下标。

【重复文本的坑】同一片段在原文出现多次时 text.find() 取首次——这是有意近似，
与老师事件抽取一致；但要注意 clean 数据为避免标注冲突已按文本切分，跨句重复已消除。

【统计】解析/对齐过程全程记账：json_valid / repaired / illegal_type / text_not_found，
这些工程指标比 F1 更能说明生成式链路的成本。

自检：python llm_lora/llm_utils.py
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

VALID_TYPES = ("Attacker", "Campaign", "Industry", "Region", "Tools")


def strip_code_fence(s: str) -> str:
    """剥掉 ```json ... ``` 围栏和前后噪声。

    模型常在 JSON 前先说一句「好的，结果如下：」再跟围栏，所以不能只处理
    「字符串以 ``` 开头」——要在任意位置找到围栏并取其内容。
    """
    s = s.strip()
    if "```" in s:
        parts = s.split("```")
        # 取第一对围栏之间的内容（parts[1]）
        if len(parts) >= 3:
            s = parts[1]
            if s.lstrip().startswith("json"):
                s = s.lstrip()[4:]
            s = s.strip()
    return s


def first_json_object(s: str) -> str | None:
    """栈式括号配平，截取第一个完整 JSON 对象（正则贪婪/非贪婪都会挂）。"""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def close_truncated(s: str) -> str:
    """输出被截断时补全：栈式扫描（区分字符串内外）括号状态，补齐缺失的括号/引号。

    结尾的 `}` 可能只是内部实体对象的闭合，外层 `{` `[ 仍未闭合——所以不能用
    endswith("}") 判断，必须真正计数。若截断落在字符串中间，先补引号再补括号。
    """
    s = strip_code_fence(s)
    in_str = False
    esc = False
    curly = square = 0
    for c in s:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            curly += 1
        elif c == "}":
            curly -= 1
        elif c == "[":
            square += 1
        elif c == "]":
            square -= 1
    out = s
    if in_str:
        out += '"'          # 截断在字符串中间：补引号
    out = out.rstrip()
    if out.endswith(","):
        out = out[:-1]      # 去掉悬空逗号
    out += "]" * max(0, square)
    out += "}" * max(0, curly)
    return out


def parse_entities(raw: str, stats: Counter | None = None) -> list[dict[str, str]]:
    """解析模型回复 -> [{type, text}]，带三层兜底与全链路计数。"""
    stats = stats if stats is not None else Counter()
    text = strip_code_fence(raw)
    if strip_code_fence(raw) != raw.strip():
        stats["noise_stripped"] += 1
    obj = first_json_object(text)
    repaired = False
    data = None
    if obj is not None:
        try:
            data = json.loads(obj)
        except json.JSONDecodeError:
            pass
    # obj 找不到配平 }，或找到但解析失败 → 都尝试截断补全
    if data is None:
        fixed = close_truncated(text)
        candidate = first_json_object(fixed) or fixed
        try:
            data = json.loads(candidate)
            repaired = True
        except json.JSONDecodeError:
            stats["empty_or_no_json" if obj is None else "unrepairable"] += 1
            return []
    stats["json_valid"] += 1
    if repaired:
        stats["repaired"] += 1

    items = data.get("entities", []) if isinstance(data, dict) else []
    out: list[dict[str, str]] = []
    for it in items:
        if not isinstance(it, dict):
            stats["bad_item"] += 1
            continue
        t, txt = it.get("type"), it.get("text")
        if t not in VALID_TYPES:
            stats["illegal_type"] += 1
            continue
        if not isinstance(txt, str) or not txt:
            stats["empty_text"] += 1
            continue
        out.append({"type": t, "text": txt})
    return out


def align_spans_to_text(items: list[dict[str, str]], text: str,
                        stats: Counter | None = None) -> list[dict[str, Any]]:
    """用 text.find() 把 [{type,text}] 对齐成字符 span；找不到就丢弃并计数。"""
    stats = stats if stats is not None else Counter()
    spans: list[dict[str, Any]] = []
    for it in items:
        txt = it["text"]
        start = text.find(txt)
        if start < 0:
            stats["text_not_found"] += 1
            continue
        spans.append({"type": it["type"], "start": start, "end": start + len(txt), "text": txt})
    return spans


def _self_test() -> None:
    print("=" * 66)
    print("llm_utils 自检")
    print("=" * 66)
    raw = '好的，结果如下：\n```json\n{"entities": [{"type":"Attacker","text":"APT-C-39"}]}\n```'
    stats: Counter = Counter()
    items = parse_entities(raw, stats)
    print(f"  剥围栏解析: {items} | stats={dict(stats)}")
    assert items == [{"type": "Attacker", "text": "APT-C-39"}]
    assert stats["noise_stripped"] >= 1

    # 截断修复：少了 ]}
    truncated = '{"entities": [{"type": "Tools", "text": "Houdini"}'
    items2 = parse_entities(truncated, stats)
    print(f"  截断修复: {items2} | stats={dict(stats)}")
    assert items2 == [{"type": "Tools", "text": "Houdini"}]

    # 类型白名单 + text.find 对齐
    items3 = parse_entities('{"entities":[{"type":"虚构类型","text":"X"},'
                            '{"type":"Region","text":"印度"}]}', stats)
    spans = align_spans_to_text(items3, "针对印度的攻击活动", stats)
    print(f"  白名单+对齐: {spans} | stats={dict(stats)}")
    assert len(spans) == 1 and spans[0]["type"] == "Region" and spans[0]["start"] == 2
    assert stats["illegal_type"] >= 1
    print("自检通过 ✓")


if __name__ == "__main__":
    _self_test()
