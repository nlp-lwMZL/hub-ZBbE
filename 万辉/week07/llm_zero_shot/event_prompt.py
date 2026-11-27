# -*- coding: utf-8 -*-
"""
事件抽取的 Prompt 构造 + 模型输出解析（zero-shot / few-shot / SFT 共用）

一句话：把「一句金融公告」→「一段聊天消息」，再把「模型返回的文本」→「结构化事件」。
训练（train_sft.py）、推理（llm_extract.py）、评估（evaluate_sft.py）都调用这里的函数，
保证**三处用的提示词和目标格式逐字一致**（格式不一致是最常见的"训练白训"原因）。

教学重点（为什么这么设计）：

  1. **为什么只让模型输出文本 span，不让它输出 offset？**
     生成式模型算字符下标极不可靠：中文里「字符数」和「token 数」不是一回事、
     从 0 还是从 1 开始数全凭模型心情、长文本里还容易算错一位。
     一旦 offset 错，span 就完全对不上 —— 而**文本内容**模型是逐字抄的，抄对很容易。
     所以约定：模型只输出原文片段，程序用 `text.find()` 反查位置（本文件的 align()）。
     找不到 = 模型编了原文里没有的字（幻觉），此时 offset 记 -1，不强行纠正。
     这就是"生成式偏移不可信 → 用原文检索对齐"这条工程原则。

  2. **为什么论元要漏掉就省略，而不是让模型编满？**
     事件抽取的评测是按 (角色, 论元文本) 精确比对的，编出来的论元只会算 FP。
     提示词里明确"没有出现的角色直接省略"，能把「编造」压下去。

  3. **为什么 few-shot 例子必须从 train 里挑？**
     从 dev 里挑示例等于把评测集答案喂进 prompt（数据泄漏），指标会虚高。
     本文件里的 6 个示例全部取自 train_extract.jsonl，且刻意选了容易混淆的类型
     （质押 vs 解除质押、增持 vs 减持）。

使用方式：
  from event_prompt import build_messages, parse_model_output
  msgs = build_messages(text, schema, few_shot=False)
  events, info = parse_model_output(raw_output, text, schema)
"""

import json
import re

# ══════════════════════════════════════════════════════════════════════════════
# 系统提示词
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = """你是金融公告事件抽取专家。输入是**一句**金融公告文本，请抽取其中发生的**一个**事件。

## 事件类型与论元角色（共 13 类）
{type_lines}

## 抽取要求
1. 先判断事件类型（每句只对应一个事件），必须是上面 13 类之一，逐字照抄类型名。
2. 触发词：原文中标志该事件发生的词，必须是从原文**逐字截取**的连续片段（如"回购""解除质押""中标"），不要改写、不要翻译。
3. 论元：抽取该类型下出现的论元角色及其原文片段，同样必须逐字来自原文：
   - 原文里没有出现的角色，直接省略，不要猜、不要补全、不要用常识编；
   - 同一角色可以有多个论元（例如多个减持方），用多条记录分别表示。
4. **不要输出字符下标（start/end）**：生成的偏移不可靠，程序会用 text.find() 在原文里重新对齐。
5. 论元里的数字、日期、比例要与原文写法一致（如"1.35亿""2019年10月15日""不超过2%"）。

## 输出格式（严格，只输出一个 JSON 对象，不要解释、不要加 markdown 代码块）
{{"events": [{{"type": "事件类型", "trigger": {{"text": "触发词"}}, "arguments": [{{"role": "论元角色", "argument": "论元文本"}}]}}]}}
如果一个事件都抽不到，输出：{{"events": []}}"""


def schema_type_lines(schema: dict) -> str:
    """
    把 schema 渲染成紧凑的类型-角色列表（一行一个类型）。

    为什么要压缩成 "类型(角色、角色、...)" 这种一行式：
      13 个类型约 70 个角色，写成多级缩进的 YAML 会有 3000+ 字符，
      而整个 prompt 要塞进 SFT 的 max_length 预算里 —— 越省 token 越安全。
      实测压缩后 schema 部分约 392 个 token（Qwen3 分词器）。

    ⚠ 为什么分隔符用「、」而不是「/」：
      DuEE-Fin 里有两个角色名**本身带斜杠** ——「质押股票/股份数量」「交易股票/股份数量」。
      最早这个函数用 "/" 分隔角色，模型就把它们拆成「质押股票」+「股份数量」两个角色，
      于是这两个角色（200 条抽样里共 43 次金标）全部对不上，论元 F1 被白白拉低。
      这是「prompt 分隔符与字段值冲突」的经典坑：**分隔符不能出现在字段本身里**。
    """
    lines = []
    for etype, roles in schema["event_types"].items():
        lines.append(f"- {etype}：" + "、".join(roles))
    return "\n".join(lines)


def build_system_prompt(schema: dict) -> str:
    """生成系统提示词（schema 变了自动跟着变，避免手抄漏类型）。"""
    return SYSTEM_PROMPT_TEMPLATE.format(type_lines=schema_type_lines(schema))


# ══════════════════════════════════════════════════════════════════════════════
# Few-shot 示例（全部取自 train，绝不从 dev 里挑，避免评测集泄漏）
# ══════════════════════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES = [
    {   # 易混对 1：解除质押（"解除质押"是触发词，不是"质押"）
        "text": "万通地产股东万通控股解除质押1.35亿股 2019年上半年净利同比下滑32%",
        "target": '{"events": [{"type": "解除质押", "trigger": {"text": "解除质押"}, '
                  '"arguments": [{"role": "质押物所属公司", "argument": "万通地产"}, '
                  '{"role": "质押方", "argument": "万通控股"}, '
                  '{"role": "质押股票/股份数量", "argument": "1.35亿"}]}]}',
    },
    {   # 易混对 2：质押
        "text": "共计成交1.22亿股，这和中州国际持有的质押股份1.18亿股非常接近，市场人士分析。",
        "target": '{"events": [{"type": "质押", "trigger": {"text": "质押"}, '
                  '"arguments": [{"role": "质押物所属公司", "argument": "中州国际"}, '
                  '{"role": "质押物", "argument": "股份"}, '
                  '{"role": "质押股票/股份数量", "argument": "1.18亿"}]}]}',
    },
    {   # 易混对 3：股东减持（注意"减持方"可能多人）
        "text": "宏川智慧：持股3.48%的股东南靖百源股权投资合伙企业拟减持不超过2%",
        "target": '{"events": [{"type": "股东减持", "trigger": {"text": "减持"}, '
                  '"arguments": [{"role": "股票简称", "argument": "宏川智慧"}, '
                  '{"role": "减持方", "argument": "南靖百源股权投资合伙企业"}, '
                  '{"role": "减持部分占总股本比例", "argument": "不超过2%"}]}]}',
    },
    {   # 易混对 4：股东增持
        "text": "申通快递：阿里拟行权增持10.35%，关注对公司的短期催化和对格局的影响",
        "target": '{"events": [{"type": "股东增持", "trigger": {"text": "增持"}, '
                  '"arguments": [{"role": "股票简称", "argument": "申通快递"}, '
                  '{"role": "增持方", "argument": "阿里"}, '
                  '{"role": "增持部分占总股本比例", "argument": "10.35%"}]}]}',
    },
    {   # 高管变动：触发词常常是"辞去/委任/退休"这类职位动作，而非"变动"
        "text": "同时，公司宣布BAN LOR辞去总裁一职，立即生效。 BAN LOR也将继续担任公司的联合CEO和联合主席。",
        "target": '{"events": [{"type": "高管变动", "trigger": {"text": "辞去"}, '
                  '"arguments": [{"role": "高管姓名", "argument": "BAN LOR"}, '
                  '{"role": "高管职位", "argument": "总裁"}, '
                  '{"role": "变动类型", "argument": "辞去"}]}]}',
    },
    {   # 企业收购：一条论元都别编 —— 原文没写收购方就省略
        "text": "北京银保监局关于徽商银行股份有限公司 收购包商银行北京分行并开业的批复",
        "target": '{"events": [{"type": "企业收购", "trigger": {"text": "收购"}, '
                  '"arguments": [{"role": "收购方", "argument": "徽商银行股份有限公司"}, '
                  '{"role": "被收购方", "argument": "包商银行北京分行"}]}]}',
    },
]


def build_messages(text: str, schema: dict, few_shot: bool = False) -> list[dict]:
    """
    组装 OpenAI 兼容的 messages 列表。

    zero-shot：system + user（只靠任务描述）
    few-shot ：system + 6 组 (user, assistant) 示例 + user —— 靠"照着做"约束格式，
               代价是 prompt 变长（约 +1200 token），API 成本与延迟都会涨。

    注意 user 消息里只放原文，不放任何"请输出 JSON"之类的重复强调：
      要求统一写进 system，user 保持干净，避免模型在长 prompt 里抓错重点。
    """
    messages = [{"role": "system", "content": build_system_prompt(schema)}]
    if few_shot:
        for ex in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": ex["text"]})
            messages.append({"role": "assistant", "content": ex["target"]})
    messages.append({"role": "user", "content": f"句子：{text}"})
    return messages


# ══════════════════════════════════════════════════════════════════════════════
# 解析：模型返回的文本 → 结构化事件（含容错修复）
# ══════════════════════════════════════════════════════════════════════════════

def _strip_code_fence(raw: str) -> str:
    """去掉 markdown 代码围栏：模型最常见的"多余礼貌"是 ```json ... ```。

    真实踩坑：不处理围栏时，JSON 解析失败率会有 5%~20%（取决于模型和 prompt），
    而这批样本其实"内容是对的，只是包了壳" —— 白丢分。所以先剥壳再解析。
    """
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", s)   # 去掉开围栏（```json / ```JSON ...）
        s = re.sub(r"\s*```\s*$", "", s)               # 去掉闭围栏
    return s.strip()


def _first_json_object(s: str) -> str | None:
    """
    从第一个 '{' 开始做括号配平扫描，返回第一个完整 JSON 对象文本。

    为什么不用正则 `re.search(r"\\{.*\\}", s)`：
      贪婪匹配会把"前后夹带的解释文字"一起吃掉；非贪婪 `.*?` 又会在第一个 '}' 就截断，
      遇到嵌套结构（事件里套 trigger/arguments 两个对象）必定解析失败。
      括号配平扫描 + 字符串状态（处理 \" 转义）才是可靠做法。

    扫描不到配平（模型输出被 max_tokens 截断）时返回 None，交给 _close_truncated() 兜底。
    """
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def _close_truncated(s: str) -> str | None:
    """
    输出被 max_tokens 截断时，尽量把 JSON 补成合法（补引号 + 补右括号）。

    这是"尽力而为"的修复：补全后最可能丢掉最后一个论元 —— 丢掉一个论元只是少一次命中，
    比整条样本判解析失败（记 0 分）要划算。修复过的事件会在 stats 里单独计 repaired。
    """
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    stack = []
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    if not stack and not in_str:
        return None  # 本来就是完整的（或压根不是 JSON）
    fixed = s[start:]
    if in_str:
        fixed += '"'
    # 去掉可能悬空的 "key": 后面没值的半截（如 ...,"role": ）
    fixed = re.sub(r',\s*"[^"]*"\s*:\s*$', "", fixed)
    fixed = re.sub(r',\s*$', "", fixed)
    for ch in reversed(stack):
        fixed += "}" if ch == "{" else "]"
    return fixed


def _light_repair(s: str) -> str:
    """轻量修复：去掉对象/数组末尾多余逗号（JSON 不允许尾随逗号）。

    只做这一种修复：改引号、改中文标点都可能把**论元文本本身**改坏
    （比如论元里就有中文逗号），宁可让它判解析失败，也不要静默改坏数据。
    """
    return re.sub(r",\s*([}\]])", r"\1", s)


def _repair_brackets(s: str) -> str | None:
    """
    栈式括号修复：模型（尤其是小模型）最常见的畸形 JSON 是**少一个右括号**，
    例如 `...}]}}` 少了 events 列表的 `]`（正确应为 `}]}]}`）。
    这种"括号不配平"用去尾逗号/补末尾括号都救不回来，必须**在缺失处插入**。

    做法：用栈记录当前未闭合的 { 和 [，
      - 遇到与栈顶不匹配的右括号时，先把栈里"该关而没关"的括号补上，再消费这个右括号；
      - 遍历结束后把栈里剩下的全部补上；
      - 字符串状态（处理 \\" 转义）必须跟踪，否则论元文本里的括号会被误判。
    修复失败（算出来仍不是合法 JSON）就返回 None，调用方按解析失败处理。

    这对应课堂上的一个真实结论：**生成式的"格式正确率"要用工程手段救，
    但救回来的内容仍然要如实参与评测**（不能只统计"能解析的部分"）。
    """
    start = s.find("{")
    if start < 0:
        return None
    out, stack = [], []
    in_str = esc = False
    for ch in s[start:]:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
        elif ch in "{[":
            stack.append(ch)
            out.append(ch)
        elif ch in "}]":
            want = "{" if ch == "}" else "["
            if stack and stack[-1] == want:
                stack.pop()
                out.append(ch)
            elif stack:                      # 不匹配：先补上缺的闭合括号
                while stack and stack[-1] != want:
                    out.append("}" if stack.pop() == "{" else "]")
                if stack:
                    stack.pop()
                out.append(ch)
            # 栈空还遇到右括号 → 多余的，直接丢掉
        else:
            out.append(ch)
    if in_str:
        out.append('"')
    fixed = re.sub(r",\s*$", "", "".join(out))
    while stack:                             # 补完剩余的闭合括号
        fixed += "}" if stack.pop() == "{" else "]"
    return fixed


def _try_loads(candidate: str):
    """按「原样 → 去尾逗号 → 栈式括号修复 → 截断补全」的顺序尝试解析，返回 (obj, 是否修复过)。"""
    for text, repaired in ((candidate, False), (_light_repair(candidate), True)):
        try:
            return json.loads(text), repaired
        except json.JSONDecodeError:
            continue
    fixed = _repair_brackets(candidate)
    if fixed:
        try:
            return json.loads(_light_repair(fixed)), True
        except json.JSONDecodeError:
            pass
    closed = _close_truncated(candidate)
    if closed:
        try:
            return json.loads(_light_repair(closed)), True
        except json.JSONDecodeError:
            return None, False
    return None, False


def align(text: str, span: str) -> tuple[int, int]:
    """
    用原文检索把「模型抄的片段」对齐到字符下标：返回 (start, end)，找不到返回 (-1, -1)。

    为什么用 `find`（第一次出现）就够：
      评测只看 (角色, 文本) 是否一致，不看 offset；offset 只用于前端高亮/服务展示。
      同一片段在句中多次出现时取第一次，会让高亮位置偶有偏差 —— 这个近似是**有意的**，
      换来的是"模型不用输出任何数字"，从根上消除偏移幻觉。

    注意 Python 的切片是左闭右开：text[start:end] == span。
    """
    idx = text.find(span)
    return (idx, idx + len(span)) if idx >= 0 else (-1, -1)


def _to_event_list(obj, schema: dict, source_text: str, info: dict) -> list[dict]:
    """
    把解析出的 JSON 对象规整成统一事件列表（容错各种"差不多的写法"）。

    兼容的写法（模型会自由发挥，能救就救，救不回来算解析失败）：
      {"events": [...]}   ← 我们要求的格式
      {"event": {...}}    ← 单事件简写
      {...}               ← 直接给一个事件对象（没有外层包装）
    字段名兼容 trigger/触发词、arguments/args、argument/text。
    """
    valid_types = schema["event_types"]
    if isinstance(obj, dict) and isinstance(obj.get("events"), list):
        raw_events = obj["events"]
    elif isinstance(obj, dict) and isinstance(obj.get("event"), dict):
        raw_events = [obj["event"]]
    elif isinstance(obj, dict) and ("type" in obj or "事件类型" in obj):
        raw_events = [obj]
    elif isinstance(obj, list):
        raw_events = obj
    else:
        raw_events = []

    events = []
    for ev in raw_events:
        if not isinstance(ev, dict):
            continue
        etype = str(ev.get("type", ev.get("event_type", ev.get("事件类型", "")))).strip()
        if not etype:
            info["type_invalid"] += 1
            continue
        if etype not in valid_types:
            # 类型不在 13 类里：**不替模型纠正**，原样留下（评测会算 FP，这才诚实）
            info["type_invalid"] += 1

        # ── 触发词 ──────────────────────────────────────────────────────────
        trig = ev.get("trigger", ev.get("触发词", ""))
        if isinstance(trig, dict):
            trig_text = str(trig.get("text", trig.get("word", "")) or "").strip()
        else:
            trig_text = str(trig or "").strip()
        if not trig_text:
            continue  # 没触发词的事件没有评测意义（trigger F1 必然判错）
        t_start, t_end = align(source_text, trig_text)
        if t_start < 0:
            info["trigger_not_in_source"] += 1

        # ── 论元 ────────────────────────────────────────────────────────────
        raw_args = ev.get("arguments")
        if raw_args is None:
            raw_args = ev.get("args")
        if raw_args is None:
            raw_args = ev.get("论元") or []
        if isinstance(raw_args, dict):  # {"角色": "文本"} 这种写法也吃
            raw_args = [{"role": k, "argument": v} for k, v in raw_args.items()]

        args, seen = [], set()
        for a in raw_args:
            if isinstance(a, str):  # 模型只给了文本没给角色
                info["role_invalid"] += 1
                continue
            if not isinstance(a, dict):
                continue
            role = str(a.get("role", a.get("type", "")) or "").strip()
            text = str(a.get("argument", a.get("text", a.get("论元", ""))) or "").strip()
            if not role or not text:
                info["role_invalid"] += 1
                continue
            if etype in valid_types and role not in valid_types[etype]:
                info["role_invalid"] += 1  # 角色不属于该事件类型
            if (role, text) in seen:      # 去重：模型偶尔把同一论元写两遍
                continue
            seen.add((role, text))
            if source_text.find(text) < 0:
                info["arg_not_in_source"] += 1  # 幻觉论元：保留但记数（评测会算 FP）
            a_start, a_end = align(source_text, text)
            args.append({"role": role, "text": text, "start": a_start, "end": a_end})

        events.append({
            "type": etype,
            "trigger": {"text": trig_text, "start": t_start, "end": t_end},
            "args": args,
        })
    return events


def parse_model_output(raw: str, source_text: str, schema: dict) -> tuple[list[dict], dict]:
    """
    模型原始输出 → (事件列表, 解析统计)。

    统计字段含义（实验表里的"JSON 合法率"就来自这里）：
      empty_reply       空回复（常见原因：推理模型把 max_tokens 全用在思考上）
      json_ok           成功解析出 JSON 对象（= JSON 合法率分子）
      repaired          需要剥代码围栏/去尾逗号/补截断才解析成功
      type_invalid      事件类型缺失或不在 13 类里
      role_invalid      角色缺失、或该角色不属于这个事件类型
      trigger_not_in_source / arg_not_in_source  文本 span 在原文里 find 不到（幻觉）
    """
    info = {"empty_reply": False, "json_ok": False, "repaired": False, "noise_stripped": False,
            "type_invalid": 0, "role_invalid": 0,
            "trigger_not_in_source": 0, "arg_not_in_source": 0}
    if not raw or not raw.strip():
        info["empty_reply"] = True
        return [], info

    cleaned = _strip_code_fence(raw)
    candidate = _first_json_object(cleaned)
    # 「噪音」= 代码围栏或 JSON 之外的废话（模型很爱说"好的，结果如下"）
    info["noise_stripped"] = bool(candidate and candidate != raw.strip())
    obj, repaired = (None, False)
    if candidate:
        obj, repaired = _try_loads(candidate)
    if obj is None:
        # 兜底：从第一个 '{' 到结尾整段做截断补全（应对 max_tokens 截断）
        obj, repaired = _try_loads(cleaned)
    if obj is None:
        return [], info

    info["json_ok"] = True
    info["repaired"] = bool(repaired)
    return _to_event_list(obj, schema, source_text, info), info


if __name__ == "__main__":
    # 自检：打印完整提示词（PPT / NOTES 里要引用 prompt 全文，直接跑这个拿）
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from event_data import load_schema

    _schema = load_schema()
    prompt = build_system_prompt(_schema)
    print(prompt)
    print("\n" + "=" * 70)
    print(f"system prompt 字符数：{len(prompt)}")
    print(f"zero-shot messages 条数：{len(build_messages('示例句子', _schema, False))}")
    print(f"few-shot  messages 条数：{len(build_messages('示例句子', _schema, True))}")
