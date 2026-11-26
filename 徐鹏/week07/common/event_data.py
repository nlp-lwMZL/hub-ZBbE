"""
事件抽取的「数据层」公共模块：标签体系 + span↔BIO 转换 + 子词对齐 + 数据集 + 解码

整个判别式链路（BERT / BiLSTM-CRF / 两段式）都只用这一个文件里的数据逻辑，
这样「谁的 F1 高」这件事才可比 —— 如果每套方法各写一份 BIO 转换，指标就不可信了。

教学重点（按学生最容易踩坑的顺序）：
  1. 事件抽取的标签体系有两种典型设计，本项目两种都实现，用来对比：
     - joint（联合）：触发词标签带类型（TRG:股份回购），论元标签跨类型共享（ARG:交易金额）
       → 一个模型一次前向同时抽「类型+触发词+论元」，标签空间大（本例 149 个）
     - typed（类型条件）：先知道事件类型 t，再用「该类型专属」的标签表（TRG + roles(t)）
       → 标签空间小很多（单个类型约 13~21 个），但必须先有类型（两段式的第二段）
  2. BERT 的「子词对齐」（word_ids）：序列标注里 token 数**不一定等于**字数，标签必须能贴回字。
     实测（bert-base-chinese + 本数据）：
       - 中文基本**一字一 token**（"质押" → ["质","押"]），dev 930 句 154,318 字 → 152,858 token，
         **## 子词 0 个** —— 网上常举的「质/##押」在这个词表上并不存在；
       - 真正让 token ≠ 字的是三类：特殊符 [CLS]/[SEP]/[PAD]、**空格/换行不产生 token**、
         [UNK]（本数据 3,026 个，生僻字/符号，连 "IPO" 都可能变 UNK）；
       - 只有**不逐字、整句 tokenize** 时，数字/英文才会被 WordPiece 切成 "005"+"##10" 这种多片；
         换 Qwen 这类 BPE 词表时，中文反而会多字合成一个 token（"股份回购" → 2 个）。
     所以规则是：只让**首个子词**背这个字的标签，非首子词与特殊 token 一律 -100（CrossEntropy 忽略）；
     空格等没有 token 的字符位置直接跳过。代码写成通用防线，换 tokenizer 也不会静默错位。
  3. BIO 解码：B-x 开新 span、I-x 续接、O 结束；非法序列（I-x 开头 / B-x 后接 I-y）要能优雅兜底，
     同时**单独统计**非法数（那是 CRF 那一节的量化对比点）。
  4. 预测输出格式必须带 id + sent_id：DuEE-Fin 的 id 是「文档 id」不唯一，
     同文档多句会互相覆盖（详见 common/evaluate_events.py 的 row_key 注释）。

使用方式：
  from event_data import (load_schema, build_joint_schema, build_typed_schema,
                          load_jsonl, build_bio_dataset, decode_tags_to_event)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows: torch 与 numpy 各带 libiomp5md，不设会 OMP Error #15

import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"
SCHEMA_PATH = DATA_DIR / "event_schema.json"

# 「触发词」在 typed 标签体系里的固定名字：和角色名放在同一层，解码时靠它区分
TRIGGER_TAG = "TRG"
# joint 标签体系里触发词标签的前缀，后面接事件类型名（例：TRG:股份回购）
TRIGGER_PREFIX = "TRG:"
# joint 标签体系里论元标签的前缀，后面接角色名（例：ARG:交易金额）
ARG_PREFIX = "ARG:"

IGNORE_INDEX = -100  # CrossEntropy / CRF 里表示「这个位置不算 loss」


# ─────────────────────────── 读数据 / 写数据 ───────────────────────────

def load_jsonl(path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ─────────────────────────── 事件类型 schema ───────────────────────────

def load_schema(path=None) -> dict:
    """读事件类型定义：{类型名: [角色名, ...]}；跳过 __enums__ 这类元信息键。"""
    path = path or SCHEMA_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    schema = {k: list(v) for k, v in raw.items() if not k.startswith("__")}
    return schema


def load_enums(path=None) -> dict:
    """读枚举型角色的取值范围：{"公司上市.环节": [...]}（目前只有这一个）"""
    path = path or SCHEMA_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("__enums__", {}) or {}


def union_roles(schema: dict) -> list[str]:
    """所有类型角色名的并集（joint 标签体系里角色跨类型共享，靠它去重）。"""
    return sorted({r for roles in schema.values() for r in roles})


def schema_stats(schema: dict) -> dict:
    """打印用的统计：13 类 / 92 个「类型-角色」槽位 / 61 个去重角色名。"""
    return {
        "num_types": len(schema),
        "num_type_role_slots": sum(len(v) for v in schema.values()),
        "num_union_roles": len(union_roles(schema)),
    }


# ─────────────────────────── 标签体系（两种）───────────────────────────

def build_joint_schema(schema: dict) -> dict:
    """
    联合标签体系：触发词标签带事件类型，论元标签跨类型共享。

    实体标签（entity tag）两类：
      TRG:<类型>   —— 该类型的触发词
      ARG:<角色>   —— 论元（类型无关，因为一句只标一个事件，类型已由触发词给出）
    再展开成 BIO：每个实体标签变成 B-xxx / I-xxx，加上 O 一共 1 + 2*74 = 149 个标签。

    为什么论元不按类型再拆一层？
      拆开会让标签数涨到 211，而很多角色名本来就在多个类型里通用
      （「披露时间」在 13 个类型里都出现），共享之后每个标签能见到更多训练样本，
      长尾角色的样本数不至于跌破两位数 —— 这是小数据集上很实际的一个取舍。
    """
    entity_tags = [f"{TRIGGER_PREFIX}{t}" for t in sorted(schema)]
    entity_tags += [f"{ARG_PREFIX}{r}" for r in union_roles(schema)]
    return _expand_bio_labels(entity_tags, name="joint",
                              note="触发词按类型拆分 + 论元跨类型共享")


def build_typed_schema(schema: dict, event_type: str) -> dict:
    """
    类型条件标签体系（两段式的第三段用）：已知事件类型 t，只要抽「触发词 + 该类型的角色」。

    实体标签：TRG + roles(t)（不含类型名，因为模型本身就是这个类型的专家模型）
    标签数随类型变化：例如 被约谈 = 1 + 2*(1+4) = 11，公司上市 = 1 + 2*(1+8) = 19。

    这就是「流水线 vs 联合」的核心差别：标签空间从 149 缩到十几，
    每个标签的样本密度高一个数量级，所以论元 F1 通常明显更高；
    代价是必须先做对类型，且三个模型各自有延迟、各自会错。
    """
    entity_tags = [TRIGGER_TAG] + list(schema[event_type])
    return _expand_bio_labels(entity_tags, name=f"typed:{event_type}",
                              note="类型条件下的小标签空间")


def _expand_bio_labels(entity_tags: list[str], name: str = "", note: str = "") -> dict:
    """把实体标签列表展开成 BIO 标签表，返回 {labels, label2id, id2label, entity_tags}。"""
    labels = ["O"]
    for tag in entity_tags:
        labels.append(f"B-{tag}")
        labels.append(f"I-{tag}")
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return {
        "labels": labels,
        "label2id": label2id,
        "id2label": id2label,
        "entity_tags": entity_tags,
        "name": name,
        "note": note,
    }


def classify_entity_tag(tag: str) -> tuple[str, str]:
    """
    把「实体标签」翻译成 (kind, payload)：
      "TRG:股份回购" → ("trigger", "股份回购")   joint 体系：类型直接由触发词标签给出
      "TRG"          → ("trigger", "")           typed 体系：类型由外部传入
      "ARG:交易金额" → ("arg", "交易金额")
      "交易金额"     → ("arg", "交易金额")       typed 体系：角色名裸写
    """
    if tag.startswith(TRIGGER_PREFIX):
        return "trigger", tag[len(TRIGGER_PREFIX):]
    if tag.startswith(ARG_PREFIX):
        return "arg", tag[len(ARG_PREFIX):]
    if tag == TRIGGER_TAG:
        return "trigger", ""
    return "arg", tag


def build_start_constraint_mask(label_scheme: dict) -> torch.Tensor:
    """
    构造「序列开头允许哪些标签」的硬约束 (C,)，True=允许。

    只有 O 和 B-x 能当第一个标签；I-x 开头是非法序列。
    光约束转移矩阵是不够的 —— 转移只管「第 2 个 token 往后」，
    「第 1 个 token 就是 I-x」要靠这个 start 约束才能堵住（实测只加转移约束时 illegal_start 仍有 42%）。
    """
    labels = label_scheme["labels"]
    return torch.tensor([not lbl.startswith("I-") for lbl in labels], dtype=torch.bool)


def build_bio_constraint_mask(label_scheme: dict) -> torch.Tensor:
    """
    构造 BIO 合法转移的硬约束矩阵 (C, C)，True=允许，False=非法。

    为什么要它：CRF 的转移矩阵是**学出来的**，它只是倾向于不产生非法序列，
    并不保证非法序列为 0（bert_ner 项目实测过，CRF 仍然会有几十条非法转移）。
    把非法转移的分数直接置成 -inf 才是「硬约束」，可以让非法序列严格归零。
    本项目默认不施加（保持和 pytorch-crf 一致的软约束，便于教学对比），
    用 --hard_constraint 打开，作为「怎么把非法数压到 0」的消融。
    """
    labels = label_scheme["labels"]
    C = len(labels)
    mask = torch.ones(C, C, dtype=torch.bool)
    for i, prev in enumerate(labels):
        for j, cur in enumerate(labels):
            if not cur.startswith("I-"):
                continue
            cur_tag = cur[2:]
            # 规则 1：I-x 不能出现在 O 之后
            if prev == "O":
                mask[i, j] = False
                continue
            # 规则 2：I-x 前面必须是 B-x 或 I-x（同一实体标签）
            if prev.startswith("B-") or prev.startswith("I-"):
                if prev[2:] != cur_tag:
                    mask[i, j] = False
    return mask


# ─────────────────────────── span → 字符级 BIO ───────────────────────────

def row_event(row) -> dict | None:
    """取出一行里的单个事件（兼容 event / events 两种写法；检测集没有事件返回 None）。"""
    if row.get("event"):
        return row["event"]
    evs = row.get("events") or []
    return evs[0] if evs else None


def span_to_char_tags(text: str, event: dict, label_scheme: dict, typed: bool = False,
                      conflict_counter: Counter | None = None) -> list[int]:
    """
    把一条样本的 span 标注（trigger + args）刷成逐字符的 BIO 标签 id。

    参数：
      typed=False → joint 体系：触发词标签写成 TRG:<类型>，论元写成 ARG:<角色>
      typed=True → typed 体系：触发词写成 TRG，论元直接写角色名（类型由调用方指定）

    覆盖顺序：先刷论元，再刷触发词。**触发词优先级更高**，
    因为触发词决定事件类型，是整条链路最关键的一个 span；
    万一数据里出现「论元与触发词重叠」（极少见），让触发词赢。
    真正的重叠会记到 conflict_counter 里，便于在 NOTES/PPT 里如实说明数据规模瑕疵。
    """
    n = len(text)
    tags = ["O"] * n
    label2id = label_scheme["label2id"]

    def _paint(start: int, end: int, entity_tag: str, kind: str) -> None:
        # span 越界说明数据有脏（本项目实测 0 条），截断到有效范围即可，不让训练崩
        start = max(0, int(start))
        end = min(n, int(end))
        if start >= end or f"B-{entity_tag}" not in label2id:
            return
        if tags[start] != "O" and conflict_counter is not None:
            conflict_counter[kind] += 1
        tags[start] = f"B-{entity_tag}"
        for i in range(start + 1, end):
            tags[i] = f"I-{entity_tag}"

    # 1) 论元（低优先级）
    for arg in event.get("args") or []:
        role = arg.get("role", "")
        # ⚠ 这里曾经写成 `TRIGGER_TAG if typed else ...`（错！）：
        # typed 模式下论元也被刷成了 TRG，于是 13 个"类型条件论元专家"训练时
        # **所有论元都变成触发词标签、一个角色标签都没学到**，推理时就会在随机单字上吐 TRG、
        # 论元全空 —— 表现为端到端 trigger="甚"、args=[]、事件级 F1 = 0。
        # typed 模式的正确语义是：触发词=TRG，论元=**角色名本身**（不含类型前缀，因为模型已按类型特化）。
        etype = role if typed else f"{ARG_PREFIX}{role}"
        # typed 模型只学本类型的角色：不属于这个类型（或别的类型同名角色）的论元直接跳过，
        # 否则 f"B-{role}" 不在 label2id 里，会静默丢标签
        if f"B-{etype}" not in label2id:
            continue
        _paint(arg.get("start"), arg.get("end"), etype, "arg")

    # 2) 触发词（高优先级，最后刷，覆盖论元）
    trig = event.get("trigger") or {}
    if trig:
        etype = TRIGGER_TAG if typed else f"{TRIGGER_PREFIX}{event.get('type', '')}"
        _paint(trig.get("start"), trig.get("end"), etype, "trigger")

    return [label2id[t] for t in tags]


# ─────────────────────────── BIO → span / 事件 ───────────────────────────

def decode_bio_spans(tags: list[str]) -> list[dict]:
    """
    BIO 标签序列 → span 列表 [{"tag","start","end"}]（字符级，end 左闭右开）。

    对非法序列的兜底策略（重要，课堂上要讲）：
      - 以 I-x 开头：没有 B 打头，但模型「想说这里有东西」，按 B-x 处理（保留实体，别丢）
      - B-x 后面跟 I-y（x≠y）：把 I-y 当成新实体的开始
    兜底是为了「别让非法序列直接导致这个 span 消失」；非法数本身另外统计，不影响兜底的正确性。
    """
    spans = []
    cur = None
    for i, tag in enumerate(tags):
        if tag == "O":
            if cur is not None:
                spans.append(cur)
                cur = None
            continue
        if tag.startswith("B-"):
            if cur is not None:
                spans.append(cur)
            cur = {"tag": tag[2:], "start": i, "end": i + 1}
        else:  # I-x
            t = tag[2:]
            if cur is not None and cur["tag"] == t:
                cur["end"] = i + 1
            else:
                # 非法：I-x 开头，或 B-x→I-y 换类。按新实体开始处理
                if cur is not None:
                    spans.append(cur)
                cur = {"tag": t, "start": i, "end": i + 1}
    if cur is not None:
        spans.append(cur)
    return spans


def illegal_bio_stats(tags_list: list[list[str]]) -> dict:
    """
    统计非法 BIO 序列（linear vs CRF 的核心量化对比点，口径沿用 bert_ner/src/evaluate.py）。

    非法两类：
      illegal_start      —— 序列以 I-x 开头
      illegal_transition —— I-y 前面是 O，或前面是 B-x/I-x 但 x≠y
    注意：这里统计的是**模型原始输出**，不要用 decode_bio_spans 兜底后的结果，
    否则兜底会把非法情况掩盖掉。
    """
    stats = {"illegal_start": 0, "illegal_transition": 0, "total_seqs": len(tags_list),
             "illegal_seqs": 0}
    for seq in tags_list:
        if not seq:
            continue
        n_bad_here = 0
        if seq[0].startswith("I-"):
            stats["illegal_start"] += 1
            n_bad_here += 1
        for i in range(1, len(seq)):
            prev, cur = seq[i - 1], seq[i]
            if not cur.startswith("I-"):
                continue
            if prev == "O":
                stats["illegal_transition"] += 1
                n_bad_here += 1
            elif prev[2:] != cur[2:]:
                stats["illegal_transition"] += 1
                n_bad_here += 1
        if n_bad_here:
            stats["illegal_seqs"] += 1
    stats["total_illegal"] = stats["illegal_start"] + stats["illegal_transition"]
    # 两个率分开报，别混：
    #   illegal_seq_rate = 有多少条**句子**里出现了非法 BIO（PPT 上最直观）
    #   illegal_per_seq  = 平均每条句子有几处非法（看严重程度）
    stats["illegal_seq_rate"] = round(stats["illegal_seqs"] / max(stats["total_seqs"], 1), 4)
    stats["illegal_per_seq"] = round(
        stats["total_illegal"] / max(stats["total_seqs"], 1), 3)
    return stats


def spans_to_event(spans: list[dict], text: str, typed_type: str = "",
                   enums: dict | None = None) -> dict | None:
    """
    span 列表 → 统一评测用的事件 dict：{"type","trigger":{"text"},"args":[{"role","text"}]}。

    只有找得到触发词才算抽出一个事件（没有触发词就没有事件类型，下游无法消费）。
    typed 体系下事件类型由外部传入（`typed_type`），joint 体系下由触发词标签自带。
    """
    trigger_text, trigger_span, event_type, args = "", None, typed_type, []
    for sp in spans:
        kind, payload = classify_entity_tag(sp["tag"])
        piece = text[sp["start"]:sp["end"]]
        if not piece:
            continue
        if kind == "trigger":
            # 一个句子只保留一个触发词（本任务一句一个事件）；取最先出现的那个
            if trigger_span is None:
                trigger_span = sp
                trigger_text = piece
                if payload:
                    event_type = payload
        else:
            args.append({"role": payload, "text": piece,
                         "start": sp["start"], "end": sp["end"]})
    if not trigger_text or not event_type:
        return None

    # 枚举型角色做一次取值校验：例如「公司上市.环节」只能是 筹备上市/正式上市/...
    # 模型抽出来的片段若不在枚举里，仍然保留（不然会白丢召回），只是把它标出来，
    # 后续 service 层再决定要不要转 needs_review。
    if enums:
        for a in args:
            allowed = enums.get(f"{event_type}.{a['role']}")
            if allowed:
                a["enum_ok"] = a["text"] in allowed

    # 同一角色同一文本的重复论元去重（模型偶发重复预测同一段）
    seen, uniq = set(), []
    for a in args:
        k = (a["role"], a["text"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(a)

    return {
        "type": event_type,
        "trigger": {"text": trigger_text, "start": trigger_span["start"], "end": trigger_span["end"]},
        "args": uniq,
    }


def make_pred_row(row: dict, events: list[dict], extra: dict | None = None) -> dict:
    """
    构造一行预测。

    必须同时带 id 和 sent_id —— DuEE-Fin 的 id 是文档 id 不唯一，
    只带 id 会让同文档的多句预测互相覆盖（评测端按 id|sent_id 对齐）。
    解析失败/没抽出事件时 events 传空列表，**不要删掉这一行**（删了分母会变小、F1 虚高）。
    """
    out = {"id": row.get("id", ""), "sent_id": row.get("sent_id", ""), "events": events}
    if extra:
        out.update(extra)
    return out


# ─────────────────────────── BERT 数据集（子词对齐）───────────────────────────

class BioDataset(Dataset):
    """
    BERT 序列标注数据集：字符级 BIO → 子词级标签（word_ids 对齐）。

    核心流程：
      text → span_to_char_tags()     得到每个汉字 1 个标签
           → tokenizer(list(text), is_split_into_words=True)   一个汉字当作一个 word
           → word_ids() 把 token 映射回「第几个汉字」
           → 只有每个汉字的**首个子词**保留标签，其余（含 ##xx、[CLS]、[SEP]、[PAD]）设 -100
    """

    def __init__(self, rows: list[dict], tokenizer, label_scheme: dict,
                 max_length: int = 256, typed: bool = False, event_type: str = "",
                 with_labels: bool = True):
        self.rows = rows
        self.tokenizer = tokenizer
        self.label_scheme = label_scheme
        self.max_length = max_length
        self.typed = typed
        self.event_type = event_type
        self.with_labels = with_labels
        # 有多少条会被 max_length 截断（会丢 span，必须如实报告）。
        # 在 __init__ 里一次算好：如果放在 __getitem__ 里自增，dev/val 集每个 epoch 都会重扫一遍，
        # 计数会变成「样本数 × 迭代次数」，报出去就是假数字（本项目第一版就踩过，报成 854/930）。
        self.n_truncated = sum(1 for r in rows if len(r["text"]) > max_length - 2)
        self.span_conflicts = Counter()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        text = row["text"]
        chars = list(text)  # 中文按字符切，一个汉字 = 一个 word

        enc = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,           # 不在这里补齐：交给 collate_bio 按 batch 内最长补（动态 padding）
            return_tensors=None,     # 返回 python list，方便用 word_ids 对齐
        )
        word_ids = enc.word_ids()

        if self.with_labels:
            event = row_event(row)
            if event is None:
                char_labels = [self.label_scheme["label2id"]["O"]] * len(text)
            else:
                char_labels = span_to_char_tags(
                    text, event, self.label_scheme, typed=self.typed,
                    conflict_counter=self.span_conflicts)
        else:
            char_labels = None

        # ── 子词对齐：token 数可能 ≠ 字数（特殊符/空格/数字英文多片/换词表），只让首片背标签 ──
        aligned = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                # [CLS] / [SEP] / [PAD]：不参与 loss
                aligned.append(IGNORE_INDEX)
            elif wid != prev_wid:
                # 该汉字的第一个子词：背上这个汉字的标签
                aligned.append(char_labels[wid] if wid < len(char_labels) else IGNORE_INDEX)
                prev_wid = wid
            else:
                # 同一汉字的后续子词：设 -100。否则同一个 span 会被重复算好几遍 loss，
                # 标签计数也被放大，模型会过度自信地往这个类别偏。
                aligned.append(IGNORE_INDEX)

        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            # 把 word_ids 也存下来（None→-1）：推理时要把「子词级预测」还原回「字符级」，
            # 否则没法把 BIO 标签拼回原文的 span。不存的话推理时得重新 tokenize 一遍，容易错位。
            "word_ids": torch.tensor([-1 if w is None else w for w in word_ids],
                                     dtype=torch.long),
        }
        if "token_type_ids" in enc and enc["token_type_ids"] is not None:
            item["token_type_ids"] = torch.tensor(enc["token_type_ids"], dtype=torch.long)
        if self.with_labels:
            item["labels"] = torch.tensor(aligned, dtype=torch.long)
        item["row_idx"] = torch.tensor(idx, dtype=torch.long)
        return item


def collate_bio(batch: list[dict]) -> dict:
    """
    动态 padding：只补到「这个 batch 里最长的那条」，而不是全局 max_length。

    为什么这点很重要（真实性能差距）：本数据集的句子平均 164 字，但最长 797 字。
    如果按 max_length=256 静态补齐，每条样本都要算满 256 个 token；
    动态补齐后平均只算 ~170 个，**训练速度约 1.5 倍**，效果完全不变。
    这是 NLP 训练里最便宜的一次提速，学生在自己机器上跑大模型时更值得知道。
    """
    max_len = max(int(b["input_ids"].numel()) for b in batch)
    B = len(batch)
    out = {
        "input_ids": torch.zeros(B, max_len, dtype=torch.long),
        "attention_mask": torch.zeros(B, max_len, dtype=torch.long),
        "token_type_ids": torch.zeros(B, max_len, dtype=torch.long),
        "labels": torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long),
        "word_ids": torch.full((B, max_len), -1, dtype=torch.long),
        "row_idx": torch.stack([b["row_idx"] for b in batch]),
    }
    for i, b in enumerate(batch):
        n = int(b["input_ids"].numel())
        out["input_ids"][i, :n] = b["input_ids"]
        out["attention_mask"][i, :n] = b["attention_mask"]
        out["labels"][i, :n] = b["labels"]
        out["word_ids"][i, :n] = b["word_ids"]
        if "token_type_ids" in b:
            out["token_type_ids"][i, :n] = b["token_type_ids"]
    return out


class CharDataset(Dataset):
    """
    BiLSTM-CRF 用的**纯字符级**数据集：没有 BERT、没有子词，直接查字表。

    和数据集的 BiLSTM 基线对齐（embedding 128 + BiLSTM 256）：
    这正是「不用预训练模型」的起点，用来量化 BERT 预训练带来多少收益。
    """

    def __init__(self, rows: list[dict], vocab: dict, label_scheme: dict,
                 max_length: int = 256, typed: bool = False):
        self.rows = rows
        self.vocab = vocab
        self.label_scheme = label_scheme
        self.max_length = max_length
        self.typed = typed
        self.unk_id = vocab.get("<UNK>", 1)
        self.n_truncated = sum(1 for r in rows if len(r["text"]) > max_length)
        self.span_conflicts = Counter()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        text = row["text"][:self.max_length]
        event = row_event(row)
        if event is None:
            labels = [self.label_scheme["label2id"]["O"]] * len(text)
        else:
            labels = span_to_char_tags(text, event, self.label_scheme,
                                       typed=self.typed,
                                       conflict_counter=self.span_conflicts)
        ids = [self.vocab.get(ch, self.unk_id) for ch in text]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "length": torch.tensor(len(ids), dtype=torch.long),
            "row_idx": torch.tensor(idx, dtype=torch.long),
        }


def make_vocab(rows: list[dict], min_freq: int = 1) -> dict:
    """
    从训练集文本建字表。固定前两个 id：
      0 = <PAD>（补齐长度用，embedding 不参与梯度更新）
      1 = <UNK>（测试集里没见过的字）
    """
    cnt = Counter()
    for r in rows:
        cnt.update(r["text"])
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for ch, c in cnt.most_common():
        if c >= min_freq:
            vocab[ch] = len(vocab)
    return vocab


def collate_char(batch: list[dict]) -> dict:
    """字符级 batch 拼接：按 batch 内最长补齐，同时给出真实长度的 mask。"""
    max_len = max(int(b["input_ids"].numel()) for b in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    row_idx = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        n = int(b["input_ids"].numel())
        input_ids[i, :n] = b["input_ids"]
        labels[i, :n] = b["labels"]
        mask[i, :n] = 1
        lengths[i] = n
        row_idx[i] = b["row_idx"]
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": mask, "lengths": lengths, "row_idx": row_idx}


# ─────────────────────────── 分类数据集（检测 / 类型）───────────────────────────

class ClsDataset(Dataset):
    """
    把句子包装成 BERT 分类样本，一套类服务两个二段式模型：
      task="detect" → 句中是否有事件（0/1，负样本占 78%）
      task="type"   → 13 分类（只在有事件的句子上做）
    """

    def __init__(self, rows: list[dict], tokenizer, max_length: int = 128,
                 task: str = "detect", type2id: dict | None = None):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        self.type2id = type2id or {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        enc = self.tokenizer(row["text"], max_length=self.max_length, truncation=True,
                             padding="max_length", return_tensors=None)
        if self.task == "detect":
            label = 1 if int(row.get("n_events", 0)) > 0 else 0
        else:
            event = row_event(row)
            label = self.type2id.get((event or {}).get("type", ""), -100)
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
            "row_idx": torch.tensor(idx, dtype=torch.long),
        }
        if enc.get("token_type_ids") is not None:
            item["token_type_ids"] = torch.tensor(enc["token_type_ids"], dtype=torch.long)
        return item


def collate_cls(batch: list[dict]) -> dict:
    """分类任务的动态 padding：补齐到 batch 内最长，理由同 collate_bio（省一半算力）。"""
    max_len = max(int(b["input_ids"].numel()) for b in batch)
    B = len(batch)
    out = {
        "input_ids": torch.zeros(B, max_len, dtype=torch.long),
        "attention_mask": torch.zeros(B, max_len, dtype=torch.long),
        "token_type_ids": torch.zeros(B, max_len, dtype=torch.long),
        "labels": torch.stack([b["labels"] for b in batch]),
        "row_idx": torch.stack([b["row_idx"] for b in batch]),
    }
    for i, b in enumerate(batch):
        n = int(b["input_ids"].numel())
        out["input_ids"][i, :n] = b["input_ids"]
        out["attention_mask"][i, :n] = b["attention_mask"]
        if "token_type_ids" in b:
            out["token_type_ids"][i, :n] = b["token_type_ids"]
    return out


# ─────────────────────────── 数据装载入口 ───────────────────────────

def load_extract_split(split: str, data_dir=None) -> list[dict]:
    """读 train_extract / dev_extract（句子级、每句恰好 1 个事件）。"""
    d = Path(data_dir) if data_dir else DATA_DIR
    return load_jsonl(d / f"{split}_extract.jsonl")


def load_detect_split(split: str, data_dir=None) -> list[dict]:
    """读 train_detect / dev_detect（全部句子，含负样本）。"""
    d = Path(data_dir) if data_dir else DATA_DIR
    return load_jsonl(d / f"{split}_detect.jsonl")


def load_docs(split: str = "dev", data_dir=None) -> list[dict]:
    """读篇章级样本（dev_docs.jsonl，一篇公告一条）。"""
    d = Path(data_dir) if data_dir else DATA_DIR
    return load_jsonl(d / f"{split}_docs.jsonl")


def dataset_overview(data_dir=None) -> dict:
    """一次性汇总数据规模，脚本启动时打印，方便确认「跑的是对的数据」。"""
    d = Path(data_dir) if data_dir else DATA_DIR
    schema = load_schema(d / "event_schema.json")
    out = {"schema": schema_stats(schema)}
    for name in ("train_extract", "dev_extract", "train_detect", "dev_detect", "dev_docs"):
        p = d / f"{name}.jsonl"
        out[name] = len(load_jsonl(p)) if p.exists() else 0
    return out


if __name__ == "__main__":
    # 自检：把两种标签体系打出来，确认标签数没算错
    sch = load_schema()
    print("schema:", schema_stats(sch))
    j = build_joint_schema(sch)
    print(f"joint 标签数 = {len(j['labels'])}（O + 2×{len(j['entity_tags'])} 实体标签）")
    t = build_typed_schema(sch, "公司上市")
    print(f"typed[公司上市] 标签数 = {len(t['labels'])} → {t['labels']}")
    print("数据规模:", dataset_overview())
