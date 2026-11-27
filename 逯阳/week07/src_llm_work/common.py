"""
src_llm_work 公共模块：数据格式转换、prompt、解析、span F1

这个文件是 src_llm_work/ 下所有脚本共享的地基。参考实现（src_llm/）把同样的
helper 在三个脚本里各抄了一份，这里抽成一个模块，改一处就够了。

和 cluener 版（src_llm/）的核心差异：
  ┌──────────────┬─────────────────────────┬──────────────────────────┐
  │              │ cluener（src_llm/）      │ 人民日报（src_llm_work/） │
  ├──────────────┼─────────────────────────┼──────────────────────────┤
  │ 原始标注      │ span {类型:{实体:[[起,止]]}} │ CoNLL，tokens + BIO 标签  │
  │ 文本字段      │ record["text"]           │ "".join(record["tokens"]) │
  │ 实体类型      │ 10 类小写英文             │ 3 类大写（PER/ORG/LOC）    │
  │ test 集标签   │ 未公开                   │ 有，可直接评估             │
  └──────────────┴─────────────────────────┴──────────────────────────┘

指标为什么要改（重要）：
  参考实现用 (surface, type, start, end) 四元组做集合交集，预测实体的位置靠
  text.find(surface) 猜。同一句里同一个实体出现两次时，两次 find 都返回第一次的
  位置，去重后只剩一个 span —— 而 gold 有两条，于是 TP 白丢一条。
  实测本数据集 val 集有 **7.57%** 的实体属于这种「句内重复」，全对也只能拿到
  92.4% 的召回。所以这里的主指标改成 **（surface, type）多重集匹配**：
      TP = Σ_{(s,t)} min(pred 里 (s,t) 的个数, gold 里 (s,t) 的个数)
  另外保留一份参考实现口径的「严格位置匹配」数字，方便和 cluener 的 0.6801 对比。
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# 路径
# ══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
# 注意：ROOT 已经是 序列标注项目/，所以是 ROOT.parent（= D:/project），不是 parent.parent。
# src_llm/ 里写的是 parent.parent，会指到 D:/pretrain_models —— 那是历史遗留 bug，这里已修正。
MODEL_PATH = ROOT.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"

# 本目录专属产出，和 outputs_work/（BERT）、outputs/（cluener）完全隔离
OUT_DIR = ROOT / "outputs_llm_work"
LOG_DIR = OUT_DIR / "logs"
ADAPTER_DIR = OUT_DIR / "sft_adapter"
FIG_DIR = OUT_DIR / "figures"

# ══════════════════════════════════════════════════════════════════════════════
# 实体类型
# ══════════════════════════════════════════════════════════════════════════════

ENTITY_TYPES = ["PER", "ORG", "LOC"]

ENTITY_TYPE_ZH = {
    "PER": "人名",
    "ORG": "机构名",
    "LOC": "地名",
}

# 模型偶尔会输出中文名、小写、或者 CoNLL 里常见的同义写法，统一归一化后再判断，
# 否则「答对了但写法不对」会被当成完全没识别出来，冤枉模型。
_TYPE_ALIASES = {
    "PER": "PER", "PERSON": "PER", "NAME": "PER", "人名": "PER", "人物": "PER",
    "ORG": "ORG", "ORGANIZATION": "ORG", "机构": "ORG", "机构名": "ORG",
    "组织": "ORG", "组织机构": "ORG",
    "LOC": "LOC", "LOCATION": "LOC", "GPE": "LOC", "地名": "LOC", "地点": "LOC",
}

SYSTEM_PROMPT = (
    "你是一个命名实体识别助手。从文本中识别命名实体，以 JSON 格式输出。\n"
    "实体类型（英文标识）：PER（人名）、ORG（机构名）、LOC（地名）\n"
    '输出格式（严格遵守，不输出其他内容）：{"entities": [{"text": "实体文本", "type": "实体类型"}]}\n'
    '无实体时输出：{"entities": []}'
)

# few-shot 示例（本次没跑 API，留在这里备用；都取自 train.json，不会泄漏到验证集）
FEW_SHOT_EXAMPLES = [
    {
        "text": "中共中央总书记、国家主席习近平在北京人民大会堂会见美国总统奥巴马。",
        "output": '{"entities": [{"text": "中共中央", "type": "ORG"}, {"text": "习近平", "type": "PER"}, '
                  '{"text": "北京", "type": "LOC"}, {"text": "人民大会堂", "type": "LOC"}, '
                  '{"text": "美国", "type": "LOC"}, {"text": "奥巴马", "type": "PER"}]}',
    },
    {
        "text": "上海市市长韩正今天会见了来访的德国西门子公司总裁。",
        "output": '{"entities": [{"text": "上海市", "type": "LOC"}, {"text": "韩正", "type": "PER"}, '
                  '{"text": "德国", "type": "LOC"}, {"text": "西门子公司", "type": "ORG"}]}',
    },
    {
        "text": "会议由国务院副总理李克强主持，国家发改委主任张平作了汇报。",
        "output": '{"entities": [{"text": "国务院", "type": "ORG"}, {"text": "李克强", "type": "PER"}, '
                  '{"text": "国家发改委", "type": "ORG"}, {"text": "张平", "type": "PER"}]}',
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# 数据读取与格式转换
# ══════════════════════════════════════════════════════════════════════════════

def load_split(split: str) -> list[dict]:
    """读 data/peoples_daily/{split}.json，split ∈ {train, validation, test}。"""
    with open(DATA_DIR / f"{split}.json", encoding="utf-8") as f:
        return json.load(f)


def record_text(record: dict) -> str:
    """tokens 拼回原句。本数据集 tokens 全是单字，但拼法不依赖这个前提。"""
    return "".join(record["tokens"])


def bio_to_spans(tokens: list[str], ner_tags: list[str]) -> list[tuple[str, str, int, int]]:
    """BIO 标签序列 → [(实体文本, 类型, 起始字符下标, 结束字符下标)]，下标含首含尾。

    和 cluener 的 span 格式对齐，是为了让「严格位置匹配」这个口径能照抄参考实现。
    tokens 可能是多字（比如一个 token 就是一个词），所以位置要按累计长度算，
    不能直接用 token 下标当字符下标。
    """
    offsets, acc = [], 0
    for tok in tokens:
        offsets.append(acc)
        acc += len(tok)

    spans = []
    i = 0
    while i < len(ner_tags):
        if ner_tags[i].startswith("B-"):
            etype = ner_tags[i][2:]
            j = i + 1
            while j < len(ner_tags) and ner_tags[j] == f"I-{etype}":
                j += 1
            surface = "".join(tokens[i:j])
            start = offsets[i]
            end = offsets[j - 1] + len(tokens[j - 1]) - 1
            spans.append((surface, etype, start, end))
            i = j
        else:
            i += 1
    return spans


def bio_to_entities(tokens: list[str], ner_tags: list[str]) -> list[dict]:
    """BIO → [{"text": 实体, "type": PER/ORG/LOC}]，这是喂给 SFT 的目标格式。"""
    return [{"text": s, "type": t} for s, t, _, _ in bio_to_spans(tokens, ner_tags)]


def record_to_target(record: dict) -> str:
    """一条样本 → SFT 目标字符串（紧凑 JSON，不带空格，省 token）。"""
    entities = bio_to_entities(record["tokens"], record["ner_tags"])
    return json.dumps({"entities": entities}, ensure_ascii=False, separators=(", ", ": "))


# ══════════════════════════════════════════════════════════════════════════════
# 模型输出解析
# ══════════════════════════════════════════════════════════════════════════════

def normalize_type(raw: str) -> str | None:
    """把模型输出的类型写法归一化成 PER / ORG / LOC；认不出来返回 None。"""
    key = str(raw).strip().strip("\"'").upper()
    if key in _TYPE_ALIASES:
        return _TYPE_ALIASES[key]
    # 兼容 "B-PER" / "I-ORG" 这种直接抄 BIO 标签的写法
    if len(key) > 2 and key[1] == "-" and key[2:] in _TYPE_ALIASES:
        return _TYPE_ALIASES[key[2:]]
    return _TYPE_ALIASES.get(str(raw).strip())


def parse_entities(raw_output: str, text: str) -> tuple[list[dict], bool]:
    """从模型输出里抽实体列表，返回 (entities, json_ok)。

    json_ok 表示是否找到了合法的 {"entities": [...]} 结构 —— 生成式 NER 的
    「格式失败率」就是从这里统计的，和分类任务里「答非所问」是同一类问题。

    注意：**不丢弃原文里找不到的实体**。参考实现会直接 continue 掉，等于幻觉零成本，
    precision 被虚高。这里保留它们，匹配不上自然算 FP，更诚实。
    """
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        return [], False
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError:
        return [], False
    if not isinstance(obj, dict) or "entities" not in obj:
        return [], False

    entities_raw = obj.get("entities")
    if not isinstance(entities_raw, list):
        return [], False

    entities = []
    for ent in entities_raw:
        if not isinstance(ent, dict):
            continue
        surface = str(ent.get("text", "")).strip()
        etype = normalize_type(ent.get("type", ""))
        if not surface or etype is None:
            continue
        entities.append({"text": surface, "type": etype})
    return entities, True


# ══════════════════════════════════════════════════════════════════════════════
# 指标
# ══════════════════════════════════════════════════════════════════════════════

def _prf(tp: int, n_pred: int, n_gold: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "tp": tp, "pred_total": n_pred, "gold_total": n_gold}


def compute_metrics_multiset(golds: list[list[dict]],
                             preds: list[list[dict]]) -> dict:
    """主指标：按 (surface, type) 多重集匹配，正确处理句内重复实体。

    golds / preds 都是「每句一个 [{"text","type"}] 列表」。
    """
    tp = 0
    n_pred = n_gold = 0
    for g, p in zip(golds, preds):
        cg = Counter((e["text"], e["type"]) for e in g)
        cp = Counter((e["text"], e["type"]) for e in p)
        tp += sum((cg & cp).values())      # 逐键取较小值再求和
        n_gold += sum(cg.values())
        n_pred += sum(cp.values())
    return _prf(tp, n_pred, n_gold)


def compute_metrics_strict(texts: list[str],
                           gold_spans: list[list[tuple]],
                           pred_entities: list[list[dict]]) -> dict:
    """参考实现口径：(surface, type, start, end) 集合交集，预测位置靠 text.find() 猜。

    只为和 cluener 版（F1 0.6801）对齐着看，**不是**本目录的主指标 ——
    句内重复实体会被低估，原因见文件头注释。
    """
    tp = 0
    n_pred = n_gold = 0
    for text, g, p in zip(texts, gold_spans, pred_entities):
        gset = set(g)
        pset = set()
        for e in p:
            idx = text.find(e["text"])
            if idx == -1:
                continue                     # 严格口径下，原文找不到的直接丢
            pset.add((e["text"], e["type"], idx, idx + len(e["text"]) - 1))
        tp += len(gset & pset)
        n_gold += len(gset)
        n_pred += len(pset)
    return _prf(tp, n_pred, n_gold)


def per_type_metrics(golds: list[list[dict]], preds: list[list[dict]]) -> dict:
    """逐实体类型的 P/R/F1，同样用多重集匹配。"""
    out = {}
    for etype in ENTITY_TYPES:
        tp = n_pred = n_gold = 0
        for g, p in zip(golds, preds):
            cg = Counter(e["text"] for e in g if e["type"] == etype)
            cp = Counter(e["text"] for e in p if e["type"] == etype)
            tp += sum((cg & cp).values())
            n_gold += sum(cg.values())
            n_pred += sum(cp.values())
        out[etype] = _prf(tp, n_pred, n_gold)
        out[etype]["gold_count"] = n_gold
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 采样 / 打印小工具
# ══════════════════════════════════════════════════════════════════════════════

def sample_records(data: list[dict], n: int, seed: int = 42) -> list[dict]:
    """抽 n 条；n <= 0 表示用全部（这时不做随机，顺序就是原文件顺序）。"""
    if n <= 0 or n >= len(data):
        return list(data)
    return random.Random(seed).sample(data, n)


def fmt_pct(x: float) -> str:
    return f"{x * 100:5.2f}%"


def _dw(s: str) -> int:
    """字符串显示宽度：中文算 2 列。f-string 的 {:<6} 按字符数补齐，对中文无效。"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, width: int, right: bool = False) -> str:
    fill = " " * max(0, width - _dw(s))
    return fill + s if right else s + fill


def load_env_file(var_name: str = "DEEPSEEK_API_KEY") -> str | None:
    """如果环境变量里没有这个 key，就从 .env 文件里读。

    为什么需要这个：`setx` 设的永久环境变量只对**之后新启动**的进程生效，
    已经开着的终端 / IDE / 应用读不到。把 key 写进项目里的 .env 最省事，
    不用重启任何东西，而且换机器时只要改这一个文件。

    查找顺序（先找到先用）：项目根目录 .env → src_llm_work/.env
    文件格式：一行一个 KEY=VALUE，# 开头是注释，值两侧的引号会被去掉。

    安全性：.env 是明文，别提交到 git（本目录目前不是 git 仓库，但以后 init 了
    记得把 .env 写进 .gitignore）。
    """
    import os

    if os.getenv(var_name):
        return os.environ[var_name]

    for path in (ROOT / ".env", Path(__file__).parent / ".env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() != var_name:
                continue
            value = v.strip().strip("\"'")
            if value:
                os.environ[var_name] = value      # 回写，后续调用直接命中
                return value
    return None
