"""
数据层 —— BERT 与 LLM 共用同一份 jsonl，只在「张量化」这一步分叉

设计要点（这是为后续 LLM 全量微调 / LoRA 微调做准备的核心）：
  同一条样本在两种范式下的差别只在最后一步：
    BERT : text  → input_ids + attention_mask            → 3 分类头
    LLM  : text  → chat 模板 + 答案，prompt 段 labels=-100 → 生成「高/中/低」
  所以 `load_rows / build_text / build_user_prompt / build_answer` 四个函数共用，
  只有 BertDataset 与 LLMSFTDataset 两个 Dataset 类分开实现。
  这样消融实验里「同一份数据、同一个 split、同一个 seed」是天然保证的——
  不会出现 BERT 用 5k、LLM 用另一批 5k 这种不可比的低级错误。

schema（data/*.jsonl 每行，由 prepare_data.py 产出）：
  {"id": "CNVD-2021-12345", "title": "...", "text": "<description>",
   "label": "高|中|低", "label_id": 0|1|2, "split": "train|val|test|pool"}

使用方式：
  from dataset import load_label_map, load_split, BertDataset, make_collate_fn
  from dataset import LLMSFTDataset            # 后续 LLM 微调直接用
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

# 输入字段拼接：训练与评测必须走同一个函数，否则「训练拼了 title、评测忘了拼」
# 会让指标静默掉几个点且不报错——这是这类脚本最贵的坑。
TITLE_SEP = "："


# ─────────────────────────────────────────────────────────────────────────────
# 读取与文本构造（BERT / LLM 共用）
# ─────────────────────────────────────────────────────────────────────────────

def load_label_map(data_dir: Path) -> tuple[dict, dict, list]:
    """返回 (name2id, id2name, labels)。映射表只从 label_map.json 来，全项目唯一来源。"""
    with open(Path(data_dir) / "label_map.json", encoding="utf-8") as f:
        name2id = json.load(f)                       # {"高": 0, "中": 1, "低": 2}
    id2name = {int(v): k for k, v in name2id.items()}
    labels = [id2name[i] for i in sorted(id2name)]
    return name2id, id2name, labels


def load_jsonl(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_text(row: dict, use_title: bool = False) -> str:
    """BERT 的输入文本：默认只用 description（与官方 MacBERT 基线口径可比）"""
    return f"{row['title']}{TITLE_SEP}{row['text']}" if use_title else row["text"]


def load_split(path: Path, use_title: bool, name2id: dict, logger=None) -> tuple[list, list]:
    """返回 (texts, labels)。加载时校验 label_id 与 label_map 一致——两套代码各查一遍
    映射表是消融实验的经典翻车点，这里直接拒载。"""
    rows = load_jsonl(path)
    texts, labels, bad = [], [], 0
    for r in rows:
        texts.append(build_text(r, use_title))
        if r["label_id"] != name2id[r["label"]]:
            bad += 1
        labels.append(r["label_id"])
    if bad:
        raise ValueError(f"{path}：{bad} 行 label_id 与 label_map 不一致，拒绝训练")
    if logger:
        logger.info(f"加载 {Path(path).name}: {len(rows)} 条（use_title={use_title}）")
    return texts, labels


def stratified_subset(rows: list, n: int) -> list:
    """
    等距抽样到 n 条。**不能用 rows[:n]**：三份 jsonl 是按类分组存放的，
    掐头会让子集全是同一个类别（冒烟时 val 只含「高」类，指标直接失真）。
    """
    if not n or n >= len(rows):
        return rows
    step = max(1, len(rows) // n)
    return rows[::step][:n]


def class_counts(labels, n_classes: int) -> list:
    counts = [0] * n_classes
    for y in labels:
        counts[int(y)] += 1
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# BERT：预分词 + 动态 padding
# ─────────────────────────────────────────────────────────────────────────────

class BertDataset(Dataset):
    """
    预分词（只截断、不 padding），padding 留给 collate 按 batch 内最长补齐。
    token P95 = 154 < 192，动态 padding 比写死 max_length 省约 20% 计算。
    """

    def __init__(self, texts, labels, tokenizer, max_length: int):
        enc = tokenizer(list(texts), truncation=True, max_length=max_length, padding=False)
        self.input_ids = enc["input_ids"]
        self.labels = [int(y) for y in labels]
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


def make_collate_fn(tokenizer):
    """动态 padding：同一 batch 内 pad 到最长，attention_mask 由 tokenizer.pad 生成"""

    def collate(features):
        batch = tokenizer.pad({"input_ids": [f["input_ids"] for f in features]},
                              return_tensors="pt")
        batch["labels"] = torch.tensor([f["labels"] for f in features])
        return batch
    return collate


def token_length_stats(texts, tokenizer, max_length: int, sample: int = 2000) -> dict:
    """token 长度分布 + 截断比例：决定 max_length 该取多少（EDA 与日志都用）"""
    import numpy as np
    sub = list(texts[:sample])
    ids = tokenizer(sub, truncation=False, padding=False)["input_ids"]
    lens = np.asarray([len(x) for x in ids])
    return {
        "n": len(lens), "mean": round(float(lens.mean()), 1),
        "median": int(np.median(lens)), "p95": int(np.percentile(lens, 95)),
        "max": int(lens.max()),
        "truncation_pct": round(float((lens > max_length).mean() * 100), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM：chat 模板 + loss mask（供后续全量微调 / LoRA 复用同一份数据）
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是网络安全漏洞等级研判专家。根据漏洞标题和描述，判断该漏洞的严重等级。"
    "等级只有三种：高、中、低。只输出等级对应的一个字，不要输出任何解释。"
)


def build_user_prompt(row: dict, use_title: bool = True) -> str:
    """LLM 的 user 消息。默认带 title——标题里「SQL注入」「命令执行」是强判级信号。"""
    if use_title:
        return (f"漏洞标题：{row['title']}\n"
                f"漏洞描述：{row['text']}\n"
                f"请判断该漏洞的严重等级（高/中/低）。")
    return (f"漏洞描述：{row['text']}\n"
            f"请判断该漏洞的严重等级（高/中/低）。")


def build_answer(row: dict) -> str:
    """监督信号：就是标签汉字本身，一个 token 对三个类，最省解码步数。"""
    return row["label"]


def build_messages(row: dict, use_title: bool = True) -> list:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row, use_title)}]


def render_messages(tokenizer, msgs: list, add_generation_prompt: bool = True) -> str:
    """
    messages → chat 模板文本。训练与评测必须走**这一个**函数，否则模板错位
    会让指标无声掉一半（老师 eval_lora.py 的教学点第一条）。

    enable_thinking=False 必须传：Qwen3-0.6B/1.7B 是混合思考模型，不传的话
    模板里 assistant 后面不闭合 think 块，训练学的是「先推理再作答」而不是
    「直接答标签」。Qwen3-4B-Instruct-2507 的模板没有思考块，这个参数
    会被无害地忽略——所以对三种模型统一传，模板不存在该变量时退回不传。
    """
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
                enable_thinking=False)
        except TypeError:                         # 老版本 transformers 不认这个 kwarg
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
    # 没有 chat 模板的底座（极少见）退化为纯文本拼接，保证脚本不崩
    text = SYSTEM_PROMPT + "\n\n"
    for m in msgs:
        text += f"{'用户' if m['role'] == 'user' else '助手'}：{m['content']}\n\n"
    return text + "助手："


def render_chat_prompt(tokenizer, row: dict, use_title: bool = True) -> str:
    """单条样本 → prompt 文本（zero-shot / SFT 训练与评测共用入口）"""
    return render_messages(tokenizer, build_messages(row, use_title))


class LLMSFTDataset(Dataset):
    """
    SFT 数据集：prompt 段 labels 置 -100（不计 loss），只有答案 token 参与训练。
    与 BERT 共用同一份 rows（同一 split / 同一 seed / 同一去重口径），保证可比。

    __getitem__ 返回：
      input_ids      : [L]
      attention_mask : [L]
      labels         : [L]，prompt 段 -100
    """

    def __init__(self, rows, tokenizer, max_length: int = 1024,
                 use_title: bool = True, label_names=None):
        self.rows = list(rows)
        self.tok = tokenizer
        self.max_length = max_length
        self.use_title = use_title
        self.label_names = label_names or ["高", "中", "低"]

    def __len__(self):
        return len(self.rows)

    def _render_prompt(self, row) -> str:
        return render_chat_prompt(self.tok, row, self.use_title)

    def __getitem__(self, i):
        row = self.rows[i]
        prompt_ids = self.tok.encode(self._render_prompt(row), add_special_tokens=False)
        answer = build_answer(row)
        ans_ids = self.tok.encode(answer, add_special_tokens=False)
        eos = [self.tok.eos_token_id] if self.tok.eos_token_id is not None else []

        input_ids = (prompt_ids + ans_ids + eos)[:self.max_length]
        labels = ([-100] * len(prompt_ids) + ans_ids + eos)[:self.max_length]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def make_pad_collate_fn(tokenizer, pad_to_multiple_of: int = 8):
    """LLM 的 collate：右侧 pad，labels 用 -100 补（否则 pad 位置会被算进 loss）"""

    def collate(features):
        maxlen = max(len(f["input_ids"]) for f in features)
        if pad_to_multiple_of > 1:
            maxlen = ((maxlen + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        ids, labs, mask = [], [], []
        for f in features:
            n = maxlen - len(f["input_ids"])
            ids.append(torch.cat([f["input_ids"], torch.full((n,), pad_id, dtype=torch.long)]))
            labs.append(torch.cat([f["labels"], torch.full((n,), -100, dtype=torch.long)]))
            mask.append(torch.cat([f["attention_mask"], torch.zeros(n, dtype=torch.long)]))
        return {"input_ids": torch.stack(ids), "attention_mask": torch.stack(mask),
                "labels": torch.stack(labs)}
    return collate


def seed_everything(seed: int):
    """BERT 与 LLM 两侧共用，保证「同一份数据、同一个 seed」可比"""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
