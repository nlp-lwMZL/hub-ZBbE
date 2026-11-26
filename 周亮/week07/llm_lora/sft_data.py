"""
SFT 数据构造：cdtier NER 记录 -> chat 样本（prompt + 目标实体 JSON）+ loss masking。

目标格式（你选的「实体 span 列表 JSON」）：
  {"entities": [{"type":"Attacker","text":"APT-C-39"}, {"type":"Tools","text":"Houdini"}]}
  - 只输出**类型 + 原文片段文本**，不输出坐标（生成模型算 offset 极不可靠，老师 PPT 的坑）；
  - 评测时用 text.find() 把文本对齐回字符 span（见 llm_utils.align_spans_to_text）。

【loss masking】只在 assistant 回复部分算 loss，prompt 全置 -100；
  否则模型花一半算力学「复述用户输入」，且 loss 被长 prompt 稀释。

【Qwen3 的坑】默认开思考会污染输出 -> apply_chat_template 必须 enable_thinking=False；
  目标必须带 EOS，否则推理时不会停。

用法：python llm_lora/sft_data.py   （自检：打印一条样本 + mask 对齐统计）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common.ner_data import IGNORE_INDEX

SYSTEM_PROMPT = """你是一个网络安全实体抽取助手。从给定文本中抽取所有下列类型的实体：
Attacker（攻击组织/攻击者）、Campaign（攻击活动/行动）、Industry（行业/领域/宗教等群体）、Region（地区/国家）、Tools（攻击工具/恶意软件/病毒名）。

输出要求：
- 只输出一个 JSON，格式：{"entities": [{"type": "类型", "text": "原文片段"}, ...]}
- "text" 必须是原文中**逐字连续出现**的片段，不要改写、不要补字、不要翻译；
- 不要输出解释、注释、Markdown 代码块或任何 JSON 之外的文字；
- 没有实体时输出 {"entities": []}。"""


def build_target(entities: list[dict[str, Any]]) -> str:
    """从 gold entities 构造目标 JSON（只留 type + text，text 取原文切片）。"""
    items = [{"type": e["type"], "text": e["text"]} for e in entities]
    return json.dumps({"entities": items}, ensure_ascii=False)


class SFTDataset:
    """把 NER 记录转成 chat 格式的 token 样本。"""

    def __init__(self, records: list[dict], tokenizer, max_length: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[dict[str, Any]] = []
        self.n_prompt_truncated = 0
        self.n_target_truncated = 0
        for rec in records:
            target = build_target(rec.get("entities", []))
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": f"文本：{rec['text']}"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            # 目标 + EOS（必须带 EOS，否则推理时不知道在哪停）
            response_ids = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
            full = (prompt_ids + response_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]
            if len(prompt_ids) >= max_length:
                self.n_prompt_truncated += 1
            elif len(prompt_ids) + len(response_ids) > max_length:
                self.n_target_truncated += 1
            if all(x == IGNORE_INDEX for x in labels):
                continue     # prompt 自己就超长，labels 全 -100 → loss=nan，丢弃
            self.samples.append({
                "input_ids": full,
                "labels": labels,
                "id": rec["id"],
                "text": rec["text"],
                "gold_spans": [{"type": e["type"], "text": e["text"]} for e in rec.get("entities", [])],
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.samples[i]


def collate_sft(batch: list[dict[str, Any]], pad_token_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attn = [], [], []
    for x in batch:
        n = len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * (max_len - n))
        labels.append(x["labels"] + [IGNORE_INDEX] * (max_len - n))
        attn.append([1] * n + [0] * (max_len - n))
    import torch
    return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)}


def _self_test() -> None:
    import sys
    from pathlib import Path
    code = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code))
    from common import paths
    from common.model_paths import resolve_qwen
    from transformers import AutoTokenizer

    recs = json.loads((paths.get_data_dir() / "train.json").read_text(encoding="utf-8"))[:3]
    tok = AutoTokenizer.from_pretrained(str(resolve_qwen()), trust_remote_code=False)
    ds = SFTDataset(recs, tok, max_length=1024)
    print(f"样本数 {len(ds)} | prompt 截断 {ds.n_prompt_truncated} | 目标截断 {ds.n_target_truncated}")
    s = ds[0]
    print(f"input_ids {len(s['input_ids'])} | labels 有效位 {sum(l != -100 for l in s['labels'])}")
    print(f"gold entities: {s['gold_spans']}")
    resp = tok.decode([t for t, l in zip(s["input_ids"], s["labels"]) if l != -100])
    print(f"回复部分解码: {resp[:200]}")
    # 断言：labels 有效位从 prompt 之后才开始；结尾是 EOS
    assert sum(l == -100 for l in s["labels"]) > 0, "prompt 没被 mask"
    assert s["labels"][-1] == tok.eos_token_id or tok.eos_token_id in s["labels"][-3:], "目标没带 EOS"


if __name__ == "__main__":
    _self_test()
