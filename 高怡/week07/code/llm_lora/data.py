"""SFT 数据集：system/user 不计 loss，只在 assistant 的 JSON 上计算。

- prompt 用 Qwen3 chat template 渲染，且 ``enable_thinking=False``（模板会注入一个
  空的 think 块，因此生成从 JSON 直接开始）。
- 目标文本只含 ``{"event_type", "trigger", "arguments"}``，不含 start/end。
- 标签：prompt 段全部 ``-100``，仅 ``JSON + <|im_end|>`` 计 loss。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .prompt import build_system_prompt, build_target, build_user_prompt
except ImportError:  # 直接运行
    from prompt import build_system_prompt, build_target, build_user_prompt  # type: ignore

IGNORE_INDEX = -100


def render_prompt(tokenizer, text: str) -> str:
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(text)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def encode_prompt(tokenizer, text: str, budget_tokens: int) -> list[int]:
    """在 token 预算内编码 prompt；超长时按比例截断原文（保留开头）。"""
    if budget_tokens <= 0:
        budget_tokens = 64
    cur = text or ""
    while True:
        ids = tokenizer(render_prompt(tokenizer, cur), add_special_tokens=False)["input_ids"]
        if len(ids) <= budget_tokens or len(cur) <= 80:
            return ids
        ratio = budget_tokens / max(1, len(ids))
        cur = cur[: max(80, int(len(cur) * ratio * 0.95))]


def get_im_end_id(tokenizer) -> int:
    tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if tid is None or tid == tokenizer.unk_token_id:
        return tokenizer.eos_token_id
    return tid


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, records, tokenizer, max_seq_len: int = 2048):
        self.records = records
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.im_end_id = get_im_end_id(tokenizer)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        text = rec.get("text") or ""
        target = build_target(rec.get("event"))
        target_ids = self.tok(target, add_special_tokens=False)["input_ids"]
        target_ids = target_ids + [self.im_end_id]

        budget = self.max_seq_len - len(target_ids)
        prompt_ids = encode_prompt(self.tok, text, budget)

        input_ids = prompt_ids + target_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(target_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_len": len(prompt_ids),
        }


def collate_fn(batch, pad_token_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    n = len(batch)
    input_ids = torch.full((n, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((n, max_len), IGNORE_INDEX, dtype=torch.long)
    attn = torch.zeros((n, max_len), dtype=torch.long)
    for i, x in enumerate(batch):
        L = len(x["input_ids"])
        input_ids[i, :L] = torch.tensor(x["input_ids"], dtype=torch.long)
        labels[i, :L] = torch.tensor(x["labels"], dtype=torch.long)
        attn[i, :L] = 1
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
