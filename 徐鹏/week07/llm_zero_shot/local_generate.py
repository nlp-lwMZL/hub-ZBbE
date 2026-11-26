# -*- coding: utf-8 -*-
"""
本地小模型的**批量**贪心生成（llm_extract.py 与 evaluate_sft.py 共用）

为什么需要这个模块（一条真实的工程坑）：
  在 Windows + RTX 4060 Laptop 上实测：Qwen3-0.6B 逐条生成只有 **1.5~2 tok/s**
  （150 条约 1~1.5 小时）。排查结论：
    - 纯算力没问题：bf16 大矩阵乘实测 21 TFLOPS；
    - 慢的不是"算"，而是"启动"：逐 token 解码每步要发 ~200 个 CUDA kernel，
      Windows(WDDM) 下每次 kernel 启动约 1~2 ms，GPU 一直停在 P8 低功耗档（7W），
      CPU 侧反而忙 —— 于是 decode 阶段被启动开销淹没。
  对策：把多条样本拼成一个 batch 一起解码。kernel 数量不变，但每个 kernel 的活变多了，
  启动开销被摊薄，实测提速 4~6 倍（batch=8）。

  这类"吞吐 vs 延迟"的取舍是工程常态：批量生成提高吞吐，但单条延迟略升、显存占用变大
  （要同时容纳 batch 份 KV cache）。评测脚本为了"跑完 200 条"用 batch；线上服务用 batch=1。

关键细节（写错就全乱）：
  1. **左 padding**：decoder-only 模型必须把 pad 补在左边，最后一位才是真正的"下一步"位置；
     right padding 会让模型从 pad 位置开始续写，输出全是垃圾。
  2. `pad_token_id = eos_token_id`：Qwen 没有单独的 pad，复用 eos；
     生成时用 attention_mask 屏蔽 pad，否则注意力会看到一串"结束符"。
  3. 解码时只取 `out[:, prompt_len:]`，要把左 padding 部分一起跳过（prompt_len 是 padding 后的长度）。
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_torch_dtype(device: torch.device, dtype: str = "auto") -> torch.dtype:
    """auto = 有 CUDA 用 bf16（Qwen3 权重原生 bf16），CPU 只能 fp32。"""
    if dtype == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]


def load_causal_lm(model_path: str, device: torch.device,
                   dtype: torch.dtype = torch.bfloat16):
    """
    加载本地基座模型 + tokenizer。

    transformers 5.x 注意：`torch_dtype=` 已废弃，要用 `dtype=`。
    离线课堂约定：只用本地已下载的权重，不联网下载（resolve_model_path 已保证路径存在）。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"     # ← 生成必须左 padding（见模块头部第 1 条）
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, trust_remote_code=True)
    model = model.to(device).eval()
    return model, tokenizer


def batch_messages(tokenizer, messages_list):
    """把 chat messages 批量渲染成 prompt 文本（tokenize=False 兼容 transformers 5.x）。"""
    return [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for msgs in messages_list
    ]


def generate_batch(model, tokenizer, messages_list, device,
                   max_new_tokens: int = 320, batch_size: int = 8,
                   max_length: int = 1280) -> list[dict]:
    """
    批量贪心生成，返回与 messages_list 对齐的列表：
      [{"text": str, "latency_s": float, "prompt_tokens": int, "completion_tokens": int}]

    为什么贪心（do_sample=False）：抽取任务要"同一输入同一输出"，评测才算得稳。
    max_length 只用于兜底超长 prompt（保留**末尾**，因为指令和句子都在尾部）。
    """
    results = []
    for start in range(0, len(messages_list), batch_size):
        chunk = messages_list[start:start + batch_size]
        prompts = batch_messages(tokenizer, chunk)
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False, truncation=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # 超长兜底：保留尾部（prompt 的尾部才是"请你抽取这句话"，头部是 schema 说明）
        keep = max_length - max_new_tokens
        if input_ids.shape[1] > keep > 0:
            input_ids = input_ids[:, -keep:]
            attention_mask = attention_mask[:, -keep:]

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,                       # 贪心
                pad_token_id=tokenizer.eos_token_id,
            )
        latency = time.time() - t0
        prompt_len = input_ids.shape[1]
        for j in range(input_ids.shape[0]):
            new_ids = out[j][prompt_len:]
            results.append({
                "text": tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
                "latency_s": latency / input_ids.shape[0],   # 摊到每条（吞吐视角）
                "prompt_tokens": int(attention_mask[j].sum().item()),
                "completion_tokens": int(new_ids.shape[0]),
            })
    return results
