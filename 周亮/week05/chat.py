"""chat.py — SFT 后的问答演示：交互式 或 --prompt 单问

两种用法（Q3 共识）：
    python chat.py --prompt "什么是人工智能？"     # 单问单答，适合报告截图
    python chat.py                                 # 交互模式，输入问题回车，q/exit 退出

生成（Q12 共识 + Q14 决策）：
    - 无 KV cache：每步把整段上文重新前向一遍（week4 generate 同款）。
      KV cache 的原理是缓存每层已算过的 K/V、新 token 只做增量计算；
      不做的权衡：模型仅 2.6M、序列 ≤128、CPU 上单步毫秒级，收益无感；
      且训练用的 nn.TransformerEncoderLayer 封装不暴露内部 K/V，
      行业形态（HF/Llama）是模型实现天生带 past_key_values 接口、训练推理一套代码。
      详见 week5作业.md。
    - temperature 0.8（传 --temperature 0 为贪心，输出确定，适合对比）
    - 单轮独立会话：训练数据全是单轮 QA，模型没学过多轮轮换
    - 生成到 <|end|> 停——这正是 SFT 教会的「说完就停」（课件 slide 29）
"""
import argparse
import os

import torch
import torch.nn.functional as F

from dataset import encode_text
from model import TransformerLM

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    a = ck["args"]
    char2idx = ck["char2idx"]
    idx2char = {int(k): v for k, v in ck["idx2char"].items()}
    special_ids = ck["special_ids"]
    model = TransformerLM(len(char2idx), a["d_model"], a["heads"], a["layers"],
                          a["max_len"], 0.0).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()   # 关 dropout，贪心输出才可复现
    return model, char2idx, idx2char, special_ids, a


def encode_prompt(question, char2idx, special_ids):
    """<|user|> + 问题 + <|assistant|>；问题里的生僻字直接跳过"""
    ids = [special_ids["<|user|>"]]
    ids += encode_text(question, char2idx)
    ids.append(special_ids["<|assistant|>"])
    return ids


@torch.no_grad()
def generate_answer(model, prompt_ids, idx2char, special_ids,
                    max_len, max_new_tokens=100, temperature=0.8):
    """逐 token 生成，总长不超过 max_len（位置编码只有 max_len 行）"""
    ids = list(prompt_ids)
    end_id = special_ids["<|end|>"]
    n_new = min(max_new_tokens, max_len - len(ids))
    for _ in range(n_new):
        logits = model(torch.tensor([ids], dtype=torch.long))[0, -1]
        if temperature <= 0.01:
            nxt = int(logits.argmax())        # 贪心：确定输出，用于对比实验
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
        ids.append(nxt)
        if nxt == end_id:
            break
    # 解码：只取新生成部分，跳过特殊标记
    return "".join(idx2char[i] for i in ids[len(prompt_ids):]
                   if not (idx2char[i].startswith("<|") or idx2char[i] == "<pad>"))


def ask(model, question, char2idx, idx2char, special_ids, args):
    ids = encode_prompt(question, char2idx, special_ids)
    return generate_answer(model, ids, idx2char, special_ids,
                           args.max_len, args.max_new_tokens, args.temperature)


def main():
    ap = argparse.ArgumentParser(description="SFT 问答演示")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "best_sft.pt"))
    ap.add_argument("--prompt", default=None, help="直接问一个问题（不进交互模式）")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="0 = 贪心（argmax），输出确定")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cpu")
    model, char2idx, idx2char, special_ids, ck_args = load_model(args.ckpt, device)
    args.max_len = ck_args.get("max_len", 128)

    if args.prompt is not None:
        print(ask(model, args.prompt, char2idx, idx2char, special_ids, args))
        return

    print(f"SFT 问答演示（{os.path.basename(args.ckpt)}，temperature={args.temperature}，"
          f"输入 q 退出）")
    while True:
        try:
            q = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("q", "exit", "quit"):
            break
        print(f"模型: {ask(model, q, char2idx, idx2char, special_ids, args)}")


if __name__ == "__main__":
    main()
