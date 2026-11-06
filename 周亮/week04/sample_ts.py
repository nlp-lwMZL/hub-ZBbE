"""
用训练好的字符级 Transformer 做续写：给几个字，模型接着往下写。

用法:
    python sample_ts.py --prompt "今天天气怎么样" --n 80
    python sample_ts.py --prompt "女朋友和我打冷战" --temperature 0.7 --top_k 20
    python sample_ts.py                      # 交互模式，逐行输入

判定标准:
    能接着往下说人话 -> 因果掩码和位置编码都是对的
    原地打转 / 重复 "的的的的" -> 温度太低，或模型没训够
    输出和输入无关 -> 掩码写反了，或者权重和词表对不上
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from week4_homework import TSLM, pick_device      # 复用同一套结构，避免超参漂移
except ImportError:
    raise SystemExit("找不到 week4_homework.py，请把本脚本和它放在同一目录")


def load_model(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise SystemExit(
            f"找不到权重: {ckpt_path}\n先训练一次: python week4_homework.py --epochs 1"
        )
    # weights_only=False: checkpoint 里除了张量还有 char2idx / args 这些普通对象
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = ckpt["args"]
    char2idx, idx2char = ckpt["char2idx"], ckpt["idx2char"]

    model = TSLM(
        vocab_size=len(char2idx),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        dim_ff=cfg["ff_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, char2idx, idx2char, cfg


def encode(prompt, char2idx):
    """逐字查表；字表里没有的字只能丢掉，但要告诉用户丢了什么。"""
    ids, dropped = [], []
    for ch in prompt:
        if ch in char2idx:
            ids.append(char2idx[ch])
        else:
            dropped.append(ch)
    return ids, dropped


@torch.no_grad()
def generate(model, ids, n_new, ctx_len, temperature, top_k, device):
    """
    逐字生成：每次只取最后一个位置的 logits 预测下一个字。
    没有 KV cache，每步都重算整个上下文 —— 对几十个字来说够用，
    真实推理才会用 cache 把已算过的 K/V 存下来。
    """
    ids = list(ids)
    for _ in range(n_new):
        ctx = torch.tensor([ids[-ctx_len:]], dtype=torch.long, device=device)
        logits = model(ctx)[0, -1].float()          # (V,)

        if temperature <= 0:                        # 贪心，容易陷入循环
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k and top_k > 0:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k).values[-1]
                logits = torch.where(logits < thresh, float("-inf"), logits)
            nxt = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))

        ids.append(nxt)
    return ids


def run_once(model, char2idx, idx2char, prompt, args, device):
    ids, dropped = encode(prompt, char2idx)
    if not ids:
        print("  [跳过] 这段话里没有一个字在词表内")
        return
    if dropped:
        print(f"  [提示] 词表外的字已丢弃: {''.join(dropped)}")

    head = "".join(idx2char[i] for i in ids)          # 实际喂进去的那段（可能比原 prompt 短）
    out = generate(model, ids, args.n, args.ctx, args.temperature, args.top_k, device)
    body = "".join(idx2char[i] for i in out[len(ids):])

    print(f"\n  输入  {head}")
    print(f"  续写  {body}")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="best_ts.pt")
    p.add_argument("--prompt", default=None, help="不填则进入交互模式")
    p.add_argument("--n", type=int, default=80, help="续写多少个字")
    p.add_argument("--temperature", type=float, default=0.8, help="0 = 贪心")
    p.add_argument("--top_k", type=int, default=40, help="0 = 不限制")
    p.add_argument("--ctx", type=int, default=0, help="最多回看多少个字，0 = 用训练时的 seq_len")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    torch.manual_seed(args.seed)

    model, char2idx, idx2char, cfg = load_model(args.ckpt, device)
    if args.ctx <= 0:
        args.ctx = cfg["seq_len"]

    n_params = sum(x.numel() for x in model.parameters())
    print(f"权重: {args.ckpt}   设备: {device}")
    print(f"配置: d_model={cfg['d_model']} nhead={cfg['nhead']} ff_dim={cfg['ff_dim']} "
          f"layers={cfg['num_layers']}  参数量={n_params:,}")
    print(f"词表: {len(char2idx)}   训练时 seq_len={cfg['seq_len']}   回看上限={args.ctx}")

    if args.prompt is not None:
        run_once(model, char2idx, idx2char, args.prompt, args, device)
        return

    print("\n交互模式：输入一段话回车，模型接着写；空行退出。\n")
    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        run_once(model, char2idx, idx2char, text, args, device)


if __name__ == "__main__":
    main()
