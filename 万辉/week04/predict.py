"""
加载训练好的字符级语言模型，预测下一个字 / 连续生成。

用法示例：
    python predict.py --prompt "床前明月光"
    python predict.py --prompt "床前明月光" --n 20
    python predict.py --prompt "床前明月光" --top_k 5
    python predict.py --prompt "床前明月光" --n 20 --temperature 0.8 --top_k 10
"""

import argparse
import torch

# 复用训练脚本里的模型定义，保证结构完全一致
from language_model import LM


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    args = ckpt["args"]

    kwargs = dict(
        vocab_size=len(char2idx),
        embed_dim=args["embed_dim"],
        hidden_dim=args["hidden_dim"],
        num_layers=args["num_layers"],
        model_type=args["model"],
        dropout=args["dropout"],
    )
    # 如果你改过 transformer 版本，LM 接受 nhead 参数
    if args["model"] == "transformer":
        kwargs["nhead"] = args.get("nhead", 4)

    model = LM(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, char2idx, idx2char, args


def encode(text, char2idx):
    """把文本转成 id 列表，过滤掉词表里没有的字符。"""
    return [char2idx[c] for c in text if c in char2idx]


def next_token_probs(model, ids, seq_len, device):
    """输入一段 id，返回最后一个位置对所有字符的概率分布。"""
    ids = ids[-seq_len:]                      # 只保留最近 seq_len 个字符
    x = torch.tensor([ids], dtype=torch.long, device=device)  # (1, T)
    with torch.no_grad():
        logits = model(x)                     # (1, T, V)
    logits = logits[0, -1]                    # 取最后一个时间步 (V,)
    return torch.softmax(logits, dim=-1)


def predict_next(model, char2idx, idx2char, text, seq_len, device,
                 temperature=1.0, top_k=None):
    """
    预测下一个字。
    temperature: 越小越保守，越大越随机
    top_k: 只从概率最高的 k 个字符里采样；None 表示全词表
    """
    ids = encode(text, char2idx)
    if not ids:
        raise ValueError("输入文本里没有任何字符在词表中，无法预测。")

    probs = next_token_probs(model, ids, seq_len, device)

    # 温度缩放
    if temperature != 1.0:
        logits = torch.log(probs) / temperature
        probs = torch.softmax(logits, dim=-1)

    # top-k 截断
    if top_k is not None and top_k < len(probs):
        top_vals, top_idx = torch.topk(probs, top_k)
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[top_idx] = True
        probs = probs * mask
        probs = probs / probs.sum()

    next_id = torch.multinomial(probs, 1).item()
    return idx2char[next_id], probs


def generate(model, char2idx, idx2char, prompt, n, seq_len, device,
             temperature=1.0, top_k=None):
    text = prompt
    for _ in range(n):
        ch, _ = predict_next(
            model, char2idx, idx2char, text, seq_len, device,
            temperature=temperature, top_k=top_k,
        )
        text += ch
    return text


def show_top_k(probs, idx2char, k=5):
    vals, idxs = torch.topk(probs, k)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        ch = idx2char[i]
        # 让空格、换行等可见
        display = {"\n": "\\n", "\t": "\\t", " ": "␣"}.get(ch, ch)
        out.append(f"{display!r}: {v:.4f}")
    return "  ".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="best_model.pt")
    parser.add_argument("--prompt", default=None, help="给定的前文，不传则运行时提示输入")
    parser.add_argument("--n", type=int, default=4, help="连续生成多少个字，0 表示只预测一个")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    # 没传 --prompt 就交互输入
    prompt = args.prompt
    if prompt is None:
        prompt = input("请输入前文（直接回车使用默认「床前明月光」）：").strip()
        if not prompt:
            prompt = "床前明月光"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, char2idx, idx2char, train_args = load_checkpoint(args.ckpt, device)

    seq_len = train_args["seq_len"]
    print(f"device: {device}  model: {train_args['model'].upper()}  vocab: {len(char2idx)}")
    print(f"prompt: {prompt!r}")

    # 预测下一个字，并显示 top-k 候选
    ch, probs = predict_next(
        model, char2idx, idx2char, prompt, seq_len, device,
        temperature=args.temperature, top_k=args.top_k,
    )
    print(f"\n预测下一个字: {ch!r}")
    k = args.top_k or 5
    print(f"Top-{k} 候选: {show_top_k(probs, idx2char, k)}")

    # 连续生成
    if args.n > 0:
        full = generate(
            model, char2idx, idx2char, prompt, args.n, seq_len, device,
            temperature=args.temperature, top_k=args.top_k,
        )
        print(f"\n连续生成 {args.n} 个字:\n{full}")


if __name__ == "__main__":
    main()