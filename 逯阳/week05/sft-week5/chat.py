"""
SFT 效果验证脚本 —— 用 chat 模板提问，看模型是否学会了「回答」。

和 generate.py 的区别：
    generate.py  验证「模型会不会说中文」（自由续写，遇 <eos> 停）
    chat.py      验证「模型会不会答题」（走 chat 模板，遇 <|end|> 停）

输入侧拼的和训练时一模一样：
    <bos> <|user|> {问题} <|assistant|>
输出侧一直采样，直到模型自己吐出 <|end|> 为止 ——
如果它停不下来，说明 SFT 的收尾没学会，脚本会明确告诉你。

用法:
    python chat.py --question "什么是人工智能？"
    python chat.py --question "计算 12 和 25 的平均值" --greedy 1
    python chat.py --interactive 1                 # 进入多轮问答
    python chat.py --model_path sft_transformer.pt --question "把这句话改成被动语态：Tom 发现了工作"

路径约定:
    --model_path 的查找顺序：cwd → SFT 目录 → 预训练工程目录
    模型定义（build_model）在 D:\\project\\循环神经网络语言模型\\generate.py 里，
    通过 --code_dir 挂进 sys.path 复用，不复制一份。
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CODE_DIR = r"D:\project\循环神经网络语言模型"


def _prescan(flag, default):
    """
    在 argparse 之前先把 --code_dir 读出来 —— 因为顶层 import 就需要它，
    等 argparse 解析完再 import 就太晚了。
    """
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == flag:
            return argv[i + 1] if i + 1 < len(argv) else default
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


CODE_DIR = _prescan("--code_dir", DEFAULT_CODE_DIR)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from generate import build_model, apply_top_k_top_p   # noqa: E402

BOS = "<bos>"
U_TAG, A_TAG, END_TAG = "<|user|>", "<|assistant|>", "<|end|>"


def resolve_file(path):
    """权重查找顺序：绝对路径 → cwd → SFT 目录 → 预训练工程目录。"""
    if os.path.isabs(path):
        return path
    for d in (os.getcwd(), SFT_DIR, CODE_DIR):
        p = os.path.join(d, path)
        if os.path.exists(p):
            return p
    return path



@torch.no_grad()
def answer(model, char2idx, idx2char, question, max_new_tokens=128,
           temperature=0.8, top_k=40, top_p=0.95, greedy=False,
           repetition_penalty=1.1, device="cpu"):
    """
    按 chat 模板拼 prompt → 逐字生成 → 遇到 <|end|> 停。

    返回 (回答文本, 生成字数, 是否正常收尾)。
    """
    end_id = char2idx.get(END_TAG)

    missing = [c for c in question if c not in char2idx]
    if missing:
        print(f"  (问题里有 {len(missing)} 个字符不在词表，已忽略: {''.join(missing)[:20]})")

    # 角色标记是单个 token，直接查 id；只有问题正文才逐字编码
    ids = [char2idx[c] for c in (BOS, U_TAG) if c in char2idx]
    ids += [char2idx[c] for c in question if c in char2idx]
    ids.append(char2idx[A_TAG])

    ctx_limit = model.max_len or 10 ** 9
    out_ids = []
    hit_end = False

    for _ in range(max_new_tokens):
        ctx = ids[-ctx_limit:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :].float()

        # 重复惩罚只作用在「已经生成过的字」上，别惩罚问题里的字
        if repetition_penalty != 1.0 and out_ids:
            uniq = torch.tensor(sorted(set(out_ids)), device=logits.device)
            sel = logits[uniq]
            logits[uniq] = torch.where(sel < 0, sel * repetition_penalty, sel / repetition_penalty)

        if greedy:
            next_id = int(torch.argmax(logits))
        else:
            scaled = apply_top_k_top_p(logits / max(temperature, 1e-6), top_k, top_p)
            probs = F.softmax(scaled, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1))

        if end_id is not None and next_id == end_id:
            hit_end = True
            break

        ids.append(next_id)
        out_ids.append(next_id)

    text = "".join(idx2char.get(i, "") for i in out_ids)
    return text, len(out_ids), hit_end


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="sft_transformer.pt", help="SFT 后的 checkpoint")
    p.add_argument("--code_dir", default=CODE_DIR,
                   help="预训练工程目录（generate.py 所在处）")
    p.add_argument("--question", default="什么是人工智能？")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--greedy", type=int, default=0, choices=[0, 1], help="1=贪心解码，结果确定")
    p.add_argument("--num_samples", type=int, default=1, help="同一个问题生成几次做对比")
    p.add_argument("--interactive", type=int, default=0, choices=[0, 1], help="1=进入多轮问答")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = resolve_file(args.model_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 {args.model_path}。先跑 sft_train.py 生成它。\n"
            f"  脚本目录    : {SFT_DIR}\n"
            f"  当前工作目录: {os.getcwd()}"
        )

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model, cfg, kind = build_model(ckpt, device)
    char2idx, idx2char = ckpt["char2idx"], ckpt["idx2char"]

    if END_TAG not in char2idx:
        raise ValueError(
            f"{args.model_path} 里没有 {END_TAG}，它不是 SFT 后的权重。\n"
            f"  SFT 权重请用 sft_train.py 生成；自由续写请用 generate.py。"
        )

    print("=" * 64)
    print(f"模型: {args.model_path}   ({'SFT 微调' if cfg.get('sft') else '未标记 SFT'})")
    print(f"类型: {kind.upper()}  层数={cfg['num_layers']}  d_model={cfg['embed_dim']}  "
          f"nhead={cfg['nhead']}  max_len={cfg['max_len']}")
    print(f"词表: {len(char2idx)} 字   参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"chat 模板: {BOS} {U_TAG} 问题 {A_TAG} 回答 {END_TAG}")
    print("=" * 64)

    def ask(q):
        print(f"\n{U_TAG}\n{q}\n{A_TAG}")
        for i in range(args.num_samples):
            text, n, hit = answer(
                model, char2idx, idx2char, q, args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                greedy=bool(args.greedy), repetition_penalty=args.repetition_penalty,
                device=device,
            )
            tag = f"样本 {i + 1}，" if args.num_samples > 1 else ""
            print(f"{text}")
            print(f"  [{tag}{n} 字，"
                  f"{'模型自己输出了 ' + END_TAG + '，正常收尾' if hit else '未触发 ' + END_TAG + '，到长度上限截断'}]")

    if args.interactive:
        print("\n输入问题回车提问，输入 :q 退出。\n")
        while True:
            try:
                q = input("你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q in (":q", "quit", "exit"):
                break
            if q:
                ask(q)
    else:
        ask(args.question)


if __name__ == "__main__":
    main()
