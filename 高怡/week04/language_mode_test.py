"""
language_mode.py 的自检脚本

对应 results.md 里的：
  题1：打印 generate_square_subsequent_mask(4)，并演示 -inf 如何把未来位置权重压成 0
  题2：扰动最后一个 token，验证因果掩码是否真的屏蔽了未来信息

题3 没有临时代码：它只是用不同的命令行参数反复运行 language_mode.py，
因此不在本文件中。

用法：
    python language_mode_test.py
    python language_mode_test.py --ckpt tf_full_80w.pt --n_chars 10
"""

import argparse
import os
import sys

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(BASE_DIR, "tf_full_80w.pt")
DEFAULT_CORPUS = os.path.join(BASE_DIR, "corpus_train.txt")


def load_language_mode_class():
    """
    导入 language_mode.py 中的 LanguageMode。

    language_mode.py 在模块级就执行了 ap.parse_args()，直接 import 会把本脚本的
    命令行参数吃掉（甚至报错），所以导入期间先把 sys.argv 换成只有脚本名。
    """
    original_argv = sys.argv
    sys.argv = [original_argv[0]]
    try:
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)
        from language_mode import LanguageMode
    finally:
        sys.argv = original_argv
    return LanguageMode


def check_mask_matrix(n=4):
    """
    题1：官方因果掩码的形状 + softmax 演示。

    主对角线及以下（j <= i）为 0，主对角线右上方（j > i）为 -inf。
    加到注意力打分上后经 softmax，-inf 位置权重精确为 0，0 位置保持原权重。
    """
    print("=" * 70)
    print(f"题1：torch.nn.Transformer.generate_square_subsequent_mask({n})")
    print("=" * 70)
    mask = torch.nn.Transformer.generate_square_subsequent_mask(n)
    print(f"shape = {tuple(mask.shape)}, dtype = {mask.dtype}")
    print(mask)

    # 演示：同一份打分加掩码前后 softmax 权重的差异
    torch.manual_seed(0)  # 固定随机打分，保证每次运行结果一致
    scores = torch.randn(n, n)  # 模拟 QKᵀ/√d
    masked = scores.masked_fill(mask == float("-inf"), float("-inf"))
    probs = torch.softmax(masked, dim=-1)

    print("\n原始打分 scores:")
    print(scores)
    print("\n加上掩码后:")
    print(masked)
    print("\nsoftmax 后（未来位置权重恒为 0）:")
    print(probs)
    print("\n每行权重和:", probs.sum(-1))
    print("exp(-inf) =", torch.exp(torch.tensor(float("-inf"))))

    # 未来位置（严格上三角）的权重必须精确为 0
    upper = torch.triu(torch.ones(n, n), diagonal=1).bool()
    assert bool((probs[upper] == 0).all()), "上三角位置 softmax 权重不为 0"
    print("判定：上三角位置权重全为 0 → 通过")


def check_causal_mask_effect(ckpt_path, corpus_path, n_chars=10):
    """
    题2：扰动最后一个 token，比较每个位置 logits 的最大变化。

    有因果掩码：前 n-1 个位置看不到末位，输出必须逐位精确为 0
    去掉掩码：末位的改动会通过注意力扩散到所有位置，全部非 0
    """
    print("\n" + "=" * 70)
    print("题2：扰动末位 token，验证因果掩码是否生效")
    print("=" * 70)
    if not os.path.exists(ckpt_path):
        print(f"未找到权重 {ckpt_path}，跳过题2（可先训练或指定 --ckpt）")
        return

    LanguageMode = load_language_mode_class()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    model = LanguageMode(
        len(vocab), ckpt["d_model"], ckpt["num_layers"], ckpt["num_heads"],
        ckpt["max_length"], ckpt["dropout"], ckpt["use_pos"], ckpt["causal"],
        arch=ckpt.get("arch", "transformer"),
        tie=ckpt.get("tie", False),
        lstm_hidden=ckpt.get("lstm_hidden", 0),
    )
    model.load_state_dict(ckpt["state_dict"])
    # 必须 eval：train 模式下 dropout 会随机置零神经元，前几位的变化量就不再是 0
    model.eval()

    with open(corpus_path, "r", encoding="utf-8") as f:
        line = next(text.strip() for text in f if len(text.strip()) >= n_chars)
    text = line[:n_chars]
    other_char = line[n_chars - 2]  # 用倒数第二个字作为扰动值，保证与末位不同
    x = torch.LongTensor([[vocab.get(c, 1) for c in text]])
    print(f"输入: {text}   末位字: {text[-1]} → 改成: {other_char}")

    def probe(causal):
        """切换掩码开关，量化末位扰动对每个位置 logits 的影响"""
        model.causal = causal
        with torch.no_grad():
            base = model(x)
            perturbed = x.clone()
            perturbed[0, -1] = vocab.get(other_char, 1)  # 只扰动最后一个 token
            changed = model(perturbed)
        return (changed - base).abs().amax(-1)[0]  # 每个位置 logits 的最大变化

    with_mask = probe(True)
    no_mask = probe(False)
    print("\n有掩码 各位置最大变化:")
    print(with_mask)
    print("\n无掩码 各位置最大变化:")
    print(no_mask)

    past_all_zero = bool((with_mask[:-1] == 0).all())
    past_all_changed = bool((no_mask[:-1] != 0).all())
    print(f"\n有掩码 前{n_chars - 1}位是否全为精确 0: {past_all_zero}")
    print(f"无掩码 前{n_chars - 1}位是否全部非 0: {past_all_changed} "
          f"（最大变化 {no_mask[:-1].max():.4e}）")
    print(f"末位变化 有掩码 {with_mask[-1]:.4e} / 无掩码 {no_mask[-1]:.4e}")
    assert past_all_zero and past_all_changed, "掩码验证未通过"
    print("判定：因果掩码已生效 → 通过")


def main():
    parser = argparse.ArgumentParser(description="language_mode.py 的自检：题1 掩码矩阵 / 题2 掩码生效验证")
    parser.add_argument("--mask_n", type=int, default=4, help="题1 的掩码边长")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="题2 使用的权重文件")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, help="题2 采样输入的语料文件")
    parser.add_argument("--n_chars", type=int, default=10, help="题2 输入长度")
    args = parser.parse_args()

    check_mask_matrix(args.mask_n)
    check_causal_mask_effect(args.ckpt, args.corpus, args.n_chars)


if __name__ == "__main__":
    main()
