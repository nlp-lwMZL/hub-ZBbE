"""model.py — SFT 用的 decoder-only Transformer 语言模型

结构与 week4 的 transformer_lm.py 完全同构（d=192 / 4 层 / 4 头 / Pre-LN /
weight tying / 可学习位置编码），参数命名也保持一致，因此 tf_mask.pt 的权重
可以直接按行前缀加载。week5 的改动只有两处：

  1) max_len 64 → 128（SFT 数据 P95=93 字 + 3 个标记位）
  2) forward 增加 pad_mask：SFT 样本长短不一，batch 内用 <pad> 补齐后，
     padding 位置不能被注意力看到。它和 causal mask 是两个独立的 mask：
        - mask（下三角 float）：注意力层面屏蔽「未来」——decoder-only 的根基
        - src_key_padding_mask（bool）：屏蔽「本 batch 内的填充位」
     两个 mask 分别传给 nn.TransformerEncoder，由 PyTorch 在注意力里合并。
"""
import os

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
WEEK4_CKPT = os.path.join(HERE, "..", "week4", "tf_mask.pt")


class TransformerLM(nn.Module):
    """nn.Embedding(字表) + 位置编码 + N × TransformerEncoderLayer + 共享输出头

    名字叫 Encoder，但加了下三角因果掩码之后，每一层就是
    「Masked Self-Attention + FFN」，等价 GPT 的一个 decoder block——
    week4 作业的核心结论，SFT 之后结构一个字都不变（课件 slide 29）。
    """

    def __init__(self, vocab_size, d_model=192, n_head=4, n_layer=4,
                 max_len=128, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        # 位置编码 128 行（week4 只有 64 行，SFT 数据更长，扩到 128）
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layer, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        # 输出头与输入 embedding 共享权重（weight tying，同 week4/GPT-2）
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight

        # 新增行（字表补缺、位置编码 64→128）用 std=0.02 初始化，同 week4 惯例
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, idx, pad_mask=None):
        """
        idx:      (B, T) long，字 id 序列
        pad_mask: (B, T) bool 或 None。True = padding 位置（注意力不可见）
        返回:     (B, T, vocab) 的 logits
        """
        B, T = idx.shape
        x = self.tok(idx)
        x = x + self.pos(torch.arange(T, device=idx.device))
        x = self.drop(x)
        # 下三角因果掩码：位置 t 只能看 0..t。SFT 不变的东西之一
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        # padding mask 统一成 float(0/-inf)，与 causal mask 同型同语义（-inf = 屏蔽）
        if pad_mask is not None:
            pad_mask = torch.zeros(pad_mask.shape, dtype=x.dtype,
                                   device=idx.device).masked_fill(pad_mask, float("-inf"))
        x = self.encoder(x, mask=mask, src_key_padding_mask=pad_mask)
        return self.head(x)


def load_pretrained(model, ckpt_path=WEEK4_CKPT):
    """
    把 tf_mask.pt 的权重按「行前缀」搬进扩展后的模型：
        tok.weight   (4000,192) → 拷进 (V,192) 的前 4000 行（head 与其共享，自动生效）
        pos.weight   (64,192)   → 拷进 (128,192) 的前 64 行，新 64 行保持新初始化
        encoder.*    形状完全一致，严格加载
    返回 ckpt dict（供 train.py 记录来源）。
    """
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["model_state"]
    with torch.no_grad():
        n_tok = sd["tok.weight"].size(0)
        model.tok.weight[:n_tok].copy_(sd["tok.weight"])
        n_pos = sd["pos.weight"].size(0)
        model.pos.weight[:n_pos].copy_(sd["pos.weight"])
    enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(enc)
    return ck


if __name__ == "__main__":
    # 自检 1：参数量与输出形状
    from dataset import DEFAULT_VAL, SFTDataset, build_vocab

    char2idx, idx2char, special_ids = build_vocab()
    V = len(char2idx)
    model = TransformerLM(V)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"vocab={V}  参数量 {n_par/1e6:.2f}M")

    ds = SFTDataset(DEFAULT_VAL, char2idx, special_ids)
    x0, _ = ds[0]
    x1, _ = ds[1]
    xb = torch.stack([x0, x1])
    pad_mask = xb == special_ids["<pad>"]
    logits = model(xb, pad_mask)
    print(f"输入 {tuple(xb.shape)} → logits {tuple(logits.shape)}，NaN: {bool(torch.isnan(logits).any())}")

    # 自检 2：因果掩码生效——扰动最后一个 token，前面位置的 logits 必须纹丝不动
    model.eval()
    seq = torch.randint(0, V, (1, 12))
    with torch.no_grad():
        base = model(seq)
        pert = seq.clone()
        pert[0, -1] = (pert[0, -1] + 123) % V
        after = model(pert)
    diff = (base[0, :-1] - after[0, :-1]).abs().max().item()
    print(f"扰动末位 token 后，前 11 个位置 logits 的最大变化: {diff}（应为 0）")

    # 自检 3：预训练权重加载
    model2 = TransformerLM(V)
    ck = load_pretrained(model2)
    same = torch.equal(model2.tok.weight[:4000], ck["model_state"]["tok.weight"])
    same_pos = torch.equal(model2.pos.weight[:64], ck["model_state"]["pos.weight"])
    new_pos_random = not torch.equal(model2.pos.weight[64:], ck["model_state"]["pos.weight"][:1].expand(64, 192))
    print(f"tok 前 4000 行一致: {same}  pos 前 64 行一致: {same_pos}")
