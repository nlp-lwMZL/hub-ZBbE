"""
L1 基线：字符级 BiLSTM + Linear / CRF（无预训练，从零训字向量）。

为什么需要这一臂（老师 PPT 的核心对照）：
  它和 L2 RoBERTa 是「同任务、同标签、同 CRF、同评测」的对照，唯一差别是特征编码器：
  BiLSTM 从零学字向量 vs RoBERTa 带着预训练知识。两者一减 = 「预训练值多少个点」。
  本机 CPU 可训完，每一步都能摊开算，适合教学。

结构：字向量(128) -> BiLSTM(hidden 256×2) -> Linear(num_labels) -> [CRF]
  - 一个汉字一个 id（查字表），无子词、无 word_ids 对齐
  - CRF 与 L2 用同一个 torchcrf 实现（保证公平，不是「换个 CRF 就变强」）

【batch 契约】model(**batch)：input_ids / attention_mask / labels，均 (B,L) LongTensor。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
from common.ner_data import IGNORE_INDEX


class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden: int = 256,
                 num_layers: int = 1, dropout: float = 0.3) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.hidden_size = hidden * 2

    def forward(self, input_ids, attention_mask=None):
        return self.lstm(self.dropout(self.emb(input_ids)))[0]       # (B, L, 2*hidden)


class BiLSTMNerBase(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, emb_dim: int = 128,
                 hidden: int = 256, num_layers: int = 1, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = BiLSTMEncoder(vocab_size, emb_dim, hidden, num_layers, dropout)
        self.classifier = nn.Linear(self.encoder.hidden_size, num_labels)
        self.num_labels = num_labels

    def emissions(self, input_ids, attention_mask=None) -> torch.Tensor:
        return self.classifier(self.encoder(input_ids, attention_mask))

    def parameter_groups(self, lr: float) -> list[dict]:
        return [{"params": self.parameters(), "lr": lr}]


class BiLSTMLinearNer(BiLSTMNerBase):
    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.emissions(input_ids, attention_mask)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1),
                                   ignore_index=IGNORE_INDEX)
        return logits, loss

    def decode(self, input_ids, attention_mask=None, labels=None) -> torch.Tensor:
        logits, _ = self.forward(input_ids, attention_mask, labels=None)
        pred = logits.argmax(-1)
        valid = labels.ne(IGNORE_INDEX) if labels is not None else attention_mask.bool()
        return pred.masked_fill(~valid, IGNORE_INDEX)


class BiLSTMCRFNer(BiLSTMNerBase):
    def __init__(self, vocab_size, num_labels, emb_dim=128, hidden=256,
                 num_layers=1, dropout=0.3):
        super().__init__(vocab_size, num_labels, emb_dim, hidden, num_layers, dropout)
        try:
            from torchcrf import CRF
        except ImportError as exc:
            raise ImportError("BiLSTM+CRF 需要 pytorch-crf：pip install pytorch-crf") from exc
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        emissions = self.emissions(input_ids, attention_mask)
        loss = None
        if labels is not None:
            mask = attention_mask.bool()          # 连续前缀，不挖洞
            tags = labels.clone()
            tags[tags == IGNORE_INDEX] = 0
            loss = -self.crf(emissions, tags, mask=mask, reduction="token_mean")
        return emissions, loss

    def decode(self, input_ids, attention_mask=None, labels=None) -> torch.Tensor:
        emissions = self.emissions(input_ids, attention_mask)
        mask = attention_mask.bool()
        paths = self.crf.decode(emissions, mask=mask)
        B, L = emissions.shape[:2]
        out = torch.full((B, L), IGNORE_INDEX, dtype=torch.long, device=emissions.device)
        for i, path in enumerate(paths):
            idx = mask[i].nonzero(as_tuple=False).flatten()
            assert len(path) == idx.numel(), "CRF 回填长度与 mask 不一致"
            out[i, idx] = torch.tensor(path, dtype=torch.long, device=emissions.device)
        if labels is not None:
            out = out.masked_fill(labels.eq(IGNORE_INDEX), IGNORE_INDEX)
        return out


def build_model(use_crf: bool, vocab_size: int, num_labels: int, **kw):
    cls = BiLSTMCRFNer if use_crf else BiLSTMLinearNer
    model = cls(vocab_size, num_labels, **kw)
    print(f"模型：字符级 BiLSTM+{'CRF' if use_crf else 'Linear'} | "
          f"字表 {vocab_size} | 标签 {num_labels} | "
          f"参数 {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    return model


def _self_test(use_crf: bool) -> None:
    torch.manual_seed(7)
    B, L, V, K = 2, 6, 50, 11
    input_ids = torch.randint(2, V, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    labels = torch.randint(0, K, (B, L))
    labels[0, 5] = IGNORE_INDEX
    model = build_model(use_crf, V, K)
    model.train()
    _, loss = model(input_ids, attention_mask, labels)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    model.eval()
    with torch.no_grad():
        pred = model.decode(input_ids, attention_mask, labels)
    assert pred.shape == labels.shape
    assert (pred[labels == IGNORE_INDEX] == IGNORE_INDEX).all()
    print(f"  自检 {'CRF' if use_crf else 'Linear'}: loss={loss.item():.4f} ✓")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_crf", action="store_true")
    a = ap.parse_args()
    _self_test(a.use_crf)
