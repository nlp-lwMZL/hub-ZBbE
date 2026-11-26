"""BiLSTM + CRF 事件抽取模型。

解码期硬约束（--constrained）
----------------------------
CRF 的 Viterbi 解码默认是「自由」的，可能产出非法 BIO 序列（如 O 后面接 I-X）。
开启约束后，在**解码阶段**对转移矩阵加掩码，只允许合法转移：
    O     -> O / B-X
    B-X   -> I-X / B-Y / O
    I-X   -> I-X / B-Y / O
且序列首标签不能是 I-X。
注意：约束仅作用于解码，训练时的 CRF loss 仍使用原始（无掩码）转移矩阵。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.label import BIO_ID2LABEL, OUTSIDE_LABEL, split_bio_label

NEG_INF = -1e4


def build_bio_transition_mask(num_tags: int) -> torch.Tensor:
    """构造 (num_tags, num_tags) 的合法转移掩码：1 合法，0 非法。"""
    labels = [BIO_ID2LABEL[i] for i in range(num_tags)]
    mask = torch.zeros(num_tags, num_tags, dtype=torch.float32)
    for i, li in enumerate(labels):
        p_i, b_i = split_bio_label(li)
        for j, lj in enumerate(labels):
            p_j, b_j = split_bio_label(lj)
            if lj == OUTSIDE_LABEL:
                mask[i, j] = 1.0
            elif p_j == "B":
                mask[i, j] = 1.0
            else:  # j 是 I-*
                if li == OUTSIDE_LABEL:
                    mask[i, j] = 0.0
                else:
                    mask[i, j] = 1.0 if (p_i in ("B", "I") and b_i == b_j) else 0.0
    return mask


def build_bio_start_mask(num_tags: int) -> torch.Tensor:
    """起始位置允许的标签：只要不是 I-* 即可。"""
    labels = [BIO_ID2LABEL[i] for i in range(num_tags)]
    mask = torch.ones(num_tags, dtype=torch.float32)
    for i, lab in enumerate(labels):
        p, _ = split_bio_label(lab)
        if p == "I":
            mask[i] = 0.0
    return mask


class CRF(nn.Module):
    """线性链 CRF。前向算 log-partition，Viterbi 解码，可选 BIO 硬约束。"""

    def __init__(self, num_tags: int, constrained: bool = False):
        super().__init__()
        self.num_tags = num_tags
        self.constrained = constrained

        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

        # 注册为 buffer（不参与训练），随 device 迁移
        self.register_buffer("bio_trans_mask", build_bio_transition_mask(num_tags))
        self.register_buffer("bio_start_mask", build_bio_start_mask(num_tags))

    # ---------------------------------------------------------- 训练用：-log P
    def neg_log_likelihood(
        self, emissions: torch.Tensor, mask: torch.Tensor, tags: torch.Tensor
    ) -> torch.Tensor:
        """emissions (B,T,C)，mask (B,T) bool，tags (B,T) long（padding=-100）。"""
        mask_tags = tags.clone()
        mask_tags[mask_tags < 0] = 0  # 非法 tag 用 0 占位，靠 mask 屏蔽
        log_z = self._compute_log_partition(emissions, mask, self.transitions,
                                            self.start_transitions, self.end_transitions)
        gold_score = self._score(emissions, mask, mask_tags, self.transitions,
                                 self.start_transitions, self.end_transitions)
        return (log_z - gold_score).mean()

    def _compute_log_partition(self, emissions, mask, trans, start_t, end_t):
        B, T, _ = emissions.shape
        alpha = start_t.unsqueeze(0) + emissions[:, 0]
        for t in range(1, T):
            scores = alpha.unsqueeze(2) + trans.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            new_alpha = torch.logsumexp(scores, dim=1)
            m = mask[:, t].unsqueeze(1)
            alpha = torch.where(m, new_alpha, alpha)
        alpha = alpha + end_t.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def _score(self, emissions, mask, tags, trans, start_t, end_t):
        B, T = tags.shape
        score = start_t[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for t in range(1, T):
            cur = (
                trans[tags[:, t - 1], tags[:, t]]
                + emissions[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            )
            m = mask[:, t]
            score = score + torch.where(m, cur, torch.zeros_like(cur))
        lengths = mask.long().sum(dim=1).clamp(min=1)
        last_tags = tags.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        score = score + end_t[last_tags]
        return score

    # ---------------------------------------------------------- 解码：Viterbi
    def decode(
        self, emissions: torch.Tensor, mask: torch.Tensor, constrained: bool | None = None
    ) -> list[list[int]]:
        if constrained is None:
            constrained = self.constrained
        trans = self.transitions
        start_t = self.start_transitions
        if constrained:
            trans = trans + (1.0 - self.bio_trans_mask) * NEG_INF
            start_t = start_t + (1.0 - self.bio_start_mask) * NEG_INF
        return self._viterbi(emissions, mask, trans, start_t, self.end_transitions)

    def _viterbi(self, emissions, mask, trans, start_t, end_t):
        B, T, _ = emissions.shape
        seq_score = start_t.unsqueeze(0) + emissions[:, 0]
        history: list[torch.Tensor] = []
        for t in range(1, T):
            scores = seq_score.unsqueeze(2) + trans.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            best_score, best_idx = scores.max(dim=1)
            m = mask[:, t].unsqueeze(1)
            seq_score = torch.where(m, best_score, seq_score)
            history.append(best_idx)

        seq_score = seq_score + end_t.unsqueeze(0)
        best_last = seq_score.argmax(dim=1)
        lengths = mask.long().sum(dim=1).clamp(min=1).tolist()

        paths: list[list[int]] = []
        for b in range(B):
            L = int(lengths[b])
            path = [int(best_last[b])]
            for t in range(L - 2, -1, -1):
                path.append(int(history[t][b, path[-1]]))
            path.reverse()
            paths.append(path)
        return paths


class BiLSTMCRF(nn.Module):
    """字符级 BiLSTM 编码 + CRF 解码。符合 common.trainer 的统一接口。"""

    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_id: int = 0,
        constrained: bool = False,
        label_prior: list[float] | None = None,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.num_tags = num_tags
        self.constrained = constrained
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_tags)
        self.crf = CRF(num_tags, constrained=constrained)
        # 类别不平衡：给每个标签一个可学习的 emission 偏置，初始化为标签先验。
        # 始终创建（无先验时初始化为 0），保证 state_dict 结构稳定。
        if label_prior is not None:
            init = torch.tensor(label_prior, dtype=torch.float32)
        else:
            init = torch.zeros(num_tags)
        self.tag_bias = nn.Parameter(init)

    def get_config(self) -> dict:
        """返回重建模型所需的全部结构超参（供推理加载 checkpoint）。"""
        return {
            "vocab_size": int(self.embedding.num_embeddings),
            "num_tags": self.num_tags,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_p,
            "pad_id": self.pad_id,
            "constrained": self.constrained,
        }

    def _emissions(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(input_ids))
        out, _ = self.lstm(emb)
        out = self.dropout(out)
        return self.classifier(out) + self.tag_bias  # (B, T, num_tags)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        emissions = self._emissions(input_ids)
        loss = None
        if labels is not None:
            loss = self.crf.neg_log_likelihood(emissions, attention_mask, labels)
        pred_ids = self.crf.decode(emissions, attention_mask)
        return loss, pred_ids


class BiLSTMSoftmax(nn.Module):
    """字符级 BiLSTM + 线性分类（无 CRF），用于消融对比。

    与 BiLSTMCRF 的唯一区别：训练 loss 用逐 token 交叉熵，解码用 argmax。
    constrained=True 时仍做 BIO 合法性约束解码（固定转移，无可学习参数），
    这样「去掉 CRF」与「去掉硬约束」可以拆成两个独立维度。
    """

    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_id: int = 0,
        constrained: bool = False,
        label_prior: list[float] | None = None,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.num_tags = num_tags
        self.constrained = constrained
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_tags)
        if label_prior is not None:
            init = torch.tensor(label_prior, dtype=torch.float32)
        else:
            init = torch.zeros(num_tags)
        self.tag_bias = nn.Parameter(init)
        self.register_buffer("bio_trans_mask", build_bio_transition_mask(num_tags))
        self.register_buffer("bio_start_mask", build_bio_start_mask(num_tags))

    def get_config(self) -> dict:
        return {
            "vocab_size": int(self.embedding.num_embeddings),
            "num_tags": self.num_tags,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_p,
            "pad_id": self.pad_id,
            "constrained": self.constrained,
            "use_crf": False,
        }

    def _emissions(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(input_ids))
        out, _ = self.lstm(emb)
        out = self.dropout(out)
        return self.classifier(out) + self.tag_bias

    def _decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        if not self.constrained:
            pred = emissions.argmax(dim=-1)
            return [p[: int(m.sum())].tolist() for p, m in zip(pred, mask)]
        # 固定转移的约束 Viterbi：合法=0，非法=-1e4，无 end 转移
        trans = (self.bio_trans_mask - 1.0) * 1e4
        start_t = (self.bio_start_mask - 1.0) * 1e4
        B, T, _ = emissions.shape
        seq_score = start_t.unsqueeze(0) + emissions[:, 0]
        history: list[torch.Tensor] = []
        for t in range(1, T):
            scores = seq_score.unsqueeze(2) + trans.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            best_score, best_idx = scores.max(dim=1)
            m = mask[:, t].unsqueeze(1)
            seq_score = torch.where(m, best_score, seq_score)
            history.append(best_idx)
        best_last = seq_score.argmax(dim=1)
        lengths = mask.long().sum(dim=1).clamp(min=1).tolist()
        paths: list[list[int]] = []
        for b in range(B):
            L = int(lengths[b])
            path = [int(best_last[b])]
            for t in range(L - 2, -1, -1):
                path.append(int(history[t][b, path[-1]]))
            path.reverse()
            paths.append(path)
        return paths

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        emissions = self._emissions(input_ids)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                emissions.reshape(-1, self.num_tags),
                labels.reshape(-1),
                ignore_index=-100,
            )
        pred_ids = self._decode(emissions, attention_mask)
        return loss, pred_ids
