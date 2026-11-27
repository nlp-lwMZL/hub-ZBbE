"""RoBERTa（chinese-roberta-wwm-ext）事件抽取模型。

结构：RoBERTa encoder -> dropout -> Linear(num_tags)，可选接线性链 CRF。

统一接口（与 BiLSTM 一致，便于复用公共指标）::

    loss, pred_ids = model(
        input_ids, attention_mask,
        labels=...,            # 子词级标签，padding/special/续接=-100
        content_mask=...,      # bool (B,T)，True=真实内容子词
        content_offsets=...,   # list[list[(start,end)]]，内容子词的字符 offset
        char_lengths=...,      # list[int]，每个 chunk 的字符数
    )

pred_ids 为 **字符级** 预测（已把子词折叠回字符），长度 == char_lengths[b]，
不含 padding，因此可直接对接 common.metrics 按字符 offset 的聚合评估。

子词对齐在 Dataset 层完成（见 train.py 的 RobertaDataset）；模型只负责
emission 计算、损失、解码与子词→字符折叠。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.label import (
    BIO_ID2LABEL,
    BIO_LABEL2ID,
    OUTSIDE_LABEL,
    split_bio_label,
)

try:
    from ..bilstm_crf.model import CRF  # python -m 方式
except ImportError:  # 直接运行
    from extract.bilstm_crf.model import CRF  # type: ignore


# 相邻两个子词之间的空隙（被空白吞掉的字符、或分词器未覆盖的字符）最多回填多少个字符。
# 只用于把"2017 年 6 月 30 日"这类被空格切断的实体重新接起来；过大的空隙不补，
# 避免把两段本不相连的内容误连成一个实体。
GAP_FILL_MAX = 8


def _base_of(label_id: int) -> str:
    """标签 id -> 基标签（O 返回 O）。"""
    name = BIO_ID2LABEL.get(int(label_id), OUTSIDE_LABEL)
    if name == OUTSIDE_LABEL:
        return OUTSIDE_LABEL
    return split_bio_label(name)[1]


def _prefix_of(label_id: int) -> str:
    """标签 id -> B / I / O。"""
    name = BIO_ID2LABEL.get(int(label_id), OUTSIDE_LABEL)
    if name == OUTSIDE_LABEL:
        return OUTSIDE_LABEL
    return split_bio_label(name)[0]


class RobertaForEventExtraction(nn.Module):
    def __init__(
        self,
        pretrain_dir: str,
        num_tags: int,
        dropout: float = 0.1,
        use_crf: bool = False,
        constrained: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.pretrain_dir = str(pretrain_dir)
        self.num_tags = num_tags
        self.use_crf = use_crf
        self.constrained = constrained
        self.dropout_p = dropout

        self.encoder = AutoModel.from_pretrained(self.pretrain_dir)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_tags)

        if use_crf:
            self.crf = CRF(num_tags, constrained=constrained)
        # 无 CRF 时的损失由外部注入（criterion），模型不持有，保持结构稳定

    def get_config(self) -> dict:
        """推理重建模型所需的结构超参。"""
        return {
            "pretrain_dir": self.pretrain_dir,
            "num_tags": self.num_tags,
            "dropout": self.dropout_p,
            "use_crf": self.use_crf,
            "constrained": self.constrained,
        }

    # ---------------------------------------------------------- 前向
    def _emissions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        return self.classifier(self.dropout(out.last_hidden_state))  # (B, T, C)

    def _pack_content(
        self, emissions: torch.Tensor, content_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """把每个样本的内容子词打包成对齐矩形 (B, Lmax, C) + mask。

        CRF 要求 t=0 是第一个有效位置，故必须去掉 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>/[SEP]/pad 后再喂入。
        """
        B, T, C = emissions.shape
        Lmax = int(content_mask.long().sum(1).max().item())
        Lmax = max(Lmax, 1)
        packed = emissions.new_zeros(B, Lmax, C)
        pmask = content_mask.new_zeros(B, Lmax, dtype=torch.bool)
        for b in range(B):
            idx = content_mask[b].nonzero(as_tuple=False).squeeze(-1)
            n = idx.numel()
            if n > 0:
                packed[b, :n] = emissions[b, idx]
                pmask[b, :n] = True
        return packed, pmask

    @staticmethod
    def _fold_to_char(
        sub_pred: list[int],
        content_offsets: list[tuple[int, int]],
        char_len: int,
        outside_id: int,
    ) -> list[int]:
        """内容子词预测 -> 字符级预测。

        两步，缺一不可：

        1. **按子词覆盖区间展开**：一个子词可能覆盖多个字符（如 BPE 把 "519005"
           合成一个 token）。必须把标签写满它覆盖的 `[s, e)` 全部字符，且首字符用
           原标签（B-X），其余字符提升为 I-X；否则 span 会被截成 1 个字符。
        2. **回填空隙**：中文 RoBERTa 的分词器会吞掉空白，"2017 年6 月30 日" 的
           字符 offset 从 4 直接跳到 5，空格位置没有任何 token。若左右两个相邻
           子词同属一个实体（右侧是 I-X），则把它们之间的空隙补成 I-X，
           否则实体在空格处断成多段。

        参数与返回值与改造前保持一致，便于 train.py / 脚本直接复用。
        """
        char_pred = [outside_id] * max(char_len, 0)

        # ---- 1) 子词 -> 字符区间展开 ----
        for k, (s, e) in enumerate(content_offsets):
            if k >= len(sub_pred):
                break
            s, e = int(s), int(e)
            if s < 0 or s >= char_len:
                continue
            e = min(e, char_len)
            if e <= s:  # 防御：异常零长度 offset 至少占 1 个字符
                e = s + 1
            lid = int(sub_pred[k])
            prefix = _prefix_of(lid)
            base = _base_of(lid)
            for c in range(s, e):
                if char_pred[c] != outside_id:
                    continue
                if c == s:
                    char_pred[c] = lid
                elif prefix == "B":
                    char_pred[c] = BIO_LABEL2ID.get(f"I-{base}", lid)
                else:
                    char_pred[c] = lid

        # ---- 2) 回填相邻子词之间被吞掉的空隙 ----
        for k in range(len(content_offsets) - 1):
            if k + 1 >= len(sub_pred):
                break
            e0 = int(content_offsets[k][1])
            s1 = int(content_offsets[k + 1][0])
            if s1 <= e0 or (s1 - e0) > GAP_FILL_MAX:
                continue
            left, right = int(sub_pred[k]), int(sub_pred[k + 1])
            lb, rb = _base_of(left), _base_of(right)
            # 只有"右侧仍是同一实体的续接（I-X）"才回填：
            # 若右侧是 B-X，说明那是另一个新实体，中间的空白不该被吞掉。
            if lb == OUTSIDE_LABEL or lb != rb or _prefix_of(right) != "I":
                continue
            iid = BIO_LABEL2ID.get(f"I-{lb}")
            if iid is None:
                continue
            for c in range(max(0, e0), min(s1, char_len)):
                if char_pred[c] == outside_id:
                    char_pred[c] = iid

        return char_pred

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        content_mask: torch.Tensor | None = None,
        content_offsets: list[list[tuple[int, int]]] | None = None,
        char_lengths: list[int] | None = None,
        criterion: nn.Module | None = None,
    ):
        emissions = self._emissions(input_ids, attention_mask)  # (B, T, C)

        if content_mask is None:
            # 退化：全部非 pad 视为内容（不含 special 时可用），正常训练不会走到
            content_mask = attention_mask.bool()

        packed, pmask = self._pack_content(emissions, content_mask)

        loss = None
        if self.use_crf:
            # CRF 需要沿 T 做数百步递推，bf16 累积误差会破坏 log-partition，
            # 统一升到 float32 计算（autocast 下会自动降级为 fp32 分支）。
            packed = packed.float()
        if labels is not None and criterion is not None and not self.use_crf:
            packed_labels = self._pack_content(
                labels.unsqueeze(-1).float(), content_mask
            )[0].squeeze(-1).long()  # (B, Lmax)
            flat_logits = packed[pmask]  # (N, C)
            flat_labels = packed_labels[pmask]  # (N,)
            loss = criterion(flat_logits, flat_labels)
        elif labels is not None and self.use_crf:
            packed_labels = self._pack_content(
                labels.unsqueeze(-1).float(), content_mask
            )[0].squeeze(-1).long()
            loss = self.crf.neg_log_likelihood(packed, pmask, packed_labels)

        # ---- 解码：子词 -> 字符级 pred_ids ----
        if content_offsets is None or char_lengths is None:
            return loss, None

        B = input_ids.size(0)
        outside_id = BIO_LABEL2ID[OUTSIDE_LABEL]
        pred_ids: list[list[int]] = []
        if self.use_crf:
            sub_paths = self.crf.decode(packed, pmask)  # list[list[int]]
        else:
            argmax = packed.argmax(dim=-1)  # (B, Lmax)
            sub_paths = [
                argmax[b, : int(pmask[b].sum().item())].tolist() for b in range(B)
            ]

        for b in range(B):
            pred_ids.append(
                self._fold_to_char(
                    sub_paths[b] if b < len(sub_paths) else [],
                    content_offsets[b] if b < len(content_offsets) else [],
                    char_lengths[b] if b < len(char_lengths) else 0,
                    outside_id,
                )
            )
        return loss, pred_ids
