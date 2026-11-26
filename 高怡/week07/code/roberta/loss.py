"""统一的 token 级分类损失。

设计原则（与用户确认）：
  - ``weight`` 是类别不平衡的**单一真源**：CE / 标签平滑 / Focal 共用同一份
    per-class 权重向量，绝不维护两套，避免口径冲突、无法归因。
  - Focal 的动态调制因子 ``(1 - p_t)^gamma`` 是样本级的（由预测概率实时算出），
    不是权重矩阵，标签平滑没有它；两者正交叠加：
        loss = weight[y] * (1 - p_t)^gamma * smoothed_CE
  - 标签平滑与 Focal 同时开启时打日志提示（两者部分对冲，通常不同时开）。

只在 **Softmax（无 CRF）** 路径使用；CRF 路径的损失由 CRF 自身负责（NLL）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenClassificationLoss(nn.Module):
    """逐 token 损失：可选类别加权 + 标签平滑 + Focal。

    参数
    ----
    weight : Tensor | None
        形状 (num_tags,) 的 per-class 权重，CE/LS/Focal 共用。None 表示不加权。
    label_smoothing : float
        标签平滑系数，0 关闭。
    focal_gamma : float
        Focal 的 gamma，0 关闭（退化为带权 CE）。
    ignore_index : int
        要忽略的标签 id（padding / <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> / [SEP] / 子词续接）。
    """

    def __init__(
        self,
        num_tags: int,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.weight = weight
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)
        self.ignore_index = ignore_index

        if self.focal_gamma > 0 and self.label_smoothing > 0:
            import logging
            logging.getLogger("event_extract").warning(
                "同时启用 Focal(gamma=%s) 与 LabelSmoothing(%s)：两者部分对冲，"
                "通常不建议同时开。", self.focal_gamma, self.label_smoothing,
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits (N, C)，targets (N,)，含 ignore_index 的位置被忽略。"""
        valid = targets != self.ignore_index
        logits = logits[valid]
        targets = targets[valid]
        if targets.numel() == 0:
            return logits.sum() * 0.0  # 保持计算图连通的零损失

        log_probs = F.log_softmax(logits, dim=-1)  # (N, C)
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)

        if self.label_smoothing > 0:
            smooth = -log_probs.mean(dim=-1)  # (N,)
            loss = (1.0 - self.label_smoothing) * nll + self.label_smoothing * smooth
        else:
            loss = nll

        if self.focal_gamma > 0:
            p_t = (-nll).exp()  # 模型对硬真值的预测概率
            loss = loss * ((1.0 - p_t) ** self.focal_gamma)

        if self.weight is not None:
            w = self.weight.to(loss.device)[targets]
            loss = loss * w

        return loss.mean()
