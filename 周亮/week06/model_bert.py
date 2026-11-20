"""
BERT 分类模型 + 对抗训练 + 损失函数

结构：BertModel → 池化（cls/mean/max）→ Dropout → Linear → logits
只改两处——**池化方式**和**损失函数**，BERT 本体 12 层不动，这就是本次消融的范围。

教学重点：
  1. 为什么手写分类头而不用 BertForSequenceClassification？
     结构一目了然，且池化方式可以自由替换（框架自带的只给 pooler_output）。
  2. 池化前必须 mask 掉 [PAD]：Mean 会被补位稀释、Max 会把 pad 的 0 当成特征。
     这是最常见的隐形 bug——模型能训，但指标莫名低几个点。
  3. 分类头是随机初始化的，需要比 BERT 大的学习率（本项目 ×5）。

使用方式：
  from model_bert import build_model, get_param_groups, FGM, focal_loss
  model = build_model(bert_dir, num_labels=3, pool="mean")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

POOL_OPTIONS = ("cls", "mean", "max")


class BertClassifier(nn.Module):
    """
    pool 三选一：
      cls  —— 取 [CLS] 向量（预训练就为它设计，但不一定最适合所有文本）
      mean —— 对所有有效 token 取均值（长文本常更稳）
      max  —— 逐维取最大（抓最显著特征，噪声也大）
    """

    def __init__(self, model_dir, num_labels: int = 3, pool: str = "cls", dropout: float = 0.1):
        super().__init__()
        assert pool in POOL_OPTIONS, f"pool 必须是 {POOL_OPTIONS} 之一，收到 {pool}"
        import transformers
        from transformers import AutoModel

        self.pool = pool
        # 加载时的 LOAD REPORT 会把 MLM 预训练头（cls.*）列为 UNEXPECTED——
        # 那是 backbone 的正常现象，但会刷满日志，这里临时降噪
        prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            self.bert = AutoModel.from_pretrained(str(model_dir))
        finally:
            transformers.logging.set_verbosity(prev)
        hidden = self.bert.config.hidden_size          # bert-base = 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # return_dict=True：transformers 5.x 里不传会拿到 tuple，容易下标取错
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids, return_dict=True)
        hs = out.last_hidden_state                     # [B, L, H]
        if self.pool == "cls":
            vec = hs[:, 0]
        else:
            m = attention_mask.unsqueeze(-1).float()    # [B, L, 1]
            if self.pool == "mean":
                vec = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            else:                                      # max：PAD 位置压到极小值再取 max
                vec = hs.masked_fill(m == 0, -1e9).max(1).values
        return self.classifier(self.dropout(vec))


def build_model(model_dir, num_labels: int, pool: str = "cls", dropout: float = 0.1,
                logger=None):
    model = BertClassifier(model_dir, num_labels=num_labels, pool=pool, dropout=dropout)
    n_all = sum(p.numel() for p in model.parameters()) / 1e6
    n_head = sum(p.numel() for p in model.classifier.parameters()) / 1e3
    msg = f"模型参数量 {n_all:.1f}M（分类头 {n_head:.1f}K）| 池化策略 {pool}"
    (logger.info if logger else print)(msg)
    return model


def get_param_groups(model, lr: float, head_lr_mult: float = 5.0):
    """分层学习率：头是随机初始化的要学得快；BERT 小步走，否则冲掉预训练知识"""
    head = list(model.classifier.parameters()) + list(model.dropout.parameters())
    bert = [p for n, p in model.named_parameters()
            if not n.startswith("classifier") and not n.startswith("dropout")]
    return [{"params": bert, "lr": lr}, {"params": head, "lr": lr * head_lr_mult}]


# ─────────────────────────────────────────────────────────────────────────────
# FGM 对抗训练
# ─────────────────────────────────────────────────────────────────────────────

class FGM:
    """
    FGM（Fast Gradient Method）三步：
      ① 正常前向 + 反传，拿到参数的梯度 g；
      ② 在 **word_embeddings** 上沿梯度方向加扰动：r = eps * g / ||g||；
      ③ 用被扰动的权重再前向 + 反传一次，两次梯度一起更新，最后把权重还原。
    为什么只扰动 embedding：文本任务的「输入」就是 embedding，扰它等于在字的表示上加噪；
    扰动全部参数既慢又容易破坏预训练知识。
    为什么必须归一化：不除 ||g||，eps 的单位随梯度大小漂，同一 eps 在不同 step 上强度完全不同。
    AMP 下用的是缩放后的梯度，方向不变、模长同乘一个 scale，除 ||g|| 时自动约掉，可直接用。
    """

    def __init__(self, model, emb_name="word_embeddings", eps=1.0):
        self.model, self.emb_name, self.eps = model, emb_name, eps
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(self.eps * param.grad / norm)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ─────────────────────────────────────────────────────────────────────────────
# 损失函数
# ─────────────────────────────────────────────────────────────────────────────

def focal_loss(logits, targets, weight=None, gamma=2.0):
    """
    Focal Loss：把「已经学会的样本」的权重压下去，让梯度集中在难样本上。
        L = -alpha_t * (1 - p_t)^gamma * log(p_t)
    alpha 直接用类别权重表——和「类别加权」共用同一张表，不额外加超参。
    """
    logp = F.log_softmax(logits, dim=-1)
    logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    p_t = logp_t.exp()
    alpha = weight[targets] if weight is not None else 1.0
    return (-alpha * ((1.0 - p_t) ** gamma) * logp_t).mean()


def compute_class_weight(labels, n_classes: int, mode: str = "balanced"):
    """
    mode: none → None；sqrt → 权重的平方根（温和）；balanced → 完全反频率（激进）
    归一化到均值 1：这样 lr 不用跟着改，消融才只有一个变量。
    """
    import numpy as np
    if mode == "none":
        return None
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    counts = np.clip(counts, 1e-9, None)
    w = counts.sum() / (n_classes * counts)
    if mode == "sqrt":
        w = np.sqrt(w)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float)


def build_loss_fn(args, weight, device):
    """返回 (criterion, 描述字符串)。Focal 与 Label Smoothing 不叠加——都在改目标分布"""
    weight = weight.to(device) if weight is not None else None
    if args.focal_gamma > 0:
        crit = lambda logits, y: focal_loss(logits, y, weight, args.focal_gamma)  # noqa: E731
        desc = f"Focal(gamma={args.focal_gamma:g})"
    else:
        crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
        desc = "CE" + (f"+LS({args.label_smoothing:g})" if args.label_smoothing > 0 else "")
    return crit, desc
