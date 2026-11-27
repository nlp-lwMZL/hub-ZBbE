"""
BertNER（线性头）和 BertCRFNER（CRF 头）两个模型

与 ../src/model.py 完全同一套结构，只是数据集换成人民日报 NER（7 个标签）：
  num_labels 由 dataset.build_label_schema() 决定，模型本身不写死类别数。

教学重点：
  1. 线性头（BertNER）：每个 token 独立 softmax
     - 忽略标签间的依赖关系，可能产生非法序列：I-PER 开头、B-ORG 后接 I-LOC
  2. CRF 头（BertCRFNER）：emissions + 转移矩阵，Viterbi 求全局最优路径
     - 转移矩阵是从数据里**学出来的**，不是硬编码约束
     - 所以"合法性"取决于训练是否充分：转移矩阵没收敛时，照样会输出非法序列
     - 想从数学上彻底杜绝，需要给非法转移位置加惩罚（如 -1e4），本实现没有加，
       这样两种头的差异才是"纯模型结构"的差异，可以直接用非法序列数对比
  3. 两者差异的量化：evaluate.py 统计非法序列数 + seqeval entity-level F1

依赖：
  pip install pytorch-crf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertModel


def _load_bert(bert_path: str) -> BertModel:
    """加载 BERT，临时把日志级别调到 error，避免每个脚本都刷一屏 loading 报告。"""
    prev = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    bert = BertModel.from_pretrained(bert_path)
    transformers.logging.set_verbosity(prev)
    return bert


class BertNER(nn.Module):
    """BERT + 线性分类头，逐 token 独立预测 BIO 标签。

    input_ids → BertModel → (B, L, 768)
              → Dropout → Linear(768, num_labels) → logits (B, L, num_labels)
    损失：CrossEntropy，ignore_index=-100 跳过特殊 token / 非首子词 / PAD
    预测：argmax(logits, -1)
    """

    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = _load_bert(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        logits = self.classifier(self.dropout(outputs.last_hidden_state))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss


class BertCRFNER(nn.Module):
    """BERT + CRF 层，全局最优序列解码。

    与 BertNER 的区别：
      - Linear 输出叫 emissions（发射分数），不直接 argmax
      - CRF 在 emissions 上叠加 (num_labels × num_labels) 转移矩阵，用 Viterbi 找全局最优
      - 损失：负对数似然（CRF 内部前向-后向算法）
      - 解码：crf.decode() 返回每条序列的实际 token 数（不含 PAD）
    """

    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        from torchcrf import CRF

        self.bert = _load_bert(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

    def _get_emissions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        return self.classifier(self.dropout(outputs.last_hidden_state))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()  # CRF 要求 BoolTensor

        loss = None
        if labels is not None:
            # CRF 不支持 ignore_index：把 -100 换成 0，这些位置已被 mask 屏蔽，不影响梯度
            labels_crf = labels.clone()
            labels_crf[labels_crf == -100] = 0
            loss = -self.crf(emissions, labels_crf, mask=mask, reduction="mean")

        return emissions, loss

    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> list[list[int]]:
        """Viterbi 解码，返回 list[list[int]]，长度等于实际 token 数（不含 PAD）。"""
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        return self.crf.decode(emissions, mask=attention_mask.bool())


def build_model(
    use_crf: bool,
    bert_path: str,
    num_labels: int,
    dropout: float = 0.1,
) -> nn.Module:
    """模型工厂函数。"""
    model_cls = BertCRFNER if use_crf else BertNER
    model = model_cls(bert_path=bert_path, num_labels=num_labels, dropout=dropout)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型：{'BERT + CRF' if use_crf else 'BERT + Linear'}")
    print(f"  标签数：{num_labels}")
    print(f"  参数总量：{total_params / 1e6:.1f}M")
    print(f"  可训练参数：{trainable_params / 1e6:.1f}M")
    return model
