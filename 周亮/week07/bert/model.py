"""
L2 判别式主线：RoBERTa-wwm-ext + Linear / CRF 序列标注（NER）。

迁移自 homework/week7/ner_model.py，关键改造：
  1. 预训练底座从 bert-base-chinese 改为 hfl/chinese-roberta-wwm-ext（可 --bert_path 覆盖）；
  2. CRF mask 用连续前缀 attention_mask.bool()（不能挖洞，否则维特比回溯错位——
     这是现有代码已踩过并修复的坑，逻辑原样保留）；
  3. loss 用 token_mean，与 CE 梯度尺度对齐，两类解码器共用同一套 lr / clip；
  4. 数据/标签/评测全部走 common/，与 BiLSTM / SFT 臂同口径。

两个模型共用主干，只在「输出头 / 损失 / 解码」处分叉：
    input_ids / attention_mask [+ token_type_ids]
      -> AutoModel(RoBERTa) -> Dropout -> Linear(hidden, num_labels)
         -> ① BertLinearNer: CrossEntropy(ignore_index=-100) + argmax
         ② BertCrfNer    : CRF NLL(logZ - 金标路径) + Viterbi decode

【batch 契约】model(**batch) 接受 input_ids, attention_mask, labels，
token_type_ids 可选（RoBERTa 不需要）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
from common.ner_data import IGNORE_INDEX
from common.model_paths import resolve_roberta

DEFAULT_TOKENIZER = "hfl/chinese-roberta-wwm-ext"


class BertNerBase(nn.Module):
    """RoBERTa + Dropout + token 级分类头，两个模型只共用这一段。"""

    def __init__(self, bert_path: str | Path, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(str(bert_path))
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.num_labels = num_labels

    def emissions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        kwargs: dict[str, torch.Tensor] = {"input_ids": input_ids, "attention_mask": attention_mask,
                                           "return_dict": True}
        # RoBERTa 无 token_type_ids；只在传入且模型支持时才带上
        if token_type_ids is not None and "token_type_ids" in self.bert.forward.__code__.co_varnames:
            kwargs["token_type_ids"] = token_type_ids
        out = self.bert(**kwargs)
        return self.classifier(self.dropout(out.last_hidden_state))  # (B, L, K)

    def parameter_groups(self, bert_lr: float, head_lr: float) -> list[dict]:
        bert_params = list(self.bert.parameters())
        bert_ids = {id(p) for p in bert_params}
        head_params = [p for p in self.parameters() if id(p) not in bert_ids]
        return [{"params": bert_params, "lr": bert_lr}, {"params": head_params, "lr": head_lr}]


class BertLinearNer(BertNerBase):
    """逐位置独立分类基线：快，但会喷非法 BIO 序列。"""

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        logits = self.emissions(input_ids, attention_mask, token_type_ids)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1),
                                   ignore_index=IGNORE_INDEX)
        return logits, loss

    def decode(self, input_ids, attention_mask, token_type_ids=None, labels=None) -> torch.Tensor:
        logits, _ = self.forward(input_ids, attention_mask, token_type_ids, labels=None)
        pred = logits.argmax(dim=-1)
        valid = labels.ne(IGNORE_INDEX) if labels is not None else attention_mask.bool()
        return pred.masked_fill(~valid, IGNORE_INDEX)


class BertCrfNer(BertNerBase):
    """在 emissions 上叠加可学习转移矩阵，联合建模整条标签序列。"""

    def __init__(self, bert_path, num_labels, dropout=0.1) -> None:
        super().__init__(bert_path, num_labels, dropout)
        try:
            from torchcrf import CRF
        except ImportError as exc:
            raise ImportError("BERT+CRF 需要 pytorch-crf：pip install pytorch-crf") from exc
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        emissions = self.emissions(input_ids, attention_mask, token_type_ids)
        loss = None
        if labels is not None:
            mask = attention_mask.bool()          # 连续前缀，不能用 labels != -100（中间会挖洞）
            tags = labels.clone()
            tags[tags == IGNORE_INDEX] = 0         # torchcrf 要求 0..K-1 的合法 id
            loss = -self.crf(emissions, tags, mask=mask, reduction="token_mean")
        return emissions, loss

    def decode(self, input_ids, attention_mask, token_type_ids=None, labels=None) -> torch.Tensor:
        emissions = self.emissions(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()
        paths = self.crf.decode(emissions, mask=mask)   # list[list[int]]，长度 = mask.sum(i)
        B, L = emissions.shape[:2]
        out = torch.full((B, L), IGNORE_INDEX, dtype=torch.long, device=emissions.device)
        for i, path in enumerate(paths):
            idx = mask[i].nonzero(as_tuple=False).flatten()
            assert len(path) == idx.numel(), "CRF 回填长度与 mask 不一致，检查 mask 口径"
            out[i, idx] = torch.tensor(path, dtype=torch.long, device=emissions.device)
        if labels is not None:
            out = out.masked_fill(labels.eq(IGNORE_INDEX), IGNORE_INDEX)
        return out


def build_model(use_crf: bool, bert_path: str | Path, num_labels: int,
                dropout: float = 0.1):
    cls = BertCrfNer if use_crf else BertLinearNer
    model = cls(bert_path=bert_path, num_labels=num_labels, dropout=dropout)
    total = sum(p.numel() for p in model.parameters())
    groups = model.parameter_groups(1.0, 1.0)
    head = sum(p.numel() for p in groups[1]["params"])
    print(f"模型：{'RoBERTa + CRF' if use_crf else 'RoBERTa + Linear'} | "
          f"标签 {num_labels} | 参数 {total / 1e6:.1f}M | head {head / 1e3:.1f}K")
    return model


def resolve_bert_path(arg: str | None) -> str:
    """--bert_path 给了就用它；否则解析本地 RoBERTa，没下过就用 hub id。"""
    return arg if arg else str(resolve_roberta())


# --------------------------------------------------------------------------- #
# 自检（本机用 bert-base-chinese 覆盖 --bert_path）
# --------------------------------------------------------------------------- #
def _self_test(bert_path: str, use_crf: bool) -> None:
    torch.manual_seed(7)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(bert_path)
    enc = tok(list("马云创立了阿里巴巴"), is_split_into_words=True, max_length=16,
              truncation=True, padding="max_length", return_tensors="pt")
    labels = torch.full_like(enc["input_ids"], IGNORE_INDEX)
    tag_by_char = [1, 2, 0, 0, 0, 3, 4, 4, 4]
    for pos, w in enumerate(enc.word_ids(0)):
        if w is not None and w < len(tag_by_char):
            labels[0, pos] = tag_by_char[w]
    batch = {k: v.repeat(2, 1) for k, v in
             {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
              "labels": labels}.items()}
    model = build_model(use_crf, bert_path, num_labels=7)
    model.train()
    _, loss = model(**batch)
    assert loss is not None and torch.isfinite(loss), "loss 非有限"
    loss.backward()
    assert next(model.bert.parameters()).grad is not None, "梯度未回传底座"
    model.eval()
    with torch.no_grad():
        pred = model.decode(**batch)
    assert (pred[batch["labels"] == IGNORE_INDEX] == IGNORE_INDEX).all()
    print(f"  自检 {'CRF' if use_crf else 'Linear'}: loss={loss.item():.4f}, "
          f"decode 形状 {tuple(pred.shape)} ✓")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="bert/model.py 自检")
    ap.add_argument("--bert_path", required=True, help="本地 tokenizer/模型目录")
    ap.add_argument("--use_crf", action="store_true")
    a = ap.parse_args()
    _self_test(a.bert_path, a.use_crf)
