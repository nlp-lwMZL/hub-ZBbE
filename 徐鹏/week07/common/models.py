# -*- coding: utf-8 -*-
"""
判别式链路的**公共模型定义**（被多方法共用，放这里避免各方法目录互相 import）

为什么单独一个文件：
  重构后每个方法一个文件夹，`bert/train.py`、`bilstm_crf/train.py`、`bert_pipeline/train.py`
  都叫 `train.py` —— 一旦把这些目录同时放进 sys.path，`import train` 就会撞名、
  拿到别的方法那个文件。而 service 既要 BERT 编码器（bert）、又要两段式的分类头（bert_pipeline），
  天然需要同时用到两边。
  所以把「与训练流程无关、纯粹是模型结构/推理还原」的部分集中到这里：
    - BertEncoder      BERT 编码器（可选只保留前 N 层）
    - build_model      BERT + (Linear | CRF) 工厂
    - BertCls          BERT + [CLS] 分类头（事件检测器 / 事件类型 13 分类）
    - build_arg_model  类型条件论元抽取模型（BERT 前 N 层 + 线性头）
    - predict_char_tags 子词级预测 → 字符级标签序列（含"只取首个子词"的关键细节）
  函数签名与实现**与重构前完全一致**，只是换了存放位置。

使用方式（各方法脚本在 sys.path 里加 common/ 后）：
  from models import BertEncoder, build_model, predict_char_tags
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows: torch 与 numpy 各带 libiomp5md

import torch
import torch.nn as nn
from transformers import AutoModel, BertModel

from crf import CRFClassifier
from event_data import build_bio_constraint_mask, build_start_constraint_mask


class BertEncoder(nn.Module):
    """
    只保留 BERT 的编码部分，输出 (B, L, H)；后面接 Linear 还是 CRF 由 CRFClassifier 决定。

    为什么要包一层：CRFClassifier 要求 encoder 是一个能「输入 ids、输出 hidden」的模块。
    把 BERT 和 BiLSTM 都适配成这个接口，两条链路就能共用同一个 CRF / 同一套解码 / 同一套统计。
    """

    def __init__(self, bert_path: str, num_hidden_layers: int = 0):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        if num_hidden_layers and num_hidden_layers > 0:
            # 消融 / 省磁盘用：**保留预训练权重，只截取前 N 层**。
            # 注意不要写成 BertConfig(num_hidden_layers=4) + BertModel(cfg) —— 那样是随机初始化 4 层，
            # 把预训练知识全丢了，结果会跟「层数」这个变量混在一起，做不出干净的消融。
            # 这里是把 pretrained 的 encoder.layer 截前 N 层，embedding 和第 N 层之后的 pooler 不用动。
            self.bert.encoder.layer = nn.ModuleList(
                list(self.bert.encoder.layer)[:num_hidden_layers])
            self.bert.config.num_hidden_layers = num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # transformers 5.x：BertModel 默认返回 tuple，必须显式 return_dict=True
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids, return_dict=True)
        return out.last_hidden_state   # (B, L, H)


def batch_to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def build_model(args, label_scheme: dict) -> nn.Module:
    """BERT + (Linear | CRF)。"""
    encoder = BertEncoder(str(args.bert_path), args.bert_layers)
    constraint = start_c = None
    if args.hard_constraint:
        # 硬约束：把非法转移 / 非法开头的分数直接置 -1e4，让模型不可能解码出非法序列。
        # 不加约束时 CRF 只是「倾向于」不非法（学出来的软约束），实测仍会有少量非法转移。
        constraint = build_bio_constraint_mask(label_scheme)
        start_c = build_start_constraint_mask(label_scheme)
    model = CRFClassifier(
        encoder=encoder,
        hidden_size=encoder.hidden_size,
        num_tags=len(label_scheme["labels"]),
        dropout=args.dropout,
        use_crf=(args.decoder == "crf"),
        constraint=constraint,
        start_constraint=start_c,
    )
    return model


def predict_char_tags(model, loader, id2label: dict, device) -> tuple[list[list[str]], dict]:
    """
    在 loader 上推理，把「子词级」预测还原成「字符级」标签序列（每条样本一个 list[str]）。

    还原逻辑：用 batch 里存下来的 word_ids，把每个 token 的预测标签写回它对应的汉字位置。
    -1（[CLS]/[SEP]/[PAD]/被截断的 token）直接跳过。
    被 max_length 截断、没被任何 token 覆盖的尾部字符，保持 "O"（会丢 span，这是截断的真实代价）。

    ⚠ 关键细节 1：**一个"字"可能对应不止一个 token**（顺序是：整句 tokenize 时的数字/英文
      会被 WordPiece 切成 "005"+"##10"；换 Qwen 这类词表时中文又会多字合成一个 token）。
      训练时只有**首个子词**背标签、其余是 -100，所以还原时也只该取首个子词。
      如果让后面的子词覆盖前面的，取到的是训练时从不背标签的位置，预测是噪声。
      （本数据实测：汉字 9,219 个、非首子词 0 个、## 子词 0 个（逐字输入），所以这个 bug
       在本数据上影响为 0；但换成整句分词或别的 tokenizer 就会中招，属于必须写对的地方。）

    ⚠ 关键细节 2（这才是"硬约束后字符级仍报非法"的真因）：
      **空白字符不产生 token**。例如 "格隆汇 9 月 8日" 里 len(text)=116 但只有 110 个 token。
      这些没有 token 的字符位置被强制标成 "O"，于是跨空格的 span 在**字符级**BIO 上
      变成 "O → I-x"（非法），但**token 级序列是完全合法的**。
      实测：硬约束后 token 级非法 = 0；字符级 155 处里有 100% 的前一字符是空白。
      所以报告非法数时必须写清口径（token 级 or 字符级），否则会得出相反结论。
    """
    model.eval()
    all_tags: list[list[str]] = []
    info = {"n_pred_tokens": 0}

    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            # 训练/评测阶段**不要**路径分数：CRF 算路径分数要对每条序列额外跑一遍前向算法
            # （甚至在 decode 里是逐样本 Python 循环），实测能让 dev 推理慢 3~5 倍。
            # 只有 service 需要「置信度 → needs_review」时才要分数。
            paths = model.predict(batch["input_ids"], batch["attention_mask"],
                                  batch.get("token_type_ids"))
            word_ids = batch["word_ids"].cpu().tolist()
            for b in range(len(paths)):
                # 这条样本覆盖了多少个汉字 = 最大的 word_id + 1（-1 是 [CLS]/[SEP]/[PAD]）
                max_wid = max(word_ids[b]) if word_ids[b] else -1
                n_chars = max_wid + 1 if max_wid >= 0 else 0
                tags = ["O"] * n_chars
                prev_wid = None
                for j, wid in enumerate(word_ids[b]):
                    if wid < 0 or wid >= n_chars:
                        prev_wid = wid
                        continue
                    # 只认这个汉字的**首个子词**（与训练时的标签对齐规则一致）
                    if wid != prev_wid and j < len(paths[b]):
                        tags[wid] = id2label.get(paths[b][j], "O")
                    prev_wid = wid
                all_tags.append(tags)
                info["n_pred_tokens"] += len(paths[b])
    return all_tags, info


class BertCls(nn.Module):
    """
    BERT + [CLS] 线性分类头。同一个类服务两件事：
      检测器（2 分类）与类型分类（13 分类），只改 num_labels。

    用 [CLS] 而不是 mean-pooling：BERT 的 [CLS] 在预训练时就被训练成「整句语义」的汇聚位，
    微调时只调它上面的线性层收敛最快；对短句子（本数据平均 114 字）两者差别很小。
    """

    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids, return_dict=True)
        cls = out.last_hidden_state[:, 0]          # [CLS] 向量
        logits = self.classifier(self.dropout(cls))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels, ignore_index=-100)
        return logits, loss


def build_arg_model(bert_path: str, scheme: dict, args):
    """
    类型条件论元模型：BERT(前 N 层) + 线性头（标签空间只有十几类，不需要 CRF）。

    为什么默认只留 4 层（工程取舍，PPT 上会讲）：
      13 个类型专家 × 12 层 BERT = 13 × 409MB ≈ 5.3GB 权重，训练还要 3 倍时间；
      截到 4 层（保留预训练权重）后 13 个模型只占 ~1.8GB，速度约 3 倍，
      而 (c) 段的任务很简单（标签空间十几类、数据是同一类型的句子），层数收益本来就小。
    """
    from crf import CRFClassifier

    encoder = BertEncoder(bert_path, getattr(args, "arg_layers", 4))
    return CRFClassifier(encoder=encoder, hidden_size=encoder.hidden_size,
                         num_tags=len(scheme["labels"]), dropout=args.dropout,
                         use_crf=False)


class _Layers:
    """把 checkpoint 里记录的 arg_layers 包成 build_arg_model 需要的参数对象。"""

    def __init__(self, n: int, dropout: float = 0.1):
        self.arg_layers = n
        self.dropout = dropout


def _layers_of(ckpt: dict, dropout: float = 0.1) -> _Layers:
    """从 checkpoint 的 args 里读出该专家用了多少层（保证推理与训练结构一致）。"""
    return _Layers(int((ckpt.get("args") or {}).get("arg_layers", 4)), dropout)

