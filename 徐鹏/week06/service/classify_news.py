"""
分流服务：把单点模型拼成可被业务调用的接口（L5 系统组合）

教学重点：
  1. 真实系统从来不是「一个模型」，而是四段串起来：
       规则快筛 → 模型分类 → 置信度闸门 → 队列路由
     每一层只负责自己最擅长的那一段。
  2. 接口契约比准确率更重要：下游要的是 {topic, relevant, conf, queue}，
     有了 conf 才能做「低置信转人工」；只有 label 的话下游没法决策。
  3. 降级要显式：BERT 权重不存在时退回规则法并打标记，
     工作台不会崩，但调用方能从 model 字段看出「这次是降级结果」。

被谁调用：
  第 9 周会把这个函数包成 Agent 的第一个工具（Function Calling / MCP / CLI 三接法）。

用法：
  python service/classify_news.py --text "贵州茅台：2023 年年度业绩预告"
  python service/classify_news.py --file inputs.txt      # 每行一条
"""

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common.labels import LABELS, is_relevant, queue_name, to_group

MODEL_NAME = "bert-cls-v1"
MODEL_VERSION = "2026-11"

# 规则快筛：公告标题有很强的固定套路，一条正则就能挡掉大量调用。
# 命中即直接给出结果，不再过模型 —— 这是生产里最常见的「省钱」手段。
FAST_RULES = [
    ("业绩预告", "财经"), ("年度报告", "财经"), ("季度报告", "财经"),
    ("回购", "证券"), ("减持", "证券"), ("增持", "证券"), ("涨停", "证券"),
]


class ShuntService:
    """分流服务：加载一次模型，之后每次调用复用"""

    def __init__(self, ckpt_path=None, conf_threshold=0.8):
        self.conf_threshold = conf_threshold
        self.ckpt_path = Path(ckpt_path) if ckpt_path else CODE_ROOT / "outputs" / "bert_ckpt.pt"
        self.model, self.tok = None, None
        self._try_load()

    def _try_load(self):
        """加载 BERT；权重缺失时**显式降级**，而不是静默出错"""
        if not self.ckpt_path.exists():
            print(f"[warn] 未找到 {self.ckpt_path}，降级为规则法（先跑 bert/train_bert.py）", flush=True)
            return
        try:
            import torch
            from transformers import AutoTokenizer
            sys.path.insert(0, str(CODE_ROOT / "bert"))
            from train_bert import BertClassifier
            from common.model_paths import bert_path

            ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            self.tok = AutoTokenizer.from_pretrained(str(bert_path()))
            self.model = BertClassifier(bert_path(), num_labels=len(LABELS), pool=ckpt.get("pool", "cls"))
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
        except Exception as e:                      # 模型坏了也不能让服务整体挂掉
            print(f"[warn] BERT 加载失败（{e}），降级为规则法", flush=True)
            self.model = None

    def _rule_hit(self, text):
        for kw, label in FAST_RULES:
            if kw in text:
                return LABELS.index(label)
        return None

    def _model_predict(self, text):
        import torch
        enc = self.tok(text, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            logits = self.model(enc["input_ids"], enc["attention_mask"])
        prob = torch.softmax(logits.float(), -1)[0]
        idx = int(prob.argmax())
        return idx, float(prob[idx])

    def classify(self, title: str) -> dict:
        """
        返回字段说明（下游按这个契约消费）：
          topic     4 组主题，给人看的
          sub_topic 15 类细标签，给统计用的
          relevant  是否与投研相关
          conf      置信度；< threshold 时 queue=review
          queue     research / review / drop
          model     用了哪条路径（rule / bert / fallback）
        """
        label_idx = self._rule_hit(title)
        if label_idx is not None:
            conf, model_used = 0.99, "rule"        # 规则命中的置信度是人为设定的，不是模型输出
        elif self.model is not None:
            label_idx, conf = self._model_predict(title)
            model_used = "bert"
        else:
            label_idx, conf, model_used = 0, 0.0, "fallback"   # 什么都没有 → 全部进 review

        return {
            "title": title,
            "topic": to_group(label_idx),
            "sub_topic": LABELS[label_idx],
            "relevant": is_relevant(label_idx),
            "conf": round(conf, 4),
            "queue": queue_name(label_idx, conf, self.conf_threshold),
            "model": model_used,
            "version": MODEL_VERSION,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="单条标题")
    ap.add_argument("--file", help="每行一条标题的文件")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--conf_threshold", type=float, default=0.8)
    args = ap.parse_args()

    svc = ShuntService(args.ckpt, args.conf_threshold)
    texts = [args.text] if args.text else Path(args.file).read_text(encoding="utf-8").splitlines()
    for t in [x for x in texts if x.strip()]:
        print(json.dumps(svc.classify(t.strip()), ensure_ascii=False))


if __name__ == "__main__":
    main()
