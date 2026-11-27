# bert/ —— L2 BERT 序列标注（一次联合抽取）

**一句话**：`bert-base-chinese` + 序列标注，**一次前向同时抽「事件类型 + 触发词 + 论元」**。
`--decoder linear|crf` 一套脚本跑两种解码器，这是本项目的主线方法。

```bash
python bert/train.py --decoder linear --max_len 256 --epochs 3 --run_name bert_linear_256
python bert/train.py --decoder crf    --max_len 256 --epochs 3 --run_name bert_crf_256
# 消融
python bert/train.py --decoder linear --max_len 128 --epochs 3 --run_name bert_linear_128
python bert/train.py --decoder linear --max_len 256 --bert_layers 4 --epochs 3 --run_name bert4L_linear_256
# 冒烟（CPU / 手造数据）
python bert/train.py --mock --device cpu --epochs 1 --bert_layers 4 --max_len 64
# 从 checkpoint 复评（不重训）
python bert/evaluate.py --ckpt outputs/checkpoints/bert_crf_256.pt --run_name bert_crf_256
# 解码期硬约束（复用已训权重，只换解码）
python bert/evaluate_hard_constraint.py
```

| 文件 / 产物 | 说明 |
|---|---|
| `train.py` | 训练：joint 149 标签、子词对齐、动态 padding、warmup、分层学习率 |
| `evaluate.py` | 从 checkpoint 复评 dev（`_eval.json` + console 日志） |
| `evaluate_hard_constraint.py` | CRF 硬约束解码对比（非法 token 级归零） |
| `outputs/checkpoints/bert_*.pt` | checkpoint |
| `outputs/logs/bert_*_train.json` / `_steps.jsonl` / `_console.log` | 训练日志三类 |

**教学重点**
1. **子词对齐（`word_ids()`）**：WordPiece 可能把一个汉字切成多片，只有**首个子词**背标签、
   其余与 `[CLS]/[SEP]/[PAD]` 设 **-100**；否则同一个 span 的 loss 会被重复计算。
   ⚠ 还原预测时也**只能取首个子词**（取末子词等于读噪声）——踩坑记录见 NOTES。
2. **joint 标签体系 = 149 个标签**：触发词按 13 类拆开 + 论元 61 个角色跨类型共享（见 `common/event_data.py`）。
3. **模型选择用内层验证集**：从 train 切 10% 早停，**dev 930 句完全不参与训练与选点**。
4. **CRF vs Linear**：论元 +1.9 点，训练慢 5.9×、推理慢 3.3×；再加**解码期硬约束**，论元再 +7.5 点。
