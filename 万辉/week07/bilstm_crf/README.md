# bilstm_crf/ —— L1 字符级 BiLSTM-CRF（无预训练）

**一句话**：字向量 128 → BiLSTM(hidden 256×2) → Linear → **手写 CRF**，从零训练，参数量约 **1.31M**。
它是 BERT 的**同任务对照**：同样的标签体系、同一个 CRF、同一套解码与评测，**唯一差别是编码器**，
所以「BERT 比它高多少」可以干净地归因到「预训练」上。

```bash
# GPU（有卡时；约 3 分钟内 3 epoch）
python bilstm_crf/train.py --decoder crf --epochs 3 --max_len 128 --run_name bilstm_crf
# 纯 CPU（本项目的做法：GPU 让给 LLM 链路，CPU 上 3 epoch 约 26 分钟）
python bilstm_crf/train.py --device cpu --decoder crf --epochs 3 --max_len 128 --run_name bilstm_crf
python bilstm_crf/train.py --mock --device cpu --epochs 1 --max_len 96   # 冒烟
```

| 产物 | 说明 |
|---|---|
| `outputs/checkpoints/bilstm_crf.pt` | 最优 checkpoint（含字表 vocab） |
| `outputs/metrics/disc/bilstm_crf.json` | dev 指标 + 非法 BIO 统计 |
| `outputs/logs/bilstm_crf_train.json` | **训练日志**：args/环境/每 epoch loss+验证+耗时/参数量/峰值显存 |
| `outputs/logs/bilstm_crf_steps.jsonl` | **每 50 步一行**：epoch/step/loss/lr/elapsed（画 loss 曲线用） |
| `outputs/logs/bilstm_crf_console.log` | 完整控制台输出 |

**教学重点**
1. 没有 tokenizer / 子词 —— 一个汉字查一个字表，所以**不需要 word_ids 对齐**（简单，但没有词表先验）。
2. LSTM 要求定长张量 → `collate_char` 补到 batch 内最长并给 mask。
3. CRF 与 `bert/train.py` **共用 `common/crf.py`**，保证两条链路的解码/统计完全同源。
4. 实测结论：**触发词几乎追平 BERT（差 1.5 个点），但论元差 11.8 个点**
   → 预训练的收益主要在「理解边界与角色」，不在「认出关键词」。
