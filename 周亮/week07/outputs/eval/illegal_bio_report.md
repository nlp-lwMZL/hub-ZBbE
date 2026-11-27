# 非法 BIO 转移统计（Linear vs CRF）

> split = `test`，387 句；生成于 2026-11-25T12:37:33

## 1. 口径（为什么这件事容易做错）

非法 BIO 指三类：**I-x 出现在序列开头**、**I-x 紧接 O**、**类型串味**（I-x 接 B-y/I-y）。
老师 PPT P25 的核心结论是：Linear 逐字独立预测会喷出大量非法转移，CRF 用转移矩阵
学到标签间约束后非法显著下降。**必须统计模型 decode 的原始标签**，不能用兜底后的 span 反推——
`decode_bio_spans` 会把 `O→I-x` 当成新实体起点，反推后非法恒为 0，是假数据。

## 2. 金标地板值（老师要求必须一起报）

- cdtier gold 重叠实体对 = 0，gold 投影 BIO 非法转移 = **0 处**
- 老师事件抽取因重叠论元投影到 BIO，金标自带约 1.9% 非法；本作业是平铺 NER，地板 = 0。
- 意义：报告的非法数全部来自模型，不含 Gold 自身的投影噪声，归因干净。

## 3. 本作业实测

| 方法 | 原始BIO非法转移 | 含非法句数 | 每句处数 | span重叠对 | 越界 | 类型幻觉 |
|---|---:|---:|---:|---:|---:|---:|
| L1 BiLSTM+Linear | — | — | — | 0 | 0 | 0 |
| L1 BiLSTM+CRF | — | — | — | 0 | 0 | 0 |
| L2 RoBERTa+Linear | — | — | — | 0 | 0 | 0 |
| L2 RoBERTa+CRF | — | — | — | 0 | 0 | 0 |
| L4 Qwen3-LoRA-SFT | — | — | — | 84 | 0 | 0 |
| L4 Qwen3-full-SFT | — | — | — | 151 | 0 | 0 |

> 原始BIO非法转移列若为 —，说明现有 preds 只存了 span（兜底后），无法反推。span 重叠对是 BIO 非法在 span 层的等价表征（同一字符被两个实体覆盖，单标签 BIO 不可能）。

## 4. 另一个可直接观测的对比：span 重叠（无需原始 BIO）

判别式四臂的 pred_spans 重叠对数 = **0**（BIO 单标签体系 + 解码兜底天然保证一个字符只属于一个实体）；而 SFT 两臂分别出现重叠——这是**生成式没有 BIO 约束**的直接证据，等价于老师的「非法 BIO」在 span 层的表征。

| 范式 | 方法 | span 重叠对 | 被重复覆盖字符 |
|---|---|---:|---:|
| 判别式 | L1 BiLSTM+Linear | 0 | 0 |
| 判别式 | L1 BiLSTM+CRF | 0 | 0 |
| 判别式 | L2 RoBERTa+Linear | 0 | 0 |
| 判别式 | L2 RoBERTa+CRF | 0 | 0 |
| 生成式 | L4 Qwen3-LoRA-SFT | 84 | 236 |
| 生成式 | L4 Qwen3-full-SFT | 151 | 294 |

这个对比不需要原始 BIO，从现有 preds 即可复现——面试时可作为「判别式 vs 生成式」
结构合法性差异的实证。注意 gold 无重叠，所以生成式的重叠全是模型过度抽取导致。

## 5. Linear vs CRF 对比

- **L1 BiLSTM: Linear → CRF**：需补采原始 BIO（见文末命令）
- **L2 RoBERTa: Linear → CRF**：需补采原始 BIO（见文末命令）

老师基准（老师 PPT P25 / outputs/metrics/disc/bert_linear_256.json + bert_crf_256.json）：Linear 1516 处 / 623 句 → CRF 1242 处 / 581 句，**−18.1%**。
注意老师特别提醒：选「含非法句占比」会看反效果（0.67→0.625 只降 4.5pp），因为长序列上任一 span 开头吐 I- 就整句非法；
看**处数**和**每句处数**才有区分度。

## 6. 补采精确统计（不重训，只重跑评测）

```bash
python bert/evaluate.py --run_name roberta_linear --split test --dump_raw_tags
python bert/evaluate.py --run_name roberta_crf    --split test --dump_raw_tags
python bilstm_crf/evaluate.py --run_name bilstm_linear --split test --dump_raw_tags
python bilstm_crf/evaluate.py --run_name bilstm_crf    --split test --dump_raw_tags
python eval/illegal_bio.py --split test
```

重跑后本脚本自动检测 `*_rawtags.jsonl`，第 3/4 节的精确列会填满。
