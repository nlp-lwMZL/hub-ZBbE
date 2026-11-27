# week7 NER 三类方法对比（split=test）

> 口径：所有数字由 `common/metrics.evaluate` 在 cdtier/clean 同一份数据上计算；实体级 P/R/F1（三元组完全匹配）。未跑的方法显式标「未跑」。

| 层级 | 方法 | run_name | N | P | R | F1 | type_macro_F1 |
|---|---|---|---|---|---|---|---|
| L1 BiLSTM | BiLSTM+crf | `bilstm_crf` | 387 |  0.2846 |  0.1620 |  0.2065 |  0.1719 |
| L1 BiLSTM | BiLSTM+linear | `bilstm_linear` | 387 |  0.2062 |  0.1551 |  0.1770 |  0.1486 |
| L4 Qwen3-SFT | Qwen3-0.6B full SFT | `qwen_full_sft` | 387 |  0.7650 |  0.6181 |  0.6837 |  0.6528 |
| L4 Qwen3-SFT | Qwen3-0.6B LoRA SFT | `qwen_lora_sft` | 387 |  0.7160 |  0.5370 |  0.6138 |  0.5627 |
| L2 RoBERTa | RoBERTa+crf | `roberta_crf` | 387 |  0.4496 |  0.5370 |  0.4895 |  0.4043 |
| L2 RoBERTa | RoBERTa+linear | `roberta_linear` | 387 |  0.4461 |  0.4977 |  0.4705 |  0.3873 |

## SFT 臂工程指标（生成式独有成本）

| run | JSON合法 | 修复 | 类型幻觉 | 文本未找到 | 平均延迟(s/条) |
|---|---|---|---|---|---|
| `qwen_full_sft` | 387 | 1 | 1 | 1 | 0.42 |
| `qwen_lora_sft` | 383 | 0 | 0 | 0 | 0.633 |

## 读表提醒

- **L1 vs L2** 的差 = 预训练值多少点（同标签、同 CRF、同评测，唯一差别是编码器）；
- **L2 vs L4** 的差 = 判别式 vs 生成式微调（SFT 召回强、边界可能糙，看 type_macro 辅助判断）；
- loss 绝对值不能跨方法比（CE vs CRF NLL vs LM loss 量纲不同），只看实体级 F1；
- SFT 的 type_macro 若异常低，检查是否有 schema 外的幻觉类型（工程指标里的「类型幻觉」）。
