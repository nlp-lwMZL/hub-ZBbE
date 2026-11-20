# outputs/results —— 分类方案评测结果汇总

所有结果都来自 **同一批 200 条验证样本**（`val.json` 随机采样，`seed=42`），
样本相同、顺序相同，所以任何两份结果都能逐条配对比较（McNemar 精确检验）。

一共两组对比：

- **四方法对比**：zero-shot / few-shot / SFT(LoRA) / BERT —— 生成式 vs 判别式
- **五模型对比**：BERT 的 3 种池化 × 2 种 loss —— 同一族内的超参对比

## 一、执行结果（每份都是 200 条逐条明细）

| 文件 | 方法 | 类型 | 做法 | 生成脚本 |
|---|---|---|---|---|
| `01_zeroshot_qwen2-0.5b_result.json` | zero-shot | 生成式 | 基础模型直接问，不给示例 | `src_llm/classify_llm.py` |
| `02_fewshot_qwen2-0.5b_result.json` | few-shot | 生成式 | prompt 里塞 30 组示例（15 类 × 2 例） | `src_llm/classify_llm_fewshot.py` |
| `03_sft_lora_qwen2-0.5b_result.json` | SFT (LoRA) | 生成式 | Qwen2-0.5B 在 5000 条上 LoRA 微调 | `src_llm/evaluate_sft.py` |
| `04_bert_cls_unweighted_result.json` | BERT cls 不加权 | 判别式 | bert-base-chinese，全量 53K 条 | `src/eval_bert_for_compare.py` |
| `05_bert_cls_weighted_result.json` | BERT cls 加权 | 判别式 | 同上 + 加权 loss | 同上 |
| `06_bert_max_unweighted_result.json` | BERT max 不加权 | 判别式 | 同上 + max 池化 | 同上 |
| `07_bert_mean_unweighted_result.json` | BERT mean 不加权 | 判别式 | 同上 + mean 池化 | 同上 |
| `08_bert_mean_weighted_result.json` | BERT mean 加权 | 判别式 | 同上 + mean 池化 + 加权 loss | 同上 |

JSON 结构（`01`~`03` 一致，`04`~`08` 多几个字段）：

```jsonc
{
  "accuracy": 0.595,        // 准确率
  "total": 200,             // 样本数
  "correct": 119,           // 答对条数
  "unparseable": 0,         // 模型答了 15 个类别以外的词，算错（BERT 恒为 0）
  "results": [
    {
      "text": "...",        // 新闻标题
      "true_label": "农业",  // 真实类别
      "pred_label": "房产",  // 解析出的类别，null = 无法解析
      "raw_output": "房产",  // 模型原始输出
      "correct": false
    }
    // ... 共 200 条
  ]
}
```

额外字段：

- `02_`：`n_shot` / `example_seed` / `seed` / `prompt_len`
- `04_`~`08_`：`pool` / `use_class_weight` / `ckpt` / `ckpt_epoch` /
  `ckpt_val_acc_full` / `ckpt_val_macro_f1_full` / `avg_confidence` / `elapsed_sec`，
  每条 result 多一个 `prob`（softmax 置信度）

> 文件名是固定的，不含样本数和 shot 数 —— 这两个参数在 JSON 内部记录。
> 如果改了 `--num_samples` 或 `--n_shot` 重跑，同名文件会被覆盖，看 JSON 里的字段确认。

## 二、对比结果

### 四方法对比（生成式 vs 判别式）

| 文件 | 内容 |
|---|---|
| `comparison_4methods_zeroshot_fewshot_sft_bert.md` | 整体指标、逐类 F1、配对检验、误差方向、类别分布、结论 |
| `comparison_4methods_zeroshot_fewshot_sft_bert.png` | 逐类 F1 + 整体指标 + 格式遵循/类别分布偏差 |
| `comparison_4methods_zeroshot_fewshot_sft_bert.json` | 同样的数据，机器可读 |

生成脚本：`src_llm/compare_llm_methods.py`（只读结果，不加载模型，几秒跑完）

### 五模型对比（BERT 内部超参）

| 文件 | 内容 |
|---|---|
| `comparison_5bert_models.md` | 整体指标（含全量 val 参考值）、逐类 F1、配对检验、误差方向、结论 |
| `comparison_5bert_models.png` | 逐类 F1 + 整体指标 + 相对基准的净变化 |
| `comparison_5bert_models.json` | 同样的数据，机器可读 |

生成脚本：`src/compare_bert_models.py`（同样只读结果）

## 三、怎么重新生成

```bash
# 1. 重跑某个方法的评测（会覆盖对应的 0x_*.json）
python src_llm/classify_llm.py
python src_llm/classify_llm_fewshot.py --n_shot 2
python src_llm/evaluate_sft.py
python src/eval_bert_for_compare.py            # 一次跑完 5 个 BERT（--only cls 可只跑一类）

# 2. 重新生成对比报告和图
python src_llm/compare_llm_methods.py
python src/compare_bert_models.py
```

## 四、结果速览

### 四方法（200 条）

| 指标 | zero-shot | few-shot (2例/类) | SFT (LoRA) | BERT cls |
|---|---|---|---|---|
| Accuracy | 0.3600 | 0.3850 | 0.5650 | **0.5950** |
| Macro F1 | 0.3022 | 0.3418 | 0.5064 | **0.5764** |
| Weighted F1 | 0.3800 | 0.4359 | 0.5647 | **0.5967** |
| 可解析率 | 0.7100 | 0.6950 | 0.9900 | **1.0000** |
| 预测分布 L1 偏差 | 0.5900 | 0.4450 | 0.2900 | **0.1700** |

| 对比 | 净变化 | p 值 |
|---|---|---|
| zero-shot → few-shot | +5 | 0.500（不显著） |
| zero-shot → SFT | +41 | < 0.001 |
| few-shot → SFT | +36 | < 0.001 |
| zero-shot → BERT | +47 | < 0.001 |
| **SFT → BERT** | **+6** | **0.461（不显著）** |

要点：few-shot 几乎无效；微调才是分水岭；**SFT 和 BERT 在 200 条上分不出胜负**，
BERT 的优势是「结构上不会答非所问」和小类公平性（macro F1 高 0.035～0.070）。

### 五个 BERT 模型（200 条）

| 指标 | cls 不加权 | cls 加权 | max 不加权 | mean 不加权 | mean 加权 |
|---|---|---|---|---|---|
| Accuracy | **0.5950** | 0.5650 | **0.5950** | 0.5850 | 0.5850 |
| Macro F1 | 0.5764 | 0.5455 | 0.5780 | **0.5871** | 0.5632 |
| Weighted F1 | **0.5967** | 0.5665 | 0.5959 | 0.5864 | 0.5921 |
| 全量 val 准确率（参考） | 0.5637 | 0.5606 | 0.5626 | 0.5622 | 0.5592 |

**4 组配对检验全部不显著**（p = 0.345 / 1.000 / 0.845 / 0.856），准确率极差只有 6 条。
池化策略和加权 loss 在这个任务上都不是关键变量。

> ⚠️ 注意：BERT 在 200 条上是 0.5950，但训练时在完整 10000 条 val 上是 0.5637。
> 说明这 200 条对 BERT 略偏容易（n=200 的标准误约 ±0.035）。
> **只有同一批样本上的配对比较才有效，跨报告的绝对数字不能混着比。**
