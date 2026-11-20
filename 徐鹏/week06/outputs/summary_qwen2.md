# 第 6 周作业：Qwen2-0.5B 替换后的分类方法对比

> 任务：更换预训练模型，做分类任务各种方法的对比实验。
> 本次更换：**Qwen3-0.6B → Qwen2-0.5B**（影响 zero-shot 与 LoRA 两路）。

## 1. 实验设置

| 项目 | 设置 |
|---|---|
| 任务 | TNEWS 中文新闻标题 15 分类 |
| 评测集 | 去重后的干净验证集，固定抽 500 条，seed=42 |
| 训练集 | TNEWS；LoRA 使用前 5000 条 |
| 替换的模型 | Qwen2-0.5B（494M 参数，base 版，不是 Instruct） |
| 微调方式 | LoRA，r=8，alpha=16，target_modules=q/k/v/o_proj |
| 可训练参数 | 1,081,344 / 495,114,112 = **0.22%** |
| 训练配置 | batch=4，grad_accum=4（等效 16），max_length=128，lr=2e-4，2 epoch |
| 硬件 | RTX 4060 8GB |
| 指标 | 准确率（含解析失败）、Macro-F1、覆盖类别数、解析失败数 |

> BERT 一行沿用课程包内已有的 `bert_pred.json`（bert-base-chinese），本次没有重新训练；规则、TF-IDF、Qwen2 两路均为本次实际运行结果。

## 2. 结果

| 方法 | 准确率(含失败) | Macro-F1 | 覆盖类别 | 解析失败 | 成本 |
|---|---|---|---|---|---|
| 规则 / 关键词 | 0.126 | 0.0716 | 10/15 | 0 | 秒级 |
| TF-IDF + LinearSVC | 0.512 | 0.4721 | 15/15 | 0 | 6.3 s (CPU) |
| BERT-base 微调 | 0.556 | **0.5436** | 15/15 | 0 | 约 14 min（已有结果） |
| Qwen2-0.5B zero-shot | **0.012** | 0.2359 | 9/15 | 483 | 约 3 min 19 s |
| **Qwen2-0.5B + LoRA** | **0.584** | 0.5355 | 15/15 | 0 | 训练 7 min 41 s + 评测 1 min 08 s |

> zero-shot 的 `acc=0.3529` 只在 17 条能解析出标签的样本上计算。按项目统一口径（解析失败算答错），准确率是 **0.012**；表中使用的是含失败口径。它的 Macro-F1 也只在 17 条有效样本上计算，不能和别的行直接比。

## 3. 与替换前 Qwen3-0.6B 的对比

| 方法 | Qwen3-0.6B（原） | Qwen2-0.5B（本次） | 变化 |
|---|---|---|---|
| zero-shot | 0.168 / F1 0.105 | 0.012 / 仅 17 条有效 | 基座模型零样本更差，基本不可用 |
| LoRA SFT | 0.530 / F1 0.484 | **0.584 / F1 0.536** | 准确率 +5.4 个点 |
| LoRA 训练时间 | 约 44 min（README） | **7 min 41 s** | 约 5.7× 更快 |
| 显存峰值 | 约 3 GB | **2.47 GB** | 略低 |
| adapter 大小 | 约 9 MB | **4.35 MB** | 更小 |

> Qwen2 是本轮实际训练/评测；Qwen3 数字来自课程包原结果，配置同为 5000 条 × 2 epoch、评测集同为 500 条 seed=42，可以看趋势，但不是严格同机同次运行。

## 4. 结论

1. **换模型后，LoRA SFT 仍是 LLM 路线里唯一可用的方案。** Qwen2-0.5B + LoRA 达到 0.584，超过 BERT-base 的 0.556。
2. **但 Macro-F1 略低于 BERT**（0.5355 vs 0.5436，差 0.8 个点）。Qwen2 的提升主要来自头部类，BERT 在长尾类上更均衡；如果业务按类考核，BERT 仍更稳。
3. **基座模型 zero-shot 不可用。** Qwen2-0.5B 是 base 版，500 条里 483 条解析失败、只覆盖 9 类。0.5B 级别必须做 SFT，不能直接零样本。
4. **成本**：LoRA 训练 7.7 分钟、2.47GB 显存、adapter 4.35MB，4060 8GB 完全够用；推理比 BERT 慢（500 条约 68 秒），适合离线/低并发场景。
5. **统计提醒**：500 条样本的 95% 置信区间约 ±4.4 个点。Qwen2 比 BERT 高 2.8 个准确率点、Macro-F1 低 0.8 个点，都还在噪声边缘；要写“显著提升”，建议用全量干净验证集（约 8,901 条）跑 3 个 seed。

## 5. 怎么跑（每条独立命令）

```bash
conda activate torch_env
cd /d/workspace/nlp_study/徐鹏/week06
MODEL="D:/workspace/nlp_study/pretrain_models/Qwen2-0.5B"

# 1) 规则（秒级）
python rule/classify_rule.py --n 500

# 2) TF-IDF（约 6 秒）
python tfidf/classify_tfidf.py --n 500

# 3) BERT（需要 outputs/bert_ckpt.pt；当前包内没有，表里沿用已有 bert_pred.json）
python bert/predict_bert.py --n 500

# 4) Qwen2-0.5B zero-shot（约 3 分 20 秒）
python llm_zero_shot/classify_zero_shot.py --model "$MODEL" --n 500

# 5) Qwen2-0.5B LoRA 训练（约 7 分 41 秒）
python llm_lora/train_lora.py --model "$MODEL" \
    --num_train 5000 --epochs 2 --batch_size 4 --grad_accum 4 --max_length 128

# 6) LoRA 评测（约 1 分 08 秒）
python llm_lora/eval_lora.py --adapter outputs/lora_adapter \
    --model "$MODEL" --n 500 --out outputs/lora_pred.json

# 7) 汇总表（可选）
python eval/eval_all.py
```

## 6. 产物

| 文件 | 内容 |
|---|---|
| `outputs/rule_pred.json` | 规则法逐条预测 |
| `outputs/tfidf_pred.json` | TF-IDF 逐条预测 |
| `outputs/bert_pred.json` | BERT 已有结果 |
| `outputs/zero_shot_pred.json` | Qwen2-0.5B zero-shot |
| `outputs/lora_pred.json` | Qwen2-0.5B + LoRA |
| `outputs/lora_train_log.json` | 2 epoch 训练日志（loss / 显存 / 耗时） |
| `outputs/lora_adapter/` | 4.35MB LoRA adapter |
| `outputs/summary.md` | `eval_all.py` 生成的汇总表 |
| `outputs/summary_qwen2.md` | 本文件 |
| `outputs/summary_qwen3_original.md` | 替换前的原始 summary 备份 |

## 7. 附：未替换的旧结果

| 方法 | 准确率(含失败) | Macro-F1 | 说明 |
|---|---|---|---|
| Qwen3 全量微调（旧） | 0.536 | 0.4919 | 旧结果，未用 Qwen2 重跑 |
| API zero-shot（deepseek-v4-flash） | 0.514 | 0.4708 | 外部 API，未替换 |
