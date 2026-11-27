# model_swap —— 换模型 / 多方法事件抽取对比

> 本周主题：**在同一个抽取任务、同一份评测集、同一把尺子上，换不同模型、用不同方法，看效果差多少、钱和时间花在哪。**

## 1. 实验设计

| 维度 | 设置 |
|---|---|
| 任务 | DuEE-Fin 金融公告事件抽取：一句 → 1 个事件（类型 + 触发词 span + 角色-论元 span） |
| 评测集 | `data/processed/dev_extract.jsonl` 的固定子集（seed=42，canonical order，n=200 为主） |
| 评测口径 | `common/evaluate_events.py`：trigger / argument / event 三级 P/R/F1 + type macro-F1 |
| 训练集 | `data/processed/train_extract.jsonl`（LoRA 用前 3500 条，1 epoch） |
| 置信区间 | 对 n 条样本做 1000 次 bootstrap，给论元 F1 的 95% CI |
| 硬件 | RTX 4060 8GB（本地 0.5B 模型，bf16 + LoRA + 梯度检查点） |

**被比较的“模型”覆盖了从零训练到大模型的完整光谱：**

| 模型 | 类型 | 说明 |
|---|---|---|
| 无模型 | 规则 | 词典 + 正则，零训练基线 |
| BiLSTM-CRF（1.31M） | 判别式、随机初始化 | 说明“预训练值多少点” |
| bert-base-chinese + Linear / CRF（102.4M） | 判别式、预训练编码器 | 项目主线 |
| bert-base-chinese 两段式流水线 | 判别式、16 个小模型 | 检测→类型→类型条件论元 |
| Qwen2-0.5B（base） | 生成式、本地 | **本次换入的模型**；课程原始为 Qwen3-0.6B |
| Qwen2.5-0.5B-Instruct | 生成式、本地、指令微调 | 对照：指令微调对 zero-shot 的影响 |
| Qwen3-0.6B（课程原始） | 生成式、本地 | 原始结果，用于换模型前后对比 |
| DeepSeek API | 生成式、外部大模型 | 大模型参照（zero / few-shot） |

**被比较的“方法”：**

- L0 规则
- L1 字符级 BiLSTM-CRF
- L2 联合抽取：BERT+Linear / BERT+CRF（149 标签）
- L2 两段式：检测 → 类型分类 → 类型条件论元
- L3 LLM zero-shot / few-shot
- L4 LoRA SFT
- L2+ 解码期硬约束（同一 checkpoint，只改解码）

## 2. 一键对比（读已有指标，不重新训练）

```bash
cd /d/workspace/nlp_study/徐鹏/week07
python model_swap/compare_models.py --bootstrap 1000
```

产物：

| 文件 | 内容 |
|---|---|
| `outputs/model_swap/REPORT.md` | 主表（n=200）+ n=60 小样本表 + LLM 换模型矩阵 + 自动关键对比 |
| `outputs/model_swap/model_swap.csv` | 同一张表的 CSV（可直接贴 Excel / PPT） |
| `outputs/model_swap/fig_model_swap.png` | 论元 F1（带 95% CI）+ 效果-延迟散点图 |

> 原则：只从 `outputs/metrics/**/*.json` 读数字；没跑的实验显式标“未跑”，不填 0、不估算。

## 3. 重新跑 LLM 两路（本机 Qwen2 系列）

```bash
MODEL_QWEN2="D:/workspace/nlp_study/pretrain_models/Qwen2-0.5B"
MODEL_QWEN25="D:/workspace/nlp_study/pretrain_models/Qwen2.5-0.5B-Instruct"

# 1) 零样本（base 模型会失败，这是结论的一部分）
python llm_zero_shot/llm_extract.py --backend local --mode zero --n 60 \
    --model_path "$MODEL_QWEN2" --tag qwen2_zero_n60 --gen_batch 8

# 2) LoRA SFT（3500 条 × 1 epoch，约 15 分钟）
python llm_lora/train_sft.py --model_path "$MODEL_QWEN2" \
    --num_train 3500 --epochs 1 --batch_size 3 --grad_accum 5 \
    --max_length 1280 --grad_ckpt --dtype bf16 \
    --output_dir outputs/llm/ckpt_qwen2 --run_name qwen2_lora

# 3) 评测 LoRA（n=200，与课程原始 Qwen3 结果同批）
python llm_lora/evaluate_sft.py --model_path "$MODEL_QWEN2" \
    --ckpt_dir outputs/llm/ckpt_qwen2/sft_adapter --n 200 --tag qwen2_lora_n200
```

把 `MODEL_QWEN2` 换成 `MODEL_QWEN25`、tag 换成 `qwen25_*` 即可跑指令微调版对照。

## 4. 读表时一定要注意

1. **n=200 和 n=60 不能直接比**：n=60 是 n=200 的**前缀子集**，但样本少了 3 倍多，置信区间更宽。
2. **置信区间重叠 ≠ 没有差别**，只说明这个样本量下分不出显著差异；要下“谁更好”的结论请跑全量 930 或 3 个 seed。
3. **LoRA 只训了 3500 条 1 epoch**，不是调参后的最优结果；它衡量的是“小模型 + 低成本微调”的性价比，不是上限。
4. **BERT/BiLSTM/pipeline 的指标沿用课程包已有结果**（本机没有 bert-base-chinese 权重，未重训）；Qwen2/Qwen2.5 两路是本次实际运行。
5. **延迟口径不同**：判别式是 batch 推理摊到每句；生成式包含逐 token 解码，API 还含网络往返，不能只看一个数。
