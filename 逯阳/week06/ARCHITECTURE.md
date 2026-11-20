# ARCHITECTURE.md — 技术方案文档

> BERT 中文文本分类项目 

---

## 一、项目定位

### 场景选型理由

| 维度 | 说明 |
|------|------|
| 任务选型 | 多分类（15类新闻标题）：标签清晰、易于量化评估、贴近真实业务 |
| 数据选型 | TNEWS（CLUE 基准）：公开可用、无法律风险、规模适中（5.3万训练条），是标准 benchmark 可横向对比 |
| 模型选型 | bert-base-chinese：参数量适中（110M），可在 CPU 上运行，是学习 fine-tuning 的标准起点 |
| 对比维度 | 三种池化策略 × fine-tuning / SFT / zero-shot LLM，三路对比让教学价值最大化 |

### 三套实现对比

| 维度 | `src/`（BERT fine-tune）| `src_llm/classify_llm.py`（LLM zero-shot）| `src_llm/train_sft.py`（LLM SFT）|
|------|------------------------|------------------------------------------|----------------------------------|
| 目的 | 展示判别式 fine-tuning 全链路 | 展示 zero-shot 分类新范式 | 展示指令微调（SFT）+ LoRA 高效微调 |
| 模型 | bert-base-chinese（110M）| Qwen2-0.5B-Instruct（500M）| Qwen2-0.5B-Instruct（500M + LoRA）|
| 训练 | 需要，3 epoch ≈ 30~90 min（CPU）| 不需要训练 | LoRA 3 epoch ≈ 18 min（RTX 4060）|
| 可训练参数 | 全量 110M（100%）| 0 | 1.08M（0.22%，LoRA r=8）|
| 准确率（实测）| 57%~62%（val, 3 epoch）| 未大批量测（5条 demo 不具代表性）| **58.0%**（val 200条, seed=42）|
| 代码规模 | ~600 行，5 个模块 | ~90 行，1 个文件 | ~200 行，2 个文件 |
| 教学重点 | tokenize / pooling / optimizer / loss | prompt 设计 / 输出解析 | chat格式 / loss masking / LoRA 原理 |

---

## 二、整体流水线

```
原始数据（HuggingFace CLUE/TNEWS）
         │
         ▼
  download_data.py          ← 下载并保存为本地 JSON + label_map.json
         │
         ▼
  explore_data.py           ← 类别分布 / 文本长度 / Token 长度分析（5张图表）
         │
         ▼
  dataset.py                ← TNEWSDataset：tokenize → input_ids / attention_mask
         │
         ▼
  model.py                  ← BertModel + 池化策略（cls / mean / max）+ Linear
         │
         ▼
  train.py                  ← AdamW（分层 lr）+ warmup + 加权 loss（可选）
         │
         ▼
  evaluate.py               ← accuracy / macro F1 / 混淆矩阵
         │
         ▼
  predict.py                ← 单条 / 批量推理，输出 top-k 置信度

────── 并行对比 ──────

  src_llm/classify_llm.py   ← Qwen2 zero-shot，同一验证集对比准确率
         │
         ▼
  src_llm/train_sft.py      ← LoRA 指令微调（chat格式 + loss masking）
         │
         ▼
  src_llm/evaluate_sft.py   ← 加载 adapter，三方准确率对比
```

---

## 三、各环节技术选型

### 3.1 数据集：TNEWS

**选型原因**：
- CLUE 官方 benchmark，学生作品简历上可写"在 CLUE TNEWS 上达到 XX%"，有横向参考价值
- 15 类短标题（均值 22 字），计算资源需求低，CPU 也能跑完
- 存在明显的类别不均衡（证券类仅 257 条，科技类 5955 条，比例 23x），是教学加权 loss 的天然场景
- test 集无标签（CLUE 惯例），使用 val 集作为最终评估

**已知特点**：
- 文本极短（P99 = 39 字），max_length=64 即可覆盖 100%
- Token/字符比值 ≈ 0.97（中文基本 1 字 = 1 token），与英文 subword 分词形成鲜明对比

### 3.2 模型：BertModel + 自定义分类头

**选型原因（不用 BertForSequenceClassification）**：
- `BertForSequenceClassification` 内部结构是黑盒，学生看不到向量提取逻辑
- 手写分类头只有 3 行核心代码，池化策略替换清晰可见
- 方便后续扩展：换 RoBERTa、加多任务头、换 pooling 策略都无需改框架代码

**三种池化策略**：

| 策略 | 实现 | 直觉解释 | 适用场景 |
|------|------|---------|---------|
| `cls` | `last_hidden[:, 0, :]` | BERT 训练时 [CLS] 就被设计为句子摘要向量 | 分类任务的默认选择 |
| `mean` | 有效 token 均值（排除 padding） | 所有词信息的平均表达 | 语义相似度、句子表示 |
| `max` | 有效 token 逐维取最大值 | 保留每个维度最显著的激活 | 情感类、关键词驱动任务 |

> **教学设计**：三种策略通过 `--pool cls/mean/max` 切换，训练结果存在不同 checkpoint，
> 最后用 evaluate.py 对比混淆矩阵，让学生量化差异。

### 3.3 优化器：分层学习率

```
BERT 层：    lr = 2e-5   （预训练权重，小步微调）
分类头：     lr = 1e-4   （随机初始化，需要更快收敛）
```

**选型原因**：
- BERT 预训练权重已经很好，用过大的学习率会 "遗忘" 预训练知识（catastrophic forgetting）
- 分类头是新加的随机初始化层，需要更大步长才能快速学到映射关系
- 统一用 AdamW（带 weight decay）是 Transformer fine-tuning 的事实标准

### 3.4 类别不均衡：加权 CrossEntropyLoss

```python
# sklearn 计算 balanced weight：
# weight_i = n_samples / (n_classes × n_samples_i)
weights = compute_class_weight("balanced", classes=classes, y=labels)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
```

**选型原因**：
- 证券类（257条）/ 科技类（5955条）≈ 1:23，不处理时模型会直接忽略小类
- Balanced weight 让每个类别对 loss 的贡献大致相等
- 通过 `--use_class_weight` 开关，学生可以对比加权前后证券类的 Recall 变化

### 3.5 学习率调度：Linear Warmup + Decay

---

### 3.6 LLM SFT：LoRA 指令微调

**核心教学点**：

#### 数据格式转换

将分类任务转化为 chat 格式（指令微调的通用范式）：

```
system:    "你是一个新闻标题分类助手，只输出类别名称，可选：故事/文化/..."
user:      "新闻标题：上课时学生手机响个不停...\n类别："
assistant: "教育"         ← 只有这 1 个 token 计算 loss
```

#### Loss Masking（SFT 与 Pretraining 的核心区别）

```python
# Pretraining：对所有 token 计算 cross-entropy loss
# SFT：只在 assistant 回复部分（类别名 + EOS）计算 loss
#       prompt 部分 labels 全设 -100，PyTorch CrossEntropyLoss 自动忽略

labels = [-100] * prompt_len + response_ids
```

#### LoRA（Low-Rank Adaptation）

```
原始矩阵 W ∈ R^{d×d}（冻结）
         ↓
输出 += B·A·x，其中 A ∈ R^{r×d}，B ∈ R^{d×r}，r=8 << d=896

Qwen2-0.5B 全参数：495,114,112
LoRA r=8 可训练参数：1,081,344（0.22%）
```

**选型原因**：
- 显存友好：RTX 4060 8GB 可跑，全量 FT 会 OOM
- 速度快：仅训练 0.22% 参数，3 epoch ≈ 18 分钟（GPU）
- 是业界 SFT 标准实践，有教学迁移价值

#### 实测结果

| 配置 | 数据量 | 训练时间 | 可训练参数 | val 准确率（200条）|
|------|--------|---------|-----------|-------------------|
| LoRA r=8，3 epoch | 5,000 条 | ~18 min（RTX 4060）| 1.08M（0.22%）| **58.0%** |

**选型原因**：
- Warmup 阶段（前 10% 步数）用很小的 lr，防止模型在训练初期不稳定时破坏预训练权重
- 之后线性衰减到 0，训练后期细粒度调整
- HuggingFace `get_linear_schedule_with_warmup` 是 BERT fine-tuning 标配

---

## 四、评估体系

### 4.1 核心指标

| 指标 | 含义 | 为什么用 |
|------|------|---------|
| Accuracy | 正确预测数 / 总数 | 直觉易懂，但不均衡时会高估 |
| Macro F1 | 各类 F1 的算术平均 | 每个类等权重，能反映小类（证券）的真实性能 |
| Per-class Precision/Recall | 每个类别单独统计 | 定位问题：哪个类被预测错了？被误判成了哪类？ |
| 混淆矩阵 | 真实类别 × 预测类别 | 可视化易混淆的类别对 |

### 4.2 消融实验矩阵

通过改变单一变量，量化每个设计决策的价值：

| 实验 | 变量 | 对比维度 |
|------|------|---------|
| 池化策略 | `--pool cls/mean/max` | 三种向量提取方式的精度差异 |
| 类别加权 | `--use_class_weight` | 加权前后证券类 Recall 变化 |
| 训练轮数 | `--epochs 1/3/5` | epoch 数对收敛和过拟合的影响 |
| 截断长度 | `--max_length 32/64/128` | 本数据集极短，此实验预期差异很小 |
| 数据量 | 取训练集子集（1k/5k/全量） | 少样本时 LLM zero-shot 是否更有优势 |

### 4.3 三方方案对比（实测数据）

| 对比维度 | BERT fine-tune | LLM Zero-shot | LLM SFT（LoRA）|
|---------|---------------|---------------|----------------|
| 模型 | bert-base-chinese（110M）| Qwen2-0.5B（500M）| Qwen2-0.5B（500M）|
| 可训练参数 | 110M（全量）| 0 | 1.08M（0.22%，LoRA r=8）|
| 训练成本 | 3 epoch，10~12 min/epoch（GPU）| 无需训练 | 3 epoch，约 18 min（RTX 4060）|
| 推理速度 | ~5ms/条（GPU）| ~2s/条（CPU 本地）| ~0.06s/条（GPU，含 generate）|
| 实测准确率 | 57%~62%（val，全量数据）| 未大批量测 | **58.0%**（val 200条，5K训练）|
| 数据需求 | 需标注数据（全量 53K）| 无需标注 | 需标注数据（5K 即有效）|
| "无法解析"问题 | 无（直接输出 logits）| 有（~2%）| 有（~1%）|
| 教学概念 | tokenize / pooling / optimizer | prompt 设计 | chat格式 / loss mask / LoRA |
| 适用场景 | 有大量标注数据、追求极致精度 | 快速原型、零标注 | 少量标注 + 大模型能力 |

> **关键洞察**：LLM SFT 用 **5K 条数据**（约 BERT 训练数据的 9.4%）就达到了与 BERT 全量训练接近的准确率，
> 体现了大模型预训练知识的迁移效率优势。

---

## 五、关键工程决策与踩坑

| 问题 | 根因 | 解法 |
|------|------|------|
| Windows 多进程 DataLoader 报错 | PyTorch 多进程 pickling 在 Windows 上与 Jupyter 有冲突 | `num_workers=0` 默认关闭多进程，Linux 可设 2/4 |
| BertTokenizer 返回 squeeze 前有多余维度 | `return_tensors="pt"` 时单条输入会多一个 batch 维度 | `encoding["input_ids"].squeeze(0)` |
| max_length=128 对此数据集偏大 | TNEWS 标题平均 22 字，P99=39 字 | 设 64 即可覆盖 100% 样本，减少显存占用 |
| Qwen2 输出可能包含类别名以外的文字 | CausalLM 生成不严格受控 | 用模糊匹配 `if name in raw_output` 而非精确匹配 |
| `apply_chat_template(tokenize=True)` 返回 BatchEncoding | transformers 5.x 改变了返回类型（原为 list）| 改用 `tokenize=False` 拿文本串，再 `tokenizer.encode()` 编码 |
| SFT 训练时 `torch_dtype` 参数报废弃警告 | transformers 5.x 将 `torch_dtype` 改名为 `dtype` | 改用 `dtype=torch.float32` |
| RTX 4060 8GB 无法全量微调 Qwen2-0.5B | AdamW fp32 优化器状态（~6GB）+ 模型权重（~2GB）超出显存 | 使用 LoRA，优化器状态仅 ~30MB，总显存 ~3GB |
| transformers 5.x BertModel 输出变为 tuple | 5.x 将默认 `return_dict` 改为 `False` | `self.bert(..., return_dict=True)` 强制返回命名对象 |
| PyTorch 2.6 `torch.load` 默认 `weights_only=True` | checkpoint 含非 tensor 对象（args dict），无法用 weights_only 加载 | 本地可信文件加 `weights_only=False` |
| transformers 5.x `AutoModelForCausalLM.from_pretrained` 参数名变化 | `torch_dtype` 参数已废弃，改名为 `dtype` | 使用 `dtype=torch.float16` |
| transformers 5.x `apply_chat_template` 返回 BatchEncoding | 5.x 返回包含 `input_ids`/`attention_mask` 的对象而非裸 tensor | 加 `return_dict=True`，用 `encoding["input_ids"]` 取值 |
| 证券类（label=12）Recall 接近 0 | 训练样本仅 257 条，验证集 45 条，模型倾向预测高频类 | 加 `--use_class_weight` 参数 |
| matplotlib 中文乱码 | 默认字体不含 CJK 字符 | 动态检测系统中文字体，找不到时降级为英文 |
| CLUE datasets 下载慢 | HuggingFace Hub 在国内访问不稳定 | 脚本将数据落盘为本地 JSON，后续全部从本地加载 |

---

## 六、目录结构

```
bert_text_classification/
│
├── src/                          # fine-tuning 实现
│   ├── download_data.py          # 下载 TNEWS，保存为 JSON + label_map
│   ├── explore_data.py           # 数据探索：类别分布 / 长度统计 / Token 分析
│   ├── dataset.py                # TNEWSDataset + build_dataloaders
│   ├── model.py                  # BertClassifier（cls / mean / max 三种池化）
│   ├── train.py                  # 训练循环（分层 lr, warmup, 加权 loss, checkpoint）
│   ├── evaluate.py               # 评估 + 混淆矩阵可视化
│   ├── predict.py                # 单条 / 批量推理
│   ├── eval_bert_for_compare.py  # 5 个 BERT 权重跑 LLM 那 200 条样本（配对比较用）
│   └── compare_bert_models.py    # 5 个 BERT 微调模型的结果对比（出表 + 图）
│
├── src_llm/
│   ├── classify_llm.py           # Qwen2-0.5B zero-shot 分类对比
│   ├── classify_llm_fewshot.py   # few-shot 版（每类 N 个示例，示例只从训练集抽）
│   ├── train_sft.py              # LoRA 指令微调（chat格式 + loss masking）
│   ├── evaluate_sft.py           # 加载 adapter，三方准确率对比
│   └── compare_llm_methods.py    # 四种方案结果对比（出表 + 图）
│
├── data/                         # 本地化数据（git 忽略大文件）
│   ├── train.json                # 53360 条
│   ├── val.json                  # 10000 条
│   ├── test.json                 # 10000 条（无标签）
│   └── label_map.json            # id2name / id2code 双向映射
│
├── outputs/
│   ├── checkpoints/              # best_{pool}.pt（仅保留验证集最优）
│   ├── figures/                  # 数据分析图表（5张）
│   ├── train_log_{pool}.json     # 每 epoch 的 loss / acc 记录
│   ├── train_log_sft.json        # SFT 每 epoch 的 train_loss / val_loss
│   ├── sft_adapter/              # LoRA adapter（adapter_model.safetensors + config）
│   └── results/                  # 评测结果汇总（见该目录 README.md）
│       ├── 01_zeroshot_qwen2-0.5b_result.json
│       ├── 02_fewshot_qwen2-0.5b_result.json
│       ├── 03_sft_lora_qwen2-0.5b_result.json
│       ├── 04~08_bert_{pool}_{weighted|unweighted}_result.json   # 5 个 BERT
│       ├── comparison_4methods_zeroshot_fewshot_sft_bert.{md,png,json}
│       ├── comparison_5bert_models.{md,png,json}
│       └── README.md
│
├── ARCHITECTURE.md               # 本文件
├── USAGE_GUIDE.md
├── RESUME_GUIDE.md
└── requirements.txt
```
