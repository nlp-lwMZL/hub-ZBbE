# USAGE_GUIDE.md — 代码调用与测试指南

> BERT 中文文本分类项目 · 教学型

---

## 一、环境准备

### 1.1 依赖安装

```bash
pip install -r requirements.txt
```

核心依赖版本要求：

| 包 | 版本（已验证）| 说明 |
|----|------------|------|
| torch | 2.6.0+cu126 | CUDA 12.6；CPU 也可用，速度慢约 10x |
| transformers | 5.5.3 | BertTokenizer / BertModel / AutoModelForCausalLM |
| peft | 0.15.0 | LoRA 微调（`src_llm/train_sft.py` 依赖）|
| datasets | 4.0.0 | 下载 TNEWS |
| scikit-learn | 1.5.1 | 加权 loss / F1 / 混淆矩阵 |
| matplotlib / seaborn | 3.10.7 / 0.13.2 | 数据分析图表 |

### 1.2 预训练模型

项目使用以下两个预训练模型（均在 `pretrain_models/` 目录下）：

```
pretrain_models/
├── bert-base-chinese/        ← src/ 使用
│   ├── config.json
│   ├── vocab.txt
│   └── pytorch_model.bin（或 model.safetensors）
└── Qwen2-0.5B-Instruct/      ← src_llm/ 使用
    ├── config.json
    ├── tokenizer.json
    └── model.safetensors
```

如需指定其他路径：
- BERT 脚本支持 `--bert_path` 参数
- LLM 脚本支持 `--model_path` 参数

### 1.3 目录约定

所有脚本在 `src/` 目录下运行（`cd src`），路径均以此为基准。

---

## 二、Step 1：数据下载

```bash
cd src
python download_data.py
```

**内部流程**：
1. 通过 HuggingFace `datasets` 库下载 CLUE/TNEWS
2. 构建 `label_map.json`（label_id ↔ 类别码 ↔ 中文名 三层映射）
3. 将 train / val / test 分别序列化为 JSON

**预期输出**：

```
label_map 已保存 → ..\data\label_map.json
类别数：15
   0 | 100 | 故事
   ...
train: 53360 条 → ..\data\train.json
val  : 10000 条 → ..\data\val.json
test : 10000 条 → ..\data\test.json
下载完成。
```

---

## 三、Step 2：数据探索分析

```bash
python explore_data.py
```

可选参数：

```bash
# 指定自定义路径
python explore_data.py --data_dir ../data --output_dir ../outputs/figures

# 跳过 Token 长度分析（不加载 tokenizer，速度更快）
python explore_data.py --skip_token
```

**生成的图表**（保存至 `outputs/figures/`）：

| 文件名 | 内容 |
|--------|------|
| `label_dist_train.png` | 训练集类别分布柱状图 |
| `label_dist_val.png` | 验证集类别分布柱状图 |
| `char_length_train.png` | 字符长度分布直方图 + 截断覆盖率曲线 |
| `length_by_label_train.png` | 各类别长度箱线图 + 均值柱状图 |
| `token_length_train.png` | Token 长度 vs 字符长度对比直方图 |

**关键结论**（供教学讲解）：
- 类别不均衡比 23x（科技 5955 条 vs 证券 257 条）
- 文本极短（均值 22 字，P99 = 39 字），`max_length=64` 已完全覆盖
- Token/字符比值 ≈ 0.97，中文基本 1 字 = 1 token

---

## 四、Step 3：模型训练

### 4.1 基本用法

```bash
# 默认参数（CLS 池化，3 epoch）
python train.py

# 均值池化
python train.py --pool mean

# 最大值池化 + 加权 loss（处理证券类不均衡）
python train.py --pool max --use_class_weight
```

### 4.2 完整参数说明

```bash
python train.py \
  --pool cls          \  # 池化策略：cls / mean / max
  --epochs 3          \  # 训练轮数
  --batch_size 32     \  # batch 大小
  --max_length 128    \  # tokenizer 截断长度（建议 64 即可）
  --lr 2e-5           \  # BERT 层学习率
  --head_lr_mult 5.0  \  # 分类头 lr = lr × 5.0
  --dropout 0.1       \  # 分类头前的 dropout
  --warmup_ratio 0.1  \  # warmup 步数 = 总步数 × 0.1
  --grad_accum 1      \  # 梯度累积（显存不足时设 2 或 4）
  --use_class_weight     # 启用加权 loss
```

### 4.3 预期训练日志

```
使用设备: cpu    （有 CUDA 则显示 cuda:0）
类别数: 15
DataLoader 构建完成
  train: 53360 条, 1668 batch
  val  : 10000 条, 313 batch
模型参数量: 102.3M  (BERT: 102.3M, 分类头: 11.5K)
池化策略: cls
...
Epoch 1/3 | train_loss=1.8234 train_acc=0.4321 | val_acc=0.5312 val_macro_f1=0.3821 | 1234s
  ✓ 新最优模型已保存 → ..\outputs\checkpoints\best_cls.pt  (val_acc=0.5312)
Epoch 2/3 | train_loss=1.2156 train_acc=0.5987 | val_acc=0.5891 val_macro_f1=0.4312 | ...
Epoch 3/3 | train_loss=0.9823 train_acc=0.6543 | val_acc=0.6012 val_macro_f1=0.4876 | ...
```

> **训练时间参考**：GPU（RTX 系列）每 epoch 约 10~12 分钟；CPU 约 60~90 分钟。  
> 建议先用 `--epochs 1` 验证流程，1 epoch val_acc 可达约 0.56。

### 4.4 消融实验（三种池化对比）

```bash
python train.py --pool cls  --epochs 3
python train.py --pool mean --epochs 3
python train.py --pool max  --epochs 3
```

三次训练各自保存 `best_cls.pt` / `best_mean.pt` / `best_max.pt`，
用 `evaluate.py` 分别加载对比。

---

## 五、Step 4：评估

```bash
# 评估 cls 池化的最优模型（默认）
python evaluate.py --pool cls

# 评估指定 checkpoint
python evaluate.py --ckpt_path ../outputs/checkpoints/best_mean.pt
```

**预期输出**：

```
分类报告：
              precision    recall  f1-score   support

          故事       0.52      0.48      0.50       215
          文化       0.55      0.61      0.58       736
          ...
        证券       0.31      0.11      0.16        45   ← 小类，注意 recall
          ...

    accuracy                           0.59     10000
   macro avg       0.52      0.49      0.50     10000
weighted avg       0.59      0.59      0.59     10000

val accuracy : 0.5912
val macro F1 : 0.5034

混淆矩阵已保存 → ..\outputs\figures\confusion_matrix_cls.png
```

---

## 六、Step 5：推理

### 6.1 单条推理

```bash
python predict.py --pool cls --text "苹果发布了最新的 iPhone 17，搭载 A19 芯片"
```

输出：

```
文本：苹果发布了最新的 iPhone 17，搭载 A19 芯片
预测：科技 (置信度 0.9231)
Top-3：
  [ 8] 科技   0.9231
  [ 6] 汽车   0.0312
  [ 2] 娱乐   0.0187
```

### 6.2 批量推理

```bash
python predict.py --pool cls \
  --input_file ../data/val.json \
  --output_file ../outputs/val_predictions.json
```

输出包含准确率统计：

```
批量推理 10000 条 ...
准确率: 5912/10000 = 0.5912
结果已保存 → ..\outputs\val_predictions.json
```

### 6.3 Python 模块调用

```python
import torch
from pathlib import Path
from transformers import BertTokenizer
from model import build_model

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_path = "../../pretrain_models/bert-base-chinese"
ckpt_path = Path("../outputs/checkpoints/best_cls.pt")

# 加载模型
ckpt  = torch.load(ckpt_path, map_location=device)
model = build_model(bert_path, num_labels=15, pool=ckpt["pool"])
model.load_state_dict(ckpt["state_dict"])
model = model.to(device).eval()

tokenizer = BertTokenizer.from_pretrained(bert_path)

# 单条推理
from predict import predict_single
import json
with open("../data/label_map.json") as f:
    id2name = {int(k): v for k, v in json.load(f)["id2name"].items()}

result = predict_single("今天 A 股大涨", model, tokenizer, id2name,
                        max_length=128, device=device, top_k=3)
print(result["prediction"])  # {'label_id': 4, 'label_name': '财经', 'prob': 0.87}
```

---

## 七、Step 6：LLM Zero-Shot 对比

```bash
cd ../src_llm

# 快速演示（5条示例）
python classify_llm.py --demo

# 正式评估（200 条随机样本）
python classify_llm.py --num_samples 200
```

**预期输出**：

```
[  1/200] ✓ 真实:科技  预测:科技  | 苹果发布了最新的 iPhone...
[  2/200] ✗ 真实:教育  预测:文化  | 上课时学生手机响个不停...
...
Zero-Shot LLM 分类结果（Qwen2-0.5B-Instruct）
准确率   : 87/200 = 0.4350
无法解析 : 5 条 (2.5%)
```

> Qwen2-0.5B 模型较小，zero-shot 准确率受限。换用 Qwen-7B 或 API 模型会大幅提升。

---

## 八、Step 7：LLM SFT 指令微调（LoRA / 全量微调）

### 8.1 训练

```bash
cd ../src_llm

# ── LoRA 微调（默认，推荐用于演示）─────────────────────────────────────────
# 5000 条，3 epoch，RTX 4060 约 18 分钟，显存约 3GB
python train_sft.py

# 全部 53K 条
python train_sft.py --num_train -1

# 快速验证流程（1 epoch）
python train_sft.py --epochs 1

# 调大 LoRA rank（更多参数，效果略好，显存略增）
python train_sft.py --lora_r 16

# ── 全量微调（Full Fine-Tuning）──────────────────────────────────────────────
# 更新所有 495M 参数，需显存 ≥ 16GB；学习率必须比 LoRA 小 10x
python train_sft.py --full_ft --lr 2e-5

# 全量微调 + 全部数据
python train_sft.py --full_ft --lr 2e-5 --num_train -1
```

**完整参数说明**：

```bash
python train_sft.py \
  --num_train 5000    \  # 训练样本数，-1 用全部
  --epochs 3          \  # 训练轮数
  --batch_size 4      \  # 每步 batch 大小
  --grad_accum 4      \  # 梯度累积，等效 batch = 4×4 = 16
  --lr 2e-4           \  # 学习率；不填则自动选：LoRA=2e-4，全量=2e-5
  --lora_r 8          \  # LoRA rank（仅 LoRA 模式有效）
  --lora_alpha 16     \  # 缩放因子（仅 LoRA 模式有效）
  --full_ft              # 加此 flag 切换为全量微调
```

**两种模式对比**：

| 维度 | LoRA（默认）| 全量微调（`--full_ft`）|
|------|------------|----------------------|
| 可训练参数 | 1.08M（**0.22%**）| 495M（**100%**）|
| 推荐学习率 | 2e-4（自动）| 2e-5（需手动指定）|
| 显存需求 | ~3GB（RTX 4060 可跑）| ~8~10GB（需 ≥ 16GB 稳定）|
| checkpoint | `outputs/sft_adapter/`（仅 adapter）| `outputs/sft_full_ckpt/`（完整模型）|
| 日志文件 | `train_log_sft.json` | `train_log_full_ft.json` |

**实测训练日志**（LoRA，5000 条，3 epoch，RTX 4060）：

```
使用设备: cuda  |  微调模式: LoRA 微调
trainable params: 1,081,344 || all params: 495,114,112 || trainable%: 0.2184
总训练步数: 937（batch=4, grad_accum=4, epochs=3, lr=0.0002）

Epoch 1/3 | train_loss=0.7318  val_loss=0.6594 | 499s
  ✓ 最优 LoRA adapter 已保存 → outputs/sft_adapter  (val_loss=0.6594)
Epoch 2/3 | train_loss=0.6115  val_loss=0.6386 | 304s
  ✓ 最优 LoRA adapter 已保存 → outputs/sft_adapter  (val_loss=0.6386)
Epoch 3/3 | train_loss=0.5184  val_loss=0.6523 | 268s

训练完成。最优 val_loss=0.6386
```

> **教学观察**：Epoch 2 val_loss 最低（轻微过拟合）；LoRA 可训练参数仅 **0.22%**，
> 18 分钟即训出接近 BERT 全量的效果。

### 8.2 评估（三方对比）

`evaluate_sft.py` 自动识别 checkpoint 类型，无需手动指定模式：

```bash
# 评估 LoRA 模型（默认，读取 outputs/sft_adapter/）
python evaluate_sft.py

# 评估全量微调模型（读取 outputs/sft_full_ckpt/）
python evaluate_sft.py --ckpt_dir ../outputs/sft_full_ckpt

# 快速演示（5 条）
python evaluate_sft.py --demo

# 采样 500 条做更可靠的评估
python evaluate_sft.py --num_samples 500
```

识别逻辑：checkpoint 目录含 `adapter_config.json` → LoRA 路径；否则 → 全量微调路径。

**实测输出**（LoRA，val 200条，seed=42）：

```
检测到 LoRA adapter，加载 base model: ...
模型加载完成（LoRA adapter 已合并）

样本数    : 200
准确率    : 116/200 = 0.5800
无法解析  : 2 条 (1.0%)
均值耗时  : 0.06s/条（GPU）

三方对比
  BERT fine-tune（53K 条，3 epoch）  val accuracy ≈ 0.57~0.62
  Qwen2-0.5B zero-shot               未大批量测（5条 demo 不具代表性）
  Qwen2-0.5B SFT（LoRA，5K 条）      val accuracy = 0.5800
```

---

## 九、调试与常见问题

**Q1：训练时报 `RuntimeError: CUDA out of memory`**

```bash
# 减小 batch_size，或使用梯度累积等效扩大 batch
python train.py --batch_size 16 --grad_accum 2
```

**Q2：CPU 训练太慢，想快速验证流程**

```bash
# 先用 1 epoch + 小 batch 跑通流程
python train.py --epochs 1 --batch_size 8
```

**Q3：`from model import build_model` 报 ModuleNotFoundError**

确保在 `src/` 目录下运行，而不是项目根目录：

```bash
cd src
python train.py  # 正确
# python src/train.py  # 错误，import 路径会找不到 model.py
```

**Q4：matplotlib 图表显示方块字（中文乱码）**

脚本已自动检测系统中文字体，通常安装 `SimHei` 或 `Microsoft YaHei` 即可：

```bash
# Windows 已有微软雅黑，无需额外安装
# Linux 安装：
sudo apt install fonts-wqy-microhei
```

**Q5：`datasets` 下载超时**

数据已在 `data/` 目录下落盘，`download_data.py` 只需运行一次。
后续 `dataset.py` 直接读取本地 JSON，不再访问网络。

**Q6：transformers 5.x 下 BertModel 输出变为 tuple，`outputs.last_hidden_state` 报 AttributeError**

transformers 5.x 将 BertModel 默认输出从命名对象改为 `tuple`。代码中已在 `self.bert()` 调用时加了 `return_dict=True`，强制返回命名对象，无需手动处理。若遇到此问题，检查 `model.py` 中的 `forward` 方法是否有该参数。

**Q7：torch.load 报 UnpicklingError（weights_only 相关）**

PyTorch 2.6 将 `torch.load` 的 `weights_only` 默认值改为 `True`，但 checkpoint 中存有 Python 基础类型（argparse.Namespace 等）。代码中 `evaluate.py` 和 `predict.py` 均已加 `weights_only=False` 处理本地可信文件。

**Q8：classify_llm.py 报 `torch_dtype` 废弃警告或 dtype 错误**

transformers 5.x 将 `from_pretrained` 的 `torch_dtype` 参数重命名为 `dtype`。代码已更新为 `dtype=`。

**Q9：证券类（label_id=12）的 Recall 非常低**

这是数据集本身的问题（45 条验证样本）。
启用加权 loss 可以提升，但不能完全解决：

```bash
python train.py --use_class_weight
```

之后对比 `confusion_matrix_cls.png` 和加权版的混淆矩阵，观察证券类行的变化。

**Q10：train_sft.py 报 `ModuleNotFoundError: No module named 'peft'`**

LoRA 微调需要 `peft` 库：

```bash
pip install peft>=0.14.0
```

**Q11：evaluate_sft.py 报 "checkpoint 目录不存在"**

需要先运行 `train_sft.py` 完成训练：
- LoRA 模式（默认）保存到 `outputs/sft_adapter/`
- 全量微调（`--full_ft`）保存到 `outputs/sft_full_ckpt/`

评估时用 `--ckpt_dir` 指定对应目录，或使用默认值（指向 `sft_adapter/`）。

**Q12：SFT 训练后 evaluate_sft.py 的准确率比 BERT 还低**

可能原因：
1. 只用了 5K 条训练数据（BERT 用了 53K）；用 `--num_train -1` 跑全量对比
2. Epoch 2 是最优（val_loss 最低），但 adapter 已自动保存最优，无需手动操作
3. 输出包含无法解析的情况（~1%），判别式 BERT 无此问题
