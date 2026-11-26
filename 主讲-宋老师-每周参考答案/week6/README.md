# week6 课后作业 · 换基座实验：BERT vs 中文 RoBERTa

> 对应课堂内容：week6「数据接入与分流」第 3 节 · BERT 微调（L2 主力）
> 作业要求（课上口头布置）：**换一个中文预训练基座自己微调一遍**，和课上 `bert-base-chinese` 的结果对比，
> 说明「差多少、差在哪、值不值得换」。
>
> 本目录只覆盖「**换基座**」这一维（也就是老师这次要求的那条线）；「同一基座换微调方式（全量 / 只训分类头 / LoRA）」
> 属于可选的进阶方向，见 §6。

---

## 1. 参考答案做了什么

| 项 | 内容 |
|---|---|
| 对照基座 | `bert-base-chinese`（课上基线） |
| 实验基座 | **`hfl/chinese-roberta-wwm-ext`**（中文 RoBERTa，哈工大讯飞） |
| 任务/数据 | TNEWS 新闻标题 15 分类（train 53,360 / clean val 8,901） |
| 训练配置 | **与课上完全一致**：CLS 池化、3 epoch、batch 64、max_len 64、基座 lr 2e-5、分类头 ×5、10% warmup、类别加权、AMP、seed 42 |
| 只改了哪里 | 加载哪个基座（`--model`），其余超参一个没动 |

**为什么选这个基座**：它的 config.json 里 `model_type` 仍然是 `bert`——因为它是「**用 RoBERTa 的训练配方
（动态掩码 / 去掉 NSP / 更大 batch / 全词掩码）去训 BERT 结构**」的产物。所以它和 `bert-base-chinese`
**层数、隐层维度、词表、参数量（102.3M）完全一样**，推理成本一模一样。

> 这就是一个干净的对照实验：**结构不变、参数量不变、算力不变，只换「预训练见过什么、怎么学」**。
> 如果换成 large（24 层 1024 维）或多语基座，提升里就混进了「模型更大 / 词表更大」的因素，说不清是谁的功劳。

### 1.1 和课件 `train_bert.py` 的差别（就四处，都是「参数化」不是「改逻辑」）

| 差别 | 为什么 |
|---|---|
| 新增 `--model`（默认仍是 `bert-base-chinese`） | 作业要求「换基座」，把模型变成参数，其余一行不动 |
| 新增 `--seed 42` | 课件脚本没固定种子，两次跑出来的差距可能比「换基座」还大 |
| 评测统一换成 **clean val（8,901）** | 课件 `train_bert.py` 直接在全量 val（含 1,099 条重复）上评，所以它打印的 ~0.55 和课件对比表里的 **0.534** 不是同一口径；本作业统一到「所有方法横向比较」的那套 clean val |
| 保存逐条预测 + 默认不存权重 | 逐条预测给配对 bootstrap 用；作业只要求对比，权重不必留（要留加 `--save`） |

实测 BERT 基线在本脚本下复现为 **Macro-F1 0.5344 / Accuracy 0.5420**，与课件里那组训练技巧消融的基线
（0.5344 / 0.5420）**完全一致**，说明「只换基座」这个变量是干净的。

---

## 2. 怎么跑（三步）

### 第 1 步：下载基座到 `pretrain_models/`

放到 **`pretrain_models/chinese-roberta-wwm-ext/`**（和 `bert-base-chinese` 并列）。
只下 PyTorch 权重就够 —— 该仓库还带 `tf_model.h5` / `flax_model.msgpack`（+800MB），本项目用不到。

```powershell
# 方式一：能连 huggingface.co（约 65 秒，393 MB）
huggingface-cli download hfl/chinese-roberta-wwm-ext --local-dir pretrain_models/chinese-roberta-wwm-ext

# 方式二：国内镜像（huggingface.co 连不上时）
$env:HF_ENDPOINT = "https://hf-mirror.com"
huggingface-cli download hfl/chinese-roberta-wwm-ext --local-dir pretrain_models/chinese-roberta-wwm-ext

# 方式三：Python 片段（只下必要文件，跳过 tf/flax/onnx）
#   把下面几行存成 dl_roberta.py 再 python dl_roberta.py
from huggingface_hub import snapshot_download
snapshot_download("hfl/chinese-roberta-wwm-ext",
                  local_dir="pretrain_models/chinese-roberta-wwm-ext",
                  allow_patterns=["*.json", "*.txt", "*.bin"],
                  ignore_patterns=["*.h5", "*.msgpack", "*.onnx"])
```

下完确认目录里**真的有权重文件**（离线课堂最常见的坑就是半下载的空目录）：

```
pretrain_models/chinese-roberta-wwm-ext/
├── config.json
├── pytorch_model.bin        ← 393 MB，必须有
├── tokenizer.json / tokenizer_config.json / vocab.txt / special_tokens_map.json
└── added_tokens.json
```

### 第 2 步：放脚本

```
code/
├── bert/
│   └── train_base_model.py      ← 从本目录复制到 bert/ 下
├── common/                      ← 课件原有，不用动
├── compare_base_models.py       ← 从本目录复制到 code/ 下
└── plot_compare.py              ← 从本目录复制到 code/ 下（可选，画对比图）
```

脚本会自动向上查找项目结构（`find_code_root()`），所以**从项目根或 `code/bert/` 启动都能 import common**；
模型的查找沿用课件里的 `common/model_paths.py`（向上逐级找 `pretrain_models/<名字>`）。

### 第 3 步：跑两个基座 + 对比

```powershell
cd code

# ① BERT 基线（复现课上数字，约 14 分钟）
python bert/train_base_model.py --model bert-base-chinese

# ② 中文 RoBERTa（同样的配置，只换基座，约 14 分钟）
python bert/train_base_model.py --model chinese-roberta-wwm-ext

# ③ 对比（含配对 bootstrap 检验，几秒）
python compare_base_models.py

# ④ 画对比图（可选）
python plot_compare.py
```

冒烟（几十秒，不训练完、只看通不通）：

```powershell
python bert/train_base_model.py --model chinese-roberta-wwm-ext --max_steps 20 --max_eval 200
```

**默认不保存训练后的权重**（作业只要求对比）。要保存加 `--save`，产物在 `code/outputs/base_models/`。

---

## 3. 结果对比

**口径完全一致**：同一份 train（53,360 条）、同一批 clean val（8,901 条）、同一套超参、同一个 seed，
只有「加载哪个基座」不同。两个基座参数量都是 **102.3M**，训练耗时也几乎一样（831s vs 824s）。

| 基座 | 参数量 | 最优 epoch | Accuracy | **Macro-F1** | 训练耗时 | ΔF1 |
|---|---|---|---|---|---|---|
| `bert-base-chinese` | 102.3M | 3 | 0.5420 | **0.5344** | 831s（13.9 min） | — |
| `chinese-roberta-wwm-ext` | 102.3M | 2 | 0.5490 | **0.5444** | 824s（13.7 min） | **+0.0100** |

逐 epoch（看收敛速度，别只看最后一轮）：

| epoch | BERT Macro-F1 | RoBERTa Macro-F1 |
|---|---|---|
| 1 | 0.5135 | **0.5226**（+0.9） |
| 2 | 0.5343 | **0.5444**（+1.0）← RoBERTa 最优 |
| 3 | **0.5344** ← BERT 最优 | 0.5435（略掉） |

显著性检验（配对 bootstrap 2,000 次，同一批样本逐条对比）：

- ΔMacro-F1 = **+0.0100**，95% 区间 **[+0.0019, +0.0178]** → 区间不跨 0，**差异显著，但只是勉强显著**
  （下界离 0 只有 0.002）
- 两模型预测相同的样本 7,114/8,901（79.9%）；只有 BERT 对 545 条、只有 RoBERTa 对 608 条，
  **净变化只有 +63 条**（占全部样本 0.7%）——1 个点的 Macro-F1 就是这么几十条样本翻出来的

各类别 F1 变化（正 = RoBERTa 更好）：

| 类别 | BERT | RoBERTa | Δ | 类别 | BERT | RoBERTa | Δ |
|---|---|---|---|---|---|---|---|
| 证券 | 0.4129 | 0.4658 | **+0.0529** | 故事 | 0.5086 | 0.5132 | +0.0046 |
| 财经 | 0.4524 | 0.4871 | **+0.0347** | 旅游 | 0.4898 | 0.4914 | +0.0016 |
| 农业 | 0.5102 | 0.5397 | **+0.0295** | 娱乐 | 0.5865 | 0.5873 | +0.0008 |
| 体育 | 0.7118 | 0.7275 | +0.0157 | 电竞 | 0.6480 | 0.6487 | +0.0007 |
| 汽车 | 0.6676 | 0.6766 | +0.0090 | 科技 | 0.4758 | 0.4715 | −0.0043 |
| 军事 | 0.5543 | 0.5625 | +0.0082 | 文化 | 0.4755 | 0.4509 | **−0.0246** |
| 教育 | 0.5544 | 0.5621 | +0.0077 | | | | |
| 房产 | 0.5691 | 0.5767 | +0.0076 | | | | |
| 国际 | 0.3994 | 0.4056 | +0.0062 | | | | |

![换基座对比](outputs/compare_base_models.png)

完整报告见 [`outputs/compare_base_models.md`](outputs/compare_base_models.md)（含 bootstrap 区间、翻转样本数、逐类表），
表格版 [`outputs/compare_base_models.csv`](outputs/compare_base_models.csv)。

---

## 4. 怎么解读（写作业时可以直接用）

### 4.1 结论一句话

**换基座能涨约 1 个点 Macro-F1（0.5344 → 0.5444），而且几乎零工程成本**——参数量、显存、训练时间都不变，
代码只改「加载哪个模型」一行，不用重新标数据。这是所有提分手段里**性价比最高的一档**。

### 4.2 但别把「1 个点」当成定论

- 95% 区间是 [+0.0019, +0.0178]，下界贴着 0；净变化的正确样本只有 **63 条**（7,114 条预测完全相同）。
  换句话说：**方向可信，幅度不可信**。报告中应写「+0.2 ~ +1.8 点」而不是「+1.0 点」。
- 这也是为什么别把单次结果的小数点后第三位当真：同 seed ≠ 逐位可复现（cuDNN/AMP 非确定性），
  更稳的做法是进阶里的「多 seed 平均 + 报区间」。

### 4.3 收益集中在「样本最少的类」，而不是全面提升

涨得最多的三类是 **证券（+5.3）、财经（+3.5）、农业（+3.0）**，恰好是课上讲的
「**样本最少（证券仅 257 条，最多类 5,955 条，差 23 倍）+ 口径最重叠（财经↔证券）**」的重灾区。
而 **文化 −2.5、科技 −0.4** 反而掉了——这两类的边界本来就模糊（故事/文化/娱乐互相渗透），
换基座没有提供额外信息，表征变化甚至带来轻微退化。

> 教学结论：**基座带来的增益是「语言知识」层面的**（见过更多财经/证券语料，词义区分更细），
> 它补不了「标签口径本身有歧义」这个根本瓶颈。所以：
> - 换基座：对**长尾类**最值钱，是「少样本类救火」的好办法；
> - 但想让整体再上一个台阶，**得回到数据和标签口径**，不是继续换模型。

### 4.4 和课上其它手段放一起看

| 手段 | 工作量 | 效果（课件实测口径） | 说明 |
|---|---|---|---|
| 类别加权（从无到有） | 极低 | 证券类 Recall **+15pp** | 尾部类从「几乎全错」到「能中一部分」；宏观 F1 的增量课件未单独量化 |
| Label Smoothing 0.1 | 低 | ΔMacro-F1 **−0.7** | 15 类软标签把信号稀释了 |
| Focal Loss γ=2 | 低 | ΔMacro-F1 **±0** | 难样本主要来自口径重叠，不是「学不会」 |
| FGM 对抗训练 | 中（训练 ×1.5） | ΔMacro-F1 **+0.02** | 几乎不动 |
| **换基座（本作业）** | **极低** | **ΔMacro-F1 +1.0** | 同参数、同耗时，只换权重 |
| 训练数据 15% → 100% | 高（要标数据） | ΔMacro-F1 **+1.9（BERT）/ +4.9（TF-IDF）** | 最有效，但成本最高 |

**落地顺序建议**：先补数据/理口径 → 再换基座 → 最后才调 loss 和对抗训练这类小旋钮。

### 4.5 两个反常识的细节（可以写进作业加分项）

1. **RoBERTa 在 epoch 2 就到顶了，epoch 3 反而略掉**（0.5444 → 0.5435）；BERT 则是 epoch 3 最好。
   预训练更强的模型收敛更快、也更早进入「过拟合小幅回退」的区间——**所以必须按验证集选 epoch，不能取最后一轮**。
2. **提升几乎不花时间**：831s vs 824s。因为结构/参数量完全一样——这正是选这个基座做对照的意义：
   如果拿 large（24 层）比，涨的点数里就混进了「模型更大」的因素，说不清是「预训练更好」还是「参数更多」。

---

## 5. 常见坑

| 坑 | 现象 | 原因/解法 |
|---|---|---|
| 换基座忘了换 tokenizer | 效果暴跌或直接报错 | 必须 `AutoTokenizer.from_pretrained(同一个目录)`；不同基座词表不同，混用等于把字编码错位 |
| 在**全量** val 上评 | 数字虚高：课里 TF-IDF 0.541→0.509、BERT 0.551→0.534（虚高幅度还因方法而异，破坏横向可比） | 课件统一用 clean val（去掉 1,099 条与训练集重复的样本），换基座也必须同一口径 |
| 只跑一次就下结论 | 差 0.01 就写「RoBERTa 更强」 | 用 `compare_base_models.py` 的配对 bootstrap 看 95% 区间；同 seed ≠ 逐位可复现 |
| 拿 large 基座和 base 比 | 提升里混了参数量因素 | 要么都 base，要么明确标注「提升来自模型规模」 |
| 8GB 显存跑 large | OOM | 降 batch size / 降 max_length，或用梯度累积；base 模型 bs 64 只占约 4.2GB |
| 找不到模型 | `FileNotFoundError: 未找到本地模型 ...` | 脚本会把尝试过的所有路径打印出来；确认目录名，或直接 `--model` 传绝对路径 |

---

## 6. 可选进阶（想拿高分可以做）

- **换更大的基座**：`hfl/chinese-roberta-wwm-ext-large`（24 层 / 1024 维 / 约 3× 计算），对比「模型变大」与「预训练变好」的收益差
- **换其他中文基座**：`hfl/chinese-macbert-base`（MacBERT）、`nghuyong/ernie-3.0-base-zh`（ERNIE 3.0）、`IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment`（领域基座）
- **池化 × 基座交互**：`--pool mean` / `--pool max`，看「更好的基座是否让池化选择变得不重要」
- **多 seed 平均**：`--seed 1/2/3` 各跑一遍，报均值±标准差（比单次结果可信得多）
- **看 per_class_f1**：哪几类提升最大？是不是正好是课上说「标签口径重叠最严重」的那几类（科技/财经、国际/军事）？
- **换微调方式（老师说的「各种方式」）**：同一个基座下对比「全量微调 / 冻结编码器只训分类头 / LoRA」，
  和「换基座」放在一起看——哪个更划算？课件 `code/llm_lora/` 里有 Qwen 的 LoRA 参考实现，BERT 侧可直接用 `peft` 的 `LoraConfig(target_modules=["query","value"])`

---

## 7. 目录说明

```
base_model_exp/
├── README.md                    ← 本文件
├── train_base_model.py          ← 参考答案：训练脚本（复制到 code/bert/ 下跑）
├── compare_base_models.py       ← 参考答案：对比脚本（复制到 code/ 下跑）
├── plot_compare.py              ← 参考答案：画图脚本（可选）
└── outputs/                     ← 本机实跑产物
    ├── bert-base-chinese_log.json        ← 逐 epoch 指标 + 逐类 F1
    ├── bert-base-chinese_pred.json       ← 逐条预测（配对检验用）
    ├── chinese-roberta-wwm-ext_log.json
    ├── chinese-roberta-wwm-ext_pred.json
    ├── compare_base_models.md            ← 对比报告（Markdown）
    ├── compare_base_models.csv           ← 对比表格（Excel 可直接打开）
    └── compare_base_models.png           ← 对比图
```

> 训练后的模型权重**没有**放进交付包（两个 base 模型各 400MB，作业只需要对比数字）。
> 需要复现的话按上面三步跑一遍即可。
