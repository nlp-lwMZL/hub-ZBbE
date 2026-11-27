# src_work —— 人民日报 NER（作业）

和主项目 `src/`（cluener2020）**完全隔离**的一套代码：数据不同、标签体系不同、输出目录不同。
两套代码互不 import，互不覆盖。

## 与 src/ 的差异

| | `src/`（主项目） | `src_work/`（本目录） |
|---|---|---|
| 数据集 | cluener2020，10 类细粒度实体 | 人民日报 NER，3 类（PER / ORG / LOC） |
| 标注格式 | span `{类型: {实体: [[起, 止]]}}` | CoNLL，`tokens` + `ner_tags`（BIO） |
| 标签数 | 21（O + 10×2） | 7（O + 3×2） |
| 需要 span→BIO 转换 | 是 | **否**，数据本身就是 BIO |
| test 集标签 | 未公开（CLUE 规则） | **有**，可以直接在 test 上评估 |
| 输出目录 | `outputs/` | `outputs_work/` |
| 权重/日志前缀 | `best_linear.pt` / `train_linear.json` | `work_best_linear.pt` / `work_train_linear.json` |

## 文件

```
src_work/
├── dataset.py          # CoNLL(BIO) → BERT 输入，word_ids() 子词对齐
├── model.py            # BertNER（线性头）+ BertCRFNER（CRF 头），结构同主项目
├── train.py            # 训练，--use_crf 切换模型，--resume 断点续训
├── evaluate.py         # seqeval entity-level F1 + 非法 BIO 序列统计
├── explore_data.py     # 数据统计 + 3 张图
├── plot_train_curve.py # 读训练日志画 F1/Loss 曲线
└── compare_results.py  # 读两份 eval 结果，出对比表/图/报告
```

## 数据

`data/peoples_daily/`（由 `src/download_data.py` 下载，本次未改动）

| split | 条数 | 实体数 | 平均文本长度 |
|---|---|---|---|
| train | 20,864 | 33,992 | 46.9 字 |
| validation | 2,318 | 3,819 | 47.4 字 |
| test | 4,636 | 7,707 | 47.3 字 |

实体分布（train）：LOC 16,571（48.7%）、ORG 9,277（27.3%）、PER 8,144（24.0%）。
原始标注里非法 BIO 转移 **0 处**，数据是干净的；超过 126 字会被 `max_length=128` 截断的占 **1.71%**。

## 用法

```bash
# 0. 环境（pytorch-crf 的模块名是 torchcrf，不是 pytorch_crf）
pip install torch transformers seqeval pytorch-crf tqdm

# 1. 数据统计 + 图 → outputs_work/figures/
python src_work/explore_data.py

# 2. 训练两个模型（各 3 epoch）
python src_work/train.py                  # BERT + Linear
python src_work/train.py --use_crf        # BERT + CRF

# 2b. 断点续训：在已有权重上再多跑 2 轮（--epochs 表示【额外】轮数）
python src_work/train.py --resume outputs_work/checkpoints/work_best_linear.pt --epochs 2
python src_work/train.py --use_crf --resume outputs_work/checkpoints/work_best_crf.pt --epochs 2

# 3. 评估（validation / test 都可以）
python src_work/evaluate.py                       # Linear，验证集
python src_work/evaluate.py --use_crf --split test

# 放开截断评估（和看全文的生成式模型对比时用，输出带 _full 后缀不覆盖基线）
python src_work/evaluate.py --max_length 512 --tag full --batch_size 16
python src_work/evaluate.py --use_crf --max_length 512 --tag full --batch_size 16

# 4. 汇总对比 → outputs_work/work_compare_{split}.md + 对比图
python src_work/compare_results.py --split validation
python src_work/plot_train_curve.py       # 训练曲线 → figures/work_train_curve.png
```

> **`max_length` 为什么重要**：训练和默认评估都用 `max_length=128`，
> 也就是只读得进前 **126** 个字（`[CLS]`/`[SEP]` 各占一位），超长句被整句截断。
> 验证集里 3819 个实体中有 **159 个**（其中 148 个是 LOC）落在这个窗口之外，
> 它们既不进 TP 也不进 FN，等于从分母里消失。要跟 SFT 这种看全文的生成式模型比，
> 必须加 `--max_length 512 --tag full` 重跑一次。

## 关键实现说明

**1. 子词对齐（`dataset.py`）**
`is_split_into_words=True` 让 `word_ids()` 的下标等于字在 `tokens` 里的下标：
`[CLS]/[SEP]/[PAD]` → `-100`；每个字的首个子词 → 真实标签；同字的后续子词 → `-100`。
实测本数据集 tokens 全是单字，但代码不依赖这个前提。

**2. 分层学习率（`train.py`）**
BERT 层 `2e-5`，分类头（含 CRF 转移矩阵）`1e-4`（`head_lr_mult=5`），
AdamW + 线性 warmup（前 10%）+ 线性衰减 + 梯度裁剪 1.0。

**3. CRF 的"合法性"**
`torchcrf.CRF` 的转移矩阵是**学出来的**，不是硬编码约束。所以 CRF 输出仍可能出现
非法 BIO（如 `B-ORG → I-LOC`），只是数量应少于线性头。想从数学上彻底杜绝，
需要给非法转移位置加惩罚（如 `-1e4`）；本作业没加，这样两种头的差异才是纯结构差异。

**4. 断点续训（`train.py --resume`）**
和从头训练有两处不同，都是刻意的：
- **学习率减半**（2e-5 → 1e-5，头 1e-4 → 5e-5）。模型已收敛到 0.95 附近，还用原来的
  学习率会把它推离最优。
- **不做 warmup**。预热是给随机初始化的分类头用的，续训时分类头早就学好了，
  重新把 LR 拉回高位再降下来纯属扰动。

另外两个细节：`best_f1` 从 checkpoint 里的历史最优起算，所以续训跑差了不会覆盖好权重；
日志是**追加**写入，前 3 轮的记录不会被冲掉。

## 实测结果（5 epoch，RTX 1660 Ti；前 3 轮从头训练 + 后 2 轮续训）

| 模型 | val F1 | val 非法序列 | test F1 | test 非法序列 |
|---|---|---|---|---|
| BERT + Linear | 0.9567（3轮 0.9481） | 44（66） | 0.9424（0.9392） | 117（114） |
| **BERT + CRF** | **0.9595**（0.9550） | **28**（33） | **0.9493**（0.9451） | **85**（98） |

逐类型（val）：LOC 0.9618 → 0.9648，ORG 0.9227 → 0.9282，PER 0.9841 → 0.9835。
CRF 的收益几乎全在 ORG 上；非法序列砍掉一半左右但没归零（原因见上面第 3 点）。

多训 2 轮后 **CRF 在 val 上的领先从 0.0069 缩到 0.0028** —— Linear 靠多训追平了
CRF 在 epoch 3 的水平。而 CRF 的 val_loss 已连涨 3 轮（0.888 → 1.080 → 1.203），
再往上加轮数只会更过拟合。

完整数字、训练曲线和报告见 `outputs_work/README.md`。

### 放开截断后的补充结果（max_length 128 → 512）

上表是训练口径（`max_length=128`）下的结果。为了和看全文的生成式模型对比，
同一份权重又用 `--max_length 512` 评了一遍：

| 模型 | 128 val F1 | 512 val F1 | 变化 | 参与统计的实体数 |
|---|---|---|---|---|
| BERT + Linear | 0.9567 | **0.9580** | +0.0014 | 3657 → 3802 |
| BERT + CRF | 0.9595 | **0.9601** | +0.0006 | 3657 → 3802 |

涨幅几乎全在 LOC 上（+0.0024 / +0.0019）—— 被 128 截掉的 159 个实体里 148 个是地名。
还剩 17 个实体看不到：`bert-base-chinese` 的位置编码上限就是 512，而验证集最长句 568 字，
这一层没法靠调参解决。结果在 `outputs_work/logs/work_eval_*_validation_full.json`。
