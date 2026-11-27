# 人民日报 NER 验证集结果汇总（work07）

一句话结论：**这个任务上，序列标注（BERT+CRF 0.9601）明显优于生成式（SFT 0.8451、DeepSeek 0.7084）；
给 few-shot 示例没有帮助，一个示例反而会让 DeepSeek 显著变差。**

本目录是**自包含的交付快照**：代码、数据集、验证集结果、对比报告、图都在里面（8 张 png）。
**不含任何模型文件**（预训练权重和训练出来的 checkpoint 都没复制），路径见文末。

---

## 1. 任务与数据

BIO 序列标注，三类实体：

| 缩写 | 含义 | 说明 |
|---|---|---|
| PER | 人名 | 张艺谋、邓小平 |
| ORG | 机构名 | 国防科技大学、联合国安理会 |
| LOC | 地名 | 北京、新墨西哥州 |

| split | 句数 | 实体数 | LOC | ORG | PER |
|---|---|---|---|---|---|
| train | 20,864 | 33,992 | 16,571 | 9,277 | 8,144 |
| **validation** | **2,318** | **3,819** | **1,951** | **984** | **884** |
| test | 4,636 | 7,707 | — | — | — |

句子很短：长度中位数 41 字，p99 = 149 字，最长 568 字。
LOC 占了验证集一半以上的实体（1951/3819 = 51%），所以整体 F1 基本由地名识别决定。

---

## 2. 四个方法

| 方法 | 范式 | 做法 |
|---|---|---|
| BERT+Linear | 判别式 | bert-base-chinese + 线性分类头，逐 token 7 分类（O/B-/I- × 3 类），5 epoch |
| BERT+CRF | 判别式 | 同上，输出层换成 CRF，用维特比解码保证标签序列连贯 |
| Qwen2-0.5B SFT | 生成式 | 本地 Qwen2-0.5B-Instruct + LoRA（r=8, alpha=16），输入句子输出实体 JSON，10000 条 × 2 epoch |
| DeepSeek API | 生成式 | deepseek-chat，zero-shot / 1-shot / 3-shot，输出 JSON（开启 JSON mode） |

判别式和生成式的关键差别：前者在**每个字上**做分类，天然知道实体边界；
后者把实体当文本生成，边界靠「猜」。

---

## 3. 结果

### 3.1 整体（同一批 2318 条验证句）

| 方法 | Precision | Recall | F1 | 备注 |
|---|---|---|---|---|
| BERT+Linear | 0.9550 | 0.9611 | 0.9580 | 非法 BIO 序列 44 条 |
| **BERT+CRF** | **0.9574** | **0.9629** | **0.9601** | 非法 BIO 序列 30 条 |
| Qwen2-0.5B SFT | 0.8734 | 0.8185 | 0.8451 | 格式失败 6 条，1350 ms/条 |
| DeepSeek zero-shot ※ | 0.7743 | 0.6528 | 0.7084 | 500 条子集，744 ms/条 |
| DeepSeek 1-shot ※ | 0.7241 | 0.6597 | 0.6904 | 500 条子集，708 ms/条 |
| DeepSeek 3-shot ※ | 0.7478 | 0.6736 | 0.7087 | 500 条子集，836 ms/条 |

※ DeepSeek 那三行是 500 条子集，**不要和上面三行比绝对值**（样本不同、且它没在这份数据上训练过）。
它内部三条之间可以配对比较，见 3.3。

### 3.2 逐类型 F1（2318 条全量）

| 类型 | 实体数 | BERT+Linear | BERT+CRF | Qwen2-0.5B SFT |
|---|---|---|---|---|
| PER | 884 | **0.9841** | 0.9830 | 0.9459 |
| ORG | 984 | 0.9229 | **0.9268** | 0.7869 |
| LOC | 1,934 | 0.9641 | **0.9667** | 0.8282 |

CRF 相对 SFT 的领先幅度：PER 0.0371、ORG 0.1399、LOC 0.1385。
**差距集中在机构名和地名**——人名靠字面规律就能认，机构和地名要判边界（「北京大学」里的「北京」不算 LOC，
「美国新墨西哥大学」里的「美国」是修饰语）。

### 3.3 DeepSeek 的 few-shot 效果（配对 bootstrap，2000 次重采样）

| 对比 | ΔF1 | 95% CI | 重采样为负的比例 | 判定 |
|---|---|---|---|---|
| 1-shot vs zero-shot | **−0.0181** | [−0.0340, −0.0027] | 98.9% | **显著变差** |
| 3-shot vs zero-shot | +0.0003 | [−0.0169, +0.0180] | 49.6% | 不显著 |

三条 prompt 跑的是同一批 500 句、同一顺序，所以是配对比较，不是各算一个区间再比重叠。

DeepSeek 的逐类型 F1：PER 0.8854 > ORG 0.7136 > **LOC 0.6193**。
漏报 301 次里 LOC 占 219 次，而 LOC 恰好是 BERT 的强项（0.9667）。
**它的短板和 BERT 正好相反**：API 模型强在边界清晰的专名，弱在需要句法/上下文判断的地名。

### 3.4 成本

实测（GTX 1660 Ti，只计模型 forward，不含加载权重）：

| 方法 | 单条推理耗时 | 相对 BERT+Linear |
|---|---|---|
| BERT+Linear（batch=16, len=512） | 42 ms | 1× |
| BERT+CRF（batch=16, len=512） | 51 ms | 1.2× |
| Qwen2-0.5B SFT（本地 GPU，逐条生成） | 1,350 ms | 约 32× |
| DeepSeek API | 708–836 ms | 约 17–20×（另有费用和网络抖动） |

SFT 的精度更低、成本还高 30 倍左右，这是本次实验最实用的结论。

> BERT 的耗时对序列长度敏感（attention 是 O(n²)）：上面是放开到 512 的数字，
> 训练口径 128 下大约是它的 1/3。生成式则是逐 token 自回归，长度影响更线性。

---

## 4. 五个结论

1. **序列标注 > 生成式，差距 0.1150**（0.9601 vs 0.8451）。方向明确，靠调参翻不了盘。
2. **CRF 比线性头只好 0.0021**（0.9601 vs 0.9580），非法序列从 44 降到 30 条但没归零
   —— 转移矩阵是学出来的，不是硬约束。
3. **few-shot 无效，1-shot 显著有害**（−0.0181，98.9% 重采样为负）。
   和文本分类项目上的结论一致（few-shot +2.5pt 但 p=0.50）。
4. **LOC 是分水岭**。BERT 三类都在 0.96 以上；SFT 的 LOC 掉到 0.83；DeepSeek 只有 0.62。
   地名识别的难点是嵌套（「北京大学」里的「北京」）和修饰（「美国新墨西哥大学」里的「美国」）。
5. **格式合法性是模型能力问题，不是 prompt 问题**。DeepSeek 开 JSON mode 后失败率 0.2%（1/500），
   本地 0.5B 模型在文本分类项目里 zero-shot 的解析失败率约 30%，SFT 之后降到 0.3%（6/2318）。

---

## 5. 三条口径说明（读数字前必看）

**① BERT 的截断：从 128 放开到 512 重跑过**

训练时 `max_length=128` 只能读进前 126 个字（`[CLS]`/`[SEP]` 各占一位），超长句被整句截断，
那些实体从 BERT 的分母里消失；而生成式模型看全文。为公平比较，用同一份权重、只改
`max_length=512` 重评了一遍，本目录用的是重评后的结果：

| 模型 | 128 F1 | 512 F1 | 变化 | 参与统计的实体数 |
|---|---|---|---|---|
| BERT+Linear | 0.9567 | 0.9580 | +0.0014 | 3,657 → 3,802 |
| BERT+CRF | 0.9595 | 0.9601 | +0.0006 | 3,657 → 3,802 |

涨幅几乎全在 LOC 上（+0.0024 / +0.0019），因为被 128 截掉的 159 个实体里 148 个是地名。
仍有 **17 个实体（0.4%）**看不到：`bert-base-chinese` 的位置编码上限就是 512，
而验证集最长句 568 字，这层天花板调参解决不了。

**② SFT 的主指标和 BERT 不是同一把尺子**

- BERT 用 seqeval：类型和边界都完全一致才算 TP（严格位置匹配）。
- SFT 主指标用（实体文本, 类型）多重集匹配，因为本数据集 **7.57% 的句子有重复实体**，
  用 `text.find()` 定位会把两个同名实体塌成一个，人为压低 recall。
- SFT 换成严格位置口径是 **0.7866**（对比 0.8451）。

也就是说：主表里 SFT 用的是**对它有利**的宽松口径，差距还有 0.1150；换成同口径差距更大。
别把 0.8451 当严格数字引用。

**③ DeepSeek 只有 500 条**

从 2318 条里按 `seed=42` 抽 500 条，三条 prompt 共用同一批。它内部可比，跨表不可比。

---

## 6. 文件清单

```
work07/
├── README.md                                   本文件（总结）
├── .env.example                                DeepSeek key 模板（复制成 .env 再填，真 key 没带进来）
├── src_work/                                   BERT 侧代码
│   ├── dataset.py                              CoNLL(BIO) → BERT 输入，子词标签对齐
│   ├── model.py                                BertNER（线性头）/ BertCRFNER（CRF 头）
│   ├── train.py                                训练（支持 --resume 续训）
│   ├── evaluate.py                             seqeval 评估 + 非法 BIO 序列统计
│   ├── compare_results.py                      Linear vs CRF 对比
│   ├── explore_data.py                         数据统计 + 分布图
│   ├── plot_train_curve.py                     训练曲线
│   └── README.md                               这一侧的实现说明和踩坑
├── src_llm_work/                               LLM 侧代码
│   ├── common.py                               公共层：路径、BIO↔span、指标、JSON 解析、.env 读取
│   ├── train_sft_work.py                       Qwen2-0.5B + LoRA 训练
│   ├── evaluate_sft_work.py                    SFT 全量评估
│   ├── llm_ner_work.py                         DeepSeek zero / one / few-shot
│   ├── compare_results_work.py                 四方法对比（出表 + 图）
│   └── README.md                               这一侧的三个必改点和用法
├── data/peoples_daily/                         数据集（35 MB，和原目录 md5 一致）
│   ├── train.json                              20,864 句
│   ├── validation.json                         2,318 句
│   ├── test.json                               4,636 句
│   └── label_names.json                        7 个标签
├── results/
│   ├── 01_bert_linear_validation.json          BERT+Linear 结果（max_length=512）
│   ├── 02_bert_crf_validation.json             BERT+CRF 结果（max_length=512）
│   ├── 03_qwen2-0.5b_sft_validation.json       SFT 结果 + 2318 条逐句明细
│   ├── 04_deepseek_0shot_1shot_3shot_validation.json  DeepSeek 三条 prompt 结果 + 1500 条明细
│   ├── 05_train_sft_epochs.json                SFT 逐 epoch 的 train/val loss
│   └── 06_summary_metrics.csv                  四方法 × 逐类型的整合指标表（25 行，可直接开 Excel）
├── reports/
│   ├── comparison_4methods_validation.md       四方法对比报告（含截断影响、bootstrap、逐类型与读法）
│   ├── comparison_4methods_validation.json     同上，机器可读版
│   └── comparison_bert_linear_vs_crf_validation.md  BERT 侧 Linear vs CRF 对比
└── figures/                                    8 张图，逐张说明见下表
    ├── comparison_4methods_validation.png
    ├── comparison_bert_linear_vs_crf_validation.png
    ├── comparison_bert_linear_vs_crf_validation_len128.png
    ├── comparison_bert_linear_vs_crf_test.png
    ├── train_curve_bert.png
    ├── data_entity_distribution.png
    ├── data_entity_length_distribution.png
    └── data_text_length_distribution.png
```

### 图索引

| 文件 | 内容 | 口径 / 说明 |
|---|---|---|
| `comparison_4methods_validation.png` | 四方法逐类型 F1 + 整体指标 | validation 2318 条，BERT 用 512 口径 |
| `comparison_bert_linear_vs_crf_validation.png` | Linear vs CRF：逐类型 + 整体 + 非法序列 | validation，512 口径（正文用的就是这张） |
| `comparison_bert_linear_vs_crf_validation_len128.png` | 同上，但用训练时的 128 口径 | 拿来和上一张对照，看截断吃掉了多少 |
| `comparison_bert_linear_vs_crf_test.png` | Linear vs CRF 在 test 集上的对比 | **128 口径**，test 没放开重跑，仅作参考 |
| `train_curve_bert.png` | 5 个 epoch 的 val F1 曲线 + train/val loss（log 轴） | 虚线是「前 3 轮从头训练 / 后 2 轮续训」的分界；训练时口径 |
| `data_entity_distribution.png` | 三个 split 的实体类型分布（按每条样本平均实体数归一化） | 看 train/val/test 是否同分布 |
| `data_entity_length_distribution.png` | 训练集实体长度分布（按类型） | LOC 集中在 2 字，ORG 长尾到 10+ 字 |
| `data_text_length_distribution.png` | 训练集文本长度分布，红虚线标 `max_length=128` 截断线（超出 1.7%） | 解释为什么要放开截断 |

> 老数据集（cluener）的 3 张图在 `outputs/figures/`，不属于这份作业，没有放进来。

**代码能不能直接在 work07 里跑？** 数据路径是对的（`ROOT/data/peoples_daily`），模型路径不是：

`src_work/{train,evaluate}.py` 的 `BERT_PATH` 和 `src_llm_work/common.py` 的 `MODEL_PATH`
都写成 `ROOT.parent / "pretrain_models" / ...`。在原目录里 `ROOT` = `序列标注项目/`，
所以指向 `D:/project/pretrain_models`（正确）；搬到 `work07/` 之后 `ROOT.parent` 变成
`序列标注项目/`，会解析到 `序列标注项目/pretrain_models`（不存在）。

两种处理方式，任选其一：

- 回原目录 `D:\project\序列标注项目\` 跑（推荐，路径全对）；
- 或者把这三处的 `ROOT.parent` 改成 `ROOT.parent.parent`，就地跑。

另外，`train.py` 可以**从零训练**（只需要预训练 BERT，不需要现成 checkpoint）；
`evaluate.py` 必须先有 `outputs_work/checkpoints/work_best_{linear,crf}.pt`（409 MB，未复制）。
DeepSeek 那部分把 `.env.example` 复制成 `.env` 填上 key 就能跑。

**明细数据在哪**：`03_*.json` 和 `04_*.json` 里各有 `detail` 字段，逐句记录了
`text`（原文）、`gold`（标注）、`pred`（预测）、`raw_output`（模型原样输出）、`json_ok`（是否合法 JSON）。
想看模型具体错在哪，从这里翻。

**文件来源**（本目录是整理后的副本，原文件都在 `outputs_work/` 和 `outputs_llm_work/`）：

| 本目录文件 | 原始路径 |
|---|---|
| src_work/ | `D:\project\序列标注项目\src_work\`（整目录复制，去掉 `__pycache__`） |
| src_llm_work/ | `D:\project\序列标注项目\src_llm_work\`（同上） |
| data/peoples_daily/ | `D:\project\序列标注项目\data\peoples_daily\`（4 个文件，md5 已核对一致） |
| results/01 | `outputs_work/logs/work_eval_linear_validation_full.json` |
| results/02 | `outputs_work/logs/work_eval_crf_validation_full.json` |
| results/03 | `outputs_llm_work/logs/eval_sft_validation.json` |
| results/04 | `outputs_llm_work/logs/eval_llm_0shot_1shot_3shot_validation.json` |
| results/05 | `outputs_llm_work/logs/train_sft.json` |
| results/06 | 本目录新生成（读上面 4 份 JSON 汇总） |
| reports/comparison_4methods_* | `outputs_llm_work/compare_ner_validation.{md,json}` |
| reports/comparison_bert_* | `outputs_work/work_compare_validation_full.md` |
| figures/comparison_4methods_* | `outputs_llm_work/figures/compare_ner_validation.png` |
| figures/comparison_bert_linear_vs_crf_validation.png | `outputs_work/figures/work_model_compare_validation_full.png` |
| figures/comparison_bert_linear_vs_crf_validation_len128.png | `outputs_work/figures/work_model_compare_validation.png` |
| figures/comparison_bert_linear_vs_crf_test.png | `outputs_work/figures/work_model_compare_test.png` |
| figures/train_curve_bert.png | `outputs_work/figures/work_train_curve.png` |
| figures/data_*.png | `outputs_work/figures/work_entity_distribution.png` 等 3 张 |

---

## 7. 复现

python 用 `C:\Users\87467\.conda\envs\py312\python.exe`（只有这个环境有 torch/seqeval/torchcrf）。

```bash
cd D:/project/序列标注项目        # 在 work07 里跑要先改模型路径，见上一节

# BERT 侧（权重已训练好，直接评估；--tag full 才不会覆盖 128 口径的基线）
python src_work/evaluate.py --max_length 512 --tag full --batch_size 16
python src_work/evaluate.py --use_crf --max_length 512 --tag full --batch_size 16
python src_work/compare_results.py --split validation --tag full

# SFT 训练 + 全量评估
python src_llm_work/train_sft_work.py                 # 约 73 分钟，峰值显存 3.78 GB
python src_llm_work/evaluate_sft_work.py --n_samples -1

# DeepSeek（需要在项目根 .env 里配 DEEPSEEK_API_KEY）
python src_llm_work/llm_ner_work.py --n_shot 0,1,3 --n_samples 500

# 出对比报告和图
python src_llm_work/compare_results_work.py --bert_tag full
```

---

## 8. 本目录**不含**什么

- **模型文件**（按要求排除，一个都没复制）。需要时在这里：
  - 预训练底座：`D:\project\pretrain_models\bert-base-chinese\`、`D:\project\pretrain_models\Qwen2-0.5B-Instruct\`
  - 训练好的 BERT：`outputs_work/checkpoints/work_best_{linear,crf}.pt`（各 409 MB，epoch=5）
  - SFT LoRA adapter：`outputs_llm_work/sft_adapter/`（4.35 MB）
- **真实 API key**：只放了 `.env.example` 模板，没有把 `.env` 复制进来。
- **test 集结果**：本次只整理了 validation 的**数据**。test 集只有一张 BERT 侧的对比图
  （`figures/comparison_bert_linear_vs_crf_test.png`，128 口径），对应的 JSON 和报告没有复制；
  SFT 在 test 上的评估还没跑（全量 4636 条约需 1.7 小时）。
- **cluener 原项目的代码**（`src/`、`src_llm/`）：那是另一个数据集的项目，和这次作业无关。
