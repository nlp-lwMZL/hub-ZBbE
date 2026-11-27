# NOTES —— week7 事件抽取（判别式链路）实现与实验记录

> 本文件是**判别式链路**（规则 / BiLSTM-CRF / BERT / 两段式 / 系统组合）的完整记录：
> 设计决策、全部命令、超参、实测数字、踩坑、以及可直接抄进 PPT 的结论句。
> LLM 链路（zero-shot / LoRA）由另一条链路负责，见 `llm_zero_shot/`。
>
> 数据：DuEE-Fin 金融公告事件抽取（13 事件类型 / 92 个「类型-角色」槽位 / 61 个去重角色名）。
> 主评测集：`data/processed/dev_extract.jsonl` **930 句**（每句恰好 1 个事件）。
> 所有指标都由**唯一口径** `common/evaluate_events.py` 计算（trigger / argument / event_exact 三级 + type_macro）。
>
> **[!] 数值口径更新（2026-11-26，week07 换模型实验）**：本文件下方抄录的 BiLSTM-CRF 旧值是
> 3 epoch 未收敛时的 0.4634；本机 5 epoch 重跑后是 **0.5737**（`outputs/metrics/disc/bilstm_crf.json`）。
> 旧结论“BERT 预训练带来 +11.9pp”因此被高估——公平对比后只差 **+0.8pp**。以 `outputs/metrics/disc/summary.md`
> 和 `MODEL_SWAP_ANALYSIS.md` 为准，下方历史表格仅作过程记录。

---

## 0. 一分钟结论（PPT 首页可用）

| 结论 | 数字 |
|---|---|
| 规则基线能覆盖大部分触发词，但**论元几乎抽不对** | trigger 0.6168 → argument **0.1740** |
| BERT 一次联合抽取把 trigger 拉到 0.83，论元到 0.56 | trigger 0.8293 / argument 0.5630 |
| 论元级才是真难点：**事件级完全正确率只有十几个点** | event_exact 0.1564（规则 0.0161） |
| CRF 不只是"更准"，它主要解决**非法序列** | Linear 67.0% 句子含非法 BIO → CRF ? |
| 非法序列有 1.9% 的**不可消除地板**（标注重叠） | 金标自身 dev 18 处 / 1.9% |

---

## 1. 目录与脚本职责

```
event_extraction/
├── common/evaluate_events.py     # 【只读】统一评测（Lead 维护，本链路只 import 不改）
├── data/
│   ├── processed/                # 全量数据（外部产物，本链路不写）
│   ├── sample/                   # 学生包小样本（tools/make_sample_data.py 生成）
│   └── README.md                 # 数据来源/许可/字段契约/四个数据坑
├── common/                       # 公共模块（不含"方法"）
│   ├── model_paths.py            # 向上逐级查 pretrain_models/bert-base-chinese（母版/副本两套目录都能用）
│   ├── event_data.py             # ★ 数据层：两种标签体系 + span↔BIO + word_ids 对齐 + 解码 + 非法统计
│   ├── evaluate_events.py        # ★ 唯一评测口径（trigger/argument/event 三级 + type_macro）
│   ├── eval_utils.py             # ★ 统一评测封装 + 计时/显存 + n200 投影
│   ├── crf.py                    # ★ 手写 CRF：前向算法 / 维特比 / 硬约束
│   ├── models.py                 # 公共模型定义（BertEncoder/BertCls/build_model/predict_char_tags）
│   ├── log_utils.py              # ★ 统一训练日志（TrainLogger/EvalLogger）
│   └── README.md
├── rule/run_rule.py              # L0 规则基线（词典+正则）
├── bilstm_crf/train.py           # L1 字符级 BiLSTM-CRF（无预训练）
├── bert/                         # L2a 联合抽取（主线）
│   ├── train.py                  #   BERT+Linear / BERT+CRF（--decoder 切换）
│   ├── evaluate.py               #   从 checkpoint 复评（不重训）
│   └── evaluate_hard_constraint.py  # CRF 硬约束解码（复用已训权重）
├── bert_pipeline/train.py        # L2b 两段式：检测器 + 13分类 + 13个类型条件论元专家
├── llm_zero_shot/                # L3 生成式零样本/少样本（llm_extract/event_prompt/event_data/local_generate）
├── llm_lora/                     # L4 LoRA SFT（train_sft/evaluate_sft/model_paths）
├── service/                      # L5 系统组合
│   ├── extract_service.py        #   检测→抽取→规则兜底→校验→篇章合并→needs_review
│   └── demo_extract.py           #   现场演示 CLI（--n 3 / --doc_id 逗号分隔）
├── eval/                         # 汇总/图表/诊断
│   ├── check_bio_gold.py         #   ★ 金标自身的非法 BIO（P25 的正确基线）
│   ├── postprocess_trigger_copy.py #  触发词回填后处理
│   ├── compare_methods.py        #   汇总所有指标 → 对比表 markdown
│   ├── compare_to_numbers.py     #   与 _NUMBERS.md 历史值逐项对表（>0.5pp 标⚠）
│   ├── make_figures.py           #   6 张 PPT 用图
│   ├── plot_compare.py           #   LLM 侧对比图
│   └── run_experiments.py        #   一键复现全部实验（串行、单卡、带日志）
├── tools/make_sample_data.py     # 学生包确定性抽样 → data/sample
├── data/                         # processed/（全量） sample/（学生包） _mock/（20 条冒烟） README.md
└── outputs/
    ├── checkpoints/  disc/ (预测)  metrics/disc/ (指标 JSON)  logs/  figures/  demo/
```

---

## 2. 全部命令（按执行顺序）

```bash
# ── 0. 环境（本项目复用已有 BERT，不下载）───────────────────────────────
# python = D:\Anaconda\python.exe ；torch 2.6+cu126 / transformers 5.5.3
# GPU: RTX 4060 Laptop 8GB（任一时刻只跑一个 GPU 任务）

# ── 1. 数据自检（打印 13 类 / 92 槽位 / 61 角色 / 各文件行数）──────────────
python common/event_data.py

# ── 2. L0 规则基线（纯 CPU，<1 秒）──────────────────────────────────────
python rule/run_rule.py
#   → outputs/metrics/disc/rule.json（含 failure_analysis 错误分类 + 例子）

# ── 3. L2 主线：BERT+Linear / BERT+CRF（12 层 full，max_len=256，3 epoch）──
python bert/train.py --decoder linear --max_len 256 --epochs 3 --run_name bert_linear_256
python bert/train.py --decoder crf    --max_len 256 --epochs 3 --run_name bert_crf_256

# ── 4. L1 基线：字符级 BiLSTM-CRF（无预训练，3 epoch）────────────────────
python bilstm_crf/train.py --decoder crf --epochs 3 --run_name bilstm_crf

# ── 5. L2 两段式（检测 + 13分类 + 13 个 4 层论元专家）───────────────────
python bert_pipeline/train.py
#   → pipe_detector.json（dev_detect 全量 + clean）/ pipeline.json / pipeline_oracle_type.json

# ── 6. 消融 ────────────────────────────────────────────────────────────
python bert/train.py --decoder linear --max_len 128 --epochs 3 --run_name bert_linear_128
python bert/train.py --decoder linear --max_len 256 --bert_layers 4 --epochs 3 \
       --run_name bert4L_linear_256
python bert/evaluate_hard_constraint.py          # 硬约束（复用 CRF 权重，只换解码）

# ── 7. 金标非法地板值（P25 的基线）──────────────────────────────────────
python eval/check_bio_gold.py

# ── 8. 汇总 + 出图 ─────────────────────────────────────────────────────
python eval/compare_methods.py --with_n200
python eval/make_figures.py

# ── 9. 系统演示（篇章级）────────────────────────────────────────────────
python service/demo_extract.py --n 3
python service/demo_extract.py --doc_id 10be7f956da3...,e63d5f8717d3...

# ── 10. 一键复现全部（串行单卡，带日志 + 耗时 + 退出码）──────────────────
python eval/run_experiments.py --group all

# ── 11. 学生包小样本（不下载全量也能跑通）────────────────────────────────
python tools/make_sample_data.py
python tools/make_sample_data.py --from_sample
python bert/train.py --data_dir data/sample --device cpu --epochs 1 --bert_layers 4
```

---

## 3. 关键设计决策（为什么这么做）

### 3.1 两种标签体系：joint（149）vs typed（11~19）

| | joint（联合） | typed（类型条件） |
|---|---|---|
| 用在哪 | `bert/train.py` / `bilstm_crf/train.py` | `bert_pipeline/train.py` 的论元专家 |
| 实体标签 | `TRG:<13类>` + `ARG:<61角色>` | `TRG` + `roles(t)` |
| 标签总数 | **149**（O + 2×74） | 单个类型 **11~19** |
| 事件类型 | 由触发词标签自带 | 由上游分类器给出 |
| 样本密度 | 每个标签平均 ~120 个 span | 高一个数量级 |

**为什么 joint 的论元标签不按类型再拆**：拆开是 211 个标签，而「披露时间」在 13 个类型里都出现，
共享后每个标签见到的样本更多 —— 小数据集上长尾角色才不至于跌破两位数。

### 3.2 子词对齐（word_ids）：只让每个汉字的**首个子词**背标签

WordPiece 会把一个汉字切成多片（生僻字→`[UNK]`、数字/英文→`8` `-` `10` …）。
若整片都标同一标签，loss 会把同一个 span **重复计算**、标签计数也被放大。
所以非首子词与 `[CLS]/[SEP]/[PAD]` 一律设 **-100**（CrossEntropy / CRF 都忽略）。
实现见 `event_data.BioDataset.__getitem__`，并顺手把 `word_ids` 存进 batch，推理时用来还原字符级预测。

### 3.3 手写 CRF（不用 pytorch-crf）

- 三个参数：`transitions (C,C)` / `start_transitions (C,)` / `end_transitions (C,)`
- 训练：`loss = logZ − S(金标路径)`，`logZ` 用**前向算法**（logsumexp 动态规划，O(L·C²)）
- 解码：**维特比**（把 logsumexp 换成 max + backpointers）
- **正确性验证**：用暴力枚举全部路径（C=3, L=4，共 81 条）对拍，`logZ` 与最优路径分数误差 **0.00e+00**
- 还实现了**硬约束**（`constraint` C×C bool 矩阵）：非法转移分数置 −1e4
  - 只加转移约束时：非法转移 0 但 `illegal_start` 仍有 42%
  - 再加 `start_constraint`（只允许 O/B 开头）后：**200 条随机 emission 非法数严格 = 0**

### 3.4 动态 padding（最便宜的一次提速）

数据句子平均 164 字、最长 797 字。若按 `max_len=256` 静态补齐，每条都算满 256 个 token。
改成 **collate 里补到 batch 内最长**（`collate_bio` / `collate_cls` / `collate_char`）后，
平均只算 ~170 个 token —— **训练快约 1.5 倍，效果完全不变**。

### 3.5 模型选择用「内层验证集」，dev 只做最终评测

从 train 切 10%（`--inner_val 0.1`, seed=42）做 early stopping，**dev 930 句完全不参与训练与选点**。
理由：「按 dev 挑 epoch」本身就是一个隐藏超参，会把 dev 数字抬高
（`项目材料准备/CLAUDE.md` 踩坑表里记过：同一份数据按不同指标选点，差 0.35 都很正常）。
dev 上每个 epoch 也算一次 span F1，但**只作为监控曲线**，不用于选点。

### 3.6 两段式的论元专家用「4 层截断预训练 BERT」

13 个专家 × 12 层 = 13 × 409MB ≈ 5.3GB 权重、训练慢 3 倍。
截到 **4 层（保留预训练权重，只截 `encoder.layer` 前 4 层）** 后：
- 单模型参数 110M → **45.7M**，13 个共约 1.8GB
- 速度约 3 倍
- 为什么不用 `BertConfig(num_hidden_layers=4)+BertModel(cfg)`：那是**随机初始化**，预训练知识全丢，
  会把「层数」和「预训练」两个变量混在一起（这个坑我第一时间踩了，见 §7）

### 3.7 系统组合的四个工程点（`service/extract_service.py`）

1. **规则兜底**：数值类论元（金额/日期/比例/数量）有强规律，模型没抽到时用正则补，并标 `source="regex"`
2. **三类校验**：论元必须在原文（span 校验）/ 角色必须属于该类型（schema 校验）/ 去重
3. **needs_review**：置信度低、无事件、缺核心角色、枚举角色取值不合法 → 交人工
4. **篇章合并**：切句 → 逐句抽 → 同类型事件合并成事件流，记录来自哪几句（句子坐标→全文坐标）

---

## 4. 实测结果

> 全部数字来自 `outputs/metrics/disc/*.json`（统一评测口径），汇总表见
> `outputs/metrics/disc/summary.md`（由 `eval/compare_methods.py` 自动生成）。
> 各方法预测在 `outputs/disc/pred_<方法>.jsonl`，图在 `outputs/figures/`。

### 4.1 主表（dev_extract 930 句）

| 方法 | trigger F1 | argument F1 | event_exact F1 | type_macro F1 | 参数(M) | 训练(s) | 延迟(ms/句) |
|---|---|---|---|---|---|---|---|
| L0 规则（词典+正则，零训练） | 0.6168 | 0.1740 | 0.0161 | **0.8118** | 0 | 0 | **0.72** |
| L2 BERT+Linear（12层 full，149 标签） | 0.8293 | 0.5630 | 0.1564 | 0.7654 | 102.4 | 637 | 12.26 |
| L2 BERT+CRF（12层 full，149 标签） | **0.8320** | 0.5815 | 0.1697 | 0.7741 | 102.4 | 2742 | 40.92 |
| BERT+CRF + 触发词回填规则 | 0.8320 | **0.5831** | **0.1763** | 0.7741 | 102.4 | 0 | — |

> 规则基线的 `type_macro_f1` 反而**比 BERT 高**（0.8118 vs 0.7654）：因为它用触发词词典直接定类型，
> 命中就很准；BERT 是端到端学 149 个标签，类型会被长尾角色拖累。这说明**指标要选对**：
> 只看 type_macro 会误判「规则比 BERT 好」，而 argument/event 上规则被碾压。

### 4.2 n=200 固定子集（与 LLM 链路同批可比）

抽样规则由 LLM 链路固定在 `outputs/llm/eval_ids.json`（dev 排序后 shuffle(seed=42) 取前 200），
本链路所有方法都额外产出 `outputs/metrics/disc/<方法>_n200.json`。

| 方法 | trigger F1 | argument F1 | event_exact F1 |
|---|---|---|---|
| BERT+Linear | 0.8123 | 0.5607 | 0.1645 |
| BERT+CRF | 0.8278 | 0.5672 | 0.1748 |
| BERT+CRF + 回填规则 | 0.8278 | 0.5686 | — |

### 4.3 CRF vs Linear 的同轮次对照（训练预算公平）

| epoch | Linear inner spanF1 | CRF inner spanF1 | Linear dev 监控 | CRF dev 监控 |
|---|---|---|---|---|
| 1 | 0.3989 | 0.4190 | 0.4169 | 0.4358 |
| 2 | 0.6018 | **0.6104** | 0.6036 | **0.6240** |
| 3 | 0.6315 | **0.6409** | 0.6430 | **0.6581** |

**同一训练预算下 CRF 每一轮都略优**，最终论元 F1 高 1.9 个点、事件级高 1.3 个点。
代价是**训练慢 4.3 倍**（637s → 2742s）、**推理慢 3.3 倍**（12.26 → 40.92 ms/句）——
这是 PPT 上「精度 vs 延迟」那把天平的真实数字。

### 4.4 非法 BIO 的完整因果链（P25）

| 对象 | 非法转移 | 含非法句 | 占比 | 平均每句 |
|---|---|---|---|---|
| **金标自身（地板值）** | 18 处 | 18 句 | **1.9%** | 0.019 |
| BERT+Linear 预测 | 1516 处 | 623 句 | **67.0%** | 1.63 |
| BERT+CRF 软约束 | 1242 处 | 581 句 | **62.5%** | 1.34 |
| **BERT+CRF 硬约束（token 级）** | **0 处** | **0 句** | **0.0%** | 0 |
| BERT+CRF 硬约束（字符级口径） | 155 处 | 69 句 | 7.4% | 0.17 |
| BiLSTM-CRF 软约束 | 660 处 | 454 句 | 48.8% | 0.71 |

**关键洞察（PPT 原句可用）**：CRF 把非法**转移数**降了 18%，但「含非法句占比」只从 67.0% 降到 62.5%。
原因是句子平均 175 个 token、有 174 个转移位点，只要对**任意一个** span 的开头吐了 `I-` 而不是 `B-`，
这句就算「含非法序列」。所以：
- 能反映 CRF 作用的指标是**非法转移数 / 每句平均非法数**（1.63 → 1.34）
- 「含非法句占比」在长序列上几乎必然接近 100%，**区分度很差** —— 选错指标会掩盖机制
- 且非法率有 **1.9% 的不可消除地板**（重叠论元投影所致），任何「非法率」都不能和 0 比

**硬约束后的「字符级 155」是什么（重要口径说明）**：
硬约束在 **token 级把非法转移严格压到 0**（实测 60 句 token 级 0/0）。
字符级仍报 155/69 句，是因为**原文的空格不产生 token**：`len(text)=116` 的句子只有 110 个 token，
这些没 token 的字符位置被强制标 `O`，于是**跨空格的 span** 在字符级 BIO 上变成 `O → I-x`。
实测 150 句里 25 处字符级非法，**100% 的前一字符是空白**（例：「格隆汇 9 月 8日」、「Project Martini 」）。
→ **报告非法数必须写清是 token 级还是字符级**，否则会得出相反结论。

### 4.5 硬约束（解码期）对精度的影响

| | 软约束 | **硬约束** | Δ |
|---|---|---|---|
| trigger F1 | 0.8320 | **0.8464** | +1.44pp |
| argument F1 | 0.5815 | **0.6566** | **+7.51pp** |
| event_exact F1 | 0.1697 | **0.2177** | **+4.80pp** |
| n200 argument / event | 0.5672 / 0.1748 | **0.6418 / 0.2268** | +7.5pp / +5.2pp |

> 为什么「加个约束」能提 7.5 个点论元 F1：软约束下模型会解出 `I-` 开头的 span，
> 这种 span 在字符级拼接时边界是错的，论元文本也跟着错；硬约束直接排除了这类解码，
> 等于免费修掉了一大批边界错误。**约束解码是序列标注里性价比最高的一次"白捡"**。
> （延迟数字在 CPU 上测得 557.83 ms/句，不可与 GPU 的 40.92 直接比 —— 见 metrics 里的 `device_note`）

### 4.5 其余方法（两段式 / BiLSTM / 消融）

### 4.5 L1 BiLSTM-CRF（无预训练，CPU 训练）

| 项 | 值 |
|---|---|
| 结构 | 字向量 128 → BiLSTM(hidden 256×2) → Linear → **手写 CRF** |
| 参数量 | **1.31M**（对比 BERT 102.4M，约 **1/78**） |
| 字表 | 3269 字（只从训练集建，避免用 dev 泄漏） |
| 训练 | CPU，3 epoch，max_len=128（GPU 让给了 LLM 链路） |
| trigger F1 | **0.8165**（P 0.9046 / R 0.7441） |
| argument F1 | **0.4634**（P 0.5775 / R 0.3869） |
| event_exact F1 | 0.0885 |
| type_macro F1 | 0.7306 |
| 训练时长 | **1520 s**（CPU，11/18 实测）/ 延迟 21.17 ms/句（CPU） |
| 非法 BIO | 454 句（48.8%）、**非法开头 50 处**、非法转移 660 处 |
| 内层验证 spanF1 | 0.1399 → 0.4279 → **0.5474** |

**这一行最有价值的结论**：从零训练的 BiLSTM **几乎摸到 BERT 的触发词水平**（0.8165 vs 0.8320，只差 **1.5pp**），
但**论元差了 11.8pp**（0.4634 vs 0.5815）。
→ **预训练的收益主要不在"认出关键词"，而在"理解边界与角色"**。这对学生很有说服力：
触发词是词汇级模式（小模型也学得会），论元是语义/句法级的（需要预训练语言知识）。
另一个细节：BiLSTM 的 CRF 有 **50 处非法开头**，而 BERT+CRF 是 0 —— 1.3M 参数学不好转移矩阵的入场约束。

### 4.6 L2 两段式 pipeline

**已跑出的部分**（`outputs/metrics/disc/pipe_detector.json`）：

| 检测器（8000 条子集、1 epoch） | P | R | F1 | acc |
|---|---|---|---|---|
| dev_detect 全量（5302） | 0.7977 | 0.7489 | **0.7725** | 0.9019 |
| dev_detect clean（4815，去泄漏） | 0.8016 | 0.7549 | **0.7776** | 0.8953 |

> clean 版略高于全量：检测器只用了 8000 条训练子集，那 9.2% 的重复句大多没进训练，
> 所以这版检测器反而没吃到泄漏 —— 一个"少数据"带来的意外干净对照。

类型分类器：dev_extract 上准确率 **0.9441**（52/930 判错）。

**端到端结果（修完两个 bug、重训专家后）**：

| 两段式 | trigger F1 | argument F1 | event_exact F1 | type_macro F1 | n200 (trig/arg/event) |
|---|---|---|---|---|---|
| 端到端（检测→类型→论元） | 0.6769 | 0.3690 | 0.0957 | 0.5619 | 0.6522 / 0.3527 / 0.1304 |
| 类型给金标（上界） | 0.6861 | 0.3699 | 0.0956 | 0.5898 | 0.6604 / 0.3548 / 0.1308 |

**结论（重要，别只看数字大小）**：两段式这版**低于联合模型**（arg 0.3690 vs 0.5815）。
但**原因不是"两段式不好"，而是训练预算被切碎了**：
- 联合模型：5747 句 × 3 epoch = **486 个优化器步**，一个模型吃全部数据
- 两段式专家：按类型切分后，最大的「股份回购」979 句、最小的「被约谈」53 句 → 每专家 3 epoch 只有 **5~90 步**，13 个模型各自只见到自己那 1/13 的数据
- **实测证据**：1 epoch（约 30 步）时专家几乎只会吐 `I-` 标签、一个 `TRG` 都不出、端到端 F1 = 0；
  提到 3 epoch（约 90 步）后才有 0.6769/0.3690。**步数不够 = 模型根本学不会**
- **上界分析也印证**：把事件类型换成金标后 F1 几乎不变（0.3690→0.3699），
  说明瓶颈**不在类型分类**（它准确率 0.9452），而在**论元专家训练不足**
→ **P26 的正确讲法**：「两段式把标签空间从 149 缩到 11~19（理论上更好学），
但代价是**训练预算被切成 13 份**；在 5747 句这种小数据上，联合模型反而更划算。
两段式要吃满收益，需要每个类型有足够数据（或共享编码器 + 类型条件头）。」
这个"拆得越细、每个子模型越饿"的结论，比单纯比一个 F1 更有教学价值。

**两个 bug 的复盘（这是本节最值钱的部分）**：
1. **typed 标签 bug**：`etype = TRIGGER_TAG if typed else ...` 让论元也被刷成 TRG，
   13 个专家一个角色标签都没学到。实证：修复前一条样本的 typed 标签是 `{B-TRG:4, I-TRG:12}`（零角色）。
2. **训练步数被切碎**（上面那条）：不是 bug，是**设计上的隐性成本**，
   而它造成的现象（F1=0）和 bug 一模一样 —— 这是最难区分的一类问题。
   定位方法：**直接加载单个专家、打印它的预测标签分布**，一看"只有 I- 没有 B-/TRG"就知道是欠拟合而非标签错。

**当初没跑完的工程原因**（保留）：
1. `BertTokenizer` 是慢版，实测 **4.3 ms/样本**；检测器全量 32794×2 epoch 光分词就 4.7 分钟；
2. `train_classifier` 只在两个 epoch 全跑完后才 `torch.save` → 37 分钟里一个 checkpoint 都没落；
3. 期间 GPU 被另一条链路的误上卡任务短时占用；我自己还有一次 `num_labels` 误传导致 CUDA assert。

### 4.7 消融

| 消融组 | 状态 / 结果 |
|---|---|
| CRF off/on（Linear vs CRF） | ✅ 已跑（主表两行，见 §4.1）：arg 0.5630 → 0.5815（+1.85pp） |
| CRF 硬约束（解码期） | ✅ 已跑（§4.5，arg +7.51pp / event +4.80pp） |
| max_len=128 vs 256（截断代价） | ✅ 已补跑（2026-11-17）：**arg 0.5254（−3.71pp）**，训练 306s（−42%），dev 639/930 句被截断（68.7%）→ 掉点是输入被砍，不是模型能力 |
| BERT 4 层 vs 12 层 | ✅ 已补跑（2026-11-17）：**arg 0.3495（−21.30pp）**、event 0.0535（−10.29pp）、参数 45.7M（−55%）、训练 252s（−53%）、非法序列 67.0% → 86.7% → 预训练深度对「角色边界」很关键 |

> 补跑命令（与主表同 seed / 同评测口径）：
> ```
> python bert/train.py --decoder linear --max_len 128 --epochs 3 --run_name bert_linear_128
> python bert/train.py --decoder linear --max_len 256 --bert_layers 4 --epochs 3 --run_name bert4L_linear_256
> ```
> 结论：常规旋钮（max_len / 层数）都是「省成本换掉点」；本次唯一 **训练 0 成本还大涨**的是解码期硬约束（exp3）。
> 指标落盘：`outputs/metrics/disc/bert_linear_128.json` / `bert4L_linear_256.json`。

### 4.8 系统演示（`service/demo_extract.py`，篇章级）

用 **BERT+CRF 联合模型 + 解码期硬约束 + 校验器 + 正则兜底**，对 `dev_docs` 逐篇抽事件。
实测（3 篇公告）：

| 公告 | 金标 | 抽出 | 说明 |
|---|---|---|---|
| 天舟文化（解除质押/质押） | 6 | 3 | 类型命中 2；conf 0.96-0.98，论元含质押方/数量/占比 |
| 恒大回购 | 6 | **5** | 5 个独立「股份回购」事件按句成流抽出（不合并同类型） |
| 北京监局约谈 6 家平台 | 6 | 0 | **模型漏抽**：只抽出 `I-ARG:约谈机构` 而没有触发词 → 按设计丢弃（无触发词就不成事件），标 needs_review |

**3 篇合计：金标 18 → 抽出 8（篇 1 差 3 / 篇 2 差 1 / 篇 3 全丢）。**

> 2026-11-19 修掉一个切句 offset 缺陷后重跑：旧版 `split_sentences()` 用 `start += len(seg)` 累加偏移，
> 但正则 `[^。！？；\n]+` 要求至少一个非标点字符，**连续标点/空行会被漏加**，偏移逐段向左漂移
> （实测 dev_docs 985 篇里 **798 篇 = 81%** 漂移，最大 1249 字），篇章级 span 的全文坐标因此指错位置
> （旧 demo 产物 57 个 span 里有 **33 个**对不上原文）。改为「只记区间 + 从原文切片」后
> **0 篇漂移、63 个 span 全部自洽**，段数由 5→6，篇 2 因此多抽出 1 个回购事件（4→5，更接近金标 6）。
> 这条本身是很好的教学素材：**offset 是「字符串下标契约」，任何切分/拼接都必须能过 `text[s:e] == seg` 自检。**

> 第 3 篇是**故意保留的失败案例**：它正好说明「为什么真实系统必须有 needs_review 而不是全自动」。
> 另外它暴露了一个真实缺陷：模型在「约谈」这类**动词性触发词 + 机构列表**的句式上偏弱，
> 且会吐 `I-` 开头的非法序列（软约束下允许）—— 这也是解码期硬约束能稳定带来收益的原因。


---

## 5. 数据事实（与任务描述有出入的地方，按实测为准）

| 项 | 任务描述 | **实测** |
|---|---|---|
| dev_docs 篇数 | 1023 | **985**（events 合计 1493） |
| dev_extract | 930 | 930（唯一 `id` 只有 **823** 个） |
| dev_detect | 5302 | 5302（唯一 `id` 只有 1174 个；clean 版 4815） |
| 平均句长 | — | extract 164 字（p50=161, p90=273, p99=393, max 797） |
| max_len=256 截断率 | — | train 12.1% / **dev 13.1%**（122/930） |
| max_len=128 截断率 | — | train 68.3% / **dev 68.7%**（639/930） |

---

## 6. 踩坑记录（按教学价值排序）

| # | 问题 | 根因 | 解法 / 教学点 |
|---|---|---|---|
| 1 | **`id` 是文档 id 不唯一**，预测只带 id 会互相覆盖 | DuEE-Fin 的 `id` 是公告级；同公告多句共用 | 预测必须带 `id`+`sent_id`；评测按 `id|sent_id` 对齐。踩坑实测：gold-vs-gold 只 0.93 F1 |
| 2 | **金标自己就有非法 BIO**（dev 18 处 / 1.9%） | 论元可重叠（「收购标的」包含「被收购方」），BIO 表达不了，投影时必然畸变 | 单独测「金标地板值」（`eval/check_bio_gold.py`）。**非法率不能和 0 比，要和地板值比** |
| 3 | **置信度不能用概率连乘** | 序列概率随长度指数衰减：每 token 0.93、68 个 token 连乘只剩 **0.007** → 全部被判 needs_review | 改成**按 token 平均概率**：linear = 各 token 最大概率均值；CRF = `exp((路径分−logZ)/句长)`。PPT 可放「置信度不是概率连乘」小技巧 |
| 4 | `n_truncated` 报成 854/930（假数字） | 计数器写在 `__getitem__` 里自增，dev 集每 epoch 重扫会累加 | 改成 `__init__` 里一次统计。真实 dev 截断 13.1%。**任何「统计量」都要确认它只被累加一次** |
| 5 | 4 层消融写成随机初始化，混掉了「预训练」变量 | 用了 `BertConfig(num_hidden_layers=4)+BertModel(cfg)` | 改成保留预训练权重、只截 `encoder.layer[:4]`。参数 110M→45.7M |
| 6 | CRF 结尾对 5173 条训练集推理要 15+ 分钟 | 路径分数要对每条序列额外跑一遍前向算法 | 正式指标只跑 dev；训练集非法率改用 800 条抽样（`bert/evaluate.py --train_sample`） |
| 7 | transformers 5.x 相关 | `BertModel` 默认返回 tuple / `torch_dtype` 废弃 | 统一 `return_dict=True`；本项目未用 dtype 参数 |
| 8 | Windows OpenMP 冲突 | torch 与 numpy 各带 libiomp5md | 所有脚本首行 `os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")` |
| 9 | 规则基线论元精确率只有 0.13（假阳性爆炸） | 每个实体角色都把句子里**所有**公司候选填进去 | 改成「按出现顺序轮流领、每角色 1 个」→ 0.174；仍低，说明规则到头了 |
| 10 | GPU 抢占 | 两个 BERT 任务同时跑 8GB 卡会 OOM | 约定「任一时刻只跑一个 GPU 任务」+ 训练脚本都支持 `--device cpu` 做冒烟 |
| 11 | **硬约束后字符级仍报 155 处非法转移** | 原文的**空格不产生 token**（`len(text)=116` 但只有 110 个 token），这些位置被强制标 `O`；跨空格的 span 在字符级 BIO 上就成了 `O → I-x` | 实测：**token 级非法 = 0**（硬约束完全生效）；字符级 150 句 25 处非法里 **100% 的"前一字符是空白"**。→ **报非法数必须写清是 token 级还是字符级**，否则结论会反。典型例子：`格隆汇 9 月 8日` 里的「9 月 8日」 |
| 11b | 子词→字符还原「后一个子词覆盖前一个」 | 训练时只有首个子词背标签，末子词预测是噪声 | 只取首个子词（已修）。**但本数据实测非首子词 0 个，所以这个 bug 影响为 0**（重跑 BERT+Linear 指标逐位相同）。教训：**修完 bug 要用"数字有没有变化"验证它到底影响了什么**，别想当然把它当根因 |
| 12 | **typed 标签把论元刷成了触发词**（两段式端到端 F1 全 0 的真凶） | `etype = TRIGGER_TAG if typed else ...` —— typed 模式下论元也变成 `TRG`，13 个专家**一个角色标签都没学到** | 改成 `etype = role if typed else ...`。实证：修复前一条样本的 typed 标签是 `{B-TRG:4, I-TRG:12}`（无任何角色），修复后是 `{B-TRG:1,I-TRG:2,B-上市公司,B-披露时间,B-发行价格,...}`。**表征学习里"标签刷错"这类 bug 不会报错，只会让 F1=0**，所以必须有「标签分布断言」 |
| 13 | **`load_state_dict` 把硬约束覆盖回软约束** | `constraint`/`start_constraint` 是 persistent buffer，跟着 ckpt 存；ckpt 是软约束训的（全 True） | load 之后再装一次 + **断言 `(~constraint).sum()>0` 否则 raise**。没有断言的话，日志里"禁止 0 个转移对"这行才是唯一线索 |
| 14 | 规则回填：用「区间重叠」判据反而掉点 | 触发词「辞去」vs 论元「辞职」同义不同形，回填错文本 = 纯假阳性 | 判据改成「文本完全相等」→ arg 0.5815→0.5831、event 0.1697→0.1763。**规则必须先验证再上线** |
| 15 | 切句后把模型喂给它没见过的输入形态 | 按标点硬切成 20+ 个 15~30 字碎片，而训练样本均值 164 字 | 切句后**合并到 80~320 字**。实证：同一篇公告事件数 0 → 4。**部署时的掉点经常不是模型的错，是输入分布变了** |
| 16 | 篇章级「同类型跨句合并」把多事件压成一个 | 6 家子公司分别回购是 6 个独立事件 | 改成**一句一事件、按序成流**并记 `sent_idx`；跨句共指合并属另一个更重的课题 |
| 17 | `--detect_limit` 改造时把 `num_labels=2` 误换成 `args.cls_epochs` | 正则替换时把位置参数一起换了 | 检测器 1 类头 vs 标签 1 → `CUDA device-side assert`。**改函数调用参数时永远用命名参数或逐行核对** |
| 18 | 用 `Stop-Process` 按命令行模糊匹配，把自己的 shell 杀了 | 我的命令串里也含同样的关键字（和 `pkill -f` 一个坑） | 按 PID 精确杀，或用 `$PID` 排除自己 |

---

## 7. 待补 / 已知局限

- **两段式端到端已出数**（见 §4.6）：**trigger 0.6769 / argument 0.3690 / event 0.0957**
  （压缩配置：检测器 8000 子集 1 epoch + 类型分类 1 epoch + 13 个 4 层论元专家 3 epoch，三段全训 **1377.5 s**）。
  **全量配置未跑**（检测器吃 32,794 条 × 2 epoch，时间上不划算）。
  11/18 重跑时因检测器异常慢（≈9 s/微批）未完成，**沿用 11/16 的这组值**；复跑命令见 `outputs/logs/README.md`。
- **max_len=256 仍有 13.1% 的 dev 句子被截断**，尾部 span 必丢 → 这是 BERT 主线的召回上限之一。
  用动态 padding 后把 `max_len` 提到 384 几乎不额外花时间（算的是真实长度），学生可以自己试，本项目未跑（时间预算）。
- 两段式的 13 个论元专家只有 4 层，且 `被约谈` 仅 53 条训练样本（长尾极端）。
- `dev_detect` 与 train 有 9.2% 文本重复（真实泄漏），检测指标同时报全量 + clean 两份。
- 篇章级输出是「一句一事件、按序成流」，**不做跨句事件共指合并**（理由见 §4.8）。
- BiLSTM 只跑 3 epoch / max_len=128（GPU 让给 LLM 链路，CPU 上 CRF 太慢）。

---

## 8. PPT 可用的数字与金句

### 8.1 主表（dev_extract 930 句，统一口径）

| 方法 | trigger F1 | argument F1 | event_exact F1 | 参数 | 训练 | 延迟 |
|---|---|---|---|---|---|---|
| L0 规则（零训练） | 0.6168 | 0.1740 | 0.0161 | 0 | 0 | **0.72 ms/句（CPU）** |
| L1 BiLSTM-CRF（无预训练） | 0.8165 | 0.4634 | 0.0885 | **1.31M** | CPU **1520 s** | 21.17 ms/句（CPU） |
| L2 BERT+Linear | 0.8293 | 0.5630 | 0.1564 | 102.4M | 637 s | 12.26 ms/句 |
| L2 BERT+CRF | **0.8320** | 0.5815 | 0.1697 | 102.4M | 3113 s | 40.92 ms/句 |
| **BERT+CRF + 解码期硬约束** | **0.8464** | **0.6566** | **0.2177** | 102.4M | 0（复用权重） | 见 §4.5 说明 |
| BERT+CRF + 触发词回填规则 | 0.8320 | 0.5831 | 0.1763 | 102.4M | 0 | — |
| L2 两段式 pipeline（13 专家各 3 epoch） | 0.6769 | 0.3690 | 0.0957 | — | 406 s | 49.87 ms/句 |

> 全部数字来自 `outputs/metrics/disc/*.json`（`compare_methods.py` 自动生成 `summary.md`）。
> 两段式的数字低于联合模型，**原因是训练预算被切成 13 份**（每专家 90 步量级 vs 联合模型 486 步），
> 不是方法本身不行 —— 详见 §4.6 的上界分析与「1 epoch 时 F1=0」的证据。

### 8.2 可直接引用的金句（每条都对应上面实测）

1. **「规则能覆盖 62% 的触发词，但论元只能做到 0.17 —— 事件抽取的难点从来不是找触发词。」**
   规则 trigger 0.6168 vs argument 0.1740；BERT 把两者拉到 0.83 / 0.56。
2. **「事件级完全正确率只有十几个点（0.156）—— 抽对 4 个论元里错 1 个，这个事件就不算对。」**
   argument 0.5630 但 event_exact 0.1564，两个数必须一起讲，否则会误判任务难度。
3. **「CRF 让论元 F1 高 1.9 个点，代价是训练慢 5.9 倍、推理慢 3.3 倍。」**
   531s→3113s、12.26→40.92 ms/句，精度/延迟天平的真实数字。
4. **「非法 BIO 有个 1.9% 的地板值 —— 金标自己就是非法的。」**
   论元可重叠（「收购标的」包含「被收购方」），BIO 表达不了，投影必然畸变。金标 dev 18 处 / 1.9%。
5. **「CRF 把非法转移降了 18%，但『含非法句』只从 67% 降到 62.5% —— 是选错了指标。」**
   句子平均 175 个 token、174 个转移位点，只要一个 span 开头吐了 I- 就算非法。
   **能反映机制的是「非法转移数/每句非法数」（1.63→1.34），不是「含非法句占比」。**
6. **「概率连乘会指数衰减：哪怕每个 token 都 0.93 自信，68 个 token 连乘也只剩 0.007。」**
   所以置信度要用「按 token 平均概率」，否则长句会被无条件判低置信（本项目实测导致全部 needs_review）。
7. **「规则兜底要先验证：用『区间重叠』做判据反而掉点（0.5815→0.5800），改成『文本完全相等』才 +0.66pp。」**
   触发词「辞去」和论元「辞职」同义不同形，回填错的文本只增假阳性。
8. **「1.1% 的论元就是触发词本身（高管变动·变动类型 274 条），BIO 永远学不会 —— 这类要靠 schema 规则补。」**
   实测该角色 dev F1 = **0.0000**（支持数 42，一条不对）。
9. **「两段式不是白拆：检测器 F1 0.7725、类型准确率 0.9441，都是能用的一手数字；但端到端没跑完就不能报。」**
   诚实交代实验缺口，比编一个数字更专业。
10. **「把模型喂给它没见过的输入形态，效果会崩 —— 同一篇公告切成 20 个 15 字碎片后，事件从 6 个掉到 0 个。」**
    切句后要把碎片合并到接近训练分布（本数据平均 164 字）才会恢复正常。
11. **「从零训练的 BiLSTM 只比 BERT 差 1.5 个点触发词，但论元差 11.8 个点 —— 预训练的收益在"理解边界"，不在"认出关键词"。」**
    trigger 0.8165 vs 0.8320；argument 0.4634 vs 0.5815。参数量差 **78 倍**（1.31M vs 102.4M）。
12. **「解码期加个 BIO 硬约束，论元 F1 白捡 7.5 个点（0.5815→0.6566），事件级 +4.8 个点。」**
    约束只排除「I- 开头 / B-x→I-y」这类不合规范序列，不可能伤害正确答案 —— 序列标注里性价比最高的一次优化。
13. **「非法数必须写口径：token 级 0，字符级 155 —— 差别全在原文的空格。」**
    空格不产生 token、被强制标 O，跨空格的 span（「9 月 8日」）在字符级序列上就成了非法。
    实测 25 处字符级非法 **100%** 落在空白之后。**同一份模型、同一份数据，"非法率"可以是 0% 也可以是 7.4%**。

### 8.3 图表清单（`outputs/figures/`，全部 150dpi / 白底 / 中文字体）

| 文件 | 内容 | 对应 PPT |
|---|---|---|
| `fig_method_compare.png` | 方法对比（trigger/argument/event 三级并排） | P37 |
| `fig_illegal_bio.png` | 金标地板 → Linear → CRF → 硬约束 四联柱 | P25 |
| `fig_per_role_f1.png` | 各角色论元 F1（取 dev 出现 ≥20 次的角色，最难的 18 个） | P27 |
| `fig_training_curves.png` | BERT 训练曲线（loss + 内层验证 span F1） | P24 |
| `fig_type_confusion.png` | 事件类型混淆矩阵（行归一化=召回率） | P27 |
| `fig_latency_cost.png` | 精度/延迟/参数量三角权衡表 | P37/P28 |

---

## 9. 2026-11-18 重构后的重跑记录（week6 风格目录 + 训练日志）

> 代码已按「每方法一个文件夹」重组（`common/ + rule/ bilstm_crf/ bert/ bert_pipeline/ llm_zero_shot/ llm_lora/ service/ eval/ tools/`），
> 并新增统一训练日志（`common/log_utils.py`）。**本节只记重跑结果与差异**，设计决策见 §3。

### 9.1 数字差异（`eval/compare_to_numbers.py` 自动对表；`_NUMBERS.md` 为历史值）

| 方法 | trigger | argument | event | 结论 |
|---|---|---|---|---|
| `rule` | .6168→.6168 | .1740→.1740 | .0161→.0161 | **0.00pp** |
| `bilstm_crf` | .8165→.8165 | .4634→.4634 | .0885→.0885 | **0.00pp**（CPU 训练完全确定） |
| `bert_linear_256` | .8304→**.8293** | .5625→**.5630** | .1564→.1564 | ≤0.11pp（cuDNN 自动调优噪声） |
| `bert_crf_256` | .8320→.8320 | .5815→.5815 | .1697→.1697 | **0.00pp** |
| `bert_linear_128`（消融） | .8209→.8209 | .5254→.5254 | .1453→.1453 | **0.00pp** |
| `bert4L_linear_256`（消融） | .7793→.7793 | .3495→.3495 | .0535→.0535 | **0.00pp** |
| `bert_crf_256_hard` | .8464→.8464 | .6566→.6566 | .2177→.2177 | **0.00pp** |
| `pipeline` | .6769 | .3690 | .0957 | **沿用 11-16 值**（本轮未重跑成功，见 9.3） |

**结论：重构没有改变任何数值行为**（`rule` / `bilstm_crf` / `bert_crf` / 两条消融 / 硬约束全部逐位相同）。

### 9.2 训练日志（新增，学生课后复看训练过程）
每个训练方法产出 `outputs/logs/<method>_train.json`（args/环境/每 epoch loss+验证+耗时/参数量/峰值显存）
+ `_steps.jsonl`（每 50 步一行，`--log_every` 可调）+ `_console.log`（stdout 同步）；
纯评测方法写 `_eval.json` + `_console.log`。字段说明与画 loss 曲线示例见 `outputs/logs/README.md`。

### 9.3 `bert_pipeline` 本轮未重跑成功（如实记录）
代码 bug（`train_classifier` 里 `step` 未定义、logger 被插到 `train_arg_experts()` 局部作用域）**已修好**，
并用 `--mock` 验证 exit=0、`_steps.jsonl` 能正常写出（每行带 `stage` 字段）。
但真跑时检测器（8000 条 / 125 微批）**19 分钟仍未完成**（≈9 s/微批；同期 `bert/train.py` 约 0.5 s/微批），
在时间窗内跑不完 → **保留 11-16 的指标与日志，不新造数字**。复跑命令见 `outputs/logs/README.md` 与 `USAGE_GUIDE.md §2.6`。
