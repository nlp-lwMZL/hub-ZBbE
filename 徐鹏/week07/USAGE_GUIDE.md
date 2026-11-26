# USAGE_GUIDE.md — 代码调用与测试指南

> **一句话结论**：这份文档告诉你**每个脚本怎么跑、跑的时候内部做了什么、跑完看到什么**。
> 所有命令都在项目根目录（`event_extraction/`）下执行，Windows 用 `D:\Anaconda\python.exe`，
> Linux/macOS 用 `python`。没装 GPU 也能跑通：规则基线、数据脚本、统一评测、`--mock` 冒烟全是 CPU。

**先看这一节**：三条"从零到出数字"的路径，选一条开始。

| 我想…… | 用这条路径 | 大概耗时 |
|---|---|---|
| **5 分钟看到东西**（不训练） | §2.1 规则基线 + §2.12 统一评测自检 + §2.21 篇章级 demo（`--mock`） | 2~5 分钟，纯 CPU |
| **跑通全流程**（教学演示，不追精度） | §2.1 → §2.14 小样本 → §2.20 LLM `--dry-run` → §2.21 demo | 20~40 分钟，CPU 可 |
| **复现 PPT 上的完整数字** | §3 一键脚本 `eval/run_experiments.py --group all` + LLM 链路 §2.16~§2.19 | 2~4 小时，需要 GPU |

---

## 1. 环境准备

### 1.1 依赖

```bash
pip install torch transformers datasets numpy pandas matplotlib tqdm
pip install openai        # LLM 链路（API 后端）
pip install peft          # LoRA 微调（train_sft.py / evaluate_sft.py 需要）
```

| 依赖 | 用在哪 | 说明 |
|---|---|---|
| `torch` + `transformers` | `bert/train.py`、`bert_pipeline/train.py`、`llm_zero_shot/*` | 本项目已在 transformers **5.x** 上适配（`return_dict=True` / `dtype=` / `tokenize=False`） |
| `numpy` / `pandas` | `data/scripts/stats_duee10_fin.py`（读 parquet） | 其余脚本不依赖 pandas |
| `matplotlib` | `data/scripts/make_figures.py`、`eval/make_figures.py`、`eval/plot_compare.py` | 已设 `matplotlib.use("Agg")`，无显示器也能出图 |
| `openai` | `llm_zero_shot/llm_extract.py`（`--backend api`） | 只走 OpenAI 兼容接口，换厂商只改环境变量 |
| `peft` | `llm_lora/train_sft.py`、`evaluate_sft.py` | 全量微调模式不强制；缺 peft 时 LoRA 会明确报 `ImportError` |

**Windows 必看**：所有脚本顶部都有

```python
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch 与 numpy 各带一份 OpenMP，不加会 OMP: Error #15
```

### 1.2 本地模型路径（不联网下载）

```
pretrain_models/                     ← 与本项目同级的上级目录里
├── bert-base-chinese/               ← 方法目录/ 判别式链路（train_bio / train_pipeline）
└── Qwen3-0.6B/                      ← llm_zero_shot/ 生成式链路（llm_extract local / train_sft）
```

路径由 `common/model_paths.py` 与 `llm_lora/model_paths.py` **从脚本所在目录向上逐级查找**
`pretrain_models/<模型名>`，并校验目录里真有 `*.safetensors` 或 `pytorch_model.bin`
（避免拿到半下载的空目录，报一堆看不懂的错）。自检：

```bash
python common/model_paths.py
python llm_lora/model_paths.py
```

找不到时会抛 `FileNotFoundError` 并打印**所有尝试过的路径**，且**不会**偷偷联网下载——
课堂离线环境的预期行为是"明确报错"，不是"卡住 10 分钟"。

### 1.3 数据从哪来

| 数据 | 路径 | 说明 |
|---|---|---|
| **全量预处理** | `data/processed/*.jsonl` | 训练/评测真正读这份（`train_extract` 5,747 / `dev_extract` 930 / `train_detect` 32,794 / `dev_detect` 5,302 / `dev_docs` 985） |
| **完整派生数据** | `data/processed/*.jsonl` | **随包发**（约 27 MB）：train_extract 5,747 / dev_extract 930 / detect 32,794+5,302 / docs 985；**所有正式指标都跑这份** |
| **快速冒烟子集** | `data/sample/*.jsonl` | 确定性抽样（seed=42）：800 / 300 / 1,500 / 600 / 50 = 3,250 条，几秒跑通全流程；`--data_dir data/sample` 即可 |
| **冒烟数据** | `data/_mock/*.jsonl` | 手造 20 条（detect 45 条），给 `--mock` 用，**指标无意义** |
| 原始镜像 | `data/raw/` | 不参与训练，仅备查/复现（许可限制，见 `data/README.md`） |

想用**小样本**跑（`--data_dir data/sample`）：

```bash
python tools/make_sample_data.py --from_sample                 # 先校验小样本齐全 + 打印统计
python bert/train.py --data_dir data/sample --device cpu --epochs 1 --bert_layers 4
```

> `--data_dir` 留空时所有脚本默认读 `data/processed/`（全量）。
> **注意**：`data/sample` 的指标和 PPT 上的数字不可比（样本不同），只用来验证"流程通不通"。

**要全量数据 / 要复现 PPT 数字？** 下载渠道与重建步骤见 **[`data/README.md`](data/README.md) §6**：
官方千言（https://www.luge.ai/ 搜 `DuEE-Fin`，需登录）或本项目实际使用的 GitHub 镜像 `maopademiao/duee_fin`；
下载后按 §6.2 把 5 个原始文件改名放进 `data/raw/DuEE_Fin_mirror/`，再跑：

```bash
python data/scripts/build_dataset.py      # 原始 json → processed/（约 30 秒）
python data/scripts/verify_processed.py   # 45,143 个 span 全量校验，应 0 错
```


### 1.4 LLM 链路的环境变量（Key 不写进代码）

```bash
# Windows PowerShell
$env:LLM_API_KEY="sk-xxx"                    # 或 $env:DEEPSEEK_API_KEY
$env:LLM_BASE_URL="https://api.deepseek.com"  # 任意 OpenAI 兼容服务；不设则用默认 DeepSeek
$env:LLM_MODEL="deepseek-v4-flash"            # 不设则用默认
```

| 变量 | 必填 | 作用 |
|---|---|---|
| `LLM_API_KEY` / `DEEPSEEK_API_KEY` | api 后端必填 | 取第一个非空的 |
| `LLM_BASE_URL` | 否 | 换本地 vLLM / 其它兼容服务时改这里 |
| `LLM_MODEL` | 否 | 模型名 |
| `LLM_EXTRA_HEADERS` | 否 | JSON 字符串，某些兼容网关需要额外身份 header 时用 |
| `LLM_DISABLE_THINKING` | 否，默认 `1` | 推理模型先输出一大段思考会把正文吃空（`max_tokens` 小时直接返回空），默认关掉 |

---

---

## 1.5 训练日志：怎么复看「训练过程」

课上的直播没时间把训练全过程演一遍，所以**每个方法跑完都会在 `outputs/logs/` 留下日志**。
三类文件（训练类方法）：

| 文件 | 看什么 |
|---|---|
| `<method>_train.json` | `args`（超参）、`env`（python/torch/GPU）、`params`（参数量）、`epochs[]`（每轮 train_loss / 内层验证指标 / 耗时）、`total_elapsed_s`、`peak_gpu_mb`、`best_epoch` |
| `<method>_steps.jsonl` | **每 50 步一行**：`epoch/step/loss/lr/elapsed_s` —— 画 loss 曲线就看这个 |
| `<method>_console.log` | 完整 stdout（屏幕上打印过的都在） |

纯评测方法（`rule/`、`llm_zero_shot/`、`bert/evaluate*.py`、`llm_lora/evaluate_sft.py`）不训练，写
`<method>_eval.json` + `<method>_console.log`。

**调细粒度**：训练脚本都支持 `--log_every N`（默认 50）。

```python
# 画 loss 曲线（10 行，见 outputs/logs/README.md 有更完整版）
import json, matplotlib.pyplot as plt
rows = [json.loads(l) for l in open("outputs/logs/bert_crf_256_steps.jsonl", encoding="utf-8")]
plt.plot([r["elapsed_s"]/60 for r in rows], [r["loss"] for r in rows], marker="o", ms=3)
plt.xlabel("训练时间 (分钟)"); plt.ylabel("loss"); plt.grid(alpha=.3)
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
```

> ⚠ **loss 绝对值不能跨方法比**：CRF 的 loss 是 `logZ − 金标路径分数`（149 标签下能到 200+），
> 线性头是交叉熵，量纲完全不同。要比就看**曲线形状**和 `epochs[].val` 里的验证指标。

---

## 1.6 改完代码先跑这两步（语法 + 冒烟）

**① 语法自检**（改任何 `.py` 之后，秒级）：

```bash
python -m py_compile $(ls common/*.py rule/*.py bilstm_crf/*.py bert/*.py bert_pipeline/*.py \
                        llm_zero_shot/*.py llm_lora/*.py eval/*.py tools/*.py service/*.py)
```
> Windows 上一条命令太长可以分组跑，例如 `python -m py_compile bert/train.py bert/evaluate.py`。
> **这一步能抓 90% 的低级错误**（漏括号、缩进、函数改名后没跟改）；本项目的日志接入 bug 里有 3 个是它先报出来的。

**② 冒烟**（用 `data/_mock/` 的 20 条手造数据，**纯 CPU、几分钟**、不碰 GPU 也不碰真实数据）：

```bash
python rule/run_rule.py --mock
python bilstm_crf/train.py --mock --device cpu --epochs 1 --max_len 96
python bert/train.py --mock --device cpu --epochs 1 --bert_layers 4 --max_len 64
python bert_pipeline/train.py --mock --device cpu --cls_epochs 1 --arg_epochs 1 --max_len 96
python eval/make_figures.py --only latency_cost
```
> `data/_mock/` 的 6 个文件是**手造的 20 条句级样本 + 45 条检测样本**，只为打通流程；
> **上面的指标没有任何意义**，不要写进报告。它的名字带下划线表示"内部冒烟数据"，
> 是项目的一部分（`--mock` 会读它），**不要当临时文件删掉**。

---

## 2. 各步骤流程（每个脚本一节）

> 每节格式统一：**命令 → 内部流程 → 预期输出**。
> 参数以脚本 `--help` 为准（每个脚本都能 `--help`，本文只列常用参数）。

### 2.1 `rule/run_rule.py` — L0 规则基线（零训练）

**它是谁**：全项目最诚实的对照组。触发词词典从 train 自动统计 + 人工同义词，论元用正则/词典抽。

```bash
# 全量 dev 930 句（CPU，秒级）
python rule/run_rule.py

# 用 20 条手造数据自检（不依赖全量数据）
python rule/run_rule.py --mock

# 提高触发词入典门槛（只收出现 >=3 次的触发词）
python rule/run_rule.py --min_trigger_count 3

# 指定数据目录 / 输出名（输出名决定 outputs/metrics/disc/<name>.json）
python rule/run_rule.py --data_dir data/sample --name rule_sample
```

**内部流程**：

1. `build_trigger_dict()`：统计 train 里每个类型的 trigger 文本频次，留 `≥min_count` 的（去噪）；
2. `merge_trigger_dict()`：合并 `MANUAL_TRIGGERS` 人工同义词，**按词长降序**排（最长优先，「解除质押」不会被「质押」抢走）；
3. `build_company_dict()`：把 train 里公司类角色的标注值收成股票简称词典（真实工程里这份词典来自行情库）；
4. `extract_one()` 逐句：先找触发词 → 定类型 → 按 schema 角色顺序填论元（数值类走正则、实体类从候选池按出现顺序"领一个"）；
5. 评测走 `eval_utils.evaluate_predictions()`，失败分析走 `analyze_failures()`。

**预期输出**（实测，dev 930）：

```
触发词炼成：99 条（自动 ... + 人工 ...，含去重）
公司名词典：3679 个
推理完成：930 句，耗时 0.67s（0.720 ms/句，纯 CPU）
样本数 930（缺预测 0 条）
  trigger      P=0.6175 R=0.6161 F1=0.6168
  argument     P=0.1738 R=0.1742 F1=0.1740
  event_exact  P=0.0162 R=0.0161 F1=0.0161
  type_macro_f1 = 0.8118
【规则基线失败分析】
  arg_spurious     517 条 (55.6%)
  wrong_trigger    214 条 (23.0%)
  wrong_type       141 条 (15.2%)
  arg_missing       41 条 (4.4%)
  exact             15 条 (1.6%)
  no_trigger         2 条 (0.2%)
  触发词命中率（整句有事件的比例）：99.8%
```

**产物**：`outputs/disc/pred_rule.jsonl`、`outputs/metrics/disc/rule.json`

---

### 2.2 `bilstm_crf/train.py` — L1 字符级 BiLSTM-CRF（无预训练）

```bash
python bilstm_crf/train.py                                  # 默认 5 epoch，CRF
python bilstm_crf/train.py --decoder linear                 # 消融：去掉 CRF（非法 BIO 对照）
python bilstm_crf/train.py --epochs 3 --batch_size 64
python bilstm_crf/train.py --mock --device cpu              # 20 条手造数据，CPU 冒烟
```

**内部流程**：

1. `make_vocab()` 只从 **train** 建字表（用 dev 建 = 泄漏），固定 `<PAD>=0 / <UNK>=1`；
2. 从 train 切 `--inner_val 0.1` 做**内层验证集**（选 epoch 用，dev 只在最后评测）；
3. 模型：`Embedding(128) → BiLSTM(256, 双向=512) → Dropout → Linear(149) → CRF`；
4. 每个 epoch：训练 → 内层验证 loss/span F1 → 内层 F1 创新高就存 checkpoint；
5. 训完在 dev 上推理、统计**非法 BIO**（`illegal_bio_stats`）、解码成事件、走统一评测。

**预期输出**（`bilstm_crf` 实测，`--epochs 3 --max_len 128`）：

```
设备：cuda
字表大小：3269（训练集字符去重；<PAD>=0, <UNK>=1）
训练 5173 句 / 内层验证 574 句 / 评测 dev 930 句
模型参数：1.31M（从零训练，无预训练），解码器=crf
epoch 1/3 | train_loss=… | inner_loss=… inner_spanF1=… | …s
  ★ 内层验证最优 → 保存 bilstm_crf.pt
epoch 3/3 | train_loss=… | inner_loss=… inner_spanF1=0.5474 | …s
样本数 930（缺预测 0 条）
  trigger      P=0.9046 R=0.7441 F1=0.8165
  argument     P=0.5775 R=0.3869 F1=0.4634
  event_exact  P=0.0980 R=0.0806 F1=0.0885
  type_macro_f1 = 0.7306
【非法 BIO 统计（dev，模型原始输出）】
  序列数 930：含非法序列 454 条（48.8%），其中非法开头 50 处、非法转移 660 处
训练耗时 1520s | 延迟 21.17 ms/句
```

> 注意两点：① 这条是**为控时间用 `max_len=128`** 跑的（dev 有 67.1% 句子被截断），
> 所以数字偏保守，不是 BiLSTM 的真实上限；② 它 **1.31M 参数**（比 BERT 的 102.4M 小 78 倍），
> 却把 trigger 做到 0.8165 —— 触发词这种「2~4 字的关键词」本来就不太吃预训练，
> 真正拉开差距的是**论元**（0.4634 vs BERT+CRF 0.5815）。

**产物**：`outputs/checkpoints/bilstm_crf.pt`、`outputs/disc/pred_bilstm_crf.jsonl`、`outputs/metrics/disc/bilstm_crf.json`，
以及训练日志三件套 `outputs/logs/bilstm_crf_{train.json,steps.jsonl,console.log}`（见 §1.5）

> 8GB 笔记本卡上，这条链路是整套实验里最适合"课上现场跑"的（无预训练、模型小、几分钟一轮）。

---

### 2.3 `common/crf.py` — 手写 CRF（可当独立组件跑自检）

```bash
python common/crf.py     # 自检：构造小例子验证前向 logZ / 维特比 / mask 是否正确
```

**内部流程**：`CRF` 类实现 4 件事——`_gold_score()`（金标路径分数）、`_forward_algorithm()`
（log-sum-exp 求 logZ）、`neg_log_likelihood()`（loss = logZ − S）、`decode()`（维特比 + 可选置信度）。
`CRFClassifier` 把「任意 encoder（BERT/BiLSTM）+ Linear + CRF」封成一套，**BERT+CRF 与 BiLSTM+CRF
共用同一份 CRF 与解码**，这是方法可比的前提。

**预期输出**：

```
loss=2.9684
解码路径=[[0, 1, 2, 2], [0, 2, 0]]（第二条长度应为 3，不含 PAD）
标签数 C=3，序列长度 L=4 → 检查：loss 与路径长度都依赖 mask 是否正确
```

**归零实验（课堂演示用）**：`CRF(num_tags, constraint=..., start_constraint=...)`
配合 `event_data.build_bio_constraint_mask()` / `build_start_constraint_mask()`。
本项目实测：随机 emission 无约束时每条序列平均 **12.2 处**非法转移，加硬约束后 **严格 0**；
**只加转移约束、不加 start 约束时非法开头仍有 42%**。

---

### 2.4 `common/event_data.py` — 数据层自检（先跑这个确认数据对）

```bash
python common/event_data.py
```

**预期输出**（实测）：

```
schema: {'num_types': 13, 'num_type_role_slots': 92, 'num_union_roles': 61}
joint 标签数 = 149（O + 2×74 实体标签）
typed[公司上市] 标签数 = 19 → ['O', 'B-TRG', 'I-TRG', 'B-上市公司', ...]
数据规模: {'schema': {...}, 'train_extract': 5747, 'dev_extract': 930, 'train_detect': 32794, 'dev_detect': 5302, 'dev_docs': 985}
```

这个文件同时是**数据层的唯一实现**：标签体系（joint/typed）、span↔BIO、子词对齐、Dataset、
解码、`make_pred_row`（预测行必须带 `id + sent_id`）。

---

### 2.5 `bert/train.py` — L2a BERT 序列标注（本周主线）

```bash
# 主线：12 层 BERT + CRF，max_len 256，3 epoch
python bert/train.py

# 对照：BERT + Linear（非法 BIO 会明显更多）
python bert/train.py --decoder linear --run_name bert_linear_256

# 消融：只保留前 4 层（保留预训练权重，只截层，约 1/3 时间）
python bert/train.py --bert_layers 4

# 消融：CRF 硬约束（把非法转移压到 0）
python bert/train.py --decoder crf --hard_constraint

# 消融：截断代价（句子 p99=393 字，128 会截掉一批）
python bert/train.py --decoder linear --max_len 128 --run_name bert_linear_128

# 20 条手造数据、CPU 冒烟（先确认流程通，再上真数据）
python bert/train.py --mock --device cpu --epochs 1 --bert_layers 4
```

**内部流程**：

1. `build_joint_schema()` → 149 标签；`BioDataset` 做「字符级 BIO → 子词对齐」（非首子词设 -100）；
2. train 切 10% 内层验证集选 epoch，dev 只做监控与最终评测；
3. `AdamW` 分层学习率（BERT `--lr 2e-5`，分类头/CRF `--head_lr_mult 5` 倍）+ warmup 10% + 梯度裁剪 1.0；
4. 每 epoch 打印 `train_loss / inner_loss / inner_spanF1 / dev_spanF1(监控)`，内层创新高存 checkpoint；
5. 用最优 checkpoint 在 dev 推理 → 子词预测还原成字符级标签 → 解码 span → 拼事件 → 统计非法 BIO → 统一评测。

**参考：`bert_linear_256` 的真实完整输出**（已实跑，dev 930；CRF 版除解码器外输出格式相同）：

```
设备：cuda
显存占用（启动时）：0 MiB
标签体系：joint，149 个标签（13 类 / 92 槽位 / 61 去重角色）
训练 5173 句 / 内层验证 574 句 / 评测 dev 930 句
模型参数：102.4M（可训练 102.4M），解码器=linear，BERT 层数=full(12)
训练步数 486（warmup 48 步，10%）
  epoch 1 step 50/162 loss=2.6494
epoch 1/3 | train_loss=1.3387 | inner_loss=0.4394 inner_spanF1=0.3989 | dev_spanF1(监控)=0.4169 | 189s
  ★ 内层验证最优 → 保存 bert_linear_256.pt
epoch 2/3 | train_loss=0.3482 | inner_loss=0.2668 inner_spanF1=0.6015 | dev_spanF1(监控)=0.6034 | 219s
epoch 3/3 | train_loss=0.2498 | inner_loss=0.2426 inner_spanF1=0.6311 | dev_spanF1(监控)=0.6432 | 220s
加载最优 checkpoint（epoch 3，内层验证 spanF1=0.6311）
dev 解码：930 句，其中没抽到触发词 44 句
样本数 930（缺预测 0 条）
  trigger      P=0.8499 R=0.8097 F1=0.8293 (tp=753 fp=133 fn=177)
  argument     P=0.5153 R=0.6203 F1=0.5630 (tp=2557 fp=2405 fn=1565)
  event_exact  P=0.1603 R=0.1527 F1=0.1564 (tp=142 fp=744 fn=788)
  type_macro_f1 = 0.7654
  n200 子集：trigger F1=0.8123 arg F1=0.5598 → bert_linear_256_n200.json
  指标已写入 → outputs/metrics\disc\bert_linear_256.json

【非法 BIO 统计（dev，模型原始输出）】
  序列数 930：含非法序列 622 条（66.9%），其中非法开头 0 处、非法转移 1520 处
  训练集对照：含非法序列 3487 条（67.4%）

训练耗时 637s | 推理延迟 11.26 ms/句 | 峰值显存 1842 MiB
```

> 上面这段是 **`outputs/logs/bert_linear_256_console.log` 的真实内容**（2026-11-18 重跑）。
> 训练时同样内容会实时写进这个 `_console.log`，所以学生跑完可以照着对照。
> 这段输出可以直接当"读到数字该看哪几行"的模板：**每 epoch 看内层验证 span F1 选点，
> 最后看三级 F1 + 非法 BIO 统计 + 延迟**。`bert_crf_256` 的输出只在解码器与非法统计上不同。
> ⚠ 这里的 **637s** 是 11/18 重跑（当时同机还有 CPU 任务并行）；**首跑无并行任务时是 531s**，
> 引用"CRF 慢多少倍"等倍率时请用首跑口径（531s → 3112s）。

**产物**：`outputs/checkpoints/<run_name>.pt`（含 label_scheme/超参/最优 epoch）、
`outputs/disc/pred_<run_name>.jsonl`、`outputs/metrics/disc/<run_name>.json`，
以及**训练日志三件套** `outputs/logs/<run_name>_{train.json,steps.jsonl,console.log}`（见 §1.5）。
`<run_name>` 就是 `--run_name` 传的值（如 `bert_linear_256`），日志文件名与它一一对应。

---

### 2.6 `bert_pipeline/train.py` — L2b 两段式（检测 → 类型 → 类型条件论元）

```bash
python bert_pipeline/train.py                       # 完整训练 + 组合评测（约 15 分钟 GPU）
python bert_pipeline/train.py --arg_epochs 2        # 少训一点
python bert_pipeline/train.py --skip_train          # 复用已有 checkpoint，只重跑组合评测
python bert_pipeline/train.py --mock --device cpu    # 20 条数据 CPU 冒烟
```

> **复现表格里那组数字（压缩配置）要用这条**——它和默认配置不同，别混：
> ```bash
> python bert_pipeline/train.py --detect_limit 8000 --cls_epochs 1 \
>        --arg_epochs 3 --arg_lr 8e-5 --arg_layers 4
> ```
> `--detect_limit 8000` = 检测器只用 8000 条 train_detect（保持负样本比例）；
> `--arg_layers 4` = 13 个专家各只留前 4 层 BERT。默认配置会把 32,794 条全量检测集训 2 epoch，**慢 3~5 倍**。
> 只重训专家、复用已训好的 detector/type：`python bert_pipeline/train.py --only_experts --arg_epochs 3 --arg_lr 8e-5 --arg_layers 4`

**内部流程**（三段）：

| 段 | 模型 | 数据 | 标签空间 |
|---|---|---|---|
| (a) 检测器 | BERT + [CLS] 2 分类 | `train_detect`（负样本 77.9%） | 2 |
| (b) 类型分类 | BERT + [CLS] 13 分类 | `train_extract` | 13 |
| (c) 类型条件论元 | 每类型一个 BERT（**只留 4 层**）+ Linear | 该类型的句子 | 11~19 |

组合 `compose()`：检测判"有事件"→ 类型分类 → 该类型专家抽触发词+论元；**依次训练 13 个专家、训完立刻 `del` + `empty_cache()`**（8GB 卡不能同时留 13 个 BERT）。
另外跑一遍 `oracle_type=True`（类型给金标）作为**上界分析**，用来定位"损失来自类型错还是论元抽不准"。

**预期输出**（压缩配置实测）：

```
设备：cuda | 数据：…\data/processed
数据：train_extract 5747 / dev_extract 930 / train_detect 32794 / dev_detect 5302 / clean 4815

(a) 训练事件检测器（二分类，负样本占大头）...
  [pipe_detector.pt] epoch …
   检测器 dev_detect：P=0.7969 R=0.7523 F1=0.7740 acc=0.9023
   检测器 clean(4815)：P=0.8009 R=0.7584 F1=0.7790 acc=0.8957

(b) 训练事件类型 13 分类器...
(c) 训练 13 个类型条件论元抽取专家（依次训练，训完即释放显存）...
  [arg:中标] 706 句 × 3 epoch loss=…   ← 每个专家每 epoch 只有 3~31 步

组合三段 → dev_extract 端到端评测...
   诊断：检测器丢掉 255 句 / 类型判错 51 句 / dev_extract 上类型准确率 0.9452
…
======================================================================
方法                          trigger F1   arg F1   event F1
----------------------------------------------------------------------
流水线（端到端）                     0.6769     0.3690     0.0957
流水线（类型金标上界）                 0.6861     0.3699     0.0956
```

> 括号里的数字是压缩配置实测：16 个模型、49.87 ms/句、三段全训 1,377.5 s。上界几乎不涨 → 瓶颈在论元专家。

**产物**：`outputs/checkpoints/pipe_detector.pt`、`pipe_type.pt`、`pipe_arg_<类型>.pt`（13 个）、
`outputs/metrics/disc/pipe_detector.json`（全量 + clean 两套检测指标）、`pipe_type.json`、
`pipeline.json`、`pipeline_oracle_type.json`；训练日志 `outputs/logs/bert_pipeline_{train.json,steps.jsonl,console.log}`
（`_steps.jsonl` 每行带 `stage` 字段，区分检测器 / 类型分类 / 13 个专家）。

> ⚠ **2026-11-18 这一轮没有重跑成功**，表里的数字与日志沿用 11-16 那次：
> 代码 bug（`step` 未定义、logger 作用域）已修好并用 `--mock` 验证（exit=0、`_steps.jsonl` 15 行），
> 但真跑时检测器（8000 条 / 125 微批）19 分钟未完成（≈9 s/微批，同期 `bert/train.py` 约 0.5 s/微批），
> 时间窗内跑不完，于是保留旧值、不新造数字。原因与复跑命令见 `outputs/logs/README.md`。

> **实测状态（压缩配置，已跑通）**：检测器 dev_detect 全量 P/R/F1 = **0.7969 / 0.7523 / 0.7740**、
> clean 版 **F1 0.7790**；类型分类准确率 **0.9452**（错 51 句）；端到端（pipeline.json）
> trigger/argument/event = **0.6769 / 0.3690 / 0.0957**、**16 个模型**、**49.87 ms/句**、三段全训 1,377.5 s。
> 类型给金标的上界只把 trigger 提到 0.6861、论元几乎不动 → 瓶颈在论元专家而不是类型。
> **核心教训**：13 个专家每个每 epoch 只有 **3~31 步**（联合模型有 486 步）——**训练预算被切碎**，
> 所以两段式在这台 8GB 卡上跑不过联合模型；这是工程结论，不是方法结论（详见 `ARCHITECTURE.md` §4.1）。
> 首次跑端到端还踩过一个 bug：`span_to_char_tags(typed=True)` 把论元也刷成 `TRG` 标签，
> 导致专家预测「随机单字触发词 + 空论元」、三级 F1 全 0（见踩坑第 32 条）。跑出 0 时先查这条。

---

### 2.7 `eval/run_experiments.py` — 一键串行跑全部判别式实验

```bash
python eval/run_experiments.py --group main        # 先出两条核心主线（BERT linear / crf）
python eval/run_experiments.py --group pipeline    # 两段式
python eval/run_experiments.py --group bilstm      # BiLSTM 两条
python eval/run_experiments.py --group ablation    # 三个消融
python eval/run_experiments.py --group all         # 全部（80~100 分钟，GPU）
python eval/run_experiments.py --only bert_crf_256 # 只跑指定的一条
python eval/run_experiments.py --dry_run           # 只打印命令，不执行
```

**内部流程**：按 `EXPERIMENTS` 列表**串行**起子进程（一张 8GB 卡绝不并行），
每条实验的完整命令、耗时、退出码写进 `outputs/logs/run_<name>.out` 与 `run_summary.json`
（这是**一键复现的运行日志**，与 §1.5 的训练日志三件套不是一回事）；
任一实验非 0 退出会**立刻中止后续**（fail-fast），不会出现"某步崩了却继续跑"。
启动前先查 `nvidia-smi`，占用 >1GB 会提醒你先等别的任务结束。

**预期输出**：

```
GPU 检查通过：当前占用 0 MiB
将串行执行 2 个实验：
  - [main] bert_linear_256: BERT(12层)+Linear，max_len=256，3 epoch
  - [main] bert_crf_256: BERT(12层)+CRF，max_len=256，3 epoch（主线最佳）

==============================================================================
▶ bert_linear_256：BERT(12层)+Linear，max_len=256，3 epoch
  python bert/train.py --decoder linear --max_len 256 --epochs 3 --run_name bert_linear_256
  epoch 1/3 | …
  ⏱ 512s  → ok  (日志 outputs/logs/run_bert_linear_256.out —— 这是 run_experiments 的**运行日志**；
                 训练日志是 outputs/logs/bert_linear_256_train.json，两者不同，见 §1.5)
…
全部完成，总耗时 … 分钟
汇总已写入 → outputs/logs/run_summary.json
```

---

### 2.8 `eval/compare_methods.py` — 汇总对比表

```bash
python eval/compare_methods.py                          # 写到 outputs/metrics/disc/summary.md
python eval/compare_methods.py --with_n200              # 附带 n=200 子集列（跨链路可比）
python eval/compare_methods.py --out outputs/metrics/disc/summary.md
```

**内部流程**：只读 `outputs/metrics/disc/*.json`，按**教学递进顺序**（L0 规则 → L1 无预训练 →
L2 BERT → L2 两段式 → 上界/消融）拼 markdown 表；缺失的方法显式列出"尚未跑出结果，绝不填估算值"；
有 ≥2 行时自动生成"预训练带来的提升 / 联合→两段式带来的提升"这类对比句（数字从 JSON 读，不手写）。

**预期输出**（现在能看到的真实版本）：

```
# 事件抽取方法对比（dev_extract 930 句，统一评测口径 common/evaluate_events.py）
### 主表：方法对比
| 方法 | 触发词F1 | 论元F1 | 事件级F1 | 类型macroF1 | 参数(M) | 训练(s) | 延迟(ms/句) | 非法序列占比 | epoch | max_len |
| L0 规则（词典+正则） | 0.6168 | 0.1740 | 0.0161 | 0.8118 | 0 | 0 | 0.72 | - | - | - |
> ⚠ 以下方法尚未跑出结果（表中不出现，绝不填估算值）：L1 BiLSTM + Linear、…
```

---

### 2.8.1 实验辅助脚本（复评 / 硬约束 / 金标地板 / 触发词回填）

这四个脚本都是"不重训、只换评测或后处理"，课堂里最适合用来演示**同一份权重的不同用法**。

```bash
# ① 从已有 checkpoint 复评（训练崩在最后一步时不用重训）
python bert/evaluate.py --ckpt outputs/checkpoints/bert_crf_256.pt --run_name bert_crf_256
python bert/evaluate.py --ckpt outputs/checkpoints/bert_crf_256.pt --run_name bert_crf_256 --train_sample 800

# ② CRF 解码期硬约束（非法转移分数置 -1e4，F1 不变、非法数归零）
python bert/evaluate_hard_constraint.py
python bert/evaluate_hard_constraint.py --ckpt outputs/checkpoints/bert_crf_256.pt --run_name bert_crf_256_hard

# ③ 金标自身的非法 BIO（"非法率地板值"，任何非法率都要和它比）
python eval/check_bio_gold.py
python eval/check_bio_gold.py --split dev

# ④ 触发词回填后处理（某角色与触发词高度重叠时，模型没抽到就用触发词文本回填）
python eval/postprocess_trigger_copy.py --base bert_crf_256 --threshold 0.5
```

| 脚本 | 内部流程 | 预期输出 |
|---|---|---|
| `bert/evaluate.py` | 加载 checkpoint → dev 推理 → 统计非法 BIO → 统一评测（可选对训练集抽样统计非法率） | 与训练脚本结尾相同的指标块；指标写到 `outputs/metrics/disc/<run_name>.json`，另有 `outputs/logs/<run_name>_eval.json` |
| `bert/evaluate_hard_constraint.py` | 读 `bert_crf_256` 的权重 → 把非法转移 mask 成 `-1e4` → 重新维特比解码 → 重新评测（**不重训**，约 90 s） | `bert_crf_256_hard.json`：**论元 F1 0.5815 → 0.6566（+7.5pp）**、事件 0.1697 → 0.2177；token 级非法转移 **严格 0**，字符级口径仍报 155 处（100% 由 span 内空格造成，见 FAQ Q17） |
| `check_bio_gold.py` | 把金标 span 刷成 BIO 再统计非法转移 | dev **18 处 / 930（1.9%）**、train **94 处 / 5,747（1.6%）**——这是「非法率地板」，不能和 0 比 |
| `postprocess_trigger_copy.py` | 统计训练集里「角色文本 == 触发词文本」的比例；推理时若该角色缺失就回填 | `bert_crf_256_rule.json`：arg **0.5815 → 0.5831**、event **0.1697 → 0.1763**；只学到 1 条规则（`高管变动.变动类型`，236/260 = 90.8%），回填 31 个论元 |
| （失败版，教学用） | 判据改成「区间重叠率 ≥0.5」→ 回填 79 个论元 | arg **0.5815 → 0.5800（−0.15pp，负收益）**：触发词「辞去」与论元「辞职」同义不同形，回填错文本只增假阳性 |

> 为什么要有第 ③ 个脚本：金标自己就有 18 处非法 BIO（论元重叠，BIO 表达不了）。
> 不测地板值，就会把"CRF 还有 62.5% 句子含非法序列"当成模型很差——
> 而真实可消除空间只有 62.5% − 1.9% 这一部分。

---

### 2.9 `common/eval_utils.py` — 评测封装自检（**改完任何评测相关代码先跑这个**）

```bash
python common/eval_utils.py
```

**预期输出**（必须全 1.0000，否则说明预测键没带 `sent_id`、同文档多句被覆盖）：

```
gold→pred 自检（必须全 1.0000）：
  trigger      F1=1.0000
  argument     F1=1.0000
  event_exact  F1=1.0000
自检 PASS
```

---

### 2.10 `common/evaluate_events.py` — 统一评测（命令行版）

```bash
python common/evaluate_events.py --gold data/processed/dev_extract.jsonl \
                                 --pred outputs/disc/pred_rule.jsonl \
                                 --json outputs/metrics/disc/rule_cli.json

# 额外算句子级"有没有事件"的二分类指标（预测里要有 has_event）
python common/evaluate_events.py --gold data/processed/dev_extract.jsonl \
        --pred outputs/disc/pred_pipeline.jsonl --detect
```

**内部流程**：`row_key()` 用 `id|sent_id` 对齐 → `as_event_list()` 兼容 `event`/`events` 两种写法 →
`compare_events()` 在**同一事件类型内按下标一对一**配对，算 trigger / argument / event 三级 → 
`type_macro_f1()` 算 13 类 macro → 输出 `per_type` / `per_role`。
**预期输出**：

```
样本数 930（缺预测 0 条）
  trigger      P=0.6175 R=0.6161 F1=0.6168 (tp=573 fp=355 fn=357)
  argument     P=0.1738 R=0.1742 F1=0.1740 (tp=718 fp=3413 fn=3404)
  event_exact  P=0.0162 R=0.0161 F1=0.0161 (tp=15 fp=913 fn=915)
  type_macro_f1 = 0.8118
written: outputs/metrics/disc/rule_cli.json
```

> **不要**为了让分数好看而删掉解析失败的行：预测里缺的样本按"全漏"处理（分母不变），
> 删行反而会让 F1 虚高——这条纪律写在脚本注释里。

---

### 2.11 `tools/make_sample_data.py` — 生成学生包小样本

```bash
python tools/make_sample_data.py                 # 从 data/processed 抽样 → data/sample
python tools/make_sample_data.py --scale 0.5     # 只跑通流程时用更小规模
python tools/make_sample_data.py --from_sample   # 不抽样，只校验已有 sample 并打印统计
```

**内部流程**：确定性洗牌（先按 `(id, sent_id)` 排序再 `shuffle(seed=42)`）；检测集用**分层抽样保持正负比**
（直接截断很可能拿到全正或全负，训出来的检测器退化成只猜多数类）；写完打印"全量% → 抽样%"分布对照。

**预期输出**：

```
从 …\data/processed 抽样 → …\data/sample
event_schema.json        原样复制（13 类事件定义）
train_extract            全量   5747 → 抽样   800
dev_extract              全量    930 → 抽样   300
train_detect             全量  32794 → 抽样  1500
dev_detect               全量   5302 → 抽样   600
dev_docs                 全量    985 → 抽样    50

【抽样后分布校验】…
  seed=42（固定）；样本合计 3250 条
```

---

### 2.12 `eval/make_figures.py` — 生成 PPT 用实验图

```bash
python eval/make_figures.py                                  # 全部重画
python eval/make_figures.py --only method_comparison per_role_f1
python eval/make_figures.py --role_method bert_crf_256 --confusion_method bert_crf_256
```

**内部流程**：只读 `outputs/metrics/disc/*.json`、`outputs/logs/<method>_train.json`（训练曲线）
与 `outputs/disc/pred_*.jsonl`（混淆矩阵），出 6 张图
（方法对比 / 非法 BIO 对比 / per-role F1 / 训练曲线 / 类型混淆矩阵 / 延迟-参数量），
白底 150dpi、中文字体、柱上标数值。**指标或日志文件不存在就跳过该图并提示，不画假图**。

> ⚠ 训练曲线读的是**新命名** `<method>_train.json`（同时兼容旧命名 `train_<run>.json`）。
> 本项目踩过一次：重构把日志改名后忘了改读取逻辑，于是日志明明有 6 个、
> 图却打印「没有训练日志」**静默跳过**——所以凡"按文件名读产物"的地方，改命名时必须一起改。

**产物**：`outputs/figures/*.png`

---

### 2.13 `data/scripts/*.py` — 数据处理四件套（可复现整条数据链）

```bash
# 在项目根目录执行即可（脚本内部按 __file__ 推导 data/ 路径，从哪运行都一样）
python data/scripts/audit_raw_index.py     # 1) 审计原始 index 的真实性质 → stats/audit_raw.json
python data/scripts/build_dataset.py       # 2) 构建 processed/*.jsonl + schema → stats/build_stats.json
python data/scripts/verify_processed.py    # 3) 验收：逐行重建 span，0 错才算过 → stats/verify_stats.json
python data/scripts/make_figures.py        # 4) 6 张数据分布图 → data/figures/*.png
```

| 脚本 | 内部流程 | 预期输出 |
|---|---|---|
| `audit_raw_index.py` | 统计原始 `start_index` 能否与 `text` 自洽、是否等于 `text.find()` 首次出现、多少论元根本没有 index | `stats/audit_raw.json`：52,395 个有 index 的 span **全部自洽、0 例与 find 不一致**；468 个无 index 论元全是「环节」 |
| `build_dataset.py` | 句子级/篇章级原始 JSON → 4 类 JSONL + schema；一律 `find()` 重算 span，定位不到的丢弃并计数 | `processed/` 六个文件；`stats/build_stats.json` |
| `verify_processed.py` | 逐行 `json.loads` + 字段校验 + `text[start:end]==span.text` + extract/detect 一致性 + train/dev 文本重叠统计 | **45,143 个 span 重建 0 错误**；`stats/verify_stats.json` 里 `errors: 0` |
| `make_figures.py` | 读 processed + stats 出 6 张分布图 | `data/figures/fig_*.png` |
| `stats_duee10_fin.py` | 读 `raw/DuEE_hf/*.parquet`，统计 HF DuEE 1.0 的金融子集规模与"同串多次出现"标注位置 | 仅用于数据集选型对比，**未参与下游** |

> 注意：`data/scripts/*.py` 都**没有命令行参数**（路径按 `__file__` 推导，从哪运行都行）。

---

### 2.14 `llm_zero_shot/event_prompt.py` — 打印 prompt 全文（PPT 要用）

```bash
python llm_zero_shot/event_prompt.py     # 直接运行会打印 system prompt + 自检行
python llm_zero_shot/llm_extract.py --print_prompt   # 等价的另一个入口
```

**内部流程**：`build_system_prompt()` 把 13 类 schema 渲染成紧凑一行式（`- 类型：角色/角色/...`，
实测 schema 部分约 392 token）；`build_messages()` 组装 messages（few-shot 时插入 6 组
user/assistant 示例，**全部取自 train，绝不从 dev 挑**）；`parse_model_output()` 负责解析/修复/对齐。

**预期输出**：完整 system prompt + 末尾

```
======================================================================
system prompt 字符数：…
zero-shot messages 条数：2
few-shot  messages 条数：14
```

---

### 2.15 `llm_zero_shot/llm_extract.py` — L3 zero-shot / few-shot（三种后端）

```bash
# ① 纯离线演练（不联网、不占 GPU，先确认整条流水线通）
python llm_zero_shot/llm_extract.py --dry-run --mode zero --n 20 --tag smoke_mock_n20

# ② API zero-shot / few-shot（先配环境变量）
python llm_zero_shot/llm_extract.py --backend api --mode zero --n 200 --tag api_zero_n200
python llm_zero_shot/llm_extract.py --backend api --mode few  --n 200 --tag api_few_n200

# ③ 本地 Qwen3-0.6B zero-shot（占 GPU；一定要开 --gen_batch，见下面「为什么」）
python llm_zero_shot/llm_extract.py --backend local --mode zero --n 150 --tag local_zero_n150 \
       --gen_batch 8 --max_new_tokens 384

# ③b 只重解析、不重跑模型（改了容错逻辑后必用；省 20 分钟 GPU）
python llm_zero_shot/llm_extract.py --backend local --from_raw outputs/llm/raw_local_zero_n60.jsonl \
       --tag local_zero_n60

# ④ 只看提示词（PPT 里要贴 prompt 就用它）
python llm_zero_shot/llm_extract.py --print_prompt

# 只生成预测、暂不评测
python llm_zero_shot/llm_extract.py --backend api --mode zero --n 200 --no_eval
```

**后端与关键参数**：

| 参数 | 取值 | 说明 |
|---|---|---|
| `--backend` | `api` / `local` / `mock` | `mock` 等价于 `--dry-run`（离线假模型） |
| `--mode` | `zero` / `few` | few-shot = 6 个难类型示例（全部取自 train） |
| `--n` | 默认 200 | 与判别式方法共用的固定抽样（n=150/100 是 n=200 的前缀） |
| `--tag` | 默认 `{backend}_{mode}_n{n}` | 决定 `pred_/raw_/report_` 与 metrics 文件名 |
| `--max_tokens` | 默认 512 | API 单次输出上限；推理模型别小于 256（思考 token 会吃光正文） |
| `--gen_batch` | 默认 8 | **本地后端必开**：Windows 上逐条解码只有 1.5~2 tok/s，批量解码实测快 4~6 倍 |
| `--dtype` | `auto`/`bf16`/`fp16`/`fp32` | 本地模型精度，`auto` = 有 CUDA 用 bf16 |
| `--max_length` | 本地 prompt 上限 | 与 SFT 的 1280 对齐 |

**为什么本地必须批量解码**：逐 token 解码每步要发 ~200 个 CUDA kernel，Windows(WDDM) 下每次
kernel 启动 1~2ms，GPU 一直停在 P8 低功耗档 —— 瓶颈是「启动」不是「算力」。
`llm_zero_shot/local_generate.py` 用 **batch=8 + 左 padding**（decoder-only 必须左 padding，
右 padding 会让模型从 pad 位置续写）把启动开销摊薄，实测 **5.32 s/条、28.5 tok/s**。

**本地 0.6B 的实测结论（n=60，dev 前缀，2026-11-18 重跑）**：trigger **0.4068** / argument **0.1167** /
event_exact **0.0000** / type_macro **0.3817** —— **低于规则基线的 argument 0.1740**，
所以零样本小模型在结构化抽取上不可用（L3 要么 API、要么上 LoRA）。
> `type_macro` 只有 0.38（远低于 API 的 0.93）：小模型会输出 schema 之外的类型名，
> 把 macro 的分母撑大 —— 与 LoRA 那次的 `0.9150→0.7862` 是同一个现象（生成式必须做类型白名单校验）。
另一个关键点：它的 JSON **合法率从 70% 提升到 100%**（18/60 条少一个右括号），
只靠改解析器的栈式括号修复，trigger F1 就从 0.3299 涨到 0.4174 —— **解析器差 8 个点**。

**内部流程**：

1. `sample_rows()` 用全项目唯一采样口径抽样（先按 `(id,sent_id)` 排序再 `shuffle(seed=42)` 取前缀），
   把样本 id 写进 `outputs/llm/eval_ids.json`、金标写进 `gold_n<k>.jsonl`；
2. 逐条：`build_messages()` → 后端生成（api / local_generate 批量 / mock）→
   `parse_model_output()`（剥围栏/配平/截断补全/`text.find` 对齐）；
3. 累计工程指标：`json_valid_rate / parse_fail / empty_reply / repaired / noise_stripped /
   type_invalid / role_invalid / trigger_not_in_source / arg_not_in_source`；
4. **用 subprocess 调 `common/evaluate_events.py`** 算指标（保证"报告里的数字 = 命令行跑出来的数字"）。

**预期输出**（API zero-shot n=200，**取自 `outputs/logs/api_zero_n200_console.log` 的 11/18 真实输出**）：

```
数据：dev_extract.jsonl → 采样 200 条（seed=42，规则见 event_data.canonical_order）
输出：…\outputs/llm\pred_api_zero_n200.jsonl
[api] base_url=https://api.deepseek.com model=deepseek-v4-flash max_tokens=512 few_shot=False
  [ 10/200] 合法 10 条，平均 3.65s/条
  …
  [200/200] 合法 200 条，平均 3.54s/条

统计 → …\outputs/llm\report_api_zero_n200.json（总耗时 724s）

==============================================================================
结果：api_zero_n200（api/zero，200 条）
==============================================================================
  trigger F1   : 0.7789   (P=0.7828 R=0.7750)
  argument F1  : 0.5224   (P=0.5026 R=0.5437)
  event  F1    : 0.0402
  type_macro_f1: 0.9281
  JSON 合法率  : 100.0%（结构性修复 0 条，围栏/废话剥离 0 条）
  解析失败     : 0 条 | 空回复 0 条
  幻觉论元     : 9 次 | 触发词找不到 2 次
  平均延迟     : 3.535s/条（p50=2.836s）
  token        : prompt 169891 + completion 21687 = 191578
统一评测 → …\outputs/metrics\llm\api_zero_n200.json
```

**产物**：`outputs/llm/pred_<tag>.jsonl`、`raw_<tag>.jsonl`（模型原始输出，做 bad case 用）、
`report_<tag>.json`（工程指标）、`gold_n<k>.jsonl`、`eval_ids.json`、
`outputs/metrics/llm/<tag>.json`

---

### 2.16 `llm_lora/train_sft.py` — L4 LoRA / 全量 SFT

```bash
# LoRA，1 epoch，全量 5,747 条（8GB 卡预计 45~70 分钟，实测耗时见 train_log_sft.json）
python llm_lora/train_sft.py

# 冒烟：只跑 20 个优化器步（先验证显存与流程）
python llm_lora/train_sft.py --max_steps 20

# 小数据快速验证
python llm_lora/train_sft.py --num_train 1000

# 全量微调（0.6B 全参数；8GB 卡大概率 OOM，仅作对比开关）
python llm_lora/train_sft.py --full_ft --lr 2e-5
```

**关键参数**（全集用 `python llm_lora/train_sft.py --help` 看）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--max_length` | **1280** | 本任务 prompt+目标平均 1,011 token、p95 1,161；**384 会让 100% 样本 prompt 被截断 → labels 全 -100 → loss=nan** |
| `--batch_size` / `--grad_accum` | 2 / 8 | 等效 batch 16；`batch=4` 在 8GB 卡实测 OOM |
| `--lr` | LoRA `2e-4` / 全量 `2e-5` | 不传按模式自动取 |
| `--lora_r` / `--lora_alpha` | 8 / 16 | 挂 `q_proj/k_proj/v_proj/o_proj`，可训练 **2,293,760 / 598,343,680 ≈ 0.38%** |
| `--grad_ckpt` | 关 | 开：省显存、慢约 20% |
| `--val_size` | 200 | 从 **train** 尾部留验证（**绝不用 dev 做模型选择**） |
| `--max_steps` | 0（不限） | >0 时只跑这么多优化器步（冒烟用） |

**内部流程**：`EventSFTDataset` 构造 `prompt(全 -100) + 目标 JSON + EOS(算 loss)` →
LoRA 挂载 → 训练循环（`loss=nan` 跳 batch、梯度裁剪 1.0、按 val_loss 存 checkpoint）→
写 `train_log_sft.json`（含 `prompt_truncated` / `target_truncated` / `gpu_peak_mem_gib` / 可训练参数）。

**预期输出**（正式 LoRA 训练实测：1 epoch × 3,499 条）：

```
设备: cuda | 模式: LoRA
训练 3499 条 | 验证(取自 train 尾部) 200 条
prompt 被完全截断（丢弃）：1 条 | 目标尾部被截断：8 条（max_length=1280）
加载基座: …\pretrain_models/Qwen3-0.6B（dtype=torch.bfloat16）
trainable params: 2,293,760 || all params: 598,343,680 || trainable%: 0.3834
训练步数: 233（等效 batch=15，lr=0.0002）
Epoch 1/1 | train_loss=0.1002 val_loss=0.0742 | 2830s（…s/step）
  ✓ LoRA adapter已保存 → outputs/llm/ckpt/sft_adapter

训练完成：38.8 分钟，峰值显存 4.0 GiB
日志 → outputs/llm/ckpt/train_log_sft.json
checkpoint → outputs/llm/ckpt/sft_adapter
```

**产物**：`outputs/llm/ckpt/sft_adapter/`（LoRA，含 `adapter_model.safetensors`）、
`outputs/llm/ckpt/train_log_sft.json`；`--full_ft` 时是 `sft_full_ckpt/` + `train_log_full_ft.json`

---

### 2.17 `llm_lora/evaluate_sft.py` — 评测 SFT checkpoint

```bash
python llm_lora/evaluate_sft.py                                     # 默认评 LoRA，n=200
python llm_lora/evaluate_sft.py --ckpt_dir outputs/llm/ckpt/sft_adapter --n 200 --gen_batch 8
python llm_lora/evaluate_sft.py --n 930                             # 全量 dev
python llm_lora/evaluate_sft.py --ckpt_dir outputs/llm/ckpt/sft_full_ckpt   # 评全量微调
```

**内部流程**：目录里有 `adapter_config.json` → 按 LoRA 加载（基座 + adapter，`merge_and_unload()`）；
否则按全量 checkpoint 加载。用**与 `llm_extract.py` 完全相同的采样、prompt、解析**，
预测落盘后 subprocess 调统一评测；最后扫 `outputs/metrics/llm/*.json` 打一张横向对比表
（自动跳过 `probe/smoke/mock/dry`）。

**预期输出**（LoRA 实测，n=200）：

```
评测样本 200 条（seed=42，与 API / 判别式同一批）
[加载] LoRA adapter: …\sft_adapter
        基座: …\Qwen3-0.6B
  [200/200] 合法 200 条，平均 3.13s/条

  trigger  F1=0.9000  (P=0.9000 R=0.9000)
  argument F1=0.6971  (P=0.6974 R=0.6958)
  event    F1=0.1850  (P=0.1850 R=0.1850)
  type_macro_f1 = 0.7862
  JSON 合法率: 100.0% | 解析失败 0 条 | 修复 0 条 | 空回复 0 条
  幻觉论元 3 次 | 平均延迟 3.133s/条 | 总耗时 628s
```

> 对照：同一个模型零样本时 argument 只有 **0.1129**、event **0.0000**；LoRA 微调后 argument **0.6971**，
> 比硬约束 BERT（n=200 口径 0.6418）还高 **5.5pp** —— 但 event_exact 低于判别式（0.1850 vs 0.2268），
> 说明**生成式召回强、边界弱**。

---

### 2.18 `eval/plot_compare.py` — LLM 侧对比图 + CSV

```bash
python eval/plot_compare.py                        # 自动扫描 outputs/metrics/llm/*.json
python eval/plot_compare.py --n 200                # 只收 200 条那批（保证同批可比）
python eval/plot_compare.py --extra_globs "outputs/metrics/disc/*.json"   # 把判别式结果一起画
```

**内部流程**：只认统一评测产出的 JSON（trigger/argument/event 三级 F1），按"讲课顺序"
（zero → few → 本地 → LoRA → 全量 SFT）排序，画 3 柱/方法的分组柱状图（英文标签避免中文方块），
同时输出 CSV 与 markdown 表（同一份数字，贴 PPT/文档用）。

**产物**：`outputs/llm/method_compare.png`、`outputs/llm/method_compare.csv`

---

### 2.19 `service/extract_service.py` — L5 系统组合（模块，不直接跑）

```bash
python service/extract_service.py     # 自检：不加载模型，只验证「切句 + 校验 + 兜底」逻辑（CPU）
```

**预期输出**：

```
切句： [(0, '公司拟回购1200万股，交易金额约1.2亿元，占总股本的3.5%。')]
校验后： [{"type": "股份回购", ..., "args": [..., {"role": "交易金额", "text": "1.2亿元", "source": "regex"}]}]
被拦下的问题： ['role_not_in_type:股份回购.不存在的角色', 'arg_not_in_text:交易金额']
```

（这一段会演示三件事：schema 不合规的角色被拦、原文找不到的论元被拦、模型漏掉的金额被正则补上。）

**内部流程**：见 `ARCHITECTURE.md` §3.7 的六步表。核心方法：
`ExtractService(backend="joint"|"pipeline")` → `.load()` → `.extract_sentence(text)` → `.extract_document(doc)`。

---

### 2.20 `service/demo_extract.py` — 篇章级现场演示（课堂投屏用）

```bash
python service/demo_extract.py --n 3                      # 抽 3 篇公告
python service/demo_extract.py --n 1 --doc_id <公告id>     # 指定某篇（多个 id 用逗号分隔）
python service/demo_extract.py --n 2 --backend pipeline    # 用两段式后端
python service/demo_extract.py --n 1 --mock                # 用 2 篇手造数据（无需真实数据/无需训练）
```

**前置条件**：需要有 checkpoint。
- 联合后端：`python bert/train.py --decoder crf --max_len 256` → `outputs/checkpoints/bert_crf_256.pt`
- 两段式后端：`python bert_pipeline/train.py` → `outputs/checkpoints/pipe_*.pt`

> 没有 checkpoint 时会明确报 `找不到联合模型 checkpoint` 并告诉你该跑哪条命令（不会静默输出空结果）。
> 只想看演示形态、不想等训练：用 `--mock`（会加载 `--mock` 训练出的 checkpoint，形态完整、数字无意义）。

**预期输出**（结构如下）：

```
加载模型（backend=joint）...
==============================================================================
公告 ID：…
标题：…
原文：…
切句数：…
------------------------------------------------------------------------------
【逐句抽取过程】
  [0] conf=0.9x 股份回购(触发词=回购)
  [1] conf=0.5x （无事件） ⚠需复核:low_confidence(0.5x)
------------------------------------------------------------------------------
【最终事件流 JSON】
[
  {"type": "股份回购", "trigger": {"text": "回购", ...},
   "args": [{"role": "回购方", "text": "…", "source": "model"}, ...],
   "confidence": 0.9x, "sent_ids": [0]}
]
✓ 全部句子置信度达标，可直接进下游

【与金标对照】
  … 金标 2 个事件 [...] | 抽出 2 个 [...] | 类型命中 2
```

**参数**：`--review_threshold 0.55`（低于它标人工复核）、`--no_fallback`（关正则兜底，看模型原始能力）、
`--out xxx.json`（把结果落盘）。

---

### 2.21 一条命令跑完"十分钟验收"

```bash
python common/eval_utils.py                                        # 1) 评测键自检：必须全 1.0000
python common/event_data.py                                        # 2) 数据/标签体系自检：13 类 / 149 标签
python rule/run_rule.py                                      # 3) L0 规则基线（秒级，出真数字）
python eval/compare_methods.py                                   # 4) 汇总表（自动列出"还没跑的方法"）
python llm_zero_shot/llm_extract.py --dry-run --mode zero --n 20      # 5) LLM 链路离线演练（不联网）
python service/extract_service.py                               # 6) 系统组合的校验/兜底逻辑自检
```

---

## 3. 作为模块调用

### 3.1 篇章级抽取（最常用）

```python
import sys
from pathlib import Path

ROOT = Path(r"event_extraction")   # 换成你的项目根
sys.path.insert(0, str(ROOT / "common"))    # 公共模块：event_data / evaluate_events / models …
sys.path.insert(0, str(ROOT / "service"))

from extract_service import ExtractService

svc = ExtractService(backend="joint")      # 或 backend="pipeline"
svc.load()                                  # 懒加载：第一次调用才吃显存

doc = {
    "id": "demo-001",
    "title": "XX公司公告",
    "text": "公司拟以集中竞价方式回购股份1200万股，占公司总股本的1.5%，交易金额约1.2亿元。"
}
result = svc.extract_document(doc)

print(result["n_sentences"])                # 切出几句
for ev in result["events"]:                 # 合并后的事件流
    print(ev["type"], ev["trigger"]["text"], ev["confidence"], ev["sent_ids"])
    for a in ev["args"]:
        print("   ", a["role"], a["text"], a["source"])   # source = model / regex
print(result["needs_review"])               # 是否建议人工复核
```

**返回结构**：

```python
{
  "id": str, "title": str, "n_sentences": int,
  "events": [{"type": str, "trigger": {...}, "args": [...], "confidence": float, "sent_ids": [int]}],
  "needs_review": bool,
  "sentences": [{"sent_idx", "offset", "text", "events", "confidence", "needs_review", "review_reasons"}],
}
```

### 3.2 评测自己的预测

```python
import sys
sys.path.insert(0, r"common")
from evaluate_events import evaluate, load_jsonl

gold = load_jsonl("data/processed/dev_extract.jsonl")
pred = load_jsonl("outputs/disc/pred_rule.jsonl")      # 每行 {id, sent_id, events:[...]}
res = evaluate(gold, pred, verbose=False)
print(res["trigger"]["f1"], res["argument"]["f1"], res["event_exact"]["f1"], res["type_macro_f1"])
print(res["per_role"]["质押方"])                        # 定位难角色
```

判别式脚本内部走的是 `common/eval_utils.py: evaluate_predictions(name, preds, meta=...)`：
它会把预测落盘到 `outputs/disc/pred_<name>.jsonl`、指标写到 `outputs/metrics/disc/<name>.json`，
并顺带算一份 n=200 子集指标（`<name>_n200.json`）用于跨链路对比。

### 3.3 只要 prompt + 解析（自己接别的模型）

```python
import sys
sys.path.insert(0, r"llm_zero_shot")
from event_data import load_schema
from event_prompt import build_messages, parse_model_output

schema = load_schema()
text = "众安集团10月15日耗资94.56万港元回购380.5万股"
messages = build_messages(text, schema, few_shot=False)    # → OpenAI 兼容 messages

raw = "<你的模型返回的文本>"
events, info = parse_model_output(raw, text, schema)        # 自动剥围栏/修复/对齐 offset
print(events, info["json_ok"], info["arg_not_in_source"])
```

### 3.4 只读数据

```python
import sys
sys.path.insert(0, r"common")   # 公共数据层在 common/，不在 src/
from event_data import load_extract_split, load_detect_split, load_docs, load_schema, dataset_overview

train_rows = load_extract_split("train")     # 5,747 条（每行 1 个事件）
dev_rows   = load_extract_split("dev")       # 930 条
detect     = load_detect_split("dev")        # 5,302 条（含负样本）
docs       = load_docs("dev")                # 985 篇篇章级
print(dataset_overview())                    # 一次打印 schema + 全部规模
```

---

## 4. 调试与常见问题（FAQ）

### Q1：`FileNotFoundError: 未找到本地模型目录：bert-base-chinese`
模型路径解析是**向上逐级查找** `pretrain_models/<模型名>` 并校验真有权重文件。
先跑 `python common/model_paths.py` 看它尝试了哪些路径；确认 `pretrain_models/bert-base-chinese/`
下有 `*.safetensors` 或 `pytorch_model.bin`（只有 `config.json` 不算）。
本项目约定**复用已有权重、不联网下载**，所以这里故意是"报错"而不是"自动下载"。

### Q2：`CUDA out of memory`（8GB 卡）
按顺序试：
1. 确认没有别的 GPU 任务：`nvidia-smi`（本项目约定同一时刻只跑一个 GPU 任务）；
2. `--batch_size 8`（BERT 序列标注）或 `--batch_size 1`（SFT）；
3. SFT 加 `--grad_ckpt`，并把 `--batch_size 2 --grad_accum 8`；
4. 两段式：`--arg_layers 4`（默认就是 4）、`--arg_batch_size 16`；
5. `train_sft.py` 已内置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（治碎片）。

### Q3：SFT 训练 `loss=nan` / 一个 token 都没学到
几乎只有两个原因：
1. `--max_length` 太小 → prompt 被截断、labels 全是 -100。本任务要用 **1280**，不是 NER 项目的 384；
2. 学习率太大（LoRA 用 `2e-4`，全量微调用 `2e-5`）。

训练开始时会打印 `prompt 被完全截断（丢弃）：N 条`，**N>0 就是危险信号**。训练循环里
遇到 `nan` 会跳过该 batch 并提示（这是兜底，不是解法）。

### Q4：本地 Qwen3-0.6B 生成特别慢（150 条要一个多小时）
在 Windows 上这是**已知现象**，不是你的环境坏了：逐 token 解码每步要发 ~200 个 CUDA kernel，
WDDM 下每次 kernel 启动 1~2ms，GPU 一直停在 P8 低功耗档，实测只有 **1.5~2 tok/s**。
解法：用 `--gen_batch 8` 走 `llm_zero_shot/local_generate.py` 的**批量解码**（左 padding），
kernel 数量不变但每个的活变多 → 实测快 **4~6 倍**。
另外：`llm_lora/train_sft.py --help` 曾在某个 `help=` 字符串的裸 `%` 上报 `ValueError: incomplete format`，
现已修好（argparse 里 `%` 必须写成 `%%`）；若你在别处加 help 文本，注意这个坑。

### Q5：`python common/eval_utils.py` 自检不是 1.0000
说明预测行的 key 没带 `sent_id`。DuEE-Fin 的 `id` 是**文档 id 不唯一**，评测按 `id|sent_id` 对齐；
少了 `sent_id` 就会同文档多句互相覆盖（本项目首跑 gold-vs-gold 只有 0.93 F1 就是这个）。
修法：一律用 `make_pred_row(row, events)` 构造预测行。

### Q6：我用 `common/evaluate_events.py` 算出来的 F1 特别低
最常见原因是 `--gold` 用了**全量 dev 930**，而 `--pred` 只有 200 条子集——剩下的 730 条按"全漏"处理，
F1 会被压到真实值的 1/5。要么两边都用同一批样本，要么用 `outputs/llm/gold_n200.jsonl` 当 gold
（`llm_extract.py` / `evaluate_sft.py` 已经自动这么做）。

### Q7：`--mock` / `--dry-run` 跑出来的 F1 能写进报告吗
**不能。** mock 是假模型/假数据，唯一用途是验证"流程通不通"。所有产物里都带 `is_mock` 或文件名带
`mock`，汇总/画图脚本默认会跳过（`--skip probe,smoke,mock,dry`）。只有 `outputs/metrics/**` 下
不带 mock 标记的 JSON 才是实验数字。

### Q8：指标每个 epoch 在涨，但 dev 数字比内层验证掉一截
这通常是**正常**的（内层验证来自 train 同分布，dev 有 201 句与 train 文本重叠但分布更"干净"）。
要警惕的是相反情况：**dev 数字明显高于同难度基线**——先查 `dev_detect` 的 9.2% 文本泄漏，
用 `dev_detect_clean.jsonl` 复核一遍。

### Q9：API 报 401 / 超时 / 大量空回复
- 401：`LLM_API_KEY`（或 `DEEPSEEK_API_KEY`）没设或没有权限；
- 超时/网络错误：`call_api()` 有 3 次指数退避重试（1s/2s/4s），仍失败会记 `error` 并把该条写成 `events: []`，**不删行**；
- 空回复：多半是推理模型的思考 token 吃光了 `max_tokens`。默认 `LLM_DISABLE_THINKING=1` 已关思考；
  若必须开思考，`--max_tokens` 给到 **256 以上**（本项目默认 512）。

### Q10：Qwen3 本地推理输出一大段 `<think>…</think>`
`enable_thinking=False` 没生效。检查是否忘了在 `apply_chat_template(...)` 里传该参数
（`llm_extract.py` 的 `LocalGenerator.generate`、`train_sft.py` 的 prompt 构造、`evaluate_sft.py`
默认都已传）。另外：关掉思考后模板仍会在 assistant 前插入**空的** think 块，这是正常行为，
SFT 里对应位置是 `-100`、不参与 loss，**不要手动删**（会造成训练/推理格式不一致）。

### Q11：切句/篇章抽取结果里论元位置对不上原文
L5 的 `extract_sentence()` 返回的是**句子内坐标**，`extract_document()` 会统一加 `offset` 换成全文坐标。
如果你在句子级别直接调用 `extract_sentence()`，拿到的是句内坐标——这是设计如此（自检脚本可见）。
另外同一片段在原文多次出现时 `text.find()` 取首次，高亮位置可能有偏差（评测只看文本，不受影响）。

### Q12：`service/demo_extract.py` 报"找不到 checkpoint"
它需要 `outputs/checkpoints/bert_crf_256.pt`（联合）或 `pipe_*.pt`（两段式）。先跑：
`python bert/train.py --decoder crf --max_len 256` 或 `python bert_pipeline/train.py`。
只想看演示形态：`--mock`（配合 `python bert/train.py --mock --device cpu` 产生的 checkpoint）。

### Q13：图里的中文变成方块
`eval/make_figures.py` 已设 `font.sans-serif = ["Microsoft YaHei", "SimHei", "Noto Sans SC"]`。
若你的机器这三个都没有，装任意中文字体或改这个列表；`eval/plot_compare.py` 干脆用英文标签，
就是为了绕开这个坑。

### Q14：换成自己公司的公告，需要改哪里
只用 L5：把 `{"id","title","text"}` 传给 `svc.extract_document()` 即可，**不需要改任何代码**。
如果要换成自己的事件体系：改 `data/processed/event_schema.json`（类型 → 角色列表）并重新训练模型；
纯提示词方案只要改 schema，`build_system_prompt()` 会自动跟着变。

### Q15：`--data_dir data/sample` 跑出来的指标和 PPT 对不上
对不上是正常的：小样本只有 800 条训练、300 条评测，且抽样后类型分布与全量不同。
`data/sample` 只用于"验证流程通不通"；PPT 上的数字全部来自全量 `data/processed`。

### Q16：`needs_review` 把几乎所有样本都标成要复核，阈值怎么定
先检查置信度是怎么算的。**绝对不能用概率连乘**：每 token 0.93、68 个 token 连乘只剩 **0.007**，
序列越长分越低，阈值无论怎么调都不对（本项目第一版就踩了这个坑）。
正确做法是**按 token 平均概率**：linear = 各 token 最大概率的均值；CRF = `exp((路径分 − logZ) / 句长)`。
换算之后 0.55 这个阈值才有意义（`ExtractService(review_threshold=0.55)`）。

### Q17：`bert/evaluate_hard_constraint.py` 跑完发现非法数和软约束一模一样

这是本项目踩过的最隐蔽的坑：约束矩阵在 `crf.py` 里注册成 **buffer（`persistent=True`）**，会跟着
checkpoint 一起保存。checkpoint 是软约束训出来的（buffer 全 `True`），于是「先装硬约束 →
`load_state_dict` 又把它覆盖回全 `True`」→ **一条约束都没生效**。
抓法：脚本里有一句防御性检查「已装载：禁止 **N** 个非法转移对」，**N=0 就是没生效**；
修法：`load_state_dict` 之后再装一遍约束，并加 `if n_forbidden == 0: raise RuntimeError`。
修好后的真实效果：argument F1 **0.5815 → 0.6566**。

**那「修好之后还报 155 处」是怎么回事？** token 级严格 **0 处**；字符级口径的 155 处里，
逐条核对发现 **100% 的前一字符是空白** —— 原文的空格（如「9 月 8日」）不产生 token、被强制标 `O`，
跨空格的 span 在字符级 BIO 上就像 `O → I-`。**这是统计粒度问题，不是模型问题。**

### Q18：本地模型 JSON 合法率只有 70%，是模型不行吗

先看原始输出（`outputs/llm/raw_*.jsonl`）。本项目 60 条里有 **18 条**是「少一个 `]`」的畸形 JSON
（典型 `...}]}}`），内容其实都对。用**栈式括号修复**（补右括号/右引号）后合法率 **70% → 100%**，
trigger F1 **0.3299 → 0.4174**。所以：**生成式抽取的指标里，「解析器」是一个真实变量**；
改了解析逻辑不必重跑模型，用 `llm_extract.py --from_raw outputs/llm/raw_xxx.jsonl` 即可。

---

## 5. 产物路径速查

| 产物 | 路径 | 谁产生 |
|---|---|---|
| 预测（判别式） | `outputs/disc/pred_<方法>.jsonl` | `rule_extract / train_bio / train_bilstm_crf / train_pipeline` |
| 指标（判别式） | `outputs/metrics/disc/<方法>.json` + `<方法>_n200.json` | 同上（经 `eval_utils`） |
| 汇总表 | `outputs/metrics/disc/summary.md` | `eval/compare_methods.py` |
| 训练日志（三件套） | `outputs/logs/<method>_{train.json,steps.jsonl,console.log}` | 各训练脚本（`common/log_utils.TrainLogger`） |
| 评测日志 | `outputs/logs/<method>_eval.json` | `rule/`、`llm_zero_shot/`、`bert/evaluate*.py`、`llm_lora/evaluate_sft.py` |
| 实验流水 | `outputs/logs/run_<name>.out`、`run_summary.json` | `eval/run_experiments.py` |
| 判别式 checkpoint | `outputs/checkpoints/*.pt` | 训练脚本 |
| PPT 图 | `outputs/figures/*.png` | `eval/make_figures.py` |
| LLM 预测/原始输出 | `outputs/llm/pred_<tag>.jsonl`、`raw_<tag>.jsonl` | `llm_extract / evaluate_sft` |
| LLM 工程指标 | `outputs/llm/report_<tag>.json` | 同上 |
| LLM 指标 | `outputs/metrics/llm/<tag>.json` | 同上（subprocess 调统一评测） |
| 固定抽样集 | `outputs/llm/gold_n{k}.jsonl`、`eval_ids.json` | `llm_extract / evaluate_sft` |
| LoRA / 全量 ckpt | `outputs/llm/ckpt/sft_adapter/`、`sft_full_ckpt/` | `llm_lora/train_sft.py` |
| LLM 对比图 | `outputs/llm/method_compare.png` / `.csv` | `eval/plot_compare.py` |
