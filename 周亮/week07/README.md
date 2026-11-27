# week7 作业：CDTier 网络安全语料命名实体识别（NER）—— 三类方法对比

## 0. 摘要（30 秒看懂）

- **任务**：在安全威胁情报文本上做字符级序列标注 NER，抽取 Attacker / Campaign / Industry / Region / Tools 五类实体。
- **数据**：CDTier，按原文分组切分 train 3091 / validation 387 / test 387 句，跨 split 文本重叠 0（无泄漏），共 5 类 / 11 个 BIO 标签。
- **方法**：三类范式共 6 个模型——字符级 BiLSTM±CRF、RoBERTa-wwm-ext±CRF、Qwen3-0.6B LoRA/full SFT（生成实体 span JSON）。
- **结果**：test 实体级 F1 从 0.177（BiLSTM-Linear）到 **0.684**（Qwen-full-SFT）；预训练增益 +28.3 点，生成式再 +19.4 点。
- **运行**：Colab T4 一键脚本 `colab/run_all.sh`，产物落 Google Drive，支持断线续跑。

完整结果见 [`outputs/eval/summary_test.md`](outputs/eval/summary_test.md)。

---

## 1. 任务与数据

### 1.1 为什么选 NER

老师课堂示例是事件抽取（类型 + 触发词 + 论元）。本作业选取其**实体识别子问题**，聚焦于课堂强调的主线——**同一评测口径下横向对比不同方法，量化预训练与大模型的收益**。为此保留了老师方法链的核心环节：BIO 标注口径、word_ids 子词对齐、CRF 转移约束、生成式输出与 span 回填、无泄漏切分、实体级 F1 评测；未展开的是事件层的触发词/角色组装。

数据集选 CDTier 而非老师用的 DuEE-Fin，因为前者是安全领域真实语料、与 NER 任务口径直接匹配，且三类方法可在同一份数据上完整对比。调研过的完整事件抽取数据集（CASIE：5 类事件 + 约 20 种论元角色，HF `mneb/casie`；中文 CCSE/CSED：前者需向作者申请，后者只有事件检测、无论元角色）记录在 §7。

### 1.2 数据与切分

| 项 | 内容 |
|---|---|
| 原始数据 | CDTier 网络安全实体标注语料 |
| 采用版本 | `data/cdtier/clean/`——按文本分组去重切分，跨 split 文本重叠 = 0 |
| 规模 | train 3091 句 / 3738 实体，validation 387 / 443，test 387 / 432 |
| 实体类型 | 5 类：Attacker / Campaign / Industry / Region / Tools |
| 标签体系 | 11 个：`O` + `B-/I-` × 5（`label_names.json`） |
| 数据探索 | 类别不均衡 14.82×（Attacker 1156 vs Campaign 78）；句长 p50=66 / max=2232；max_len=256 截断率 5.66% |

切分依据：原始语料存在 269 组重复文本、24 组标注冲突；若随机按句切分，同一文本会同时出现在 train 与 dev/test，造成指标虚高（老师 PPT 反复强调的坑）。采用 `clean/` 后重叠为 0，且 span 偏移一律用 `text.find()` 从原文重算，不信任原始 offset。

数据探索详情见 [`outputs/eval/DATA_REPORT.md`](outputs/eval/DATA_REPORT.md)。

---

## 2. 方法

### 2.1 三类方法（6 个模型）

| 层级 | 方法 | 目录 | 回答的问题 |
|---|---|---|---|
| **L1** | 字符级 BiLSTM + Linear / CRF（无预训练） | `bilstm_crf/` | 从零训字向量能到多少？预训练值多少点？ |
| **L2** | RoBERTa-wwm-ext + Linear / CRF | `bert/` | 预训练判别式的主线水平？CRF 比 Linear 强多少？ |
| **L4** | Qwen3-0.6B LoRA / full SFT | `llm_lora/` | 生成式能否追上判别式？LoRA vs 全量、代价（幻觉/延迟）如何？ |

SFT 输出格式是实体 span 列表 JSON；评测时用 `text.find()` 把生成的实体文本回填成 `(type, start, end)`，再做类型白名单校验与 JSON 容错解析。

**L4 两种微调模式**：

| 模式 | 训什么 | 学习率 | 显存 |
|---|---|---|---|
| `lora` | 只训旁路（r=8） | 单一 lr | 宽松 |
| `full` | 全量微调 | **分层学习率**（layer-wise decay 0.95 + embed/head/norm 单独组） | 需 `--grad_ckpt --optim adamw8bit --dtype fp32 --lr 2e-5`（fp16 + 2e-4 会 loss=nan） |

**措辞说明**：老师 PPT 里的"消融"严格定义为"一次只改一个旋钮"。上述三类是**不同范式**，称"方法对比"；类内的 `Linear vs CRF`、`BiLSTM vs RoBERTa`、`LoRA vs full` 才是单变量消融。

### 2.2 关键实现

- **子词对齐**：`common/ner_data.py` 用 `word_ids()` 只让首子词背标签，非首子词 / CLS/SEP/PAD / 截断残段一律 -100 不参与 loss。
- **长句滑窗**：`common/sliding_window.py` 对超长样本切带重叠窗口分别推理，span 映射回原句坐标后按 IoU≥0.5 去重合并，救回被截断的尾部实体（test 有 22 条 >256）。
- **统一评测**：所有臂输出同一种 `pred_spans` JSONL，喂同一个 `common/metrics.evaluate`，实体级 `(type,start,end)` 完全匹配；另报 type_macro_F1 观察长尾类。
- **生成式可靠性**：JSON 三层容错（剥围栏→括号配平→截断补全）、类型白名单、offset 回填失败单独计数。
- **工程**：warmup + linear decay、梯度剪裁、断点续跑；产物落 Drive，底座权重留本地。

**可比性的三条纪律**：
1. 换方法不换评测——六臂共用同一个评测函数；
2. 选模用 train 切 10% 的内层验证集，dev/test 只在最终评测使用；
3. loss 绝对值不跨方法比（CE / CRF-NLL / LM-loss 量纲不同），只看实体级 F1。

---

## 3. 实验结果

### 3.1 主表（test，387 句）

| 层级 | 方法 | P | R | F1 | type_macro_F1 |
|---|---|---:|---:|---:|---:|
| L1 | BiLSTM+linear | 0.2062 | 0.1551 | 0.1770 | 0.1486 |
| L1 | BiLSTM+crf | 0.2846 | 0.1620 | 0.2065 | 0.1719 |
| L2 | RoBERTa+linear | 0.4461 | 0.4977 | 0.4705 | 0.3873 |
| L2 | RoBERTa+crf | 0.4496 | 0.5370 | 0.4895 | 0.4043 |
| L4 | Qwen3-0.6B LoRA SFT | 0.7160 | 0.5370 | 0.6138 | 0.5627 |
| L4 | Qwen3-0.6B full SFT | 0.7650 | 0.6181 | **0.6837** | **0.6528** |

### 3.2 关键结论

1. **预训练收益**：BiLSTM-CRF 0.2065 → RoBERTa-CRF 0.4895（+28.3 点），是本作业最大的单项收益。
2. **CRF 边际收益递减**：BiLSTM 上 +2.95 点（0.1770→0.2065），RoBERTa 上仅 +1.9 点（0.4705→0.4895）——强编码器下 CRF 的约束作用变小。
3. **生成式收益**：RoBERTa-CRF 0.4895 → full SFT 0.6837（+19.4 点），代价是单条延迟 0.42s 与重叠抽取。
4. **错误模式差异**（[`errors_report.md`](outputs/eval/errors_report.md)）：判别式的错误集中在**边界**（98–139 处），类型错极少（1–8 处）；生成式边界错仅 18/21 处，但**多预测 129/135**，且 span 重叠 84/151 对（判别式为 0）——生成式没有 BIO 的单标签约束。
5. **长尾问题**：Campaign 仅 78 条训练实体，type_macro_F1 明显低于 overall F1；case 表中可见它被误判为 Attacker（"血茜草活动"）。

成本对比（参数量 / 训练时长 / 延迟）见 [`cost_table.md`](outputs/eval/cost_table.md)；六臂并排推理 case 见 [`case_table.md`](outputs/eval/case_table.md)；6 张图在 `outputs/figures/`。

---

## 4. 目录结构

```
code/
├── common/                公共层
│   ├── paths.py           路径解析：产物→Drive，底座→本地
│   ├── ner_data.py        BIO 数据层 + word_ids 对齐 + 字符级 CharDataset
│   ├── metrics.py         ★唯一评测口径 + 预测 JSONL 读写 + 非法 BIO 统计
│   ├── sliding_window.py  长句滑窗推理
│   └── model_paths.py     RoBERTa / Qwen3 权重解析与下载
├── bilstm_crf/            L1：model.py / train.py / evaluate.py
├── bert/                  L2：model.py / train.py / evaluate.py
├── llm_lora/              L4：sft_data.py / train_sft.py / evaluate_sft.py / llm_utils.py
├── eval/                  汇总表 / 一键调度 / 错误分析 / 成本表 / 图表
├── tools/                 数据探索 / 不变量自检
├── colab/                 Colab 环境、下载、一键脚本（详见 colab/README.md）
├── data/cdtier/clean/     train / validation / test / label_names
├── outputs/               eval 指标 / preds 预测 / figures 图表
├── logs/                  各臂训练日志与超参
├── week7.ipynb            VSCode Colab 扩展用的 notebook
└── requirements.txt
```

---

## 5. 复现方式

### Colab（正式实验，T4 GPU）

```python
from google.colab import drive; drive.mount('/content/drive')
```

```bash
!mkdir -p /content/week7 && cp -r /content/drive/MyDrive/week7/* /content/week7/
!cd /content/week7 && bash colab/setup_env.sh       # 装依赖 + 下模型 + 体检
!cd /content/week7 && bash colab/run_all.sh --quick # 冒烟
!cd /content/week7 && bash colab/run_all.sh         # 正式训练（断线重跑同一命令即续）
!cd /content/week7 && bash colab/run_all.sh --eval  # 评测 + 汇总 + 分析
```

完整流程、断连策略、报错对照见 [`colab/README.md`](colab/README.md)。本机 CPU 只能跑冒烟（有 bert-base-chinese，无 RoBERTa/Qwen3 权重）。

### 离线分析（不重跑模型）

```bash
python tools/analyze_data.py        # → DATA_REPORT.md
python eval/analyze_errors.py       # → errors_report.md / case_table.md
python eval/illegal_bio.py          # → illegal_bio_report.md
python eval/cost_table.py           # → cost_table.md
python eval/make_figures.py         # → outputs/figures/*.png
```

---

## 6. 已落地的老师 PPT 要点

| # | 要点 | 代码落点 |
|---|---|---|
| 1 | 数据泄漏：dev 混入 train 文本 | 用 `clean/` 按文本切分，重叠 = 0 |
| 2 | 子词对齐：只有首子词背标签 | `ner_data.py` 的 `word_ids()` + `-100` |
| 3 | CRF mask 不能挖洞 | `bert/model.py`、`bilstm_crf/model.py` 用连续 attention_mask |
| 4 | 用 dev 选 epoch 会抬指标 | 两处 `train.py` 从 train 切 10% 内层验证 |
| 5 | loss 量纲不同不能横向比 | `compare.py` 只报实体级 F1 |
| 6 | 生成式 offset 不可靠 | `llm_utils.align_spans_to_text` 用 `text.find()` |
| 7 | JSON 畸形/围栏/截断 | `llm_utils` 三层容错 |
| 8 | 类型幻觉撑大分母 | `VALID_TYPES` 白名单 + `illegal_type` 计数 |
| 9 | Qwen3 默认开思考污染输出 | `enable_thinking=False` |
| 10 | 解析失败不能删行 | `metrics.evaluate` 对缺预测直接报错 |
| 11 | 截断率必须报 | `ner_data.py` 统计 truncated + 评测期滑窗 |
| 12 | 产物落 Drive、断线可续 | `paths.py` + `run_all.py` 断点续跑 |

---

## 7. 局限与讨论

- **任务口径**：本作业做 NER，未做事件抽取的触发词/论元角色组装。调研过的完整事件抽取数据集：CASIE（英文，5 类事件 + 约 20 种角色，HF `mneb/casie`，无官方 train/dev 划分）、CCSE/CBSEE（中文，需向作者申请）、CSED（只有事件检测，无论元角色）。更换数据意味着重写标签体系与评测口径，本次保留 NER 任务。
- **长尾**：Campaign 78 条，class weight / focal loss / logit 调整未实跑，是明确列出的后续改进方向。
- **未实测项**：显存峰值、判别式推理延迟未采集（报告中标"未采集"而非估算）；非法 BIO 的 Linear vs CRF 精确对比需重跑评测导出原始 BIO（`--dump_raw_tags`），当前报告给出的是可从产物直接观测的 span 重叠对比。
- **成本表口径**：LoRA 可训比例约 0.15% 为按 r=8 的结构估算，非 PEFT 实测；RoBERTa 102.3M / Qwen3 0.6B 为公开模型规格。
