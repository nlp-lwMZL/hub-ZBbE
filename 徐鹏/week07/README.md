# event_extraction —— 金融公告事件抽取（week7 教学项目）

> **任务（本项目口径）**：给定**一句**公告原文，抽出**一个**结构化事件 —— 事件类型、**触发词** span、
> 以及若干「**角色 → 论元** span」（公司、金额、时间…）。**一行 = 一句话 + 恰好 1 个事件。**
> 数据：DuEE-Fin 派生（13 事件类型 / 92 个「类型-角色」槽位 / 61 个去重角色名）；
> `data/processed/` 随包发：train **5,747** 句 / dev **930** 句（主评测集）/ 检测集 32,794+5,302 / 篇章 985 篇。
> `data/sample/` 只是 3,250 条**快速冒烟**子集，正式指标一律跑 `data/processed/`。

这个项目的目标是**同一口径下横向比较 6 种做法**，让每个数字都能写进简历：

| 层次 | 方法 | 目录 | 一句话 |
|---|---|---|---|
| L0 | 规则（词典+正则） | `rule/` | 零训练基线，说明"不用模型能拿到多少" |
| L1 | 字符级 BiLSTM-CRF | `bilstm_crf/` | 无预训练，说明"预训练值多少点" |
| L2 | BERT+Linear / BERT+CRF | `bert/` | 主线：一次联合抽取（149 标签） |
| L2 | 两段式流水线 | `bert_pipeline/` | 检测→类型→类型条件论元，说明"拆细的收益与代价" |
| L3/L4 | LLM zero/few-shot | `llm_zero_shot/` | 生成式零样本 / 少样本 |
| L4 | LoRA SFT | `llm_lora/` | 生成式微调（小模型 + 9MB adapter） |
| L5 | 系统组合 | `service/` | 检测→抽取→规则兜底→校验→needs_review→篇章事件流 |

> **week07 作业增量：换模型实验**。`model_swap/` 把「无模型 / BiLSTM / BERT / 两段式 /
> Qwen3-0.6B / Qwen2-0.5B / Qwen2.5-0.5B-Instruct / API」放在同一张 n=200 表上，
> 带 bootstrap 95% 置信区间、延迟与 JSON 合法率。运行 `python model_swap/compare_models.py`
> 生成 `outputs/model_swap/REPORT.md`；结论见根目录 `MODEL_SWAP_ANALYSIS.md`。

---

## 目录结构（每个方法一个文件夹）

```
event_extraction/
├── common/            # 公共模块（不含"方法"）：数据层/评测/CRF/模型/日志
│   ├── event_data.py      标签体系 + span↔BIO + 子词对齐 + 解码 + 非法统计
│   ├── evaluate_events.py ★唯一评测口径（trigger/argument/event 三级 F1）
│   ├── eval_utils.py      评测封装 + 计时/峰值显存 + n=200 子集投影
│   ├── crf.py             手写 CRF（前向算法/维特比/硬约束）
│   ├── models.py          公共模型定义（BertEncoder/BertCls/build_model/predict_char_tags）
│   ├── model_paths.py     定位本地预训练模型（不联网下载）
│   └── log_utils.py       统一训练日志（TrainLogger / EvalLogger）
├── rule/              run_rule.py          L0 规则基线
├── bilstm_crf/        train.py             L1 字符级 BiLSTM-CRF
├── bert/              train.py             L2 BERT+Linear / BERT+CRF（--decoder）
│                      evaluate.py          从 checkpoint 复评
│                      evaluate_hard_constraint.py  解码期硬约束
├── bert_pipeline/     train.py             L2 两段式（检测器+13分类+13专家）
├── llm_zero_shot/     llm_extract.py       zero/few-shot 抽取（本地/API）
│                      event_prompt.py      prompt 模板 + 输出解析
│                      event_data.py        LLM 链路数据工具
│                      local_generate.py    本地批量生成
├── llm_lora/          train_sft.py         LoRA / 全量微调
│                      evaluate_sft.py      评测（自动识别 LoRA/全量）
│                      model_paths.py       定位 Qwen3-0.6B
├── service/           extract_service.py   系统组合（校验/兜底/篇章合并）
│                      demo_extract.py      现场演示 CLI
├── eval/              compare_methods.py   汇总对比表
│                      make_figures.py      6 张 PPT 用图
│                      check_bio_gold.py    金标非法 BIO 地板值
│                      postprocess_trigger_copy.py  触发词回填后处理
│                      plot_compare.py      LLM 链路对比图
│                      run_experiments.py   一键复现全部实验
├── tools/             make_sample_data.py  学生包小样本抽样
├── data/              processed/（全量） sample/（学生包）
│                      _mock/（**手造 20 条，给 --mock 冒烟用，别删**） README.md
└── outputs/           checkpoints/ disc/（预测） metrics/ logs/ figures/ demo/ llm/
```

**依赖关系**：方法目录 → `common/`（只允许单向依赖，方法之间互不 import）。
方法脚本开头统一：

```python
ROOT = Path(__file__).resolve().parents[1]   # 方法目录的上一级 = 项目根
sys.path.insert(0, str(ROOT / "common"))
```

---

## 快速开始

```bash
# 0) 环境：torch 2.6 + transformers 5.5；预训练模型复用 pretrain_models/，不下载
pip install -r requirements.txt

# 1) 用学生包小样本跑通全流程（几个 MB，几分钟）
python tools/make_sample_data.py --from_sample
python rule/run_rule.py                                   # L0（秒级，纯 CPU）
python bert/train.py --data_dir data/sample --device cpu --epochs 1 --bert_layers 4
python eval/compare_methods.py                            # 汇总表

# 2) 复现 PPT 上的完整数字（需要全量 data/processed/ 与 GPU）
python eval/run_experiments.py --group main               # BERT+Linear / BERT+CRF 主线
python bilstm_crf/train.py --device cpu --epochs 3
python bert_pipeline/train.py
python llm_lora/train_sft.py --epochs 1 --num_train 3500
python eval/compare_methods.py --with_n200
python eval/make_figures.py

# 3) 现场演示（篇章级事件流）
python service/demo_extract.py --n 3
```

---

## 训练日志（学生可复看训练过程）

每次训练都会在 `outputs/logs/` 留下三类文件：

- `<method>_train.json` —— 超参、环境、每 epoch 的 loss/验证/耗时、参数量、峰值显存
- `<method>_steps.jsonl` —— **每 50 步一行**（`--log_every` 可调），用来画 loss 曲线
- `<method>_console.log` —— 完整控制台输出

纯评测方法（规则 / zero-shot / evaluate_*）写 `<method>_eval.json` + `_console.log`。
字段说明、画图示例与「新旧数值复现说明」见 **`outputs/logs/README.md`**。

---

## 结果与文档

| 文档 | 内容 |
|---|---|
| `NOTES_discriminative.md` | **判别式链路**的完整记录：设计决策、全部命令、实测数字、踩坑表、「PPT 可用的数字与金句」 |
| `NOTES_llm.md` | **LLM 链路**（zero/few-shot、LoRA）的记录 |
| `ARCHITECTURE.md` | 技术方案（选型对比、流水线图、实验结果、Bad Case、优化方向） |
| `USAGE_GUIDE.md` | 各脚本的调用与测试指南（命令、参数、预期输出、FAQ） |
| `RESUME_GUIDE.md` | 求职简历指导（可量化的数字、岗位写法、面试题） |
| `data/README.md` | 数据来源/许可/字段契约/**四个数据坑** |
| `outputs/logs/README.md` | 训练日志字段说明 + 画 loss 曲线示例 |

**评测口径的唯一性**：所有方法（含 LLM 链路）都把自己的预测写成同一种 JSONL，
交给 `common/evaluate_events.py` 算指标。任何"看起来更好"的结论都必须能在这同一张表上复现。
