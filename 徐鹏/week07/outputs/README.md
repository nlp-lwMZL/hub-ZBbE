# outputs/ 说明（产物与数字溯源）

> **一句话结论**：`outputs/` 里每个数字都能反查到一条命令。
> 看到文档/PPT 上的某个 F1，按 `方法名 → metrics/<方法名>.json → meta` 一路查下去就能找到
> 它的超参、训练时长、延迟、数据口径。

## 1. 目录速查

| 路径 | 内容 | 谁写 |
|---|---|---|
| `checkpoints/` | 判别式模型：`bert_linear_256.pt` / `bert_crf_256.pt` / `bilstm_crf.pt` / `pipe_detector.pt` / `pipe_type.pt` / `pipe_arg_<类型>.pt` | `方法目录/train.py` |
| `logs/<method>_train.json` | **训练日志**：`argv` / `args`（超参）/ `env`（python·torch·GPU）/ 每 epoch 的 train_loss + 内层验证指标 + 耗时 / 总耗时 / 参数量 / 峰值显存 | 各训练脚本（经 `common/log_utils.TrainLogger`） |
| `logs/<method>_steps.jsonl` | **逐步曲线**：每 `--log_every`（默认 50）步一行 `{epoch, step, loss, lr, elapsed_s}`，画 loss 曲线就读它 | 同上 |
| `logs/<method>_console.log` | 该次运行的完整 stdout（屏幕上看到的一样），排错用 | 同上 |
| `logs/<method>_eval.json` | **纯评测方法**（rule / zero-shot / evaluate_*）的日志：`args` / `env` / `runtime_s` / `metrics` + 特有统计 | `common/log_utils.EvalLogger` |
| `logs/run_<name>.out`、`run_summary.json` | **一键复现的运行日志**（每条命令的完整输出、耗时、退出码）—— 与上面的"训练日志"是两回事，别混 | `eval/run_experiments.py` |
| `disc/pred_<方法>.jsonl` | 判别式预测（统一评测输入，`{id, sent_id, events}`） | 各方法脚本（经 `common/eval_utils.save_predictions`） |
| `metrics/disc/<方法>.json` | 该方法的全部指标 + **meta 溯源字段**（超参/训练时长/延迟/显存/非法 BIO） | 各方法脚本（`common/eval_utils.evaluate_predictions`） |
| `metrics/disc/<方法>_n200.json` | 同一份预测**投影到固定 n=200 子集**后的指标（跨链路可比） | `common/eval_utils.evaluate_nested_subset` |
| `metrics/disc/summary.md` | 汇总对比表（含主表 + 消融；缺失方法会显式标注"尚未跑出"） | `eval/compare_methods.py` |
| `figures/*.png` | PPT 用图（方法对比/非法 BIO/per-role/曲线/混淆矩阵/延迟） | `eval/make_figures.py` |
| `llm/pred_<tag>.jsonl` | LLM 预测（解析失败的行保留 `events: []`） | `llm_zero_shot/llm_extract.py` / `llm_lora/evaluate_sft.py` |
| `llm/raw_<tag>.jsonl` | 模型**原始输出**（bad case 分析用） | 同上 |
| `llm/report_<tag>.json` | 工程指标：JSON 合法率/修复/空回复/幻觉数/延迟/token | 同上 |
| `llm/gold_n<k>.jsonl`、`eval_ids.json` | 固定抽样集的**金标**与 **id 列表**（seed=42，n=150/100 是 n=200 的前缀） | `llm_zero_shot/event_data.py` |
| `llm/ckpt/sft_adapter/` | LoRA adapter（`--full_ft` 时为 `sft_full_ckpt/`） | `llm_lora/train_sft.py` |
| `llm/ckpt/train_log_sft.json` | SFT 训练日志（含 `prompt_truncated` / `gpu_peak_mem_gib` / 可训练参数） | 同上 |
| `llm/method_compare.png` / `.csv` | LLM 侧方法对比图与同一份数字的 CSV | `eval/plot_compare.py` |
| `llm/badcase_unit_denominator.md` | 数量类论元口径 bad case（`9.18万` vs `9.18万股`） | 人工/脚本整理 |

> **训练日志三件套的字段含义、`--log_every` 用法、"画 loss 曲线"的 10 行代码**，见
> **`outputs/logs/README.md`** —— 那是学生课后复看训练过程的入口。

### 1.1 不进交付包的内容

课件包发**代码 + 文档 + `data/processed/` 完整派生数据 + `data/sample/` 冒烟子集 + 图表/指标**，下面这些**不要打进包**（体积大、含权重或机器相关产物）：

| 不打包 | 原因 |
|---|---|
| `checkpoints/`（各 `.pt`，单个 390MB+） | 权重文件，学生自己训练即可生成 |
| `llm/ckpt/`（含 `sft_adapter/`、`sft_full_ckpt/`） | 同上 |
| `llm/raw_*.jsonl`（原始回复，可能很大） | 体积大，且含模型原始输出文本 |
| `logs/run_*.out`（一键复现的流水） | 只在复现实验时需要，机器相关 |
| ~~`logs/_legacy/`~~ | **已在 2026-11-18 清理时删除**；旧一轮 stdout 可从对应 `_console.log` 追溯 |

## 2. 怎么从数字反查命令

```bash
# 1) 看某个方法的结果与超参
python -c "import json;d=json.load(open('outputs/metrics/disc/bert_linear_256.json',encoding='utf-8'));print(d['trigger'],d['argument'],d['meta'])"

# 2) 找到产生它的命令
python -c "import json;print(json.load(open('outputs/logs/run_summary.json',encoding='utf-8')))"

# 3) 重新算一遍（任何预测都可以用同一个评测函数重评）
python common/evaluate_events.py --gold data/processed/dev_extract.jsonl \
        --pred outputs/disc/pred_bert_linear_256.jsonl
```

## 3. 三条纪律（读数字前先看）

1. **先看 `is_mock` / 文件名**：带 `mock`、`smoke`、`probe`、`dry` 的是流程验证，**数字无意义**，
   汇总与画图脚本默认跳过它们。
2. **先看口径**：`metrics/disc/*.json` 默认是 dev 930 全量；`*_n200.json` 是固定 200 条子集；
   `metrics/llm/*.json` 默认是 n=200/150 子集。**不同口径的数字不能直接并排比**。
3. **先看 `meta`**：`epochs / max_len / decoder / bert_layers / hard_constraint / selection_note`
   都在里面。PPT 图注与文档结论必须写"读到的"而不是"记得的"参数。
