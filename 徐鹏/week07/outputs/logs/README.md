# outputs/logs/ —— 训练/评测日志说明

> 课的**训练过程**没法在直播里完整演示，所以每个方法跑完都留一份结构化日志，学生可以自己复看。

## 文件命名

| 文件 | 谁产生 | 内容 |
|---|---|---|
| `<method>_train.json` | 训练类方法 | 结构化总结：`argv`、`args`（超参）、`env`（python/torch/GPU）、`params`（参数量）、`epochs[]`（每 epoch 的 train_loss / 验证指标 / 耗时）、`total_elapsed_s`、`finished`，以及收尾字段（best_epoch / dev_metrics / peak_gpu_mb / illegal_dev…） |
| `<method>_steps.jsonl` | 训练类方法 | **每 `--log_every`（默认 50）步一行**：`{"epoch","step","loss","lr","elapsed_s","step_elapsed_s"}` |
| `<method>_console.log` | 全部方法 | 该次运行的完整 stdout（屏幕上看到的一样） |
| `<method>_eval.json` | 纯评测方法（rule / zero-shot / evaluate_*） | `argv`、`args`、`env`、`runtime_s`、`metrics`（三级 F1）、以及该方法的特有统计（如规则基线的失败分布、LLM 的 JSON 合法率/幻觉数） |

`--log_every` 只在训练脚本里有（默认 50）。想看得更细：`--log_every 10`。

## 为什么日志要分三类

- `_train.json` 回答「**这次到底用什么参数、跑了多久、最好在哪一轮**」——复现和写报告用；
- `_steps.jsonl` 回答「**loss 是怎么降的**」——画曲线、看有没有发散/平台期；
- `_console.log` 回答「**当时屏幕上一共打印了什么**」——排错用（比如某次 OOM 之前的最后几行）。

## 画 loss 曲线（10 行）

```python
import json
import matplotlib.pyplot as plt

rows = [json.loads(l) for l in open("outputs/logs/bert_crf_256_steps.jsonl", encoding="utf-8")]
xs = [r["elapsed_s"] / 60 for r in rows]                 # 横轴用「训练分钟」比用 step 更直观
ys = [r["loss"] for r in rows]
plt.plot(xs, ys, marker="o", ms=3)
plt.xlabel("训练时间 (分钟)"); plt.ylabel("loss")
plt.title("BERT+CRF 训练 loss 曲线"); plt.grid(alpha=.3)
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
```

> 注意：本数据集里 **loss 的绝对值不可跨方法比较**（CRF 的 loss 是 `logZ − 金标路径分数`，
  量纲和线性头的交叉熵完全不同，149 标签下能到 200+）。**要比就看曲线形状与验证指标。**

## 画「每 epoch 验证指标」

```python
import json
doc = json.loads(open("outputs/logs/bert_crf_256_train.json", encoding="utf-8").read())
for e in doc["epochs"]:
    print(e["epoch"], e["train_loss"], e["val"])
```

---

## 数值复现说明（先看这段）

本目录是 **2026-11-18 代码按方法重构后的重跑记录**。判别式 8 个方法（rule / bilstm_crf / bert_crf / 两条消融 / 硬约束）
**0.00pp 复现**；少数方法差 <0.5pp，属训练/采样噪声：

| 方法 | 11/16 旧值 | 11/18 重跑 | 说明 |
|---|---|---|---|
| bert_linear_256 | 0.8304 / 0.5625 / 0.1564 | 0.8293 / 0.5630 / 0.1564 | ≤0.11pp |
| api_zero_n200 | arg 0.5246 | arg 0.5224 | ≤0.22pp |
| api_few_n200 | 0.8291 / 0.5347 / 0.0905 | 0.8321 / 0.5395 / 0.0852 | event −0.53pp（n=200，1 条≈0.5pp） |
| local_zero_n60 | 0.4174 / 0.1129 | 0.4068 / 0.1167 | n=60，1 条≈1.7pp |
| llm_lora | 0.9100 / 0.6966 / 0.2050 | **0.9000 / 0.6971 / 0.1850** | 训练可复现；eval 差异属生成随机性 |
| llm_lora type_macro | 0.9150 | **0.7862** | 模型输出了 schema 外类型名，macro 分母被撑大 → 必须做类型白名单校验 |

> PPT 上：LoRA 行已按 11/18 重训值更新；其余行保留 11/16 首跑值（与重跑差异在噪声内）。
> 旧的 `_legacy/` 目录（11/16 的 run_*.log 等历史流水）已在 2026-11-18 清理时删除；
> 需要追溯旧一轮的完整 stdout，看对应的 `_console.log` 即可（新版日志已覆盖同样信息）。

> **读日志前两句说明**：本目录日志里的路径已规范成**仓库相对路径**（如 `data/processed/...`、
> `pretrain_models/bert-base-chinese`），API 日志中的 `base_url` 是**公开的 DeepSeek 官方地址**，
> 便于阅读与复现，不含任何机器相关的私有信息。

### ⏱ 关于训练耗时：11/18 的耗时被同机任务争用，不能用来算「慢多少倍」

11/18 重跑时 CPU 上**同时**在跑 `bilstm_crf`（3 epoch CPU 训练），而 `bert_linear_256` / `bert_crf_256`
都要靠 CPU 做 tokenize / dataloader，因此这两条的耗时**被争用抬高**：

| 指标 | 首跑（11/16，无并行任务） | 11/18 重跑（有并行 CPU 任务） |
|---|---|---|
| `bert_linear_256` 训练 | **531 s** | 637 s |
| `bert_crf_256` 训练 | **3112 s** | 2742 s |

→ 所以凡是「**CRF 训练慢 5.9×**」「**推理慢 3.3×**」这类对比，一律采用**首跑（无并行任务）口径：
531 s → 3112 s**。11/18 的 637 s / 2742 s 只能说明**该次运行的实际情况**，
**不能拿来重算倍率**（一次偏高、一次偏低，方向都不一致，正是争用造成的）。

## 本轮重跑的「日志可用性」清单（2026-11-18 重构后）

| 方法 | 日志状态 | 说明 |
|---|---|---|
| `rule` | ✅ 新 `_eval.json` + `_console.log` | 纯评测 |
| `bilstm_crf` | ✅ 新 `_train.json` / `_steps.jsonl` / `_console.log` | CPU 3 epoch |
| `bert_linear_256` / `bert_crf_256` / `bert_linear_128` / `bert4L_linear_256` | ✅ 新三件套 | 数字与历史值逐位一致 |
| `api_zero_n200` / `api_few_n200` | ✅ 新 `_eval.json` + `_console.log` | 云端 API |
| `llm_lora` | ✅ 新三件套（重训 3500 条 1 epoch） | 见下面「配置坑」 |
| `local_zero_n60` | ✅ 新 `_eval.json` | 本地 Qwen3-0.6B |
| **`bert_pipeline`** | ⚠ **沿用 2026-11-16 的指标与日志** | 本次重跑未成功，原因见下 |

### ⚠ `bert_pipeline` 本次为什么没有新训练日志
代码本身已修好（`train_classifier` 的 `step` 未定义 → 改 `enumerate`；`logger` 被误插到
`train_arg_experts()` 局部作用域 → 移到 `main()`），并用 `--mock` 验证过 exit=0、
`_steps.jsonl` 能正常写出（每行带 `stage` 字段区分检测器/类型/13 个专家）。

但**真跑时异常慢**：压缩配置下检测器（8000 条、125 个微批）跑了 19 分钟仍未完成
（≈9 s/微批；同期 `bert/train.py` 只要 ~0.5 s/微批）。在当晚 21:00 的硬截止前无法完成，
于是**保留 2026-11-16 的实测指标**（trigger .6769 / arg .3690 / event .0957）与其对应日志，
不新造数字。复跑命令（与历史值同配置）：

```bash
python bert_pipeline/train.py --detect_limit 8000 --cls_epochs 1        --arg_epochs 3 --arg_lr 8e-5 --arg_layers 4
```

### ⚠ LLM（LoRA）训练的配置坑：少一个 `--grad_ckpt` 就差 4 倍
同一张 8GB 卡、同一个脚本、同样 3500 条 1 epoch，两种配置实测：

| 配置 | 每微批耗时 | 峰值显存 | 预计总时长 |
|---|---|---|---|
| `--num_train 3500`（batch 默认 2、**无** `--grad_ckpt`） | **8.3 s/it** | **7.9 GiB** | ≈ 4 小时（不可用） |
| `--batch_size 3 --grad_accum 5 --grad_ckpt --dtype bf16` | **1.9 s/it** | **4.0 GiB** | ≈ 38 分钟 |

结论：**8GB 卡上跑 SFT，梯度检查点不是"可选项"而是"能不能跑完"的分水岭**；
另外 `--dtype auto` 在 CUDA 上已经解析为 bf16（不是 fp32，这一点用计速排除了）。
