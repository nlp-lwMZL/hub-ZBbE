# 成本表：六臂对比（老师 PPT P42 的五个数）

> split = `test`；生成于 2026-11-25T12:37:35

**数据来源纪律**：每个数字标注来源。`实测`=从产物读，`结构`=从模型结构算，
`规格`=公开模型规格，`未采集`=产物里没有（不编造）。

| 方法 | F1 | 总参数 | 可训参数(比例) | 训练时长 | 显存峰值 | 推理延迟 |
|---|---:|---:|---:|---:|---:|---:|
| BiLSTM+Linear | 0.177 | 1.1M | 1.1M (100.00%) | 10s | — | — |
| BiLSTM+CRF | 0.2065 | 1.1M | 1.1M (100.00%) | 105s | — | — |
| RoBERTa+Linear | 0.4705 | 102.3M | — | 253s | — | — |
| RoBERTa+CRF | 0.4895 | 102.3M | — | 340s | — | — |
| Qwen3-LoRA-SFT | 0.6138 | 600M | 900000 (0.15%) | 1257s | — | 0.633s |
| Qwen3-full-SFT | 0.6837 | 600M | 600M (100.00%) | 3004s | — | 0.42s |

## 逐列来源与解读

| 方法 | F1 来源 | 参数量来源 | 时长来源 | 延迟来源 |
|---|---|---|---|---|
| BiLSTM+Linear | 实测:eval JSON | 结构计算:Emb(2021×128)+BiLSTM(1层×256)+Head(11)+CRF(0) [字表按2021估] | 实测:history时间戳差(epoch间) | 未采集(判别式未记延迟) |
| BiLSTM+CRF | 实测:eval JSON | 结构计算:Emb(2021×128)+BiLSTM(1层×256)+Head(11)+CRF(121) [字表按2021估] | 实测:history时间戳差(epoch间) | 未采集(判别式未记延迟) |
| RoBERTa+Linear | 实测:eval JSON | [规格]hfl/chinese-roberta-wwm-ext = bert-base 结构 12L/768H，~102.3M | 实测:history时间戳差(epoch间) | 未采集(判别式未记延迟) |
| RoBERTa+CRF | 实测:eval JSON | [规格]hfl/chinese-roberta-wwm-ext = bert-base 结构 12L/768H，~102.3M | 实测:history时间戳差(epoch间) | 未采集(判别式未记延迟) |
| Qwen3-LoRA-SFT | 实测:eval JSON | [规格]Qwen3-0.6B 标称；可训：结构估算:r=8 全线性层旁路 ≈0.15%（[规格]底座0.6B） | 实测:history时间戳差(epoch间) | 实测:SFT engineering |
| Qwen3-full-SFT | 实测:eval JSON | [规格]Qwen3-0.6B 标称 | 实测:history时间戳差(epoch间) | 实测:SFT engineering |

## 能直接讲的数字（面试用）

- **预训练收益**：BiLSTM-CRF F1 0.2065 → RoBERTa-CRF 0.4895（+28.3 点）
- **参数量代价**：LoRA 只训约 0.15% vs full 100%；full 用 fp32+8bit 优化器（约 5.7GB 参数侧，[规格]经验值），需 --grad_ckpt。
- **推理延迟**：LoRA 0.633s vs full 0.42s（生成式逐条 decode，判别式未记延迟）。

## 未采集字段怎么补（下次跑训练时）

显存峰值和判别式延迟本次没记。要补只需在训练/评测脚本里加两行：
```python
# 训练结束前：
peak_gb = torch.cuda.max_memory_allocated() / 1024**3
# 评测循环里：
import time; t0 = time.perf_counter()
...  # 推理
latency = (time.perf_counter() - t0) / num_samples
```
写进 summary JSON 后重跑本脚本，`未采集`列会自动填实。

> 面试提醒：老师要求成本数字可溯源。**宁可写「未采集」也不要估一个显存数**——
被追问「怎么测的」答不上来反而扣分。
