# 第 7 周 信息抽取实验记录

任务：DuEE-Fin 金融公告事件抽取（13 类事件 / 92 个类型×角色槽位 / 句子级单事件）
数据：`data/processed/train_extract.jsonl` 5,747 句 → 训练；`dev_extract.jsonl` 930 句 → 评测
口径：所有实验统一走 `common/evaluate_events.py`，三级指标都报 —— 触发词 F1（词找对没）、论元 F1（角色+文本对没，主指标）、事件级 F1（类型+触发词+全部论元全对）；辅助报类型 macro F1 与非法 BIO 句占比。
环境：Python 3.12.14 / torch 2.3.1+cu121 / transformers 4.51.3 / peft 0.15.0 / RTX 3060 Laptop 6 GB
说明：`pretrain_models/` 内无 `bert-base-chinese`，本记录 L2 各行的编码器实际加载的是 **`roberta-base-chinese`**（BERT 结构、vocab 21128），因此方法名一律写作 RoBERTa；代码目录与产物名仍是老师包里的 `bert/`、`bert_*.pt`、`bert_crf_256` 等，未作改动。LLM 实验加载本地 `Qwen3-0.6B`。
运行位置：全部从 `week07/` 副本执行，产物落在 `week07/outputs/`。

---

## 实验一览

| # | 方法 | 层级 | 要训练吗 | 实测（dev 930 / LLM n=200） | 一句话结论 |
|---|---|---|---|---|---|
| 1 | 触发词词典 + 数值正则 | L0 | 不用（0 训练，推理 0.75 s） | 0.617 / 0.174 / 0.016 | 触发词命中 99.78%，输在论元"硬填" |
| 2 | 字符级 BiLSTM + Linear | L1 | 29.5 s | 0.742 / 0.422 / 0.068 | 1.29 M 参数就把论元做到 0.42 |
| 3 | 字符级 BiLSTM-CRF | L1 | 172.4 s | **0.828 / 0.569 / 0.117** | 本周判别式最好的一行；CRF 单独值 +0.147 |
| 4 | RoBERTa-12L + Linear（联合 149 标签） | L2 | 884.5 s | 0.551 / 0.311 / 0.043 | 逐 token 独立解码，非法句 87.2% |
| 5 | RoBERTa-12L + CRF（联合 149 标签） | L2 | 1342.3 s | 0.632 / 0.324 / 0.062 | CRF 没学到转移：非法句仍 86.1% |
| 6 | └ 解码期硬约束 | L2 技巧 | 1328.2 s（带约束重训） | 0.559 / **0.434** / 0.097 | 非法句 86.1% → 10.8%，论元 +0.110 |
| 7 | 两段式：检测 → 类型 → 类型条件论元 | L2b | 234.7 s（13 个专家）/ 16 模型 | 0.546 / 0.289 / 0.057 | 论元低于 L1；瓶颈是训练预算被切成 13 份 |
| 8 | 本地 Qwen3-0.6B zero-shot | L3 | 不用 | 0.365 / 0.080 / 0.005 | JSON 合法率 98%，但字段填不对 |
| 9 | 本地 Qwen3-0.6B few-shot | L3 | 不用 | 0.576 / 0.211 / 0.035 | 示例主要修格式：repaired 75 → 1 |
| 10 | LoRA SFT（Qwen3-0.6B，r=8，800 句） | L4 | 327.1 s 训练 + 180.9 s 评测 | **0.870 / 0.528** / 0.045 | 微调是本周唯一让 0.6B 变可用的一步 |

---

## 实验 1｜L0 规则基线（词典 + 正则）

**运行**

```
python rule\run_rule.py
```

完成 2026-11-24 21:49 ｜ 推理耗时 **0.75 s**（930 句，0.805 ms/句，纯 CPU，见 `outputs/logs/rule_console.log`）｜ 参数量 0
产物：`outputs/disc/pred_rule.jsonl`、`outputs/metrics/disc/rule.json`、`outputs/logs/rule_*`
词典规模（本次实测）：触发词 **99** 条（train 中出现 ≥2 次 + 人工同义词，按词长降序）、公司/机构名 **3,679** 个（取自 train 的公司类角色论元值）。

**做法**

1. 从 train 统计各事件类型的触发词表（`min_count=2` 去噪），并从公司类角色论元收集实体词典。
2. 定类型：逐句扫触发词，**最长优先**（否则"解除质押"会被"质押"抢走），命中即确定类型与触发词 span。
3. 填论元：按 `event_schema.json` 该类型的角色清单逐格填。角色名先归类为 date/money/percent/share/enum/entity，前四类走固定正则，entity 从公司候选池按出现顺序领一个（领走即删），enum 留空；每个角色最多 1 个论元，且不得与已占用 span（含触发词）重叠。

**结果（dev 930 句）**

| 指标 | P | R | F1 | tp / fp / fn |
|---|---|---|---|---|
| 触发词 | 0.6175 | 0.6161 | **0.6168** | 573 / 355 / 357 |
| 论元 | 0.1738 | 0.1742 | **0.1740** | 718 / 3,413 / 3,404 |
| 事件级 | 0.0162 | 0.0161 | **0.0161** | 15 / 913 / 915 |
| 类型 macro F1 | — | — | **0.8118** | — |

失败分类（930 句互斥归因）：

| 类别 | 句数 | 占比 | 含义 |
|---|---|---|---|
| arg_spurious | 517 | 55.6% | 类型与触发词都对，论元**多抽** |
| wrong_trigger | 214 | 23.0% | 类型对，触发词文本/边界错 |
| wrong_type | 141 | 15.2% | 触发词命中但类型判错 |
| arg_missing | 41 | 4.4% | 事件对，论元漏抽 |
| exact | 15 | 1.6% | 全对 |
| no_trigger | 2 | 0.2% | 整句没扫到触发词 |

**小结**

1. **触发词命中率 99.78%** —— 词典覆盖不是瓶颈，真正的问题是扫到词之后边界和类型定不准（wrong_trigger 23.0% + wrong_type 15.2% 合计 38.2%）。
2. **输在"每个格子都倾向填一个值"**。55.6% 的句子是论元多抽，精确率被压到 0.1738 —— 这是逐格贪心、没有置信度的必然结果。
3. **类型 macro 0.8118 很高、事件级 0.0161 很低**：事件级要求 5 个槽位全对，论元一崩整条就崩。本任务的难点在论元，不在分类。
4. 数值类论元（日期/金额/比例/股数）写法固定，正则又准又快，所以规则在系统里适合当兜底补充，不适合当主链路。

---

## 实验 2｜L1 字符级 BiLSTM（Linear / CRF）

**运行**

```
python bilstm_crf\train.py                        # CRF
python bilstm_crf\train.py --decoder linear       # 逐 token 对照
```

完成 2026-11-24 23:03:31（CRF）/ 23:11:33（Linear）｜ 训练 **172.4 s** / **29.5 s** ｜ 显存峰值 206.7 / 205.9 MB
配置：字向量 128 + BiLSTM(hidden 256, 1 层) + 分类头；字符词表 **3,269**、标签 **149**（O + 2×74）、lr 1e-3、epochs 5、max_len 256、从 train 切 10% 做内层验证选 epoch
产物：`outputs/checkpoints/bilstm_{crf,linear}.pt`、`outputs/disc/pred_bilstm_*`、`outputs/metrics/disc/bilstm_*.json`、`outputs/logs/train_bilstm_*`

**结果（dev 930 句）**

| 指标 | CRF | Linear |
|---|---|---|
| 触发词 F1 | **0.8282**（P 0.9222 / R 0.7516，tp 699） | 0.7416（P 0.8084 / R 0.6849，tp 637） |
| 论元 F1 | **0.5685**（P 0.6646 / R 0.4966，tp 2,047） | 0.4217（P 0.4309 / R 0.4129，tp 1,702） |
| 事件级 F1 | **0.1173**（tp 99） | 0.0675（tp 58） |
| 类型 macro F1 | 0.7742 | 0.7631 |
| 非法 BIO 句占比 | **32.37%**（411 处 / 301 句，0.442 处/句） | **79.46%**（2,030 处 / 739 句，2.183 处/句） |
| 参数量 | 1.31 M | 1.29 M |
| 延迟 | 3.03 ms/句 | 0.71 ms/句 |
| 内层验证 spanF1 / best_epoch | 0.6204 / 第 4 轮 | 0.4624 / 第 5 轮 |

**小结**

1. **相对 L0 规则：论元 0.1740 → 0.5685（+0.394）**，事件级 0.0161 → 0.1173（7.3 倍）。从"硬填槽位"换成"学序列标注"是本周增益最大的一格。
2. **CRF 层单独值 +0.147 论元 F1**（0.4217 → 0.5685），同时把非法序列从 79.46% 压到 32.37%。代价是维特比解码让延迟涨 4 倍，但 930 句只多花 2 秒，完全可接受。
3. **32.37% 仍远高于金标地板（1.94%）**：CRF 的转移矩阵是从标注里学出来的，而金标本身因重叠论元投影就有非法，模型又在 149 标签空间里没收敛够，所以只降不归零。
4. `n_truncated=117`（12.6% 的句子在 max_len 256 下被截断），这是论元召回的硬上限之一。
5. 1.3 M 参数、3 分钟训练就能到论元 0.57 —— **这个任务的信号主要在标注模式里，不在模型规模里**。

---

## 实验 3｜L2 RoBERTa 联合标注（Linear / CRF，12 层，max_len 256）

**运行**

```
python bert\train.py                                          # RoBERTa+CRF（主线）
python bert\train.py --decoder linear --run_name bert_linear_256
```

完成 2026-11-24 23:54:38（CRF）/ 11-25 00:27:18（Linear）｜ 训练 **1342.3 s（22.4 分钟）** / **884.5 s** ｜ 显存峰值 1848.9 / 1847.0 MB
配置：编码器 12 层 + 149 标签头，参数量 **102.4 M**；epochs 3、bs 32、lr 2e-5（头部 ×5 = 1e-4）、warmup 10%、max_len 256、train 切 10% 内层验证选点
产物：`outputs/checkpoints/bert_{crf,linear}_256.pt`（409 MB ×2）、`outputs/disc/pred_*`、`outputs/metrics/disc/*`、`outputs/logs/train_*`

**结果（dev 930 句）**

| 指标 | RoBERTa+CRF | RoBERTa+Linear |
|---|---|---|
| 触发词 F1 | 0.6316（P 0.7017 / R 0.5742，tp 534） | 0.5508（P 0.5992 / R 0.5097，tp 474） |
| 论元 F1 | 0.3242（P 0.3529 / R 0.2999，tp 1,236） | 0.3112（P 0.3213 / R 0.3018，tp 1,244） |
| 事件级 F1 | 0.0615（tp 52） | 0.0430（tp 37） |
| 类型 macro F1 | 0.3792 | 0.4077 |
| 非法 BIO 句占比 | **86.13%**（3,019 处 / 801 句） | **87.20%**（3,535 处 / 811 句） |
| 延迟 | 18.83 ms/句 | 9.47 ms/句 |
| 内层验证 spanF1（dev 监控） | 0.4282（0.4247） | 0.3933 |
| `n_truncated` | 122 | 122 |

**小结**

1. **这一层相对 L1 是净退步**：论元 0.5685 → 0.3242，触发词 0.8282 → 0.6316。内层验证 0.4282 与 dev 监控 0.4247 几乎相等，说明不是过拟合也不是数据泄漏，模型本身只到这个水平。
2. **原因一：编码器不是原版权重**。本次加载的是 `roberta-base-chinese`（本机无 `bert-base-chinese`），类型 macro 从 BiLSTM 的 0.7742 掉到 0.3792 —— 类型判不准会连带毁论元（论元标签空间以类型为前缀）。
3. **原因二：CRF 在 149 标签联合空间 + 3 epoch 下没学到结构**。CRF 版非法句 86.13% 与 Linear 的 87.20% 基本相同（对比实验 2：BiLSTM-CRF 32.37%），转移矩阵等于没起作用。
4. 因此**联合方案的瓶颈是标签空间太大 + 训练预算太小**，不是 RoBERTa 本身不行。这直接引出实验 4（解码期硬约束）和实验 5（把 149 标签拆成 11–21 的类型条件专家）。
5. 成本参考：12 层 102.4 M 参数、22 分钟训练、18.83 ms/句，比 1.3 M 参数的 BiLSTM-CRF 贵 7.8 倍训练、6.2 倍延迟。

---

## 实验 4｜L2 消融：max_len 128 / 前 4 层 / 硬约束

**运行**（一次只改一个旋钮，其余同实验 3）

```
python bert\train.py --decoder linear --max_len 128 --run_name bert_linear_128
python bert\train.py --bert_layers 4
python bert\train.py --decoder crf --hard_constraint
```

完成 11-25 21:14:25 / 00:35:48 / 21:08:33 ｜ 训练 252.2 s / 381.8 s / 1328.2 s

**结果（dev 930 句，Δ 相对同解码器的 12 层基线）**

| # | 配置 | 论元 F1 | Δ | 触发词 F1 | 事件 F1 | 类型 macro | 非法句占比 | 参数量 | 训练耗时 | 延迟 |
|---|---|---|---|---|---|---|---|---|---|---|
| 基线 | RoBERTa-12L + Linear，256 | 0.3112 | — | 0.5508 | 0.0430 | 0.4077 | 87.20% | 102.4 M | 884.5 s | 9.47 ms |
| exp1 | 只改 max_len → 128 | 0.3116 | **+0.0004** | 0.6384 | 0.0360 | 0.4821 | 84.95% | 102.4 M | **252.2 s（-71%）** | 5.10 ms |
| 基线 | RoBERTa-12L + CRF，256 | 0.3242 | — | 0.6316 | 0.0615 | 0.3792 | 86.13% | 102.4 M | 1342.3 s | 18.83 ms |
| exp2 | 只改层数 → 前 4 层（CRF） | 0.2133 | **−0.1109** | **0.7178** | 0.0073 | 0.5460 | 85.59% | **45.7 M** | **381.8 s** | 11.44 ms |
| exp3 | 12 层 RoBERTa+CRF + 硬约束（带约束重训） | **0.4342** | **+0.1100** | 0.5590 | 0.0968 | 0.3996 | **10.75%** | 102.4 M | 1328.2 s | 18.97 ms |

**小结**

1. **max_len 256 → 128 省 71% 训练时间，论元一点没掉**（0.3112 → 0.3116），但截断句数从 122 涨到 **639（68.7%）**。说明在这份单句数据上，后半句被切掉对论元影响很小 —— 论元集中在句子前中部，`max_len` 不是该调的旋钮。
2. **4 层版出现"触发词最好、论元最差"的分裂**：触发词 0.7178 比 12 层 CRF 的 0.6316 还高，论元却只有 0.2133、事件 0.0073。浅层编码器足以做"有没有事件/边界在哪"，撑不起细粒度角色判别 —— 这正是两段式把检测和论元分开的依据。
3. **硬约束是本周性价比最高的一招**：非法句 86.13% → 10.75%（0.217 处/句），论元 +0.110、事件 +0.035。它把 CRF 该做而没做到的事（结构合法性）从"学出来"改成"规定出来"。
4. 硬约束的代价是**触发词掉 0.073**（0.6316 → 0.5590）：约束收紧了 span 边界，宁可漏不可错，于是精确率 0.7635 换召回率 0.4409。
5. 口径说明：本次硬约束是**训练期就 mask 非法转移**（`--hard_constraint`，花 1328 s 重训，内层验证也涨到 0.4747）；若只想看解码收益，可用 `python bert\evaluate_hard_constraint.py` 直接读现成 `bert_crf_256.pt` 重解码，约 90 秒、0 训练成本。

---

## 实验 5｜L2b 两段式流水线（检测 → 类型 → 类型条件论元）

**运行**（同一个流水线跑了两轮，唯一变量是专家学习率）

```
python bert_pipeline\train.py --detect_limit 8000 --cls_epochs 1 --cls_batch_size 32 --log_every 20
python bert_pipeline\train.py --only_experts --arg_epochs 3 --arg_lr 8e-5 --arg_layers 4 --log_every 20
```

| 轮次 | 完成 | 命令要点 | 端到端结果 |
|---|---|---|---|
| 第一轮 | 11-25 22:23:43 | `--arg_lr` 用脚本默认 **3e-5**；三阶段全训，合计 610.5 s（检测 137.9 s + 类型 + 专家 262.4 s） | 三级 F1 **全部 0.0000** |
| 第二轮 | 11-25 23:57:41 | **`--arg_lr 8e-5`** + `--only_experts`（复用第一轮的 `pipe_detector.pt` / `pipe_type.pt`，只重训 13 个专家，总耗时 **235 s**） | 0.5459 / 0.2892 / 0.0566 |

**配置与产物**

| 项 | 内容 |
|---|---|
| 模型数 | **16 个**（`pipe_detector.pt` 12 层二分类 + `pipe_type.pt` 13 类 + `pipe_arg_<类型>.pt` ×13，专家为 4 层 RoBERTa + Linear） |
| 专家数据量 | 每类 **53–979 行**、标签 **11–21 个**、每轮 **3–31 步**；本轮各专家 3 epoch 后的 loss 0.3057（股份回购 979 行）– 1.2309（被约谈 53 行），合计训练 **178.2 s** |
| 端到端成本 | 25.96 ms/句 ｜ 显存峰值 **1594.0 MB**（比第一轮 2372.9 MB 低，因为省掉了 12 层检测器的重训） |
| 产物 | `outputs/metrics/disc/{pipe_detector,pipeline,pipeline_oracle_type}.json` + 两份 `*_n200.json`、`outputs/logs/bert_pipeline_*` |

**分阶段结果**

| 阶段 | 口径 | P | R | F1 | 备注 |
|---|---|---|---|---|---|
| (a) 事件检测 | dev_detect 5,302 句 | 0.7974 | 0.7778 | **0.7875** | acc 0.9066；tp 917 / fp 233 / fn 262 |
| (a) 事件检测 | dev_detect_clean 4,815 句（剔除与 train 重复的 9.2% 文本） | 0.7991 | 0.7841 | **0.7915** | 更能代表真实泛化 |
| (b) 事件类型 | dev_extract 930 句，13 类 | — | — | **acc 0.9581** | 仅 39 句类型判错 |
| (c) 端到端·触发词 | dev 930 句 | **0.9444** | 0.3839 | **0.5459** | tp 357 / fp 21 / fn 573 |
| (c) 端到端·论元 | dev 930 句 | 0.4165 | 0.2215 | **0.2892** | tp 913 / fp 1,279 / fn 3,209 |
| (c) 端到端·事件级 | dev 930 句 | 0.0979 | 0.0398 | **0.0566** | tp 37 |
| (c) 检测口径（端到端复算） | dev_detect | 1.0000 | 0.5912 | 0.7431 | acc 0.9091 |
| 上界 | 类型给金标 | 0.9495 / 0.4180 / 0.0984 | — | **0.5467 / 0.2896 / 0.0567** | 与端到端差 **≤0.0004** |
| n=200 子集 | 与 LLM 同一批 200 句 | — | — | **0.5199 / 0.2828 / 0.0722** | 上界 0.5236 / 0.2846 / 0.0727 |
| 类型 macro | dev 930 | — | — | 0.2552（上界 0.2559） | — |

**小结**

1. **两轮构成一次干净的单变量对照**：只把 `arg_lr` 从 3e-5 提到 8e-5，端到端论元 F1 从 **0.0000 → 0.2892**。原因是每个专家只拿到 53–979 行、每轮 3–31 步，低学习率下直接欠拟合塌缩——第一轮探针显示 13 个专家连自己的训练行都不输出 `B-TRG`（金标 40 → 预测 0），解码塌成 `I-中标标的` 这类多数角色标签。
2. **归零被评测口径放大**：`common/event_data.py:389` 的 `spans_to_event` 拿不到触发词文本时直接 `return None`，所以"专家不标 TRG"会让论元和事件级同时变成 0，即使论元本身蒙对一部分。
3. **触发词精确率极高（0.9444）但召回只有 0.3839** —— 专家的策略是"没把握就不标 B-TRG"，漏掉 573 个。对照阶段 (a)：检测器单独跑 dev_detect 的召回是 0.7778，进入端到端后召回掉到 0.5912（检测口径）/ 0.3839（触发词口径），**损失全部发生在类型条件专家这一层**。
4. **类型给金标的上界几乎不涨**（论元 0.2892 → 0.2896）：39 句类型判错造成的损失可以忽略，**瓶颈确认在论元专家的训练预算，不在检测或分类**。
5. **两段式仍低于 L1 BiLSTM-CRF**（论元 0.2892 vs 0.5685）：标签空间从 149 缩到 11–21 让每个任务更容易，但训练数据被切成 13 份的代价更大 —— 这正是 PPT 里"训练预算被切成 13 份"那一格的本地复现。
6. 专家数从 22:20 那批换成 23:5x 这批后合计训练时间 262.4 s → 178.2 s，但显存峰值降到 1594.0 MB；端到端延迟 25.96 ms/句 是全部判别式方法里最高的（16 次前向）。

---

## 实验 6｜L3 生成式抽取：本地 Qwen3-0.6B zero-shot / few-shot

**运行**

```
python llm_zero_shot\llm_extract.py --backend local --mode zero --n 200 --tag local_zero_n200 --gen_batch 8 --max_new_tokens 384
python llm_zero_shot\llm_extract.py --backend local --mode few --n 200 --tag local_few_n200 --gen_batch 8 --max_new_tokens 384 --max_length 2048
```

完成 11-25 22:50:23 / 22:55:25 ｜ 端到端 **213.3 s / 201.4 s**（各 200 句，含解析与评测）＝ 1.07 / 1.01 s/句
逐句生成延迟（`outputs/llm/raw_local_*_n200.jsonl` 的 `latency_s`，n=200）：zero 均值 **1.066 s**、中位 0.938、p95 2.061、最大 2.077；few 均值 **1.005 s**、中位 0.887、p95 1.525、最大 3.103
产物侧输出规模：zero 平均 190 字符/条、few 平均 185 字符/条
抽样口径：dev_extract 930 句按 (id, sent_id) 规范化排序 + `seed=42` 洗牌取前 **200** 句（`outputs/llm/eval_ids.json` + `gold_n200.jsonl`）
产物：`outputs/llm/{pred,raw,report}_local_{zero,few}_n200.*`、`outputs/metrics/llm/local_{zero,few}_n200.json`、`outputs/logs/*_eval.json`
生成设置：贪心解码（`do_sample=False`）、`enable_thinking=False`、左填充

**结果（同一批 200 句）**

| 指标 | zero-shot | few-shot |
|---|---|---|
| 触发词 F1 | 0.3654（P 0.3519 / R 0.3800，tp 76） | **0.5764**（P 0.5779 / R 0.5750，tp 115） |
| 论元 F1 | 0.0802（P 0.1253 / R 0.0590，tp 52） | **0.2105**（P 0.2849 / R 0.1669，tp 147） |
| 事件级 F1 | 0.0048（tp 1） | **0.0351**（tp 7） |
| 类型 macro F1 | 0.3094 | **0.4579** |
| JSON 一次合法率 | 98.0%（解析失败 4） | **99.5%**（解析失败 1） |
| 被修复的样本 / 类型越界 | 75 条 / 19 处 | 1 条 / 9 处 |

**小结**

1. **few-shot 全面优于 zero-shot**：触发词 +0.211、论元 +0.130、类型 macro +0.149。
2. **示例修的主要是格式，不只是内容** —— 需要事后修复的样本从 **75 条降到 1 条**、类型越界从 19 处降到 9 处。6 个示例把"输出成什么形状"这件事讲清楚了，剩下的差距才是语义理解。
3. **未微调的 0.6B 本地模型不适合结构化抽取**：论元 0.2105 远低于 L1 BiLSTM-CRF 的 0.5685，事件级 200 句只全对 7 句。JSON 能写对 ≠ 字段能填对 —— 补上这一步要靠微调（实验 7：同批 0.5282）。
4. **`--max_length` 必须按提示实际长度设**：保留给提示的长度是 `keep = max_length − max_new_tokens`（[llm_zero_shot/local_generate.py:85](file:///F:/ai/LLM/7/第七周 抽取任务/week07/llm_zero_shot/local_generate.py#L85)）。用 `Qwen3-0.6B` 的 tokenizer 对本批 200 句实测（脚本见文末附录）：zero-shot 提示平均 **924 token**（最大 1,120）、few-shot 平均 **1,576**（最大 1,772），而默认 1280−384=896 意味着 **few-shot 全部样本会被截掉示例**，所以 few-shot 必须 `--max_length 2048`；zero-shot 保持 1280。
5. 显存可以按配置算而不必试：Qwen3-0.6B 是 28 层 / 8 个 KV 头 / head_dim 128（`pretrain_models/Qwen3-0.6B/config.json`），bf16 下 KV 缓存 **= 2 × 28 × 8 × 128 × 2 B = 112 KB/token/序列**，所以 `--gen_batch 8 × 2048 token ≈ 3.5 GB`，加权重（bf16 约 1.2 GB）正好落在这张 6 GB 卡的安全区。

---

## 实验 7｜L4 LoRA 监督微调（Qwen3-0.6B）

**运行**

```
python llm_lora\train_sft.py --num_train 800 --batch_size 1 --grad_accum 16 --run_name lora_n800 --log_every 1
python llm_lora\evaluate_sft.py --n 200 --gen_batch 8
```

训练完成 11-25 23:15:03 ｜ 总耗时 **327.1 s（训练 302.4 s）**＝ **6.05 s/优化步 × 50 步** ｜ 显存峰值 **4.63 GiB**
配置：LoRA r=8 / α=16 / dropout 0.05，目标模块 q/k/v/o_proj；lr 2e-4、1 epoch、bs 1 × 梯度累积 16（有效 batch 16）、`max_length 1280`、bf16、未开梯度检查点
数据：从 train 抽 800 句训练（`ordered[200:1000]`）+ 200 句验证（`ordered[:200]`），**dev 不参与训练与选点**
可训练参数 **2,293,760 / 598,343,680 = 0.3834%**
产物：`outputs/llm/ckpt/sft_adapter/`（`adapter_model.safetensors` **8.98 MB** + `adapter_config.json`）、`outputs/llm/ckpt/train_log_sft.json`、`outputs/logs/lora_n800_{steps.jsonl,train.json,console.log}`

**结果（训练阶段）**

| 项 | 数值 |
|---|---|
| train_loss（1 epoch 末） | **0.1566** |
| best_val_loss | **0.1111** |
| 优化步数 | 50（= 800 ÷ (1×16)）；`steps.jsonl` 800 条 = 每个 micro-batch 一条 |
| 逐 micro-batch loss | 首条 0.2153 → 末条 0.1035（全程区间 0.0175 – 0.6144） |
| prompt 段截断 / assistant 段截断 | **0 句** / 1 句 |
| adapter 体积 | 8.98 MB（基座权重冻结，未改动） |

**结果（端到端评测，dev 同一批 200 句）**

评测完成 11-26 00:01:39 ｜ **180.9 s / 200 句**＝ 均值 **0.903 s/句**（p50 0.884）｜ 产物 `outputs/metrics/llm/sft_lora_n200.json`、`outputs/llm/{pred,raw,report}_sft_lora_n200.*`

| 指标 | P | R | F1 | tp / fp / fn |
|---|---|---|---|---|
| 触发词 | 0.8700 | 0.8700 | **0.8700** | 174 / 26 / 26 |
| 论元 | 0.5418 | 0.5153 | **0.5282** | 454 / 384 / 427 |
| 事件级 | 0.0450 | 0.0450 | **0.0450** | 9 / 191 / 191 |
| 类型 macro F1 | — | — | **0.8468** | — |

输出质量（`report_sft_lora_n200.json`）：JSON 一次合法率 **100.0%**（解析失败 0）、被修复 2 条、空回复 0、类型越界 2 处、角色越界 15 处、触发词不在原文 6 处、**论元不在原文（幻觉）33 处**。
逐类型最弱三项：回购 F1 0.000（金标 0 条）、公司上市 0.750（6 条）、解除质押 0.824（9 条）。

**小结**

1. **loss masking 是对的**：prompt 段 0 截断、assistant 段只 1 句截断，逐 micro-batch loss 稳定落在 0.02–0.61 区间且整体下行（0.2153 → 0.1035），说明只有 assistant 段在算损失；若截断把 label 刷成全 -100，这里会直接 loss=nan。
2. **`max_length 1280` 不能为了省显存而调小**：一调小 prompt 就被截、label 全变 -100，训练看似在跑其实什么都没学。
3. **6 GB 卡上的可行组合是 `--batch_size 1 --grad_accum 16`**，而不是脚本默认的 bs2/ga8。同一张卡上的对照实测（每个 micro-batch 一条，单位 s）：bs2/ga8 探针为 **4.2 – 7.2 s/micro-batch**（`outputs/logs/lora_probe_steps.jsonl`，8 条 → 约 50 s/优化步），bs1/ga16 探针为 **0.32 – 0.74 s/micro-batch**（`outputs/logs/lora_probe3_steps.jsonl`，16 条 → 约 6 s/优化步），正式跑同样是 6.05 s/步且 loss 与探针一致（val 0.1111 vs 探针同批 0.3307 的下降趋势）。梯度检查点在另一台同代卡上是反向收益，本机也未采用。**省显存的旋钮在这个规模上不免费，先保证不溢出再谈别的**。
4. **LoRA 的成本量级**：5.5 分钟、8.98 MB 产物、0.38% 参数 trainable，就得到一个可加载可评测的 adapter；推理时把 B·A 合并回 W，零额外延迟。
5. **微调是本周唯一让 0.6B 变可用的一步**（同一批 200 句，口径与实验 6 完全一致）：论元 0.2105 → **0.5282（+0.3177）**、触发词 0.5764 → **0.8700**、类型 macro 0.4579 → **0.8468**。相对 zero/few-shot，改善的不只是内容 —— JSON 合法率 99.5% → 100%、类型越界 9 → 2 处。
6. **事件级仍然只有 0.0450（9 句全对）**，瓶颈从"填不对"变成了"填得多"：论元 fp 384 里有 33 处是原文根本不存在的 span（幻觉论元），加上角色越界 15 处，只要一格错整条事件就判负。800 句、1 epoch 属"起步体验"规模，要再往上走应当加大数据量而不是加轮数。
7. 与判别式同批对照：同一 200 句上两段式（n=200 口径）论元 0.2828、LoRA SFT **0.5282**，但 L1 BiLSTM-CRF 在 dev 930 全量上已有 0.5685 —— **0.6B + LoRA 追平了小模型，代价是 0.9 s/句 vs 3.03 ms/句 的 300 倍延迟**。

---

## 同口径汇总

### 主表（判别式 = dev 930 全量；LLM = 同一份 `evaluate_events.py` 在 200 句子集上）

| 方法（层级） | trigger F1 | argument F1 | event F1 | type macro | 非法句占比 | 训练 / 成本 | 延迟 |
|---|---|---|---|---|---|---|---|
| 规则：词典 + 正则（L0） | 0.6168 | 0.1740 | 0.0161 | 0.8118 | — | 0（纯 CPU） | 0.805 ms/句 |
| BiLSTM + Linear（L1） | 0.7416 | 0.4217 | 0.0675 | 0.7631 | 79.46% | 29.5 s / 1.29 M | 0.71 ms/句 |
| **BiLSTM-CRF（L1）** | **0.8282** | **0.5685** | **0.1173** | 0.7742 | 32.37% | 172.4 s / 1.31 M | 3.03 ms/句 |
| RoBERTa + Linear（L2） | 0.5508 | 0.3112 | 0.0430 | 0.4077 | 87.20% | 884.5 s / 102.4 M | 9.47 ms/句 |
| RoBERTa + CRF（L2） | 0.6316 | 0.3242 | 0.0615 | 0.3792 | 86.13% | 1342.3 s / 102.4 M | 18.83 ms/句 |
| └ + 硬约束（L2 技巧） | 0.5590 | 0.4342 | 0.0968 | 0.3996 | **10.75%** | 1328.2 s / 102.4 M | 18.97 ms/句 |
| 两段式 检测→类型→论元（L2b） | 0.5459 | 0.2892 | 0.0566 | 0.2552 | — | 234.7 s / 16 模型 | 25.96 ms/句 |
| 本地 Qwen3-0.6B zero-shot（L3，n=200） | 0.3654 | 0.0802 | 0.0048 | 0.3094 | — | 0 训练 / 213.3 s 推理 | 1.07 s/句 |
| 本地 Qwen3-0.6B few-shot（L3，n=200） | 0.5764 | 0.2105 | 0.0351 | 0.4579 | — | 0 训练 / 201.4 s 推理 | 1.01 s/句 |
| **LoRA SFT（L4，800 句 1 epoch，n=200）** | **0.8700** | **0.5282** | 0.0450 | **0.8468** | — | 327.1 s 训练 + 180.9 s 评测 / +2.29 M | 0.903 s/句 |

金标自身的非法 BIO 地板：**1.94%**（930 句里 18 句非法，源于重叠论元只能压成一条序列；`python eval\check_bio_gold.py` 可在本副本复算）—— 模型的非法率应该和这个地板比，不是和 0 比。

同一批 200 句的横向数（`*_n200.json`，与 LLM 三行严格同口径）：两段式 0.5199 / 0.2828 / 0.0722（类型给金标 0.5236 / 0.2846 / 0.0727）→ few-shot 0.5764 / 0.2105 / 0.0351 → **LoRA SFT 0.8700 / 0.5282 / 0.0450**。

### 读表姿势

1. **先看 argument F1**（"字段填对没"），event F1 天然低得多，因为它要求类型 + 触发词 + 全部槽位同时全对；只报 event 会掩盖真实水平。
2. **口径要标清楚**：判别式是 dev 930 全量，LLM 三行是同一份评测脚本跑在 200 句子集上，不同口径不能横向比大小；要横比就读每行的 `*_n200.json`（两段式已同批复评）。
3. **三级都报，少报一级结论就能被任意解读** —— 例如 L0 规则只报 trigger 0.617 会显得可用，报上 argument 0.174 就知道它填不对槽位。

### 本周主线的结论

| 台阶 | 论元 F1 | 结论 |
|---|---|---|
| L0 规则 → L1 BiLSTM-CRF | 0.1740 → **0.5685**（+0.3945） | 从"贪心填槽"到"学序列标注"，是本周最大的一格 |
| L1 Linear → L1 CRF | 0.4217 → 0.5685（+0.1468） | 结构约束单独就值这些；非法句 79% → 32% |
| L2 联合 → L2 + 硬约束 | 0.3242 → 0.4342（+0.1100） | 训练学不到的结构，可以在解码期规定出来 |
| L2b 两段式 `arg_lr` 3e-5 → 8e-5 | 0.0000 → **0.2892** | 专家每轮只有 3–31 步，学习率不配套就塌成多数标签 |
| L3 zero → L3 few | 0.0802 → 0.2105（+0.1303） | 提示能修格式（repaired 75→1），修不好 0.6B 的理解力 |
| L3 few → **L4 LoRA SFT**（同批 200 句） | 0.2105 → **0.5282**（+0.3177） | 提示工程给不了的能力，800 句 + 0.38% 参数的微调给得了 |
| L1 BiLSTM-CRF vs L3 few-shot | 0.5685 **vs** 0.2105 | 1.3 M 的小模型在结构化抽取上完胜 0.6 B 大模型的提示工程 |
| L1 BiLSTM-CRF vs L4 LoRA SFT | 0.5685（930 全量）**vs** 0.5282（200 子集） | 微调后追平小模型，代价是 300 倍延迟（0.903 s/句 vs 3.03 ms/句） |

---

## 汇总与出图

```
python eval\compare_methods.py --with_n200   → outputs/metrics/disc/summary.md（L0–L2b 主表 + 消融 + 自动关键对比）
python eval\plot_compare.py --n 200          → outputs/llm/method_compare.png / .csv（L3、L4 同批对比图）
python eval\compare_to_numbers.py            → 本次结果与历史台账逐项对差
python eval\make_figures.py                  → outputs/figures/fig_*.png（六张 PPT 用图）
python data\scripts\make_figures.py          → data/figures/fig_*.png（五张数据分布图）
```

## 本次配图（`实验记录/图表/`，全部由本次 `week07/outputs/` 产物现画）

| 文件 | 内容 | 数据来源 |
|---|---|---|
| `fig_method_compare.png` | L0–L2b 八个配置的 trigger/argument/event 三级 F1 | `outputs/metrics/disc/*.json` |
| `fig_illegal_bio.png` | 金标地板 18 句(1.9%) → Linear 811/739 → CRF 801/301 → 硬约束 100(10.8%) | 各 json 的 `meta.illegal_dev` + `bio_gold_baseline.json` |
| `fig_per_role_f1_bilstm_crf.png` / `fig_per_role_f1_roberta_crf.png` | 最难的 18 个角色论元 F1（括号内为 dev 金标次数） | `per_role`（两个方法各一份） |
| `fig_training_curves.png` | 训练损失 + 内层验证 span F1（选 epoch 用，dev 不参与） | `outputs/logs/train_{bert_linear_256,bert_crf_256,bilstm_crf}.json` |
| `fig_type_confusion_bilstm_crf.png` / `fig_type_confusion_roberta_crf.png` | 13 类事件类型混淆（行归一化=召回率） | `outputs/disc/pred_*.jsonl` × `data/processed/dev_extract.jsonl` |
| `fig_latency_cost.png` | 精度 / 延迟 / 参数量 / 训练时长对照表 | 各 json 的 `meta` |
| `fig_llm_compare.png` + `.csv` | L3 zero / L3 few / L4 SFT 在同一批 200 句上的三级 F1 | `outputs/metrics/llm/*.json` |
| `数据分布/` 五张 | 事件类型分布、每篇事件数、角色 top30、句长、每事件论元数 | `data/processed/*.jsonl`、`data/stats/build_stats.json` |

出图口径说明（两处与脚本默认不同，为保证画的是本次真实产物）：
① 4 层消融这次跑的是 **CRF** 解码、文件名 `bert4L_crf_256`，脚本默认找 `bert4L_linear_256`，已按实际文件名画（图例写作「RoBERTa+CRF(前 4 层)」）；
② `plot_compare.py` 的图例表没有 `local_few` 前缀，已补成「Qwen3-0.6B few-shot」并按 zero → few → SFT 排序。
`fig_doc_length.png` 未重画：它读 `data/raw/` 原始语料，本次分发只有 `data/processed/`。

记录约定：本文件所有数字均取自 `week07/outputs/` 下的实际产物（`metrics/*.json` 的 F1 与 meta、`logs/*_eval.json` 的 argv 与环境、`llm/report_*.json` 的质量统计）；命令行为 `argv` 原样复原，完成时间取产物写入时刻。

---

## 附录：两个非产物数字的复测方式

1. **金标非法 BIO 地板（1.94%）** —— 跑 `python eval\check_bio_gold.py`，产物 `outputs/metrics/disc/bio_gold_baseline.json`（930 句里 18 句非法，重叠论元只能投影成一条序列）。
2. **提示 token 长度（zero 924 / few 1,576）** —— 本次用 `Qwen3-0.6B` 的 tokenizer 对 `gold_n200.jsonl` 那同一批 200 句统计，脚本：

```python
# tools/measure_prompt.py  （纯 CPU，约 20 秒）
import json, sys
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llm_zero_shot"))
from event_prompt import build_messages                      # event_prompt.py:135

tok = AutoTokenizer.from_pretrained(ROOT / "pretrain_models" / "Qwen3-0.6B")
schema = json.loads((ROOT / "data" / "processed" / "event_schema.json").read_text(encoding="utf-8"))
rows = [json.loads(l) for l in (ROOT / "outputs" / "llm" / "gold_n200.jsonl").read_text(encoding="utf-8").splitlines() if l]
for few in (False, True):
    L = [len(tok.apply_chat_template(build_messages(r["text"], schema, few_shot=few),
                                     tokenize=True, add_generation_prompt=True)) for r in rows]
    print(f"{'few' if few else 'zero'}-shot: 均值 {sum(L)/len(L):.0f}, 最大 {max(L)}, "
          f"超过 keep=896 的比例 {sum(x > 896 for x in L)/len(L):.1%}")
```

`keep = max_length - max_new_tokens` 的逻辑在 [llm_zero_shot/local_generate.py:86](file:///F:/ai/LLM/7/第七周 抽取任务/week07/llm_zero_shot/local_generate.py#L86) —— 超长的 prompt 只保留**末尾**，因为指令和待抽句子都在尾部，被砍掉的是前面的 schema 与示例。
