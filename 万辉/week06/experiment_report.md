# 第 6 周实验结果汇总（自跑数据）

整理日期：2026-11-19
数据来源：`week06/outputs/` 下各实验的 `log_exp*.json`（训练日志）、`pred_exp*.json`（干净集评测）、`rule_pred.json`、`tfidf_pred.json`、`lora_pred.json`。
环境：py312（torch 2.3.1+cu121 / transformers 4.51.3 / peft 0.15.0 / scipy 1.13.1），RTX 3060 Laptop 6GB。
模型：判别式 = iic RoBERTa backbone（`pretrain_models/roberta-base-chinese`，model_type=bert）；生成式 = Qwen3-0.6B（本地）。

## 表 1：RoBERTa 消融实验

主口径：去重后干净验证集 500 条（seed=42），与其他方法可比。
参考口径：全量 val（含 11% 与训练集重复样本）上最佳 epoch 的 Macro-F1，取自训练日志。

| # | 配置 | Acc（干净500） | Macro-F1（干净500） | 相对基线 | Macro-F1（全量val最佳epoch） | 最佳epoch |
|---|---|---|---|---|---|---|
| exp1 | CLS 池化（基线） | 0.556 | 0.5514 | — | 0.5695 | 3 |
| exp2 | Mean 池化 | 0.534 | 0.5325 | −1.9 | 0.5661 | 3 |
| exp3 | Max 池化 | 0.548 | 0.5432 | −0.8 | 0.5629 | 2 |
| exp5 | 学习率 3e-5 | 0.566 | **0.5616** | **+1.0** | **0.5752** | 3 |
| exp6 | max_length 32 | 0.550 | 0.5466 | −0.5 | 0.5703 | 3 |
| exp7 | + FGM 对抗训练 | 0.564 | 0.5574 | +0.6 | 0.5655 | 3 |
| exp8 | + Label Smoothing 0.1 | 0.546 | 0.5489 | −0.3 | 0.5691 | 3 |
| exp9 | Focal Loss (γ=2) | 0.562 | 0.5575 | +0.6 | 0.5622 | 3 |

读表说明：

1. 最优配置为 exp5（lr 3e-5），两个口径均第一；单旋钮最大提升 +1.0 个点，复现了老师「单点提升只有 1~2 个点、任务贴近数据上限」的结论。
2. 两个口径排序不完全一致（exp2 干净集 −1.9、全量 val 仅 −0.3）。500 条样本的波动约 ±1 个点，小于 1 个点的差异不应过度解读。
3. 与老师基准趋势的差异：老师表中 mean/max 池化上涨，本实验下跌。原因包括模型不同（本实验 RoBERTa checkpoint 的 LayerNorm 为旧式 gamma/beta 键名，from_pretrained 静默随机初始化，见「已知问题」）与抽样波动。
4. 类别加权默认开启（全部实验一致），未做 --no_class_weight 对照。
5. 校准（ECE，干净500）：exp1 0.1109 / exp2 0.1503 / exp3 0.1172 / exp5 0.1231 / exp6 0.1251 / exp7 0.0945 / exp8 0.0570 / exp9 0.0760。exp8 校准最好但 F1 略降，exp7/exp9 兼顾两者。

## 表 2：方法横向汇总（eval_all.py 口径）

| 方法 | 准确率(含失败) | Macro-F1 | 样本数 | 覆盖类别 | 解析失败 |
|---|---|---|---|---|---|
| 规则 / 关键词 | 0.126 | 0.0716 | 500 | 10/15 | 0 |
| TF-IDF + SVM | 0.512 | 0.4721 | 500 | — | 0 |
| RoBERTa 微调（exp5 最优） | 0.566 | 0.5616 | 500 | — | 0 |
| Qwen3 zero-shot | 0.168 | 0.105 | 500 | 8/15 | 1 |
| Qwen3 LoRA SFT | 0.556 | 0.5084 | 500 | 14/15 | 0 |

口径警告与说明：

1. **口径已统一**：全部方法均在 500 条去重干净样本（seed=42）上评测。全量 8901 条版本另存为 `rule_pred_full8901.json` / `tfidf_pred_full8901.json`（rule 0.1319/0.0688、tfidf 0.5296/0.5087），仅作留档参考。
2. **zero-shot 塌缩实测**：500 条中 408 条（81.6%）被预测为「文化」，「科技」57 条，其余 6 类合计 34 条，7 类从未输出；解析失败 1 条（0.2%）。与 LoRA（14/15 类、0.5084）对照，证明塌缩是模型能力问题而非任务问题，SFT 可修复。
3. LoRA（0.5084）高于老师基准 0.484，覆盖 14/15 类、解析失败 0；但仍低于 RoBERTa 微调（0.5616），支持「封闭标签 + 短文本 → 判别式微调最划算」的主结论。
4. RoBERTa 行取 exp5 占「判别式微调」槽位；exp7/exp9 的 ECE 更低，若重视置信度校准可替换。
5. 「覆盖类别」列：规则 10/15、zero-shot 8/15、LoRA 14/15；判别式方法不输出 pred_distribution（该列为 —），实际覆盖见各 pred json 的混淆矩阵。
6. 本环境 rule/tfidf/zero-shot 三项数字与老师 README 基准（0.126/0.072、0.512/0.472、0.168/0.105）几乎完全一致，说明数据口径与评测实现复现正确。

## 已知问题（影响结果解读）

1. **RoBERTa LayerNorm 随机初始化**：iic backbone 的 checkpoint 键名为旧式 gamma/beta（带 bert. 前缀），transformers from_pretrained 不报错但将全部 LayerNorm 随机初始化。本组 RoBERTa 数字是在该起点上微调得到的，与老师 bert-base-chinese 基准不完全可比。处理方式待定（修权重键名 / 改加载代码 / 换标准格式模型）。
2. **transformers 版本**：为绕 CVE-2025-32434（torch<2.6 拒载 .bin）降级至 4.51.3；老师代码中 `dtype=` 写法在自跑副本中改为 `torch_dtype=`（llm_lora 两处文件共 3 行）。
3. **数据泄漏对照未跑**：tfidf 的 --leak 对照实验未执行，全量 val 与干净集差异（老师实测 0.541 vs 0.509）未在本环境复现。

## 产物索引

- 消融训练日志：`log_exp{1,2,3,5,6,7,8,9}_*.json`；对应权重：`ckpt_exp*_*.pt`（每个约 390MB）
- 消融干净集评测：`pred_exp*_*.json`（含逐条预测、混淆矩阵、ECE）
- LoRA：`lora_adapter/`（9MB）、`lora_train_log.json`（2 epoch，峰值显存 3.0GB）、`lora_pred.json`
- 基线：`rule_pred.json`、`tfidf_pred.json`（500 条口径）；全量版 `rule_pred_full8901.json`、`tfidf_pred_full8901.json`
- zero-shot：`zero_shot_pred.json`（500 条，含逐条 raw 生成文本，可查塌缩与解析失败案例）
