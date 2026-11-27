# week07 换模型 / 多方法事件抽取对比（简版）

> 任务：金融公告单句事件抽取（13 类 / 触发词 + 角色-论元）。
> 评测：`dev_extract` 固定 200 条（seed=42，与课程统一口径 `common/evaluate_events.py` 一致）。
> 本文只保留结果和结论；完整表在 `outputs/model_swap/REPORT.md`，脚本在 `model_swap/`。

## 1. 比了什么

| 维度 | 覆盖 |
|---|---|
| 模型 | 无模型 / BiLSTM-CRF（1.3M）/ BERT-base（102M）/ DeepSeek API / Qwen3-0.6B / Qwen2-0.5B / Qwen2.5-0.5B-Instruct |
| 方法 | 规则、联合 BIO（Linear / CRF / 解码硬约束）、两段式流水线、LLM zero-shot / few-shot、LoRA SFT |
| 训练 | LoRA 统一 3500 条 × 1 epoch；判别式沿用课程包 / 前次运行结果 |

## 2. 主结果（n=200，同一批样本）

| 方法 | 模型 | 触发词 F1 | 论元 F1 | 事件级 F1 | 延迟 | 备注 |
|---|---|---|---|---|---|---|
| 规则 | 无模型 | 0.575 | 0.181 | 0.020 | 2 ms | 零训练 |
| BiLSTM-CRF | 随机初始化 | 0.822 | 0.565 | 0.172 | 10 ms | 5 epoch |
| BERT+Linear | BERT-base | 0.812 | 0.560 | 0.165 | 11 ms | 非法 BIO 66.9% |
| BERT+CRF | BERT-base | 0.828 | 0.567 | 0.175 | 74 ms | 非法 BIO 62.5% |
| **BERT+CRF+硬约束** | BERT-base | **0.840** | **0.642** | **0.227** | 40 ms | 同一 checkpoint，仅解码约束 |
| 两段式 | BERT-base ×16 | 0.652 | 0.353 | 0.130 | 50 ms | 检测→类型→论元 |
| LLM zero-shot | DeepSeek API | 0.779 | 0.522 | 0.040 | 3.5 s | 100% JSON |
| LLM few-shot | DeepSeek API | 0.832 | 0.540 | 0.085 | 4.1 s | 6 个示例 |
| **LoRA SFT** | **Qwen3-0.6B** | **0.900** | **0.697** | 0.185 | 5.4 s | 课程原始结果 |
| LoRA SFT | Qwen2-0.5B | 0.875 | 0.623 | 0.100 | 0.65 s | 本次换模型 |
| LoRA SFT | Qwen2.5-0.5B-Instruct | _待补_ | _待补_ | _待补_ | _待补_ | 同配置 |
| LLM zero-shot | Qwen2.5-0.5B-Instruct | 0.022 | 0.005 | 0.000 | 2.5 s | 94% JSON 但内容错 |
| LLM zero-shot | Qwen2-0.5B（base） | 0.000 | 0.000 | 0.000 | 1.5 s | 0% JSON |
| LLM zero-shot | Qwen3-0.6B（n=60） | 0.407 | 0.117 | 0.000 | 7.9 s | 本地小模型 |

### 全量 dev 930（判别式部分，避免只看 200 条）

| 方法 | 触发词 F1 | 论元 F1 | 事件级 F1 | 训练 |
|---|---|---|---|---|
| 规则 | 0.617 | 0.174 | 0.016 | 0 |
| BiLSTM-CRF | 0.837 | 0.574 | 0.155 | 199 s |
| BERT+Linear | 0.829 | 0.563 | 0.156 | 637 s |
| BERT+CRF | 0.832 | 0.582 | 0.170 | 2743 s |
| BERT+CRF+硬约束 | 0.846 | 0.657 | 0.218 | 0（复用 checkpoint） |
| 两段式 | 0.677 | 0.369 | 0.096 | 406 s |

## 3. 结论

1. **性价比最高的是 BERT+CRF + 解码硬约束**：论元 0.642、事件 0.227、40 ms/句，零额外训练。
   同一 checkpoint 只改解码，就比普通 CRF 高 **+7.5pp 论元 / +5.2pp 事件**，非法 BIO 从 62.5% 降到 7.4%。
2. **“预训练能涨很多点”在本数据上不成立**：5 epoch 的 BiLSTM-CRF 论元 0.574，BERT+CRF 0.582，
   全量只差 **+0.8pp**。旧文档里的 +11.9pp 来自 3 epoch 未收敛的 BiLSTM（val F1 还在涨），是训练预算混淆。
3. **两段式拆细反而更差**：0.582 → 0.369（−21pp）。类型给金标只 +0.001，说明瓶颈不是类型分类，
   而是检测器漏召回（`股东增持` 31/31、`高管变动` 48/48 被整类丢掉）和每类专家数据不足。
4. **0.5B 小模型 zero-shot 不可用，只有 SFT 能救**：
   - Qwen2-0.5B base：0% JSON，F1=0；
   - Qwen2.5-0.5B-Instruct：94% JSON 合法，但 421/566 个 role 直接照抄占位符「论元角色」，
     论元 F1 0.005；追加一句“占位符要替换”后仍是 0.004（n=60）→ 是能力问题，不是措辞问题；
   - 同尺寸 LoRA 后，Qwen2 升到 0.623，说明格式和 schema 必须靠监督微调教。
5. **换模型有明确代价，不是“换新就更好”**：Qwen3-0.6B → Qwen2-0.5B，LoRA 后论元 −7.4pp、
   事件 −8.5pp；Qwen2 LoRA（0.623）甚至低于 BERT+CRF+硬约束（0.642）。Qwen3 论元第一，但事件级仍不如 BERT。
6. **没有全能模型**：规则在模板型类型上仍是第一（`股东增持` 0.83、`高管变动` 0.88、`被约谈` 1.0）；
   BERT 强在数字/日期（每股价格、回购数量、披露时间）；LLM 强在语义/易混角色（回购方、减持方、收购方）。
   生产上适合组合：LLM/规则定类型与语义槽，BERT+CRF 抽数字日期，规则兜底。

## 4. 口径提醒

- n=200 的 95% CI 约 ±4~5pp。配对 bootstrap 显示：CRF vs Linear（+0.7pp）、API zero vs few（+1.7pp）
  **不显著**；硬约束、联合 vs 两段式、Qwen3 vs Qwen2 LoRA **显著**。
- LoRA 只训 3500 条 1 epoch，未调参；BERT/BiLSTM 的指标来自课程包/前次运行，不是同一次 seed 重跑。
- 事件级 F1 要求“类型 + 触发词 + 论元集合”全对，所以远低于论元 F1；汇报时两个都要给。

## 5. 产物与复现

```bash
cd /d/workspace/nlp_study/徐鹏/week07
python model_swap/compare_models.py --bootstrap 1000   # 生成对比表/CSV/图
python model_swap/per_type_analysis.py                 # 分类型/分角色拆解
```

| 文件 | 内容 |
|---|---|
| `outputs/model_swap/REPORT.md` | 完整对比表 + 配对 bootstrap + 自动结论 |
| `outputs/model_swap/model_swap.csv` | 同表 CSV |
| `outputs/model_swap/fig_model_swap.png` | 论元 F1（带 CI）与效果-延迟图 |
| `outputs/model_swap/per_type.md` | 13 类事件 × 方法、Top15 角色 × 方法 |
| `model_swap/run_all.sh` | LLM 两路一键复现 |
