# bert_pipeline/ —— L2 两段式流水线

**一句话**：把「一次联合抽取」拆成三段 ——
(a) **事件检测器**（句中有没有事件，二分类，负样本约 78%）→
(b) **事件类型 13 分类** →
(c) **类型条件论元抽取**（已知类型 t，只抽该类型的角色，标签空间从 149 缩到 11~19，每类型一个专家模型）。

```bash
python bert_pipeline/train.py                     # 完整：检测器 + 13分类 + 13专家 + 组合评测
python bert_pipeline/train.py --only_experts      # 只重训 13 个专家（复用 detector/type，省 20 分钟）
python bert_pipeline/train.py --mock --device cpu --cls_epochs 1 --arg_epochs 1   # 冒烟
python bert_pipeline/train.py --skip_train        # 复用 checkpoint，只重新组合评测
```

| 产物 | 说明 |
|---|---|
| `outputs/checkpoints/pipe_detector.pt` `pipe_type.pt` `pipe_arg_<类型>.pt` | 16 个模型 |
| `outputs/metrics/disc/pipe_detector.json` | 检测器 P/R/F1（dev_detect 全量 + clean 去泄漏版） |
| `outputs/metrics/disc/pipeline.json` | 端到端三级 F1 + 诊断（检测丢多少句/类型错多少） |
| `outputs/metrics/disc/pipeline_oracle_type.json` | **类型给金标的上界**（定位瓶颈在类型还是论元） |
| `outputs/logs/bert_pipeline_train.json` / `_console.log` | 训练日志 |

**教学重点（这一页比一个 F1 更值钱）**
1. 两段式的**理论优势**：标签空间小一个数量级，每个标签样本密度高 → 论元更容易学准。
2. 两段式的**隐性成本**：**训练预算被切成 13 份**。5747 句按类型切分后，
   最大的「股份回购」979 条、最小的「被约谈」53 条 → 每个专家每 epoch 只有 3~31 步。
   实测：专家 1 epoch（约 30 步）时**只会吐 `I-` 标签、一个 `TRG` 都不出、端到端 F1 = 0**；
   提到 3 epoch 才有正常数字。**「拆得越细、每个子模型越饿」**。
3. **上界分析**：把类型换成金标后 F1 几乎不变 → 瓶颈不在类型分类（准确率 0.9452），而在论元专家训练不足。
