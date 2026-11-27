# common/ —— 全项目共用的公共模块

> **这里不放任何"方法"**。判别式链路（规则 / BiLSTM / BERT / 两段式）和系统组合都 import 这里的代码，
> 保证「同一份标签体系、同一份评测口径、同一份日志格式」——否则 6 种方法的数字不可比。

| 文件 | 作用 | 谁在用 |
|---|---|---|
| `event_data.py` | **数据层核心**：两种标签体系（joint 149 标签 / typed 11~19 标签）、span↔BIO 转换、`word_ids()` 子词对齐、BIO 解码、非法序列统计、数据集类 | 所有方法 |
| `evaluate_events.py` | **唯一评测口径**（只读，Lead 维护）：trigger / argument / event_exact 三级 P/R/F1 + type_macro + per_type/per_role | 所有方法（`eval/eval_utils.py` 调它） |
| `eval_utils.py` | 评测封装：`evaluate_predictions()` 落盘预测+指标、`Timer` 计时/峰值显存、n=200 子集投影 | 所有方法 |
| `crf.py` | **手写 CRF**：前向算法 + 维特比 + BIO 硬约束（不依赖 pytorch-crf） | `bert/`、`bilstm_crf/` |
| `models.py` | 公共模型定义：`BertEncoder` / `BertCls` / `build_model` / `build_arg_model` / `predict_char_tags` | `bert/`、`bert_pipeline/`、`service/` |
| `model_paths.py` | 定位本地 `pretrain_models/bert-base-chinese`（母版/交付副本两套目录都能找到，不联网下载） | 判别式方法 |
| `log_utils.py` | **统一训练日志**：`TrainLogger` / `EvalLogger` → `outputs/logs/*_train.json`、`*_steps.jsonl`、`*_console.log` | 所有方法 |

## 各方法脚本怎么引公共模块

```python
# 每个方法脚本开头都长这样（方法目录的上一级 = 项目根目录）
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))     # 有了这一行，下面才能 import 公共模块
from event_data import load_schema, ...
```

## 为什么 `models.py` 要单独拆出来

重构后 `bert/train.py`、`bilstm_crf/train.py`、`bert_pipeline/train.py` **都叫 `train.py`**。
一旦把这些目录同时放进 `sys.path`，`import train` 就会撞名、拿到别的方法那个文件。
而 `service/extract_service.py` 天然要同时用到「BERT 编码器」（bert）和「两段式分类头」（bert_pipeline），
所以把**与训练流程无关、纯粹是模型结构/推理还原**的部分集中到 `common/models.py`：

- 各方法的训练循环、超参、CLI **完全不受影响**（函数签名与实现和重构前一致，只是换了存放位置）
- `service/` 只需 `from models import BertEncoder, BertCls, build_arg_model`
