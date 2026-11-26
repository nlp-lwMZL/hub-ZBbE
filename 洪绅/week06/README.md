# TNEWS 新闻分类方法对比

在同一份数据上比较规则、TF-IDF、BERT、训练技巧、LLM 提示分类和 SFT，共 12 组。
复用预训练 BERT 和 Qwen，默认使用小数据集和训练上限，整轮约 15～30 分钟。

```bash
pip install -r requirements.txt
python run_experiments.py --dry-run   # 只查看方法列表
python run_experiments.py             # 准备数据，依次运行全部方法
python run_experiments.py --methods bert weighted focal  # 只跑选中的方法
```

第一次运行会下载缺少的数据和模型，这部分耗时不计入估计。后续使用本地缓存。

| 文件 | 内容 |
|---|---|
| `l0_rules.py` | 关键词规则 |
| `l1_tfidf.py` | TF-IDF + 逻辑回归 / SVM |
| `l2_bert.py` | 预训练 BERT 分类微调 |
| `l2_bert_tricks.py` | 类别加权、Focal Loss、标签平滑、FGM |
| `l3_llm_prompt.py` | Qwen zero-shot / 15-shot |
| `l4_sft.py` | Qwen LoRA / 全量 SFT |
| `common.py` | 数据准备和评测 |
| `run_experiments.py` | 批量运行，汇总分数 |

训练集取 3000 条，验证集和测试集各取 150 条。各类别至少保留两条，再按原始数据随机补足。
三个子集分别从原来的 train/dev/test 抽取，固定种子 42，保存到 `data/tnews_quick/`。
原始数据保存在 `data/tnews/`，不会覆盖。数据量可在 `common.py` 顶部修改。

BERT 和四种技巧最多训练 3 轮、200 步或 120 秒；LoRA 和全量 SFT 最多 3 轮、300 步或 240 秒。
达到任一上限就停止训练，再评测同一个测试子集。上限参数在对应算法文件顶部。
时间在每批结束后检查，不包括模型加载和评测；CPU 或设备繁忙时整轮可能超过预计时间。

BERT 使用 `hfl/chinese-roberta-wwm-ext`，LLM 使用 `Qwen/Qwen2.5-0.5B-Instruct`。
LLM 每条最多生成 4 个 token；使用内置 eager 注意力以兼容本机 MPS。

运行后查看 `results/对比表.md`，里面有 accuracy、macro-F1、实际训练步数和总耗时。
逐类召回和报错在 `results/train.log`。每次运行会更新这两个文件，某个方法失败会继续运行其余方法。
小测试集和短训练的分数只用于观察方法差异，不代表充分训练后的排名。

各算法也可以单独运行，例如 `python l2_bert.py`。代码注释说明了主要外部 API 的参数和数据形状。
