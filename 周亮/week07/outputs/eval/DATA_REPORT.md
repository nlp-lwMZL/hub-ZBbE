# cdtier/clean 数据探索报告

> 数据目录：`/Users/zhouliang/Downloads/badou/nlp/homework/week7/code/data/cdtier/clean`
> 所有统计由 `tools/analyze_data.py` 从 clean 数据计算，可复现。

## 1. 三个 split 概览

| split | 句数 | token 数 | 实体数 | 类别数 |
|---|---:|---:|---:|---:|
| train | 3091 | 312352 | 3738 | 5 |
| validation | 387 | 40793 | 443 | 5 |
| test | 387 | 37720 | 432 | 5 |

**类别不均衡比（train，最多类/最少类）= 14.82×**

## 2. 实体类别分布

| 类型 | train | validation | test |
|---|---:|---:|---:|
| Attacker | 1156 | 132 | 129 |
| Tools | 963 | 135 | 133 |
| Region | 891 | 102 | 95 |
| Industry | 650 | 66 | 67 |
| Campaign | 78 | 8 | 8 |

## 3. 句子长度分位数（token 数）

| split | min | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| train | 1 | 66 | 183 | 276 | 694 | 2232 |
| validation | 2 | 69 | 187 | 284 | 738 | 2857 |
| test | 6 | 64 | 186 | 280 | 764 | 1257 |

## 4. 实体长度（字符数）

| 类型 | 个数(train) | 平均 | 中位 | max |
|---|---:|---:|---:|---:|
| Attacker | 1156 | 5.81 | 5 | 25 |
| Campaign | 78 | 5.79 | 4 | 17 |
| Industry | 650 | 3.58 | 3 | 39 |
| Region | 891 | 2.88 | 2 | 9 |
| Tools | 963 | 7.16 | 7 | 26 |

## 5. 截断率（不同 max_length）

> 判据：token 数 + 2 个特殊符 > max_length 即被截断。截断率就是召回上限。

| split | max_len=256 | 384 | 512 |
|---|---:|---:|---:|
| train | 175 (5.66%) | 98 (3.17%) | 62 (2.01%) |
| validation | 22 (5.68%) | 12 (3.10%) | 7 (1.81%) |
| test | 22 (5.68%) | 11 (2.84%) | 8 (2.07%) |

→ 句子 p99 远低于 256，但 max 到 2232；少数极长句必须靠**评测期滑窗**救回尾部实体。

## 6. 切分无泄漏核查

| split 对 | 重叠文本数 |
|---|---:|
| train-validation | 0 |
| train-test | 0 |
| validation-test | 0 |

## 7. 可写进简历的一句话

- 数据探索：cdtier clean 3091 句训练集，5 类实体 **14.82× 不均衡**（Attacker 1156 vs Campaign 78），句长 p50=66 / max=2232，overlap=0 → 定下按文本分组切分 + 评测期滑窗。
