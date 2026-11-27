# data/peoples_daily —— 人民日报 NER（BIO）

本目录的数据来自过去 NER 教学项目 `bert_ner`（上一期 week7 序列标注课的配套数据），
原始来源是公开的「人民日报命名实体识别」语料：
GitHub [`OYE93/Chinese-NLP-Corpus`](https://github.com/OYE93/Chinese-NLP-Corpus) 的
`NER/People's Daily/example.train|dev|test`。**仅限课程学习使用，请勿商用或再次分发。**

> 原始发布是 **CoNLL 格式**：每行 `字符 BIO标签`，空行分句。
> 本目录的 JSON 是旧项目 `download_data.py` 的 `_parse_conll()` 解析后的结果。
> 学生如果自己下原始文件，第一步要先解析 CoNLL——这本身也是「格式处理」的一部分。

## 文件

| 文件 | 说明 |
|---|---|
| `train.json` | 训练集，20,864 条 |
| `validation.json` | 验证集，2,318 条 |
| `test.json` | 测试集，4,636 条 |
| `label_names.json` | 7 个 BIO 标签的顺序（**训练/评估必须共用这一份**） |

## 单条格式

```json
{
  "tokens": ["海", "钓", "比", "赛", "地", "点", "在", "厦", "门", "与", "金", "门", "之", "间", "的", "海", "域", "。"],
  "ner_tags": ["O", "O", "O", "O", "O", "O", "O", "B-LOC", "I-LOC", "O", "B-LOC", "I-LOC", "O", "O", "O", "O", "O", "O"]
}
```

- `tokens`：已经切好的**单字列表**（不是原始句子字符串）。
- `ner_tags`：与 `tokens` **等长**的 BIO 标签：`O` / `B-PER` / `I-PER` / `B-ORG` / `I-ORG` / `B-LOC` / `I-LOC`。
- 没有 `text` 字段，也没有 `start` / `end` 偏移——这就是和课上 DuEE「start/end 事件标注」最大的区别。
  句子原文用 `"".join(tokens)` 还原；实体的 start/end 由标签序列**推导**（见 `src/bio_labels.py`）。

## 标签

| 标签 | 含义 |
|---|---|
| `O` | 非实体 |
| `B-PER` / `I-PER` | 人名（Begin / Inside） |
| `B-ORG` / `I-ORG` | 机构名 |
| `B-LOC` / `I-LOC` | 地名 |

## 数据规模

| split | 句子数 | 实体数 | 句长 p95 | 截断 @128 |
|---|---|---|---|---|
| train | 20,864 | 33,992 | 97 字 | 357 条（1.71%） |
| validation | 2,318 | 3,819 | 97 字 | 41 条（1.77%） |
| test | 4,636 | 7,707 | 97 字 | 79 条（1.70%） |

> 详细统计（标签分布、O 占比、每句实体数、重复检查）由 `src/explore_data.py` 生成，
> 结果写在 `outputs/data_report.json`。
