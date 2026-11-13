# Week05：基于语言模型的 SFT 训练

衔接 week05 的 Decoder-only 范式与自回归生成演示，复用 week04 已训练的字符级因果 Transformer，在人工编写的中文情感指令数据上做全参数监督微调（SFT）。这是可以在 CPU 上完成的教学实验，不需要调用 DeepSeek API 或下载大模型。基础权重来自 `../week04/output/transformer_lm.pt`，并非通用预训练大模型。

## 运行

Python 3.10+，在本目录执行：

```bash
uv run --with-requirements requirements.txt python -m unittest -v
uv run --with-requirements requirements.txt python sft_lm.py train
uv run --with-requirements requirements.txt python sft_lm.py evaluate
uv run --with-requirements requirements.txt python sft_lm.py generate \
  --instruction '判断情感，回答积极或消极。' --input '这家店值得推荐'
uv run --with-requirements requirements.txt python sft_lm.py generate \
  --instruction '判断情感，回答积极或消极。' --input '这家店值得推荐' \
  --sample --temperature 0.8 --top-k 20 --top-p 0.9
```

也可 `python -m pip install -r requirements.txt` 后直接运行。默认 CPU、2 线程；训练可指定 `--device auto` 自动选择 CUDA / MPS / CPU。线程数放在子命令前，如 `python sft_lm.py --threads 4 train`。默认 40 轮、batch size 8、学习率 0.0003，可通过 `--epochs`、`--batch-size`、`--lr` 修改。

默认数据和模型路径相对于脚本目录；显式传入的相对路径相对于当前工作目录。若基础权重不存在，先按 week04 README 训练，或用 `--base-model /path/to/transformer_lm.pt` 指定兼容权重。此实现直接导入 week04 模型代码，请保留相邻目录结构。

## 数据与监督目标

`data/train.jsonl`、`val.jsonl`、`test.jsonl` 分别有 32、4、4 条人工示例，包含积极、消极两类，按不同句子固定划分。问题不跨集合重复。仅用于验证训练流程，规模不足以评估真实泛化能力。

每行格式：

```json
{"instruction":"判断情感，回答积极或消极。","input":"电影很好看","output":"积极"}
```

`input` 可省略，`instruction` 和 `output` 必须为非空字符串。同一提示不能重复或跨划分出现；数据格式错误和超长样本直接报错，避免静默丢掉回答。换任务时可指定 `--train`、`--val`、`--test` 三个 JSONL 路径。

统一序列模板：

```text
<BOS>指令：{instruction}\n输入：{input}\n回答：{output}<EOS>
```

空 input 会省略“输入”行。字符级编码，不把 `<BOS>` 等特殊项拆成字符。

- 与预训练预测全部后续字符不同，SFT 仅监督 output 和 EOS；提示仍参与注意力并提供条件。
- 输入 `x = ids[:-1]`，标签与 `ids[1:]` 对齐。提示对应标签设为 `-100`，因此提示末字的 logits 恰好预测答案首字。
- batch 右补齐的标签同样设为 `-100`。模型沿用因果注意力，真实位置无法读取未来字符和右侧补齐。损失为有效答案 token 的 NLL 总和除以数量，PPL 为其指数。
- 保留基础模型字符 ID、全部旧权重及位置嵌入，只扩展训练集字符和 PAD/BOS/EOS 对应的词表行；验证和测试数据不参与建表。未知字符映射到 UNK，并在报告中记录比例。推理遇到提示词表外字符时明确报错。
- 全参数 AdamW 更新，梯度裁剪 1.0。每轮验证，按最低验证 loss 保存权重；训练结束重新加载最佳权重后评测测试集。

忽略标签的行为参考 [PyTorch CrossEntropyLoss 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。

## 生成与输出

默认使用贪心解码，方便复核情感任务；`--sample` 使用 temperature → top-k → top-p → 抽样，与 week05 采样课件对应。屏蔽 UNK/PAD/BOS，遇到 EOS 停止；同时设置 `--max-new-tokens` 上限。超过上下文长度后保留最近的字符，未实现 KV cache 或 beam search。

- `output/sft_lm.pt`：最佳权重、扩展词表、模型配置、最佳轮次及模板版本。权重文件已加入本目录忽略规则。
- `output/sft_lm.json`：参数、随机种子、数据路径和 SHA-256、未知字符比例、初始验证指标、每轮 train/val 指标、最佳验证与最终测试指标，以及测试集逐条生成和完全匹配率。
- `evaluate --data ...`：独立加载权重，在指定数据上重新计算答案区域 loss / PPL。

测试覆盖答案首字对齐、提示与补齐掩码、超长样本拒绝、原权重保留、未来字符不可见、保存加载一致性、答案 token 计数、EOS 停止及重复问题拒绝。验证和测试各仅 4 条，完全匹配率以 25% 为步长变化；PPL 下降不代表模型具有通用问答能力。

## 本次实际运行结果

使用默认参数、seed=42、PyTorch 2.15.0，在 CPU 上完成 40 轮训练：

| 指标 | 结果 |
| --- | ---: |
| 微调前验证 loss / PPL | 6.9815 / 1076.5094 |
| 最佳轮次 | 40 |
| 最佳验证 loss / PPL | 0.0820 / 1.0855 |
| 测试 loss / PPL | 0.4708 / 1.6012 |
| 测试集生成完全匹配率 | 50%（2/4） |
| 各集合未知字符比例 | 0% |

独立执行 `evaluate`，重新加载权重得到相同测试指标；5 项单元测试全部通过。

| 测试输入 | 参考答案 | 贪心生成 |
| --- | --- | --- |
| 这家店值得推荐 | 积极 | 积极 |
| 这家店太让人失望 | 消极 | 积极 |
| 今天的天气真好 | 积极 | 积极 |
| 今天的天气真糟糕 | 消极 | 积极 |

模型在这 4 条测试数据上都生成了“积极”，因此本次结果主要验证 SFT 流程和格式学习；尚不能认为模型学会了可靠的情感判断。后续可扩充表达多样、类别平衡的训练数据，再根据验证集选择训练参数。测试集只用于最终报告。
