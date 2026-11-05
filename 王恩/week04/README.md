# 字符级 Transformer 生成式语言模型

使用 PyTorch 预测下一个字符，以验证集困惑度（PPL）选择最优模型，再在独立测试集报告 PPL。代码、说明和训练产物均在 week04 内。

## 运行

Python 3.10+，在本目录执行：

```bash
python -m pip install -r requirements.txt
python transformer_lm.py train
python transformer_lm.py evaluate
python transformer_lm.py generate --prompt "今天" --max-new-tokens 100
python -m unittest -v
```

也可以使用 uv（无需修改系统 Python）：

```bash
uv run --with-requirements requirements.txt python transformer_lm.py train
uv run --with-requirements requirements.txt python transformer_lm.py evaluate
uv run --with-requirements requirements.txt python transformer_lm.py generate --prompt "今天"
uv run --with-requirements requirements.txt python -m unittest -v
```

默认语料为 `../../courseware/week04/rnn_lm/corpus_general.txt`，只取前 200000 个字符，便于本地完成一次实验。默认语料和模型路径相对于脚本位置解析，从其他工作目录调用脚本也能运行。显式传入的相对路径相对于当前工作目录。

自定义语料、模型或训练规模：

```bash
python transformer_lm.py train --corpus /path/to/corpus.txt --max-chars 0 \
  --epochs 20 --seq-len 128 --d-model 256 --nhead 8 \
  --num-layers 4 --dim-feedforward 1024 --save output/large.pt --device auto
python transformer_lm.py evaluate --model output/large.pt --split val
python transformer_lm.py generate --model output/large.pt --prompt "今天" \
  --temperature 0.8 --top-k 20 --seed 42
```

`--max-chars 0` 使用完整语料；默认 CPU、2 个线程，`--device auto` 自动选择 CUDA / MPS / CPU。生成提示不能为空，且须由训练字符表中的字符组成；提示超过上下文长度时保留最近的字符作为模型输入。`--top-k 0` 使用全词表采样，temperature 越低采样越集中。

## 数据与模型

- 将 UTF-8 文本按顺序切为训练 90%、验证 5%、测试 5%，先切分再切块，避免重叠窗口跨越数据集边界。读取时 Python 将 CRLF 统一为 LF，保留空格、标点和换行。
- 字符表仅从训练集构建，`<UNK>=0`。验证和测试的未知字符映射到该项，并报告未知比例，不删除字符；词表始终保存在模型文件中。
- 每个样本为 `x = text[start:end]`、`y = text[start+1:end+1]`。每段长度最多 64；各目标位置仅计算一次，最后不足 64 的块也保留，右补齐的标签使用 `-100`，不参与 loss。
- 字符嵌入 + 可学习位置嵌入 → 2 层因果自注意力 Transformer → LayerNorm → Linear 字符 logits。默认 d_model=128，4 个注意力头，前馈层 256，dropout=0.1。
- 通过 `TransformerEncoderLayer` 实现只有自注意力的生成模型：使用上三角布尔掩码阻止关注未来字符，形成 decoder-only 的因果预测结构，无需 encoder-decoder 交叉注意力。各层独立初始化。
- 使用 AdamW（lr=0.001，weight_decay=0.01）、batch size 64、梯度裁剪 1.0、随机种子 42，默认训练 10 轮。交叉熵直接接收 logits，训练时无需先 softmax。
- 每轮关闭 dropout 计算验证指标，保存验证 loss 最低的权重；训练结束恢复该权重后评测测试集。测试集不参与模型选择。

## 如何用 PPL 判断模型好坏

对所有有效目标字符累加负对数似然（NLL），然后除以字符数：

```text
loss = -sum(log p(真实的下一个字符 | 当前块内的前文)) / 有效目标字符数
PPL  = exp(loss)
```

使用自然对数。先累加 NLL、再计算 PPL，不能直接平均各 batch 的 PPL，补齐位置也不能计入分母。PPL 越低，表示模型给真实下一个字符分配的概率整体越高；均匀预测 V 个字符时 PPL=V，理想确定预测的 PPL 趋近 1。

比较模型必须使用相同语料划分、字符表、未知字符处理和评估上下文长度。这里采用不重叠目标分块的评估方式，每块的首个预测只能看到块内第一个字符，每个划分的首字符不作为目标；PPL 不是逐字符滑窗满上下文评估结果。未知比例较高时，仅比较 PPL 可能有误导，因此报告中同时保留未知比例。

小语料和小模型用于教学，PPL 下降并不保证生成文本连贯，不能仅凭一个 PPL 数值认定模型达到实用水平。可根据验证 PPL 调整训练轮数、语料规模和模型容量；不要反复根据测试 PPL 调参。

## 输出与复核

- `output/transformer_lm.pt`：最佳权重、模型配置、字符表、最佳轮次、种子、验证指标、语料路径、划分比例和文本 SHA-256。
- `output/transformer_lm.json`：每轮 train/val loss 与 PPL、初始验证 PPL、最佳轮次、最终测试 PPL、字符数和未知比例。
- `evaluate` 独立加载模型并按原划分重新计算 PPL；会校验所用语料的 SHA-256。如果语料移到其他路径，可用 `--corpus /new/path/corpus.txt` 指定相同内容。
- `generate` 每次只取最后位置的分布，按 temperature 和 top-k 抽取下一个字符，追加后继续生成；屏蔽 `<UNK>` 的生成。
- 测试覆盖：训练/推理时未来字符不可见、全部目标只计算一次、短块补齐与均匀预测的 PPL、数据切分及未知字符、权重加载一致性、超长提示与生成参数检查。

实现参考：[PyTorch TransformerEncoderLayer 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)。

## 本次实际训练结果

使用上述默认参数和课程语料前 20 万字符，Python 3.12、PyTorch 2.15.0、CPU 训练 10 轮。词表 3299 项（含 `<UNK>`），模型参数 1121251 个。

| 指标 | 结果 |
| --- | ---: |
| 初始验证 PPL | 3927.350 |
| 第 1 轮验证 PPL | 458.478 |
| 最佳轮次 | 10 |
| 最佳验证 loss / PPL | 4.7566 / 116.349 |
| 测试 loss / PPL | 4.7752 / 118.535 |
| 验证 / 测试未知字符比例 | 0.34% / 0.40% |

验证、测试集各 10000 字符，各统计 9999 个目标字符。训练后独立执行 `evaluate` 重新加载权重，测试 PPL=118.534665，与训练报告一致。5 项单元测试全部通过。不同设备或 PyTorch 版本重训可能有数值差异。

以“今天”为提示生成的节选（temperature=0.8、top-k=20、seed=42）：

```text
今天
哪些好看看
谁唱电影
我不好看的世界钻怎么做啊
怎么样
怎么下载好看
```

PPL 明显下降说明模型学到了部分局部字符规律，但输出仍有重复、语义衔接较弱；本次只使用完整语料的一小部分，结果属于教学基线。
