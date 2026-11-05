# 多语种识别分类器：12 语种模型训练

参考 `../../courseware/week03/language_recognize/train_langid_cls.py` 的字符级 RNN 思路，使用本目录已有的 v12 数据完成训练、独立评测和单句预测。

## 数据与标签

CSV 格式为 `text,label`。训练集 9600 条（每类 800），验证集 1200 条（每类 100），测试集 1800 条（每类 150）。

| 标签 | 语言 | 标签 | 语言 |
| --- | --- | --- | --- |
| ar | 阿拉伯语 | en | 英语 |
| es | 西班牙语 | hi | 印地语 |
| ja | 日语 | ko | 韩语 |
| pt | 葡萄牙语 | ru | 俄语 |
| th | 泰语 | tr | 土耳其语 |
| vi | 越南语 | zh | 中文 |

相较课堂的 8 语种，新增 ko、th、tr、vi。标签按训练集中的标签排序生成，输出层维度使用标签数量；字符表也仅从训练集重新统计，不使用 v8 字符表。验证集、测试集遇到训练集未见标签时明确报错。

文本先做 NFKC 归一化、转小写，只保留 Unicode 字母和组合标记；保留组合标记可避免丢失印地语、泰语的元音信息。字符按频次降序编号，同频按字符排序，`<PAD>=0`、`<UNK>=1`。默认截断/补齐到 40 字符。

## 模型和训练

模型结构：Embedding(64) → RNN(64) → 掩码 MaxPool → LayerNorm → Dropout(0.3) → Linear(12)。补齐位置不参与池化。使用 CrossEntropyLoss（直接输入 logits），Adam，学习率 0.001，batch size 64，默认 8 轮，随机种子 42，CPU 单线程。

每轮只在验证集上评测并按准确率选择最优模型。训练结束后恢复最优权重，再评测测试集。输出 accuracy、macro-F1、逐类 precision/recall/F1 和主要混淆组合。softmax 仅用于单句预测时展示概率。

## 运行

Python 3.10+，在本目录运行：

```bash
python -m pip install -r requirements.txt
python train_langid_cls.py
python eval_langid.py
python eval_langid.py --text "今天我们一起学习深度学习。"
```

也可以使用 uv，无需给系统 Python 安装依赖：

```bash
uv run --with torch --with numpy python train_langid_cls.py
uv run --with torch --with numpy python eval_langid.py --text "This is a language classifier."
```

默认数据和产物路径均相对于脚本位置，从其他工作目录执行也可运行。可调整参数：

```bash
python train_langid_cls.py --epochs 12 --batch-size 64 --max-len 60 --save output/langid_rnn_v12.pt
python eval_langid.py --model output/langid_rnn_v12.pt --split test
```

训练产物：

- `langid_rnn_v12.pt`：最优权重、12 类标签、字符表、模型维度、最大长度和训练种子；评测和推理直接读取这些配置。
- `langid_rnn_v12.json`：训练参数、每轮 loss 和验证指标、最优轮次、测试指标及混淆矩阵（行真实、列预测，顺序见 labels）。指定 `--save` 时 JSON 同名保存。

## 本次实测结果

Python 3.12、PyTorch 2.15.0，默认 8 轮参数。字符表 3276 项（含两个特殊标记），模型参数 218892 个。

- 最优轮次：8；验证准确率：97.08%。
- 测试准确率：96.06%（1729/1800）；macro-F1：0.9605。
- 主要混淆：es → pt 29 条，pt → es 13 条。
- 保存后重新加载并独立评测，指标一致；12 类输出、反向传播、组合字符、未知字符、空输入及未知标签检查通过。

以上结果对应提供的固定测试集；西班牙语和葡萄牙语共享大量字符，是当前模型的主要薄弱项。不同 PyTorch 版本和硬件下重训结果可能略有差异。单句预测限于这 12 类，未实现未知语种拒识。
