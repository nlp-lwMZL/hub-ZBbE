# 第六周作业：更换数据集的分类方法对比实验

将课程的 **TNEWS 中文新闻 15 分类**，更换为项目已有的 **12 语种文本识别**。沿用 `courseware/week06/code` 中“规则基线 → TF-IDF → 神经网络/预训练模型 → 统一评测”的实验思路，使用 PyTorch 和 scikit-learn 实现。

基础实验无需下载数据或预训练权重，默认在 CPU 上运行。另提供多语种 DistilBERT 的冻结特征和全量微调实验入口。

## 与课程代码的对应

| 课程参考 | 本作业实现与变化 |
| --- | --- |
| `common/data.py` | `data.py`：CSV 读取、NFKC/大小写/空白归一化去重、固定 seed、数据指纹 |
| `rule/classify_rule.py` | `baselines.py`：文字系统与功能词规则，替换新闻关键词 |
| `tfidf/classify_tfidf.py` | `baselines.py`：统一字符 1–3 gram，比较 NB、LR、LinearSVC |
| `bert/train_bert.py` | `neural.py`：掩码池化、分类头、分层学习率、Dev 选择最佳权重；新增字符 CNN |
| `common/metrics.py`、`eval/eval_all.py` | `run_experiments.py`：统一 Accuracy/Macro-F1/逐类指标/混淆矩阵/耗时 |

这里选择“更换数据集”完成作业要求；不直接搬用中文 BERT 识别所有语种。可选预训练模型也换为 `distilbert-base-multilingual-cased`（覆盖 104 种语言）。本作业不包含课程中的 LLM zero-shot、LoRA 和外部 API 实验。

## 数据与公平比较

默认读取 `../week03/v12/{train,dev,test}.csv`，列为 `text,label`，不改动原文件。12 类为 `ar/en/es/hi/ja/ko/pt/ru/th/tr/vi/zh`。当前数据训练 9,600 条、验证 1,200 条、测试 1,800 条，归一化去重后数量不变。

- 按 Train → Dev → Test 顺序去重，同时去除集合内部重复；同一归一化文本出现冲突标签时直接报错。
- 字符词表、TF-IDF 词表和 IDF 均仅使用 Train 拟合。归一化仅用于去重，模型使用原始文本（TF-IDF 默认小写、CNN 使用 casefold）。
- 神经网络每轮只查看 Dev，按 Dev Macro-F1 保存最优权重，恢复后才评测 Test。传统方法超参数预先固定，不按 Test 调参。
- 同一运行所有方法使用相同标签顺序和测试样本；不同运行先核对 `manifest.json` 的数据指纹。混淆矩阵为“行真实、列预测”。
- `--train-per-class` 只缩小训练集；`--eval-per-class` 同时缩小 Dev/Test，适合冒烟，不应与完整测试指标混为一谈。

## 方法与默认配置

| 方法参数 | 模型 | 训练配置 |
| --- | --- | --- |
| `rule` | 文字范围/功能词命中；无法区分时回退 Train 多数类 | 无参数学习；规则以当前 12 类为目标 |
| `tfidf_nb` | TF-IDF + MultinomialNB | alpha=0.1 |
| `tfidf_lr` | TF-IDF + LogisticRegression | C=4，max_iter=1000 |
| `tfidf_svm` | TF-IDF + LinearSVC | C=1 |
| `char_cnn` | Embedding(64) → Conv1d(3/5/7，各64通道) → masked MaxPool → Dropout → Linear | AdamW，lr=1e-3，5 epoch |
| `encoder_frozen` | 多语种 DistilBERT → masked MeanPool → 分类头 | encoder 冻结且保持 eval；只训练分类头，lr=1e-3 |
| `encoder_finetune` | 同上，训练全部参数 | encoder lr=2e-5，分类头 lr=1e-4 |

三种 TF-IDF 方法共用 char(1,3)、最多 100,000 特征、sublinear TF。神经网络默认 batch=64、最大长度=96（CNN 为字符数，Encoder 为 token 数），梯度裁剪=1，weight decay=0.01。PyTorch 默认 CPU 两线程，seed=42；传统方法使用依赖库的默认线程设置。GPU 跨设备结果不保证逐位相同。冻结/微调使用同一 seed 初始化分类头，形成可解释的对照。

## 运行

Python 3.10+，在本目录执行：

```bash
python -m pip install -r requirements.txt
python run_experiments.py
python -m unittest -v test_experiments.py
```

也可以使用 uv，避免给系统 Python 安装依赖：

```bash
uv run --with-requirements requirements.txt python run_experiments.py
uv run --with-requirements requirements.txt python -m unittest -v test_experiments.py
```

脚本默认路径基于自身位置，从其他工作目录运行也成立。自定义相对路径按当前工作目录解释。

快速检查和单独运行传统方法：

```bash
python run_experiments.py --train-per-class 20 --eval-per-class 5 --epochs 1 --output-dir outputs/smoke
python run_experiments.py --methods rule tfidf_nb tfidf_lr tfidf_svm --output-dir outputs/classical
```

预训练对照（首次运行会下载模型，需要额外磁盘、内存和时间；也可以传入完整本地模型目录）：

```bash
python -m pip install -r requirements-pretrained.txt
python run_experiments.py --methods encoder_frozen encoder_finetune --device mps --batch-size 16 --output-dir outputs/encoder
# CUDA 主机可用 --device cuda；CPU 可用 --device cpu
python run_experiments.py --methods encoder_frozen encoder_finetune --encoder /absolute/path/to/multilingual-model --batch-size 16 --output-dir outputs/local-encoder
```

使用默认同一数据、seed 时，预训练结果的数据指纹应与基础实验一致。需汇总到同一张表，可一次显式列出全部七种方法；默认仅运行前五种。缺依赖、无权重或设备不支持会标记失败并返回非零退出码，其他方法仍继续，不会伪造分数。

数据量对比示例（保留完整 Dev/Test，重复 seed=42/43/44 可估计波动）：

```bash
python run_experiments.py --train-per-class 100 --seed 42 --output-dir outputs/n100-s42
python run_experiments.py --train-per-class 400 --seed 42 --output-dir outputs/n400-s42
```

## 输出与结论

- `outputs/summary.md`：本次运行各方法的测试指标和耗时。
- `outputs/results.json`：完整指标、逐类表现、混淆矩阵、训练日志及最佳 epoch。
- `outputs/*_predictions.json`：样本 ID、原文、真实标签、预测标签与数据指纹，可逐条复核。
- `outputs/manifest.json`：命令参数、环境版本、各类样本量、去重数量与各集合样本 ID。
- `--save-models`：额外保存传统模型 `.joblib` 或神经模型 `.pt`，默认不保存大权重。神经模型文件包含标签、字符表、构造参数；Encoder 额外保存 tokenizer/config，复原时仍需指定原始预训练模型。

同一输出目录的报告会被覆盖；当前汇总只列当前运行的方法，旧的逐条预测文件不会自动删除。请为不同配置指定不同输出目录。

实际测得的基础实验结果见 [实验报告](EXPERIMENT_REPORT.md) 和 [自动汇总](outputs/summary.md)。预训练模型完整实验需要权重与计算资源，不能将随机模型的流程检查当成预训练效果。

本地语种数据属于学习样本；高分只能说明此固定测试集上的表现，不代表新闻分类或真实多语种业务能力。文字系统规则容易识别阿拉伯语、泰语等，西/葡等共享拉丁字母的语种更能体现模型的区别。比较时应同时看 Macro-F1 和逐类混淆，而非只看整体 Accuracy。
