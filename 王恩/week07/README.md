# 第七周作业 · 中文事件抽取对比

每条文本抽取事件类型、触发词和角色论元。沿用 [week07 课件](../../courseware/week07/code/README.md) 的任务结构与 [评测脚本](../../courseware/week07/code/common/evaluate_events.py)，将金融公告与通用事件放到同一套实验流程中。

| 数据集 | 领域 | 标注类型数 | 数据入口 |
| --- | --- | ---: | --- |
| DuEE-Fin | 金融公告 | 13 | 课件的 `data/processed` |
| DuEE 1.0 | 通用事件 | 65 | [百度发布的 ZIP](https://dataset-bj.cdn.bcebos.com/qianyan/DuEE_1_0.zip)，首次运行自动缓存至 `data/` |

两个数据集均使用原划分中的 train 和 dev。DuEE 1.0 只保留单事件行，与课件任务同口径。抽样前，按 Unicode NFKC、大小写和空白归一化剔除与完整 train 重复的 dev 文本。原文和标注偏移保持原样。默认种子为 42，每个数据集抽取 2,400 条训练和 40 条评测样本。词典、CRF 和 few-shot 示例均使用抽中的训练集。数据及包含原文的推理输出留在本地，DuEE 1.0 的使用范围见 ZIP 内的 `License.pdf`。

## 对比方法

| 方法 | 实现 |
| --- | --- |
| `lexicon` | 用训练标注建立触发词和论元词典。优先匹配最长触发词，从对应类型的词典查找论元 |
| `crf` | 字符 TF-IDF + LinearSVC 判别事件类型。字符级 CRF 识别触发词，再由该类型的 CRF 标注论元 |
| `--api … 0` | 向 OpenAI 兼容接口发送完整类型和角色表及待抽取原文，做 zero-shot |
| `--api … 1` | 加入一条训练示例，用字符二元组 Jaccard 相似度检索，做 one-shot |

同一数据集的模型都评测同一批样本。触发词和论元均按同类型事件中的文本精确匹配计算 micro F1。完整事件要求类型、触发词和整组论元一致。预测的字符位置用于复核，重复片段按首次出现对齐。CRF 的单层 BIO 对重叠论元优先保留较长的区间。API 的无效回复留在分母内，未知类型、未知角色和不在原文中的片段保留为预测，按课件口径计分并单独计数。

## 运行

在本目录使用 Python 3.11+。以下命令用 uv 临时安装依赖。

```bash
uv run --with-requirements requirements.txt python run_experiments.py
uv run --with-requirements requirements.txt python run_experiments.py --eval-n 0 --output-dir outputs/full-dev
```

`--train-n 0` 使用全量训练集，`--eval-n 0` 评测全部去重后的 dev 样本。可用 `--datasets duee-fin` 或 `--datasets duee` 选择单个数据集。已下载官方数据包时可用 `--duee-archive /path/to/DuEE_1_0.zip` 指定文件。

### 本地 API 对照

Apple Silicon 可用 [MLX LM](https://github.com/ml-explore/mlx-lm) 启动两台本机服务。以下服务命令分别在两个终端执行。先按指定修订下载权重，再以本地权重目录启动，以固定实验模型。

```bash
# Qwen2.5-0.5B-Instruct，下载原模型后以 MLX LM 量化为 4-bit
uv run --with huggingface-hub hf download Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 --local-dir data/qwen05-original
uv run --with mlx-lm==0.32.2 mlx_lm.convert \
  --hf-path data/qwen05-original --mlx-path data/qwen05-4bit -q
uv run --with mlx-lm==0.32.2 mlx_lm.server \
  --model data/qwen05-4bit --host 127.0.0.1 --port 18082
```

```bash
# Qwen2.5-1.5B-Instruct，下载社区发布的 4-bit 权重
uv run --with huggingface-hub hf download mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --revision 8b403126fc14f14cfc99bb4cfa72ecbc129ea677 --local-dir data/qwen15-4bit
uv run --with mlx-lm==0.32.2 mlx_lm.server \
  --model data/qwen15-4bit --host 127.0.0.1 --port 18083
```

```bash
uv run --with-requirements requirements.txt python run_experiments.py \
  --api qwen05-zero http://127.0.0.1:18082/v1 default_model - 0 \
  --api qwen05-one  http://127.0.0.1:18082/v1 default_model - 1 \
  --api qwen15-zero http://127.0.0.1:18083/v1 default_model - 0 \
  --api qwen15-one  http://127.0.0.1:18083/v1 default_model - 1 \
  --output-dir outputs/compare
```

若要复现 BF16 对照，另用 `uv run --with mlx-lm==0.32.2 mlx_lm.server --model data/qwen05-original --host 127.0.0.1 --port 18081` 启动原权重服务，并追加 `--api qwen05-bf16-zero http://127.0.0.1:18081/v1 default_model - 0`。

各 API 通过相同的 Chat Completions 协议访问。参数依次为实验名、基础 URL、服务端模型 ID、密钥环境变量名和示例数。连字符 `-` 用于无需鉴权的本地服务。换成云端 OpenAI 兼容 API 时，可以在同一条命令追加配置。例如设置好 `DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL` 后使用 `--api deepseek https://api.deepseek.com "$DEEPSEEK_MODEL" DEEPSEEK_API_KEY 0`。接口须接受 `messages`、`temperature` 和 `max_tokens`。只跑 API 时加 `--api-only`，可直接使用 Python 标准库。该入口会向所选服务发送原文，one-shot 请求还会带一条训练示例。

## 结果文件

- `summary.md` 汇总四项 F1、单条平均/P95 延迟和 API 的 JSON 结构合法率。后者只校验 JSON 和字段结构，类型、角色及片段的正确性由 F1 衡量。
- `results.json` 包含样本数与指纹、P/R/F1、训练或初始化耗时、无效输出和请求错误统计，以及服务返回的 token 用量。
- `<dataset>/<method>.jsonl` 记录样本 ID、金标、预测、偏移、API 原始回复和 few-shot 示例 ID。API 部分请求失败时仍以全部样本计算指标，全部请求失败时显示 `failed` 且无分数。存在请求失败时进程返回非零。

实测结果和错误分析见 [实验报告](EXPERIMENT_REPORT.md)。
