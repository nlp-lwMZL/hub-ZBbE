# src_llm_work —— 人民日报 NER 的 LLM 方案（作业）

参考 `../src_llm/`（cluener 版）重写，数据换成 `data/peoples_daily`，产出放 `outputs_llm_work/`。
和 `src_work/`（BERT 序列标注）、`src_llm/`（cluener）、`outputs_work/` 完全隔离。

## 与 src_llm/ 的差异

| | `src_llm/`（cluener） | `src_llm_work/`（本目录） |
|---|---|---|
| 数据 | span 格式 `{类型:{实体:[[起,止]]}}` | CoNLL，`tokens` + BIO `ner_tags` |
| 实体类型 | 10 类小写（address/book/company/…） | **3 类大写**（PER / ORG / LOC） |
| 文本字段 | `record["text"]` | `"".join(record["tokens"])` |
| 目标 JSON 长度 | 20~150 token | 6~167 token（P50=18） |
| 空实体样本 | 少 | **35%**（必须学会输出 `{"entities": []}`） |
| max_length | 256 | **384**（P99=321，256 会切掉目标 JSON 的尾巴） |
| helper 组织 | 三个脚本各抄一份 | 抽成 `common.py`，改一处就够 |
| API 后端 | Qwen（dashscope） | **DeepSeek**（OpenAI 兼容） |
| 评估指标 | (surface,type,start,end) 集合 | **（surface,type）多重集**，另存严格口径 |
| 评估规模 | 抽 100 条 | 默认整个 split，和 BERT 同批样本 |

## 文件

```
src_llm_work/
├── common.py                  # 数据转换 / prompt / 解析 / 指标 / .env 读取
├── train_sft_work.py          # LoRA 指令微调（10000 条 × 2 轮）
├── evaluate_sft_work.py       # 加载 adapter，在整个 split 上算 entity-level F1
├── llm_ner_work.py            # DeepSeek zero / one / few-shot
└── compare_results_work.py    # BERT+Linear / BERT+CRF / SFT 三方法对比（出表+图）
```

## 三个必须改、不改就出错的地方

**1. `max_length` 256 → 384，并且超长样本要丢弃而不是截断**

实测「system prompt + 原文 + 目标 JSON」的总长：P50=143、P95=225、P99=321、max=356。

参考实现的 `(prompt_ids + response_ids)[:max_length]` 会**切掉目标 JSON 的尾巴** ——
等于拿残缺 JSON 当标签去训练模型。这里改成：超过 `max_length` 的样本直接丢弃
（10000 条里丢了 23 条，0.23%），并且把丢弃条数打出来。少 0.2% 的数据，
远比往训练集里塞坏标签划算。

**2. 指标换成多重集匹配**

参考实现用 `(surface, type, start, end)` 四元组做集合交集，预测实体的位置靠
`text.find(surface)` 猜。问题在于：同一句里同一个实体出现两次时，两次 `find`
都返回第一次的位置，`set` 去重后只剩一个 span，而 gold 有两条 —— TP 白丢一条。

**实测本数据集 val 集有 7.57% 的实体属于这种句内重复**（208/2318 句受影响），
也就是说即使模型全答对，这套口径下的召回上限也只有约 92.4%。

所以主指标改成：

```
TP = Σ_{(surface,type)} min(pred 里的个数, gold 里的个数)
```

同时保留一份参考实现口径的「严格位置匹配」数字（存在 `metrics_strict` 里），
方便和 cluener 的 0.6801 对照着看。

**3. `MODEL_PATH` 用 `ROOT.parent` 不是 `ROOT.parent.parent`**

`ROOT` 已经是 `序列标注项目/`，再 `parent.parent` 就跑到 `D:\` 去了。
`src_llm/train_sft.py` 和 `src/train.py` 里都有这个历史遗留 bug，
本目录已修正为 `ROOT.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"`。

## 显存：6 GB 卡上必须开梯度检查点

这台机器是 GTX 1660 Ti 6 GB。Qwen2-0.5B fp32 + 长度 384 + batch 2，
**不开梯度检查点时峰值显存 5.8 GB，会顶到物理上限**。表现很有欺骗性：

- 日志里**一条 OOM 都没有**（`expandable_segments` 让分配器「成功」了）
- `nvidia-smi` 显示 5844 / 6144 MiB
- 训练到第 710 步后**卡死**，60 秒零进展、GPU 100%

这是 WDDM 在往系统内存静默倒页。开梯度检查点后峰值降到 **3.78 GB**，
速度反而正常（0.48 s/step）：

```python
model.gradient_checkpointing_enable()
model.enable_input_require_grads()   # PEFT 冻结了 embedding，得手动打开输入梯度
model.config.use_cache = False       # 和检查点不兼容
```

教训：看显存要看 `nvidia-smi` 的 reserved，不能只看 `torch.cuda.max_memory_allocated()`
（后者报的是 4.36 GB，看起来还有余量）。

## 用法

```bash
pip install openai                      # DeepSeek 走 OpenAI 兼容协议

# ── SFT ──────────────────────────────────────────────────────────────────
python src_llm_work/train_sft_work.py                     # 10000 条 × 2 轮，约 73 min
python src_llm_work/train_sft_work.py --num_train -1      # 全量 20864 条（约 2.5 h）
python src_llm_work/evaluate_sft_work.py                  # 验证集全量
python src_llm_work/evaluate_sft_work.py --split test     # 测试集

# ── DeepSeek zero / one / few-shot ───────────────────────────────────────
python src_llm_work/llm_ner_work.py --n_shot 0,1,3 --n_samples 500
python src_llm_work/llm_ner_work.py --n_shot 1 --demo     # 5 条，打印 prompt 和输出

# ── 三方法对比 ────────────────────────────────────────────────────────────
python src_llm_work/compare_results_work.py --split validation
python src_llm_work/compare_results_work.py --bert_tag full   # 读放开截断的那份 BERT 结果
```

`compare_results_work.py` 除了出表出图，还会自己算两件容易骗人的事：

- **BERT 的截断口径**：训练时 `max_length=128` 只能读前 126 个字，超长句整句截掉，
  那些实体从 BERT 的分母里消失，而 SFT 看全文。脚本会自动读评估 JSON 里的
  `max_length` 判断有没有这个问题：没放开就多写一节「口径校正」，
  放开了就只写一节「截断长度的影响」（128 vs 512 的 ΔF1）。
  注意 `bert-base-chinese` 的位置编码上限是 512，放开也只能到 512，
  验证集最长句 568 字，所以永远剩 17 个实体看不到。
- **DeepSeek 的配对 bootstrap**：500 条子集上同一批句子配对重采样 2000 次，
  算 ΔF1 的置信区间。结果 1-shot 是**显著更差**（98.9% 的重采样为负）。
  这一节单独成节，不混进 2318 条的主表。

DeepSeek 的 key 放在项目根目录 `.env`：

```
DEEPSEEK_API_KEY=sk-xxx
```

`common.load_env_file()` 的查找顺序是：**已存在的环境变量 → 项目根 `.env` →
`src_llm_work/.env`**。用 `.env` 的好处是不用重启任何进程（`setx` 设的永久变量
只对之后新启动的进程生效）。

## 关于 few-shot 的示例抽取

`pick_examples()` 只从 `train.json` 抽（脚本里 assert 了和评测集零重叠），
并且**优先挑同时含 PER/ORG/LOC 的句子**：

- 只给一个例子时，如果那个例子只有人名，等于另外两类没教。
- 实测抽到的是「美国新墨西哥大学的古人类学家布鲁斯·哈克尔评论说……」，
  一句话里三类实体齐全。

`--seed`（抽评测样本）和 `--example_seed`（抽示例）故意分开，
改示例不会挪动评测样本的抽样序列。
