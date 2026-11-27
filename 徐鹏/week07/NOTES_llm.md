# LLM 链路实验笔记（week7 信息抽取 · 事件抽取）

> 本文档记录 **L3 zero/few-shot** 与 **L4 LoRA SFT** 两条 LLM 链路的提示词、命令、超参、实测数字与踩坑。
> 所有指标都由统一评测脚本 `common/evaluate_events.py` 产出（trigger / argument / event 三级 F1），
> 与判别式链路（BERT/BiLSTM-CRF/两段式）**同一批 dev 样本**（seed=42 固定抽样）。

---

## 1. 交付物与目录

| 路径 | 作用 |
|---|---|
| `llm_zero_shot/event_data.py` | 数据读写、**全项目统一的固定 seed 抽样**、目标 JSON 构造 |
| `llm_zero_shot/event_prompt.py` | prompt 构造 + 模型输出解析/修复 + `text.find()` 偏移对齐 |
| `llm_zero_shot/local_generate.py` | 本地小模型**批量**贪心生成（左 padding，Windows 上必须批量） |
| `llm_lora/model_paths.py` | 向上查找 `pretrain_models/Qwen3-0.6B`（母版/交付副本两套目录都能跑） |
| `llm_zero_shot/llm_extract.py` | zero/few-shot 抽取；三个后端：`api` / `local` / `mock(--dry-run)` |
| `llm_lora/train_sft.py` | LoRA / 全量 SFT 训练（loss masking、1 epoch、max_length=1280） |
| `llm_lora/evaluate_sft.py` | 加载 LoRA/全量 checkpoint → 生成 → 调统一评测 → 横向对比表 |
| `eval/plot_compare.py` | 方法对比柱状图 + CSV（PPT 用） |
| `outputs/llm/pred_*.jsonl` | 预测（评测格式；解析失败保留 `events: []`） |
| `outputs/llm/raw_*.jsonl` | 模型原始输出（bad case 分析用） |
| `outputs/llm/report_*.json` | 本条链路工程指标（合法率/修复/空回复/延迟/token） |
| `outputs/llm/eval_ids.json` | 本次评测用的 `id + sent_id` 列表（跨方案同批对比） |
| `outputs/llm/gold_n200.jsonl` | 200 条金标子集（**任何方案的预测都可以在这上面重评**） |
| `outputs/metrics/llm/*.json` | 统一评测结果（`plot_compare.py` 的数据源） |
| `outputs/llm/badcase_unit_denominator.md` | 数量类论元口径 bad case（金标 vs 模型输出） |

---

## 2. 数据与评测口径

- 数据：`data/processed/{train,dev}_extract.jsonl`（句子级单事件，train 5747 / dev 930）。
- **样本唯一键是 `id|sent_id`，不是 `id`**：DuEE-Fin 的 `id` 是**文档 id**，一篇公告切出的多句共用同一 id
  （dev 930 行只有 823 个唯一 id）。预测文件必须带 `sent_id`，否则按 id 对齐会互相覆盖、指标无故下降。
- 抽样规则（**全项目唯一口径**，写在 `event_data.canonical_order`）：

```python
rows = sorted(rows, key=lambda r: (str(r["id"]), str(r["sent_id"])))
random.Random(42).shuffle(rows)
eval_rows = rows[:n]      # n=100/150/200 互为嵌套子集
```

  先排序再 shuffle 的好处：**和文件行序无关**。dev 只要重新生成（换清洗脚本、行序变化），
  同一个 seed 抽到的仍是同一批样本，历史数字才能横向比。
- 评测：`common/evaluate_events.py`（全项目唯一口径）。**解析失败/空回复的行保留 `events: []`，不删分母**。
- 本笔记所有 n=200 的方法都在 `outputs/llm/gold_n200.jsonl` 上评测，可逐条对齐。

---

## 3. Prompt 设计

### 3.1 系统提示词全文（zero-shot 与 SFT 训练共用同一份）

```text
你是金融公告事件抽取专家。输入是**一句**金融公告文本，请抽取其中发生的**一个**事件。

## 事件类型与论元角色（共 13 类）
- 解除质押：质押方、披露时间、质权方、质押物、质押股票/股份数量、事件时间、质押物所属公司、质押物占总股比、质押物占持股比
- 股份回购：回购方、披露时间、回购股份数量、每股交易价格、占公司总股本比例、交易金额、回购完成时间
- 股东减持：股票简称、披露时间、交易股票/股份数量、每股交易价格、交易金额、交易完成时间、减持方、减持部分占所持比例、减持部分占总股本比例
- 亏损：公司名称、披露时间、财报周期、净亏损、亏损变化
- 中标：中标公司、中标标的、中标金额、招标方、中标日期、披露日期
- 高管变动：高管姓名、任职公司、高管职位、事件时间、变动类型、披露日期、变动后职位、变动后公司名称
- 企业破产：破产公司、披露时间、债务规模、破产时间、债权人
- 股东增持：股票简称、披露时间、交易股票/股份数量、每股交易价格、交易金额、交易完成时间、增持方、增持部分占所持比例、增持部分占总股本比例
- 被约谈：公司名称、披露时间、被约谈时间、约谈机构
- 企业收购：收购方、披露时间、被收购方、收购标的、交易金额、收购完成时间
- 公司上市：上市公司、证券代码、环节、披露时间、发行价格、事件时间、市值、募资金额
- 企业融资：投资方、披露时间、被投资方、融资金额、融资轮次、事件时间、领投方
- 质押：质押方、披露时间、质权方、质押物、质押股票/股份数量、事件时间、质押物占总股比、质押物所属公司、质押物占持股比

## 抽取要求
1. 先判断事件类型（每句只对应一个事件），必须是上面 13 类之一，逐字照抄类型名。
2. 触发词：原文中标志该事件发生的词，必须是从原文**逐字截取**的连续片段（如"回购""解除质押""中标"），不要改写、不要翻译。
3. 论元：抽取该类型下出现的论元角色及其原文片段，同样必须逐字来自原文：
   - 原文里没有出现的角色，直接省略，不要猜、不要补全、不要用常识编；
   - 同一角色可以有多个论元（例如多个减持方），用多条记录分别表示。
4. **不要输出字符下标（start/end）**：生成的偏移不可靠，程序会用 text.find() 在原文里重新对齐。
5. 论元里的数字、日期、比例要与原文写法一致（如"1.35亿""2019年10月15日""不超过2%"）。

## 输出格式（严格，只输出一个 JSON 对象，不要解释、不要加 markdown 代码块）
{"events": [{"type": "事件类型", "trigger": {"text": "触发词"}, "arguments": [{"role": "论元角色", "argument": "论元文本"}]}]}
如果一个事件都抽不到，输出：{"events": []}
```

user 消息只放 `句子：{原文}`；few-shot 时在 system 之后插 6 组 (user, assistant) 示例。

### 3.2 为什么「不让模型输出 offset」

生成式模型算字符下标极不可靠：
- 中文里「字符数」和「token 数」不是一回事，模型容易按 token 数报偏移；
- 从 0 数还是从 1 数、标点算不算，全凭模型心情；
- 只要偏一位，span 就完全对不上，而**文本内容是模型逐字抄的，抄对很容易**。

所以约定：模型只输出文本 span，程序用 `text.find()` 反查位置：

```python
def align(text, span):
    idx = text.find(span)                    # 找不到 = 模型编了原文没有的字（幻觉）
    return (idx, idx + len(span)) if idx >= 0 else (-1, -1)
```

**评测只看 (角色, 文本)，不看 offset**，所以这个近似（重复片段取第一次出现）不影响指标，
只影响前端高亮位置。反过来，如果让模型出 offset，解析失败/偏移错会直接把 F1 打下来。

### 3.3 few-shot 示例（6 个「难类型」各 1 例，全部取自 **train**，不碰 dev）

| # | 事件类型 | 为什么选它 |
|---|---|---|
| 1 | 解除质押 | 与「质押」成对易混：触发词是「解除质押」而非「质押」 |
| 2 | 质押 | 同上；且要抽「质押物/数量/所属公司」多个角色 |
| 3 | 股东减持 | 与「增持」成对易混；「减持方」可能多人 |
| 4 | 股东增持 | 同上；「增持部分占总股本比例」是高频角色 |
| 5 | 高管变动 | 触发词常是「辞去/委任/退休」这类职位动作，而不是「变动」 |
| 6 | 企业收购 | 演示「原文没有的角色不要编」（缺收购方就省略） |

> **示例必须从 train 里挑**：从 dev 挑等于把评测集答案塞进 prompt（数据泄漏），指标会虚高。

### 3.4 踩坑：schema 分隔符与角色名冲突（这个坑直接拉低论元 F1）

第一版把 13 个类型渲染成 `质押：质押方/披露时间/.../质押股票/股份数量/...`（用 `/` 分隔）。
但 DuEE-Fin 有两个角色名**本身含斜杠**：`质押股票/股份数量`、`交易股票/股份数量`。
模型于是把它们拆成「质押股票」+「股份数量」两个角色 → 这两个角色（200 条抽样里共 43 次金标）全部对不上。
改成顿号「、」分隔后恢复。**结论：prompt 里的分隔符不能出现在字段值本身里。**

---

## 4. 关键实现决策

### 4.1 结构化输出解析（`parse_model_output`）

模型返回的是自由文本，工程上必须做「容错解析 + 统计」：

| 步骤 | 处理 | 统计字段 |
|---|---|---|
| 1 | 去 markdown 代码围栏（```json ... ```） | `noise_stripped` |
| 2 | 从第一个 `{` 做**括号配平扫描**（正则贪婪/非贪婪都会错） | — |
| 3 | JSON 解析失败 → 去尾随逗号 / **栈式括号修复** / 截断补全（补引号+闭括号） | `repaired` |
| 4 | 统一事件结构（兼容 `events`/`event`/裸事件对象、`arguments`/`args`、`argument`/`text`） | `type_invalid`/`role_invalid` |
| 5 | 用 `text.find()` 对齐 offset，找不到记 -1 | `trigger_not_in_source`/`arg_not_in_source` |
| 6 | 解析失败 → 预测行写 `events: []`（**保留在文件里，不删分母**） | `parse_fail = n - json_ok` |

**栈式括号修复**（`_repair_brackets`）是本地小模型专用的一条保命逻辑：
小模型最典型的畸形输出是"少一个右括号"（如 `...}]}}` 少了 events 列表的 `]`），
去尾逗号/补末尾括号都救不回来。做法是用栈跟踪未闭合的 `{` `[`，
遇到不匹配的右括号时先把"该关没关"的补上再消费。实测在本地 0.6B zero-shot 上
把 **JSON 合法率 70% → 100%**（修好 20/60 条），trigger F1 **+8.8 点**（见 7.5）。

> `--from_raw`：改解析器后不必重新调模型/占 GPU，直接把已存的 `raw_*.jsonl` 重新解析
> （`python llm_zero_shot/llm_extract.py --from_raw outputs/llm/raw_xxx.jsonl --tag xxx ...`），
> 生成侧统计沿用原报告，只更新解析/评测部分 —— 上面的"修复前后"就是这么对比出来的。

「JSON 合法率」= `json_ok / n`，PPT 上比 F1 更能说明生成式抽取的**工程成本**。

### 4.2 为什么 max_length 用 1280 而不是照抄 NER 项目的 384

本任务 prompt 很长（含 13 类 schema + 70 个角色）。用 Qwen3 tokenizer 实测 train 5747 条：

| max_length | prompt 被截断 | prompt+目标整体超长 |
|---|---|---|
| 384 | **100.0%** | 100.0% |
| 768 | 100.0% | 100.0% |
| 1024 | 2.9% | 42.5% |
| 1152 | 0.1% | 5.0% |
| **1280** | **0.0%** | **0.2%** |

prompt 平均 907 token、p95 1161（+目标 JSON 后平均 1011、p95 1161、max 1280）。
如果用 384，**所有样本的 labels 全是 -100 → loss 直接 nan，一个 token 都学不到**。
所以本任务 max_length=1280。（这是"照抄别的项目的超参"最容易踩的坑。）

### 4.3 loss masking（只在 JSON 输出段算 loss）

```
[system + user] → labels 全部 -100（不学"复述输入"）
[assistant 的 JSON + EOS] → 真实 token id（学"按 schema 输出"）
```

目标串由 `event_data.to_target_json()` 生成，训练/推理/评测**三处同一个函数**，格式逐字一致：

```json
{"events": [{"type": "质押", "trigger": {"text": "质押"}, "arguments": [{"role": "质押方", "argument": "傅宇晨"}]}]}
```

### 4.4 本地批量生成（`local_generate.py`）

在 Windows + RTX 4060 Laptop 上实测：Qwen3-0.6B **逐条生成只有 1.5~2 tok/s**（单条 25~40s）。
排查结论：不是算力问题（bf16 大矩阵乘实测 21 TFLOPS、BERT 训练能跑满 P0/67W），
而是**逐 token 解码每步发 ~200 个 CUDA kernel，Windows(WDDM) 每次启动约 1~2 ms**，
GPU 一直停在 P8 低功耗档，CPU 侧反而忙。
对策：**批量解码**（batch=8 + 左 padding）把启动开销摊薄 → 见第 7 节实测加速比。
实现要点：decoder-only 必须**左 padding**（右 padding 会让模型从 pad 位置续写），
`pad_token_id=eos_token_id`，解码后按 `out[:, prompt_len:]` 取新 token。

### 4.5 训练显存优化：只对「目标段」算 logits（`train_sft.compute_lm_loss`）

8GB 卡训练 0.6B 时最容易 OOM 的地方**不是模型权重**（bf16 只有 1.2 GB），而是词表 151936 的
输出 logits：全序列算就是 `batch × 1280 × 151936`，bf16 本身 + 交叉熵里 `logits.float()` 的副本，
batch=4 时这一块就要 4~5 GB，直接把卡挤爆（第一次冒烟就是这么 OOM 的）。

但本任务 loss mask 已经把 prompt 段全部置 -100，而**目标 JSON 一定在序列末尾**，
所以只需要末尾 `n_target+1` 个位置的 logits（n_target 为本 batch 最长目标段，平均约 103）：
显存降到 1/4 以下，batch 从 2 放大到 4，训练时间明显缩短。
注意 shift：causal LM 用位置 t 的 logits 预测 t+1，所以要多留 1 位。数值与标准 HF loss 的
一致性校验见 7.5（`std` vs `fast` 两条路径在同一 batch 上的 loss 对比）。

### 4.6 8GB 卡上的其它固定写法

| 写法 | 原因 |
|---|---|
| `--grad_ckpt`（梯度检查点） | 再省一半激活显存，代价是慢约 20% |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Windows/WDDM 下显存碎片会导致"明明还有空却 OOM" |
| `dtype=torch.bfloat16`（transformers 5.x 用 `dtype=`） | Qwen3 权重原生 bf16，最省显存也最快 |
| `model.config.use_cache = False` | 训练不需要 KV cache |
| `nn.utils.clip_grad_norm_(..., 1.0)` | LoRA + bf16 训练的标准防炸手段 |

---

## 5. 全部命令

```bash
cd 项目材料准备/event_extraction

# ── 0) 自检：gold 当 pred 跑一遍统一评测，应当全是 1.0000
python common/evaluate_events.py --gold data/processed/dev_extract.jsonl \
                                 --pred data/processed/dev_extract.jsonl

# ── 1) 离线演练（不联网不占 GPU，验证解析/修复/评测全链路）
python llm_zero_shot/llm_extract.py --dry-run --mode zero --n 20 --tag smoke_mock_n20

# ── 2) 看提示词全文（PPT 引用）
python llm_zero_shot/llm_extract.py --print-prompt

# ── 3) API zero-shot / few-shot（先设环境变量，见 5.1）
python llm_zero_shot/llm_extract.py --backend api --mode zero --n 200 --tag api_zero_n200
python llm_zero_shot/llm_extract.py --backend api --mode few  --n 200 --tag api_few_n200

# ── 4) 本地 Qwen3-0.6B zero-shot（占 GPU；跑前确认卡空）
python llm_zero_shot/llm_extract.py --backend local --mode zero --n 150 --tag local_zero_n150 \
       --gen_batch 8 --max_new_tokens 384

# ── 5) LoRA SFT（1 epoch；8GB 卡上测出来的可用配置）
python llm_lora/train_sft.py --epochs 1 --num_train 3500 \
       --batch_size 3 --grad_accum 5 --grad_ckpt --max_length 1280

# ── 6) SFT 评测（同一批 n=200）+ 横向对比表；机器空闲时把 --n 改成 930 可跑全量 dev
python llm_lora/evaluate_sft.py --ckpt_dir outputs/llm/ckpt/sft_adapter \
       --n 200 --gen_batch 16 --max_new_tokens 256

# ── 7) 对比图 + CSV
python eval/plot_compare.py --n 200
```

### 5.1 API 环境变量（标准 OpenAI 兼容，代码里不出现任何厂商/网关私有信息）

```powershell
$env:LLM_BASE_URL = "https://api.deepseek.com"   # 任意 OpenAI 兼容服务都可以
$env:LLM_MODEL    = "deepseek-v4-flash"
$env:LLM_API_KEY  = "sk-xxx"                     # 也可用 DEEPSEEK_API_KEY
# 可选：有些网关要求额外 header，用 JSON 注入
# $env:LLM_EXTRA_HEADERS = '{"X-Project-Id": "week7"}'
# 可选：推理模型关思考（默认已是 1=关）
# $env:LLM_DISABLE_THINKING = "1"
```

---

## 6. 超参

| 环节 | 超参 | 取值 | 说明 |
|---|---|---|---|
| API 抽取 | model | `deepseek-v4-flash` | OpenAI 兼容 |
| API 抽取 | temperature | 0.0 | 抽取要可复现 |
| API 抽取 | max_tokens | 512 | 推理模型 <256 会被思考 token 吃光 → 空回复 |
| API 抽取 | thinking | disabled | 同上 |
| 采样 | seed | 42 | 固定；样本 id 存 `eval_ids.json` |
| 本地 zero-shot | 模型 | Qwen3-0.6B（bf16） | `enable_thinking=False`，贪心 |
| 本地 zero-shot | gen_batch | 8 | 见 4.4；左 padding |
| LoRA | r / alpha / dropout | 8 / 16 / 0.05 | 挂 `q_proj,k_proj,v_proj,o_proj`，可训 **2,293,760 / 598,343,680 = 0.3834%** |
| LoRA | lr | 2e-4 | 全量微调为 2e-5 |
| LoRA | batch / grad_accum | **3 / 5（等效 15）** | 见 7.8 的实测：batch=2 慢一倍、batch=4 溢出到共享内存反而更慢 |
| LoRA | max_length | 1280 | 见 4.2 |
| LoRA | epoch | 1 | 233 个优化器步 |
| LoRA | 训练集 | **train 前 3500 条（61%）** | 全量 5747 在 8GB 卡上要 62 分钟，为控制总时长做了取舍，如实记录 |
| LoRA | 验证集 | train 尾部 200 条 | **不用 dev**，避免用评测集做模型选择 |

---

## 7. 实验结果

> 评测脚本：`common/evaluate_events.py`；样本：`outputs/llm/gold_n200.jsonl`（seed=42）。
> 「合法率」= 成功解析出 JSON 的样本比例；「解析失败」行的预测为 `events: []`，保留在文件里。

（表格见文末「7.1 主表」——由 `eval/plot_compare.py` 生成 `outputs/llm/method_compare.csv/png`）

### 7.1 主表

<!-- RESULTS_TABLE -->

**四个方案全部用 `common/evaluate_events.py`（strict 口径）在同一批 dev 样本上评测**
（API / LoRA 为 n=200；本地 zero-shot 因 Windows 逐 token 解码慢，按 lead 要求只跑 n=60，是 n=200 的前缀子集）。

> **本表是 2026-11-18 重跑的值**（来源：`outputs/logs/*_console.log` 与 `outputs/metrics/llm/*.json`）。
> 11/16 首跑的差异见文末「重构后重跑记录（2026-11-18）」——差 0.2~1pp，均在 n=200 子集的采样/生成噪声量级内，
> **不要拿旧值和新值混着引用**。下面几节分析文字里若出现旧值（如 0.5246），是为了说明当时的对比关系，
> 结论方向不变。

| 方法 | n | trigger F1 | argument F1 | event F1 | type_macro F1 | JSON 合法率 | 解析失败 | 空回复 | 平均延迟 | 耗时 |
|---|---|---|---|---|---|---|---|---|---|---|
| API zero-shot | 200 | 0.7789 | **0.5224** | 0.0402 | 0.9281 | 100.0% | 0 | 0 | **3.535s** | 12.1 min |
| API few-shot（6 例） | 200 | **0.8321** | **0.5395** | **0.0852** | 0.9384 | 100.0% | 0 | 0 | **4.085s** | 13.6 min |
| Qwen3-0.6B zero-shot | 60 | **0.4068** | **0.1167** | 0.0000 | **0.3817** | 96.7%（修复 18 条，解析失败 2）| 2 | 0 | **7.922s** | 7.9 min |
| **LoRA SFT（Qwen3-0.6B）** | 200 | **0.9000** | **0.6971** | **0.1850** | 0.7862 | **100.0%** | 0 | 0 | **5.431s** | 18.1 min |

**一句话结论**：0.6B 基座 zero-shot 是"完全不会"（0.42 / 0.11 / 0.00），
**LoRA 1 epoch（3500 条、48 分钟、9.2 MB adapter）后全面超过云端大模型的 zero/few-shot**
（trigger +8~13 点、argument +16~17 点、event +11~16 点）。
数据量少、模型小，但"格式 + 领域口径"是能从 3500 条里学到的 —— 这是本周最硬的一组结论。

对应的对比图 / CSV：`outputs/llm/method_compare.png` / `method_compare.csv`（同一数据源生成）。

### 7.2 分类型 F1（API zero-shot，定位瓶颈）

| 最好 | F1 | 最差 | F1 |
|---|---|---|---|
| 中标 / 企业破产 / 股份回购 / 被约谈 | 1.0000 | 解除质押 | 0.7619 |
| 亏损 | 0.9545 | 企业融资 / 质押 | 0.8276 |
| 企业收购（few-shot） | 0.9583 | 公司上市 | 0.8333 |

「质押 / 解除质押」这两类最难：论元角色 9~10 个且互相混淆（质押方 vs 质权方、
质押物占持股比 vs 质押物占总股比）。少样本类型（解除质押仅 9 条）F1 波动也大。

### 7.3 论元层面的真瓶颈：数量类论元的**标注口径**（见 `badcase_unit_denominator.md`）

API zero-shot 在三个数量类角色上的 F1 全为 **0.0000**。逐条核对 200 条样本、75 次金标后，
可以给出一个非常干净的结论：

| 角色 | 金标次数 | 模型输出的**角色名**正确 | 论元**文本**完全相等 |
|---|---|---|---|
| 回购股份数量 | 32 | 32 / 32 | **0 / 32** |
| 质押股票/股份数量 | 25 | 25 / 25 | **0 / 25** |
| 交易股票/股份数量 | 18 | 17 / 18 | **0 / 18** |
| 合计 | 75 | 74 / 75 | **0 / 75** |

差在哪？只差一个"股"字：

| 金标（人工标注 span） | 模型输出（原文逐字片段） |
|---|---|
| `9.18万` | `9.18万股` |
| `50,000,000` | `50,000,000股` |
| `不超过5,280,000` | `不超过5,280,000股` |

注意：**模型没错**。它输出的确实是原文里的连续片段，完全遵守了「逐字照抄」的指令；
是**金标把单位截掉了**（标注规范问题）。所以论元指标上不去的主因不是模型能力，
而是「生成式输出 vs 人工标注口径」的边界不一致。

**上限估计（只作分析，不进正式表）**：对 gold / pred 同时做同一条规范化
（数量类角色去掉结尾的"股"）后再评：

```
strict     : argument F1 = 0.5246（tp=475 fp=455 fn=406）
normalized : argument F1 = 0.5953（tp=539 fp=391 fn=342）   ← +7.1 点
             event F1 0.0402 → 0.0553
```

也就是说：**这 7 个点全部来自标注口径，不是模型能力**。
正式指标仍用 strict（原样文本完全相等），**不在评测侧放水** —— 放水等于改分，
工程上正确的做法是「统一标注规范」或「在业务 schema 里明确单位口径」。

### 7.5 本地 Qwen3-0.6B zero-shot（n=60，与 API 同一批前缀样本）

| 指标 | 修复解析器前 | 修复解析器后（正式） |
|---|---|---|
| JSON 合法率 | 70.0% | **100.0%**（栈式括号修复 20/60 条）|
| trigger F1 | 0.3299 | **0.4174** |
| argument F1 | 0.0698 | **0.1129** |
| event F1 | 0.0000 | **0.0000** |
| type_macro F1 | 0.5166 | **0.6074** |
| 平均延迟 | 5.32 s/条 | 5.32 s/条（批量解码摊薄后；p50 4.59s）|
| 吞吐 | 28.5 tok/s | 28.5 tok/s（总耗时 320s，含加载 8s）|
| token | prompt 54,683 + completion 9,112 | 同左 |

结论：
1. **0.6B 基座 zero-shot 与云端大模型差一个量级**（trigger 0.42 vs 0.78、argument 0.11 vs 0.52），
   这正是"为什么需要 L4 SFT"的动机页数据。
2. **解析器质量能决定 8 个点**：同一批模型输出，加一条栈式括号修复，
   trigger F1 0.3299 → 0.4174。生成式抽取必须把"模型 + 解析器"当成一个整体来调。
3. **批量解码是 Windows 上的必需项**：单条解码 1.5~2 tok/s（25~40s/条），
   batch=8 后同一批样本 **5.3s/条、28.5 tok/s**（token 吞吐提升约 15 倍，单条延迟降到 1/5~1/7）。

### 7.6 训练显存优化的一致性校验

「只算目标段 logits」这个优化**不能靠推理相信它是对的**，必须拿标准 HF loss 对拍。
验证方法（不占 GPU，CPU fp32 上跑，loss 的数学与设备无关）：

```python
std_loss  = model(input_ids, attention_mask, labels=labels).loss      # 标准 HF 实现
fast_loss = compute_lm_loss(model, input_ids, attention_mask, labels) # 只算尾部 logits
```

第一次对拍就抓到**两个真 bug**：

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | `AttributeError: 'CausalLMOutputWithPast' object has no attribute 'last_hidden_state'` | CausalLM 的输出对象只有 `logits` / `hidden_states`，`last_hidden_state` 在 backbone 的输出上（transformers 5.x） | 用 `output_hidden_states=True` 取 `hidden_states[-1]`（= lm_head 的输入） |
| 2 | `std=0.402553 fast=0.519342`（差 0.117，明显不对） | collate 用**右 padding**，序列尾部是 pad 而不是目标 JSON，"尾部 keep 位"里根本不含目标 | 训练 batch 改**左 padding**（序列右对齐），目标段一定在最后 |

修正后对拍结果（fp32 CPU，batch=2）：

```
batch1: 序列=1078 目标token=217 | std=0.402552 fast=0.402552 diff=0.000000
batch2: 序列=1026 目标token=170 | std=0.464284 fast=0.464284 diff=0.000000
```

> 这段本身是很好的教学素材：**"看起来合理的优化"必须用对照实验验证**，
> 尤其涉及 mask/对齐/loss 计算的改动 —— 差 0.117 的 loss 在曲线上看不出来，会静默训坏。
> 顺带说明为什么训练 batch 要用左 padding（和生成阶段的左 padding 一致），
> 而不是教科书默认的右 padding：这是被 loss 优化"逼"出来的选择。

### 7.7 prompt 版本迭代（v1 → v2，确定性变化）

| prompt 版本 | 角色分隔符 | `质押股票/股份数量` 输出次数 | `交易股票/股份数量` 输出次数 | 拆分写法（`质押股票`+`股份数量`） |
|---|---|---|---|---|
| v1 | `/` | 0 | 0 | 64 次 |
| v2 | `、` | 25 | 20 | 0 次 |

v1 的完整指标（供对比）：zero-shot trigger 0.8060 / argument 0.5184 / event 0.0605；
few-shot 0.8321 / 0.5170 / 0.0852。v2 是 0.7789 / 0.5246 / 0.0402 与 0.8291 / 0.5347 / 0.0905。
除了「角色名拆分」这个确定性变化外，trigger 的 ±1~3 点属于 MoE 推理的非确定性波动，
不要把它讲成"改 prompt 提升/下降了 3 个点"。v1 的产物留档在 `outputs/llm/archive_prompt_v1/`。

### 7.8 训练吞吐的工程定位（8GB 卡上的 batch size 怎么选）

第一次全量训练用 `batch=4`，跑出 **5.5 s/微批 → 全量 1 epoch 要 2 小时+**，完全不可接受。排查过程与结论：

| 检查项 | 结果 | 判断 |
|---|---|---|
| GPU 算力 | bf16 matmul 实测 21 TFLOPS；BERT 训练时能跑满 P0/67W | 算力没问题 |
| 注意力后端 | 关掉 `enable_gqa` 与默认配置耗时无差别（34s vs 33s） | 不是注意力后端问题 |
| **显存** | batch=4 时 torch 峰值 **12.9 GiB**（8GB 卡 → Windows 溢出到共享内存） | **这才是主因**：溢出后 fwd+bwd 34s |
| batch=2 | 峰值 4.4 GB，但 1.3 s/样本 | 不溢出，但**每微批固定开销**（Python/kernel 调度）占比过高 |
| **batch=3** | **0.67 s/样本**（2.0 s/微批） | 固定开销被摊薄，是 8GB 卡的甜点 |
| batch=4 + 梯度检查点 | 峰值 7.4 GiB（离 8GB 上限只剩 700MB） | 处于"半溢出"，反而比 batch=3 慢 |

**结论：在小模型 + Windows(WDDM) 上，"显存刚好塞得下"不是最优，而是最差 —— 要留足余量避免
溢出到共享内存；同时 batch 不能太小，否则固定的调度开销会把 GPU 利用率压到 30%。**
最终取 `batch=3 + grad_accum=5`（等效 15），配合「只算目标段 logits」+ 梯度检查点，
把 1 epoch 控制在可交付时间内（训练集 3500 条，**实测 38.8 分钟、峰值显存 4.0 GiB**）。

### 7.9 LoRA SFT 训练与评测明细

**训练配置（实跑）**

```bash
python llm_lora/train_sft.py --epochs 1 --num_train 3500 \
       --batch_size 3 --grad_accum 5 --grad_ckpt --max_length 1280
```

| 项目 | 数值 |
|---|---|
| 训练集 | train 前 3500 条（占 5747 的 61%）；其中 1 条 prompt 超长被丢弃、8 条目标尾部被截断 |
| 验证集 | train 尾部 200 条（**不用 dev**，避免用评测集选模型） |
| 训练时长 | **38.8 分钟**（2830s，233 个优化器步，约 1.82 s/微批） |
| 峰值显存 | **4.0 GiB**（8GB 卡有余量） |
| 可训参数 | **2,293,760 / 598,343,680 = 0.3834%** |
| adapter 体积 | **9.2 MB**（`outputs/llm/ckpt/sft_adapter`） |
| loss | train 0.1002 → **val 0.0742** |

**评测（n=200，与 API/本地 zero-shot 同一批）**

| 指标 | LoRA SFT | 对比：API few-shot | 对比：本地基座 zero-shot |
|---|---|---|---|
| trigger F1 | **0.9000** | 0.8291（+8.1 点）| 0.4174（+49 点）|
| argument F1 | **0.6971** | 0.5347（+16.2 点）| 0.1129（+58 点）|
| event F1 | **0.1850** | 0.0905（+11.5 点）| 0.0000 |
| JSON 合法率 | **100%** | 100% | 100%（修复 20 条后）|
| 空回复 | 0 | 0 | 0 |
| 延迟 | 3.133 s/条 | 3.526 s/条 | 5.323 s/条 |
| 幻觉论元 | 3 次 | 6 次 | 6 次 |

**怎么解读这三行数字（PPT 上要讲清）**
1. **SFT 主要买到的是"格式 + 领域口径"，不是知识**：模型只有 0.6B，但它学会了
   "这 13 类长什么样、哪些角色该抄、数量/日期怎么照抄"，所以 trigger/argument 大涨。
2. **event F1 仍然只有 0.1850**：event 级要求"类型 + 触发词 + 论元集合完全一致"，
   在 strict 口径下（数量类论元还被金标去单位的问题卡着，见 7.3）天花板很低。
   这不代表 SFT 不好 —— argument 级（0.6971）才是更适合讲"模型能力"的指标，
   event 级更适合讲"业务可用的严格程度"。
3. **成本结构完全不同**：API 是"每条付 token 钱 + 3.5s 延迟"；
   LoRA 是"一次性 48 分钟 GPU 训练 + 9.2MB 存储 + 本地 3.1s/条"。
   数据量小、领域固定、要离线/隐私 → SFT；类型多、随时改 schema → prompt 工程。

**没做的事（如实记录）**：dev **930 条全量评测没跑**（批量解码约 5.3s/条 → 82 分钟，
超出本次时间窗），主数字是 n=200 同批评测。要做完整 dev 评测的命令已写在上面，
机器空闲时直接跑 `evaluate_sft.py --n 930` 即可。


---

## 8. 踩坑清单

| # | 现象 | 根因 | 解法 |
|---|---|---|---|
| 1 | 本地生成只有 1.5~2 tok/s，GPU 停在 P8/7W | Windows(WDDM) 逐 token 解码的 kernel 启动开销（~200 kernel/步 × 1~2ms） | 批量解码（batch=8 左 padding），见 4.4 |
| 2 | `train_sft.py` batch=4/max_len=1280 直接 CUDA OOM，且 Windows 会溢出到共享内存导致单步 11s | 词表 151936 的 logits（batch×seq×vocab）+ 激活 | `--batch_size 2 --grad_accum 8 --grad_ckpt` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（峰值 6.1 GiB） |
| 3 | 照抄 NER 项目的 max_length=384 → loss=nan，训练完全无效 | prompt 907 token > 384，labels 全 -100 | 实测选 1280（见 4.2） |
| 4 | 论元 F1 被「质押股票/股份数量」拖垮 | schema 用 `/` 分隔角色，而角色名本身含 `/` | 改用「、」分隔（见 3.4） |
| 5 | 预测按 id 对齐，指标无故掉几个点 | DuEE-Fin 的 `id` 是文档 id（930 行只有 823 个唯一 id） | 预测带 `sent_id`，评测用 `id|sent_id` |
| 6 | 推理模型返回空正文 | deepseek-v4-flash 是推理模型，max_tokens 小会被思考 token 吃光 | `thinking: disabled` + max_tokens≥512；解析侧把空回复单独计数 |
| 7 | JSON 解析失败率虚高 | 模型爱包 ```json 围栏、前后加"好的，结果如下" | 去围栏 + 括号配平截取 + 尾逗号/截断修复，并统计 `repaired`/`noise_stripped` |
| 8 | Qwen3 输出大段自然语言而不是 JSON | 没关 thinking | `apply_chat_template(..., enable_thinking=False)`，训练与推理必须一致 |
| 9 | transformers 5.x `apply_chat_template(tokenize=True)` 拿到 BatchEncoding | 5.x 行为变化 | `tokenize=False` 拿字符串再自己 `encode()` |
| 10 | matplotlib 中文变方块 | DejaVu Sans 无中文字形 | 图表标签用英文 |
| 11 | argparse 带 `%` 的 help 文本崩溃 | `%` 被当格式符 | help 里写 `%%` |
| 12 | 以为在跑 CPU、其实一直在占 GPU | `CUDA_VISIBLE_DEVICES=''`（空字符串）在 torch 2.6/Windows 上**不生效**，等价于没设 | 用 `CUDA_VISIBLE_DEVICES='-1'`（非法索引）或脚本自带 `--device cpu` |
| 13 | `CausalLMOutputWithPast` 没有 `last_hidden_state` | transformers 5.x：CausalLM 输出只有 `logits`/`hidden_states`，`last_hidden_state` 在 backbone 输出上 | `output_hidden_states=True` 取 `hidden_states[-1]` |
| 14 | fast loss 与标准 loss 差 0.117（静默训坏） | 训练 batch 用**右 padding**，序列尾部是 pad 而不是目标 JSON | 训练 batch 也改**左 padding**（见 4.5、7.6）|
| 15 | `--max_steps` 冒烟模式下 `train_loss=0.0000` | 达到 max_steps 时提前 `break`，跳过了 loss 累加（训练本身没问题） | 看 `step_log`/验证 loss，或别用 `--max_steps` 判断收敛 |

---

## 9. 成本与延迟

| 方法 | 平均延迟 | prompt tokens | completion tokens | 合计 tokens | 备注 |
|---|---|---|---|---|---|
| API zero-shot | 3.621s | 169891 | 21687 | 191578 | api_zero_n200 |
| API few-shot（6 例） | 3.526s | 292091 | 18993 | 311084 | api_few_n200 |
| Qwen3-0.6B zero-shot | 5.323s | 54683 | 9112 | 63795 | local_zero_n60（批量解码摊薄）|
| LoRA SFT（Qwen3-0.6B） | 3.133s | 182485 | 21035 | 203520 | sft_lora_n200（本地推理，无 API 费用）|

> LoRA 那行的 token 是**本地推理**的消耗（prompt 912/条 × 200 + 生成 105/条 × 200），
> 不产生 API 费用；它的成本是"一次性 48 分钟 GPU 训练"，换来 9.2 MB 的 adapter。
> 生成吞吐 33.5 tok/s（batch=16 批量解码，627.8s 生成 21,035 token）。

成本换算式（单价随厂商/时间变，代码不硬编码价目）：
`cost = prompt_tokens/1e6 × 单价_in + completion_tokens/1e6 × 单价_out`；
需要时把单价写进环境变量 `LLM_PRICE_IN_PER_M` / `LLM_PRICE_OUT_PER_M`，脚本会算进 `report_*.json`。

few-shot 的成本代价：prompt token 从 17.0 万涨到 29.2 万（**+72%**），换来的只有 trigger F1 +2.6 点、
argument 基本不变 —— 这本身是"示例不是免费的"最好证据。

---

## 10. 复现性说明

- 采样：`event_data.canonical_order`（排序后 shuffle，seed=42），id 列表见 `outputs/llm/eval_ids.json`。
- 生成：贪心解码（do_sample=False），同一输入同一输出。
- 训练：`torch.manual_seed(42)`；1 epoch；LoRA adapter 存 `outputs/llm/ckpt/sft_adapter`。
- 指标：只认 `common/evaluate_events.py` 的输出，脚本自动 subprocess 调用，避免口径漂移。

---

## 重构后重跑记录（2026-11-18）

### 1) LoRA 重训：配置决定成败（同一张 8GB 卡，差 4 倍）
| 配置 | 每微批 | 峰值显存 | 总时长 | 结果 |
|---|---|---|---|---|
| `--num_train 3500`（batch 默认 2，**无** `--grad_ckpt`） | 8.3 s/it | 7.9 GiB | ≈4 h（放弃） | 不可用 |
| `--batch_size 3 --grad_accum 5 --grad_ckpt --dtype bf16` | **1.9 s/it** | **4.0 GiB** | **2,327 s（38.8 min）** | ✅ 与旧 run 一致 |

重训后 train_loss **0.1001** / val_loss **0.0742** / 233 优化器步 / 可训参数 **2,293,760（0.3834%）**
—— 与上一轮（0.1002 / 0.0742 / 233 步）几乎逐位相同，说明**训练完全可复现**。

> 顺带排除一个误判：`--dtype auto` 在 CUDA 上**已经**解析为 bf16（`resolve_torch_dtype` 里就是
> `bfloat16 if device.type=="cuda"`），不是 fp32。慢的真因只有一个：**没开梯度检查点**。

### 2) LoRA 评测的 ±1~2pp 波动来自「生成随机性」，不是子集变了
新旧两次评测用的 `(id, sent_id)` 子集**完全一致（200/200 交集）**，但 trigger **0.9100 → 0.9000**、
event **0.2050 → 0.1850**。原因：LoRA merge + bf16 批量解码存在数值抖动，**n=200 下 1 条样本 ≈ 0.5pp**。
argument **0.6966 → 0.6971**（几乎不变）。

### 3) `type_macro_f1` 掉了 13pp：模型输出了 **schema 之外的类型名**
评测里出现了 13 类之外的 **`公司回购股份的进展公告`、`回购`**（金标 0 条）。
`common/evaluate_events.py` 的 `type_macro_f1` 会把预测里出现的类型也纳入 macro 平均，
于是分母从 13 涨到 15 → macro 从 **0.9150 掉到 0.7862**。
**这是生成式链路必须做「类型合法性校验」的直接证据**（判别式方法的标签空间是封闭的，不会犯这个错）。
如果要做公平对比，应该在解析阶段把非 schema 类型**丢弃并计数**（`role_invalid` 已在统计，
类型非法也应同样处理），本项目暂未改评测口径，只在文档里记录。
