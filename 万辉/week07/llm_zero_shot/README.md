# llm_zero_shot/ —— LLM 链路的公共层 + zero/few-shot 抽取

**一句话**：把生成式（LLM）链路的**公共部分**放这里 —— prompt 构造、输出解析、本地批量生成、数据工具，
以及 zero-shot / few-shot 抽取脚本本体。

```bash
# 本地 Qwen3-0.6B zero-shot（不花钱，慢）
python llm_zero_shot/llm_extract.py --backend local --mode zero --n 60 --gen_batch 8 --tag local_zero_n60
# 云端 API（OpenAI 兼容；公司网络下需要能访问 api.deepseek.com）
python llm_zero_shot/llm_extract.py --backend api --mode zero --n 200 --tag api_zero_n200
python llm_zero_shot/llm_extract.py --backend api --mode few  --n 200 --tag api_few_n200
python llm_zero_shot/llm_extract.py --print-prompt          # 只看 prompt，不调用
```

| 文件 | 作用 |
|---|---|
| `llm_extract.py` | 抽取主脚本（本地/API、zero/few-shot、批量解码） |
| `event_prompt.py` | system/user prompt 模板 + 模型输出解析（JSON 修复） |
| `event_data.py` | **LLM 链路自己的**数据工具（target JSON 序列化、候选排序等）；与 `common/event_data.py` 职责不同 |
| `local_generate.py` | 本地 transformers 批量生成（batch decode、dtype 处理） |

**产物**：`outputs/llm/raw_*.jsonl`（原始回复）、`outputs/llm/pred_*.jsonl`（预测）、
`outputs/llm/report_*.json`（合法率/延迟/token/幻觉统计）、`outputs/metrics/llm/*.json`（统一评测）、
`outputs/logs/<tag>_eval.json` + `_console.log`。

**教学重点**
1. 生成式链路最脆的地方是**输出格式**：JSON 合法率、修复次数、空回复都要单独统计
   （否则「指标虚高」往往来自"解析失败被悄悄丢掉"）。
2. 本地小模型 zero-shot 的触发词能力断崖式下降（0.6B 只有 0.42），**必须靠 SFT**（见 `llm_lora/`）。
3. 评测一律走 `common/evaluate_events.py`，与判别式**同一口径**才能横向比。
