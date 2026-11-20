"""
方法四·外部版：直接用大模型 API 做零样本（不训练，走网络）

教学重点：
  1. 这是**最省事的一条路**：不加载本地权重、不占显存，只要一个 key。
     冷启动验证、跨领域临时任务，它都是第一选择。
  2. 但成本和延迟都是「按次计费」的：同一批 500 条跑一遍要几十秒到几分钟，
     单条延迟是本地 0.6B 的数十倍，因此高 QPS 场景必须换成本地小模型或微调。
  3. 代码写法就是 OpenAI 兼容接口的事实标准：
          client = OpenAI(base_url=..., api_key=...)
          client.chat.completions.create(model=..., messages=[...])
     换任何 OpenAI 兼容服务（DeepSeek / 本地 vLLM / 其他云厂商）只改环境变量，代码不动。

环境变量（只需要这一个 Key，其余有默认值）：
  LLM_API_KEY        必填，服务商的 key（DeepSeek 用 DEEPSEEK_API_KEY 也行）
  LLM_BASE_URL       默认 https://api.deepseek.com
  LLM_MODEL          默认 deepseek-v4-flash
  LLM_EXTRA_HEADERS  可选，JSON 字符串。某些网关要求额外的 header（如路由用的 provider 标识），
                     用这个传即可，代码本身不绑定任何一家网关。

用法：
  python llm_api_zero_shot/classify_api_zero_shot.py --n 500
  python llm_api_zero_shot/classify_api_zero_shot.py --n 500 --workers 8
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common.data import load_clean_val
from common.labels import LABELS
from common.metrics import classification_report, pred_distribution, to_markdown_table

# 可选：DeepSeek 推理模型支持关闭思考（省 token、降延迟）。其它服务商不认这个参数，默认 None。
EXTRA_BODY = {'thinking': {'type': 'disabled'}} if os.getenv('LLM_DISABLE_THINKING', '0') in ('1', 'true') else None

SYSTEM = ('你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，'
          '只输出类别名称，不要输出任何其他内容。' + chr(10) + '可选类别：' + '、'.join(LABELS))


def make_client():
    """
    构造 OpenAI 兼容客户端。

    为什么 base_url / model 都从环境变量读：
      「同一份代码跑通不同服务商」是工程能力的一部分——
      线上换模型时只改配置不改代码，这一点在面试里经常被问到。
    """
    from openai import OpenAI

    key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("缺少 LLM_API_KEY（或 DEEPSEEK_API_KEY）环境变量")
    extra = json.loads(os.getenv("LLM_EXTRA_HEADERS", "{}"))     # 可选：网关路由 header
    return OpenAI(
        api_key=key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        default_headers=extra or None,
        timeout=60,
        max_retries=2,          # 偶发 429/网络抖动自动重试，不要把整批跑挂
    )


def parse_label(text: str) -> int:
    """从返回文本里抽类别名；抽不到返回 -1（统计解析失败率）"""
    hits = [n for n in LABELS if n in text]
    return LABELS.index(max(hits, key=len)) if hits else -1


def classify_one(client, model, title, max_tokens=256):
    """单条请求：返回 (预测 id, 原始输出, 用量)。异常一律吞掉并记为失败，保证整批能跑完。"""
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": title}],
            temperature=0,          # 分类要稳定，不要随机性
            max_tokens=max_tokens,  # 推理模型会先生成思考 token，留出余量；64 太小会被截断成空回复
            extra_body=EXTRA_BODY,  # 见文件头：可选关闭推理模型的思考过程
        )
        raw = (r.choices[0].message.content or "").strip()
        usage = {
            "prompt_tokens": getattr(r.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(r.usage, "completion_tokens", 0),
        }
        return parse_label(raw), raw, usage
    except Exception as e:          # 网络错误 / 限流 / 超时 → 记为解析失败，不中断整批
        return -1, f"ERROR: {e}", {"prompt_tokens": 0, "completion_tokens": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8, help="并发数；太高会被限流")
    ap.add_argument("--out", default="outputs/api_zero_shot_pred.json")
    args = ap.parse_args()

    model = os.getenv("LLM_MODEL", "deepseek-v4-flash")
    client = make_client()
    samples, meta = load_clean_val(args.n)
    meta["api_model"] = model
    meta["base_url"] = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda d: classify_one(client, model, d["sentence"]), samples))
    elapsed = round(time.time() - t0, 1)

    preds = [r[0] for r in results]
    raws = [r[1] for r in results]
    tin = sum(r[2]["prompt_tokens"] for r in results)
    tout = sum(r[2]["completion_tokens"] for r in results)

    y_true = [d["label"] for d in samples]
    report = classification_report(y_true, preds)
    report.update({
        "method": f"api_zero_shot:{model}",
        "elapsed_s": elapsed,
        "sec_per_item": round(elapsed / len(samples), 2),
        "tokens_in": tin,
        "tokens_out": tout,
        "pred_distribution": pred_distribution(preds),
    })

    out = CODE_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "report": report,
                               "preds": [{"text": d["sentence"], "true": d["label"], "pred": p, "raw": r}
                                         for d, p, r in zip(samples, preds, raws)]},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"模型 {model}｜{meta['n_eval']} 条｜{elapsed}s（{report['sec_per_item']}s/条）"
          f"｜tokens in/out = {tin}/{tout}")
    print(to_markdown_table({f"API zero-shot（{model}）": report}))
    print(f"预测覆盖类别数：{report['pred_distribution']['used_classes']}/{len(LABELS)}")
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()

