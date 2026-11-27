"""
DeepSeek API 做 NER：zero-shot / one-shot / few-shot 对比

和 ../src_llm/llm_ner.py（Qwen 版）的区别：
  1. **后端换成 DeepSeek**：OpenAI 兼容接口，只改 base_url + model 名，
     代码结构完全一样。默认 https://api.deepseek.com + deepseek-chat。
  2. **新增 one-shot**：--n_shot 0 / 1 / 3 都支持，也可以一次跑多个做对比
     （--n_shot 0,1,3）。
  3. **示例只从 train.json 抽**，并且强制覆盖 PER/ORG/LOC 三类 —— 只给一个例子时，
     那个例子必须同时含三类实体，否则等于没教另外两类。
  4. **JSON 模式**：DeepSeek 支持 response_format={"type":"json_object"}，
     能显著降低「输出不是合法 JSON」的比例；接口不支持时自动降级为普通模式。
  5. 指标复用 common.py 的多重集匹配（本数据集 7.57% 的实体是句内重复，
     参考实现的 text.find 口径会低估召回）。

为什么要用 DeepSeek 而不是本地 Qwen2-0.5B 做 few-shot：
  0.5B 模型连「只能从 15 个词里选」都学不会（见 text_classification 项目的实测），
  NER 要输出更长的 JSON，格式失败率只会更高。上下文学习是小模型最不擅长的场景，
  few-shot 真正需要的是 7B 以上的模型 —— 走 API 是最省事的路。

准备：
  pip install openai
  set DEEPSEEK_API_KEY=sk-xxx          # Windows CMD
  $env:DEEPSEEK_API_KEY="sk-xxx"       # PowerShell
  export DEEPSEEK_API_KEY=sk-xxx       # bash

使用方式：
  python src_llm_work/llm_ner_work.py --n_shot 0            # zero-shot
  python src_llm_work/llm_ner_work.py --n_shot 1            # one-shot
  python src_llm_work/llm_ner_work.py --n_shot 3            # few-shot（3 例）
  python src_llm_work/llm_ner_work.py --n_shot 0,1,3        # 三种一起跑并对比
  python src_llm_work/llm_ner_work.py --n_shot 1 --demo     # 只跑 5 条看输出
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from common import (
    LOG_DIR, SYSTEM_PROMPT, ENTITY_TYPES, ENTITY_TYPE_ZH,
    load_split, record_text, bio_to_entities,
    parse_entities, compute_metrics_multiset, per_type_metrics,
    load_env_file, _dw, _pad,
)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


# ══════════════════════════════════════════════════════════════════════════════
# 客户端
# ══════════════════════════════════════════════════════════════════════════════

def build_client(base_url: str):
    """DeepSeek 走 OpenAI 兼容协议，用官方 openai SDK 即可。"""
    from openai import OpenAI

    # 先看环境变量，没有再读项目里的 .env（见 common.load_env_file）
    api_key = load_env_file("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit(
            "\n缺少 DEEPSEEK_API_KEY。三种设置方式，任选一种：\n"
            "\n  【推荐】在项目根目录建 .env 文件，写一行：\n"
            "      DEEPSEEK_API_KEY=sk-xxx\n"
            "    不用重启任何东西，脚本会自动读。\n"
            "\n  当前 PowerShell 窗口临时生效：\n"
            "      $env:DEEPSEEK_API_KEY=\"sk-xxx\"\n"
            "\n  Windows 永久生效（对之后新开的窗口才有效）：\n"
            "      setx DEEPSEEK_API_KEY \"sk-xxx\"\n"
            "\n  key 在 https://platform.deepseek.com/api_keys 申请\n"
        )
    return OpenAI(api_key=api_key, base_url=base_url)


# ══════════════════════════════════════════════════════════════════════════════
# 示例抽取（只从 train.json，且强制覆盖三类实体）
# ══════════════════════════════════════════════════════════════════════════════

def pick_examples(train_data: list[dict], n_shot: int, seed: int) -> list[dict]:
    """挑 n_shot 个标注示例。

    n_shot=0 不抽。
    其余情况优先挑「同时含 PER/ORG/LOC」的句子 —— 一个例子就能示范全部三类，
    这比给三个只含单一类型的例子更省 token，也更不容易让模型以为「一类一句」。
    找不到全覆盖的句子时退化为按类型补齐。
    """
    if n_shot <= 0:
        return []

    rng = random.Random(seed)
    with_types = []
    for rec in train_data:
        ents = bio_to_entities(rec["tokens"], rec["ner_tags"])
        types = {e["type"] for e in ents}
        if len(types) == len(ENTITY_TYPES):
            with_types.append((rec, ents))

    picked = []
    if len(with_types) >= n_shot:
        for rec, ents in rng.sample(with_types, n_shot):
            picked.append({"text": record_text(rec), "entities": ents})
        return picked

    # 全覆盖句子不够：先用掉它们，再按缺失的类型补
    for rec, ents in with_types:
        picked.append({"text": record_text(rec), "entities": ents})
    covered = {e["type"] for p in picked for e in p["entities"]}

    by_type = defaultdict(list)
    for rec in train_data:
        for e in bio_to_entities(rec["tokens"], rec["ner_tags"]):
            by_type[e["type"]].append(rec)

    for etype in ENTITY_TYPES:
        if len(picked) >= n_shot:
            break
        if etype in covered:
            continue
        for rec in rng.sample(by_type[etype], min(len(by_type[etype]), 1)):
            picked.append({"text": record_text(rec),
                           "entities": bio_to_entities(rec["tokens"], rec["ner_tags"])})
            covered.add(etype)
            break
    return picked[:n_shot]


def build_messages(text: str, examples: list[dict]) -> list[dict]:
    """system + (user/assistant) × n_shot + 待标注文本。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        messages.append({"role": "user", "content": ex["text"]})
        messages.append({
            "role": "assistant",
            "content": json.dumps({"entities": ex["entities"]},
                                  ensure_ascii=False, separators=(", ", ": ")),
        })
    messages.append({"role": "user", "content": text})
    return messages


# ══════════════════════════════════════════════════════════════════════════════
# API 调用
# ══════════════════════════════════════════════════════════════════════════════

def call_api(client, messages: list[dict], model: str,
             max_tokens: int = 256) -> tuple[str, dict]:
    """返回 (输出文本, usage)。失败重试 3 次，仍失败返回空串。

    先尝试 JSON 模式（DeepSeek 支持 response_format=json_object），
    如果接口报「不支持该参数」，本次运行内降级为普通模式并记住，不再重试。
    """
    kwargs = dict(model=model, messages=messages, temperature=0.0, max_tokens=max_tokens)
    if not call_api.json_mode_broken:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(**kwargs)
            usage = {"prompt_tokens": resp.usage.prompt_tokens,
                     "completion_tokens": resp.usage.completion_tokens} if resp.usage else {}
            return resp.choices[0].message.content or "", usage
        except Exception as e:
            msg = str(e)
            if "response_format" in msg and not call_api.json_mode_broken:
                call_api.json_mode_broken = True
                kwargs.pop("response_format", None)
                print("  [提示] 该接口不支持 JSON 模式，已降级为普通模式")
                continue
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  API 调用失败：{msg[:200]}")
                return "", {}
    return "", {}


call_api.json_mode_broken = False


# ══════════════════════════════════════════════════════════════════════════════
# 单个 n_shot 设置的评测
# ══════════════════════════════════════════════════════════════════════════════

def run_one_setting(client, args, samples: list[dict], examples: list[dict],
                    train_data: list[dict]) -> dict:
    golds, preds = [], []
    detail = []
    n_fail = 0
    tokens_in = tokens_out = 0
    t0 = time.time()

    for i, record in enumerate(samples, 1):
        text = record_text(record)
        g_ents = bio_to_entities(record["tokens"], record["ner_tags"])
        raw, usage = call_api(client, build_messages(text, examples), args.model)
        p_ents, json_ok = parse_entities(raw, text)

        if not json_ok:
            n_fail += 1
        tokens_in += usage.get("prompt_tokens", 0)
        tokens_out += usage.get("completion_tokens", 0)

        golds.append(g_ents)
        preds.append(p_ents)
        detail.append({"text": text, "gold": g_ents, "pred": p_ents,
                       "raw_output": raw, "json_ok": json_ok})

        if args.demo:
            print(f"[{i}] {text}")
            print(f"    gold: {g_ents}")
            print(f"    pred: {p_ents}")
            print(f"    raw : {raw[:200]}")
            print()
        elif i % 20 == 0 or i == len(samples):
            print(f"  已处理 {i}/{len(samples)}  ({time.time() - t0:.0f}s)")

    elapsed = time.time() - t0
    metrics = compute_metrics_multiset(golds, preds)
    return {
        "n_shot": len(examples),
        "model": args.model,
        "split": args.split,
        "n_samples": len(samples),
        "metrics": metrics,
        "per_type": per_type_metrics(golds, preds),
        "format_fail": n_fail,
        "elapsed_s": round(elapsed, 1),
        "ms_per_item": round(elapsed / len(samples) * 1000, 1),
        "tokens": {"input": tokens_in, "output": tokens_out},
        "examples": examples,
        "detail": detail,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    shots = [int(s) for s in str(args.n_shot).split(",") if s.strip() != ""]

    client = build_client(args.base_url)

    val = load_split(args.split)
    n = 5 if args.demo else args.n_samples
    samples = random.Random(args.seed).sample(val, min(n, len(val)))
    print(f"评测集: {args.split}，{len(samples)} 条 | 模型: {args.model} | "
          f"n_shot: {shots}\n")

    train_data = load_split("train")
    # 示例只从 train 抽，和评测集天然不重叠；这里再断言一次，防止以后改坏
    train_texts = {record_text(r) for r in train_data}
    assert not (train_texts & {record_text(s) for s in samples}), \
        "示例池和评测集有重叠，会泄漏"

    results = []
    for k in shots:
        examples = pick_examples(train_data, k, args.example_seed)
        label = "zero-shot" if k == 0 else f"{k}-shot"
        print(f"{'=' * 70}")
        print(f"{label}（示例 {len(examples)} 个，prompt 里 "
              f"{sum(len(e['text']) for e in examples)} 字）")
        print(f"{'=' * 70}")
        if examples and args.demo:
            for ex in examples:
                print(f"  示例: {ex['text']}")
                print(f"        {ex['entities']}")
        res = run_one_setting(client, args, samples, examples, train_data)
        results.append(res)

        m = res["metrics"]
        print(f"  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  "
              f"格式失败 {res['format_fail']} 条  "
              f"{res['ms_per_item']:.0f} ms/条  "
              f"tokens in/out={res['tokens']['input']}/{res['tokens']['output']}")
        print()

    # ── 汇总表 ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"{'DeepSeek NER 对比（' + args.split + '，' + str(len(samples)) + ' 条）':^66}")
    print(f"{'=' * 72}")
    head = _pad("设置", 14) + _pad("Precision", 11, right=True) \
        + _pad("Recall", 9, right=True) + _pad("F1", 9, right=True) \
        + _pad("格式失败", 10, right=True) + _pad("ms/条", 9, right=True)
    print(head)
    print("-" * _dw(head))
    for r in results:
        label = "zero-shot" if r["n_shot"] == 0 else f"{r['n_shot']}-shot"
        m = r["metrics"]
        print(_pad(label, 14) + _pad(f"{m['precision']:.4f}", 11, right=True)
              + _pad(f"{m['recall']:.4f}", 9, right=True)
              + _pad(f"{m['f1']:.4f}", 9, right=True)
              + _pad(str(r["format_fail"]), 10, right=True)
              + _pad(f"{r['ms_per_item']:.0f}", 9, right=True))
    print("-" * _dw(head))

    if len(results) > 1:
        base = results[0]["metrics"]["f1"]
        print("\n  相对 zero-shot 的变化：")
        for r in results[1:]:
            print(f"    {r['n_shot']}-shot: {r['metrics']['f1'] - base:+.4f}")

    # ── 保存 ──────────────────────────────────────────────────────────────────
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_".join(f"{r['n_shot']}shot" for r in results)
    out_path = LOG_DIR / f"eval_llm_{tag}_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "base_url": args.base_url,
                   "split": args.split, "n_samples": len(samples),
                   "seed": args.seed, "example_seed": args.example_seed,
                   "runs": results}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存 → {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek NER zero/one/few-shot")
    parser.add_argument("--n_shot", default="0,1,3",
                        help="示例数，可给单个（3）或列表（0,1,3）")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="deepseek-chat（V3.1 非思考）或 deepseek-reasoner（思考模式）")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--n_samples", default=300, type=int,
                        help="评测条数；300 条 95%% 置信区间约 ±0.05")
    parser.add_argument("--seed", default=42, type=int, help="抽样种子")
    parser.add_argument("--example_seed", default=1234, type=int,
                        help="抽示例的种子，和 --seed 分开，互不干扰")
    parser.add_argument("--demo", action="store_true", help="只跑 5 条并打印 prompt/输出")
    return parser.parse_args()


if __name__ == "__main__":
    main()
