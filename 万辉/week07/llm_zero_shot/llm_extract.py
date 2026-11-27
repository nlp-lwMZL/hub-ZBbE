# -*- coding: utf-8 -*-
"""
LLM zero-shot / few-shot 事件抽取（本地 Qwen3-0.6B + 任意 OpenAI 兼容 API）

对应课程里的 L3「提示词工程」层：
  同一个 prompt，换模型/换示例数，看 F1 怎么变 —— 这才是"用大模型做抽取"的基线。

教学重点：
  1. zero-shot vs few-shot：示例是"免费"的领域适配，但也吃 prompt 长度（成本）
  2. **让模型只输出文本 span，不让它输出 offset**：生成式偏移不可信，
     程序用 text.find() 回到原文对齐（详见 event_prompt.py 顶部说明）
  3. 结构化输出的健壮性：剥 ```json 围栏、括号配平截取、截断补全，
     并统计「JSON 合法率 / 修复率 / 空回复 / 幻觉论元」——
     这些数字在 PPT 上比 F1 更能说明"生成式抽取的工程成本"
  4. 成本与延迟：记录每条的 token 与耗时（API 视角）；本地模型记录 tok/s

三种后端：
  api   任意 OpenAI 兼容 API（默认 DeepSeek；base_url / model / key 全部走环境变量）
  local 本地 Qwen3-0.6B（GPU；enable_thinking=False）
  mock  --dry-run 离线假模型（不联网、不占 GPU，用于打通流程/课堂演示）

使用方式：
  # 1) 纯离线演练（不联网不占 GPU，先确认整条流水线是通的）
  python llm_zero_shot/llm_extract.py --dry-run --mode zero --n 20

  # 2) API zero-shot / few-shot（先 export 环境变量）
  set LLM_API_KEY=sk-xxx
  python llm_zero_shot/llm_extract.py --backend api --mode zero --n 200
  python llm_zero_shot/llm_extract.py --backend api --mode few  --n 200

  # 3) 本地 Qwen3-0.6B zero-shot（占 GPU，跑前确认卡空闲）
  #    --gen_batch 是批量解码：Windows 上逐条解码会被 kernel 启动开销拖到 1.5 tok/s，
  #    batch=8 实测快 4~6 倍（详见 local_generate.py 顶部说明）
  python llm_zero_shot/llm_extract.py --backend local --mode zero --n 150 --gen_batch 8

  # 4) 只看提示词全文（PPT 里要贴 prompt）
  python llm_zero_shot/llm_extract.py --print-prompt

输出：
  outputs/llm/pred_{tag}.jsonl    预测（统一评测格式；解析失败的行保留 events=[]）
  outputs/llm/raw_{tag}.jsonl     模型原始输出（做 bad case 分析用）
  outputs/llm/report_{tag}.json   本条链路的统计（合法率/延迟/token/幻觉数）
  outputs/metrics/llm/{tag}.json  common/evaluate_events.py 算出的三级指标
  outputs/llm/eval_ids.json       本次评测用到的样本 id+sent_id（跨方案同批对比用）
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Windows 上 torch 与 numpy 的 OpenMP 冲突

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

# 让 `python llm_zero_shot/llm_extract.py` 直接跑也能 import 同目录模块

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录
sys.path.insert(0, str(ROOT / "common"))        # 公共模块
sys.path.insert(0, str(ROOT / "llm_zero_shot"))  # 生成式链路共用：event_data / event_prompt / local_generate
sys.path.insert(0, str(Path(__file__).resolve().parent))  # llm_lora 自己的 model_paths.py
# 生成式链路的 model_paths（Qwen3-0.6B 路径解析）按目录方案放在 llm_lora/ 下，
# 零样本脚本也要用它，所以把那个目录也加进 import 路径（不复制代码）。
sys.path.insert(0, str(ROOT / "llm_lora"))

from log_utils import EvalLogger  # noqa: E402
from event_data import (  # noqa: E402
    OUTPUT_LLM_DIR, ROOT, SEED, canonical_order, load_schema, load_split,
    make_pred_row, sample_rows, uid, write_eval_ids, write_gold_subset,
)
from event_prompt import (  # noqa: E402
    FEW_SHOT_EXAMPLES, build_messages, build_system_prompt, parse_model_output,
)
from local_generate import generate_batch, load_causal_lm, resolve_torch_dtype  # noqa: E402

# ── API 接入配置：只用环境变量，代码里不出现任何厂商/网关私有信息 ──────────────
DEFAULT_BASE_URL = "https://api.deepseek.com"   # 任意 OpenAI 兼容服务都可以改这个
DEFAULT_MODEL    = "deepseek-v4-flash"          # 官方 = deepseek-chat
MAX_TOKENS_DEFAULT = 512                        # 推理模型太小会被"思考"吃光，给 512+

MODEL_PATH_DEFAULT = None   # None = 用 model_paths.resolve_model_path() 自动找 Qwen3-0.6B

# ══════════════════════════════════════════════════════════════════════════════
# 后端 1：OpenAI 兼容 API
# ══════════════════════════════════════════════════════════════════════════════

def build_api_client():
    """
    构造 openai 客户端。

    为什么全部走环境变量：
      交付给学生的代码不能硬编码任何 key 和内网域名；学生换成自己的服务
      只改环境变量即可（LLM_BASE_URL / LLM_MODEL / LLM_API_KEY），代码一行不动。

    LLM_EXTRA_HEADERS（可选）：JSON 字符串，用于某些 OpenAI 兼容网关要求带
      额外的身份/渠道 header 的场景（例：{"X-Project": "course"}）。
    LLM_DISABLE_THINKING（可选，默认 1）：deepseek-v4-flash 这类推理模型会先输出
      大段思考 token，max_tokens 小的时候正文直接为空（实测 500 条里 85 条空回复、
      指标反而虚高）。这里默认关掉思考，保证"一条请求 = 一条抽取结果"。
    """
    from openai import OpenAI

    api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "未设置 API Key。请先设置环境变量：\n"
            "  Windows:  set LLM_API_KEY=sk-xxx   （或 DEEPSEEK_API_KEY）\n"
            "  Linux/mac: export LLM_API_KEY=sk-xxx"
        )
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)

    headers = {}
    raw_headers = os.getenv("LLM_EXTRA_HEADERS", "").strip()
    if raw_headers:
        try:
            headers = json.loads(raw_headers)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM_EXTRA_HEADERS 不是合法 JSON：{raw_headers}") from exc

    return OpenAI(api_key=api_key, base_url=base_url,
                  default_headers=headers or None, timeout=60.0)

def call_api(client, messages, model: str, max_tokens: int, retries: int = 3) -> dict:
    """
    调一次对话补全，返回 {"text", "prompt_tokens", "completion_tokens", "latency_s", "error"}。

    重试策略：指数退避（1s, 2s, 4s）。限流(429)/超时/网络抖动都在这里吃掉，
    连续失败 3 次返回空文本 —— 上层会把它记成"空回复/解析失败"，**不删这一行**。
    """
    disable_thinking = os.getenv("LLM_DISABLE_THINKING", "1") != "0"
    extra_body = {"thinking": {"type": "disabled"}} if disable_thinking else None
    last_err = ""
    t0 = time.time()
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,        # 抽取任务要可复现，不要随机性
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            usage = getattr(resp, "usage", None)
            return {
                "text": (resp.choices[0].message.content or "").strip(),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                "latency_s": time.time() - t0,
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001 —— 网络/SDK/限流异常都要兜住
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return {"text": "", "prompt_tokens": 0, "completion_tokens": 0,
            "latency_s": time.time() - t0, "error": last_err}

# ══════════════════════════════════════════════════════════════════════════════
# 后端 2：本地 Qwen3-0.6B（实现在 local_generate.py：批量解码 + 左 padding）
# ══════════════════════════════════════════════════════════════════════════════
# 为什么单独抽一个模块：llm_extract.py（zero-shot）和 evaluate_sft.py（SFT 后评测）
# 都要用同一套生成逻辑，抽出来才能保证"生成参数、thinking 开关、批量策略"三处一致。

# ══════════════════════════════════════════════════════════════════════════════
# 后端 3：mock（--dry-run，离线演示）
# ══════════════════════════════════════════════════════════════════════════════

def mock_generate(row: dict, idx: int, seed: int) -> str:
    """
    假模型：不联网不占 GPU，按固定概率"模拟"真实模型的各种毛病。

    为什么 mock 要故意制造解析失败：
      只有这样才能验证「解析修复 + 统计」这条工程链路真的在工作
      （剥围栏、括号配平、截断补全、空回复计数）。
      注意：**mock 跑出来的 F1 没有任何意义**，只用于打通流程 / 课堂演示，
      绝不能写进 PPT 的对比表。
    """
    rng = random.Random(seed * 100000 + idx)
    event = row["event"]
    n_args = len(event["args"])
    keep = max(0, n_args // 2)              # 故意只抽一半论元 → F1 适中
    args = [{"role": a["role"], "argument": a["text"]} for a in event["args"][:keep]]
    obj = {"events": [{"type": event["type"],
                       "trigger": {"text": event["trigger"]["text"]},
                       "arguments": args}]}
    body = json.dumps(obj, ensure_ascii=False)
    roll = rng.random()
    if roll < 0.05:                          # 5% 空回复（模拟 max_tokens 被思考吃光）
        return ""
    if roll < 0.10:                          # 5% 输出被截断（模拟 max_tokens 不够）
        return body[: int(len(body) * 0.8)]
    if roll < 0.20:                          # 10% 包了 markdown 围栏（最常见）
        return f"```json\n{body}\n```"
    if roll < 0.25:                          # 5% 前后夹带解释文字
        return f"好的，抽取结果如下：\n{body}\n（如需调整请告诉我）"
    if roll < 0.30:                          # 5% 类型抽错
        types = list(SCHEMA["event_types"])
        obj["events"][0]["type"] = rng.choice([t for t in types if t != event["type"]])
        return json.dumps(obj, ensure_ascii=False)
    return body

SCHEMA = {}  # 全局 schema（main 里加载；mock_generate 会用到）

# ══════════════════════════════════════════════════════════════════════════════
# 评测（统一口径：调用 common/evaluate_events.py，绝不自己算一套）
# ══════════════════════════════════════════════════════════════════════════════

def run_shared_evaluation(gold_path: Path, pred_path: Path, metrics_path: Path) -> dict:
    """
    跑项目的统一评测脚本，返回其 JSON 结果。

    为什么要 subprocess 调脚本而不是 import 函数：
      保证"报告里的数字 = 命令行 `python common/evaluate_events.py ...` 跑出来的数字"，
      学生复现时不会因为调用方式不同得到略有差异的指标。多花一次进程启动很值。
    """
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "common" / "evaluate_events.py"),
           "--gold", str(gold_path), "--pred", str(pred_path),
           "--json", str(metrics_path), "--quiet"]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _pct(a: int, b: int) -> float:
    return round(100.0 * a / b, 2) if b else 0.0

def _print_metrics(res: dict, report: dict) -> None:
    """把评测结果 + 工程指标打印成一张人看得懂的表（PPT 直接抄）。"""
    print("\n" + "=" * 78)
    print(f"结果：{report['tag']}（{report['backend']}/{report['mode']}，"
          f"{report['n']} 条）")
    print("=" * 78)
    print(f"  trigger F1   : {res['trigger']['f1']:.4f}"
          f"   (P={res['trigger']['precision']:.4f} R={res['trigger']['recall']:.4f})")
    print(f"  argument F1  : {res['argument']['f1']:.4f}"
          f"   (P={res['argument']['precision']:.4f} R={res['argument']['recall']:.4f})")
    print(f"  event  F1    : {res['event_exact']['f1']:.4f}")
    print(f"  type_macro_f1: {res['type_macro_f1']:.4f}")
    print(f"  JSON 合法率  : {report['json_valid_rate']}%（结构性修复 {report['repaired']} 条，"
          f"围栏/废话剥离 {report['noise_stripped']} 条）")
    print(f"  解析失败     : {report['parse_fail']} 条 | 空回复 {report['empty_reply']} 条")
    print(f"  幻觉论元     : {report['arg_not_in_source']} 次 | "
          f"触发词找不到 {report['trigger_not_in_source']} 次")
    print(f"  平均延迟     : {report['latency_mean_s']}s/条（p50={report['latency_p50_s']}s）")
    print(f"  token        : prompt {report['prompt_tokens']} + completion "
          f"{report['completion_tokens']} = {report['total_tokens']}")

# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def check_gpu_idle(threshold_mib: int = 1500) -> None:
    """
    本地推理前先看一眼显存占用（8GB 卡要和同学/其它脚本共用，必须排队）。

    只做"提示"不做"拦截"：显存被别人占着时程序仍能启动（可能 OOM），
    但至少能让人立刻意识到"不是模型的问题，是卡被占了"。
    """
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        used = int(out.splitlines()[0].split(",")[0])
        print(f"[gpu] 当前显存占用 {used} MiB（阈值 {threshold_mib} MiB）")
        if used > threshold_mib:
            print("[gpu][警告] 显存占用较高，可能有其它 GPU 任务在跑；"
                  "8GB 卡同时跑两个任务很容易 OOM。")
    except Exception:  # noqa: BLE001 —— 没装 nvidia-smi 就跳过，不阻塞主流程
        print("[gpu] 未获取到显存信息（nvidia-smi 不可用），跳过检查")

def parse_args():
    p = argparse.ArgumentParser(description="LLM zero-shot / few-shot 事件抽取")
    p.add_argument("--backend", default="api", choices=["api", "local", "mock"],
                   help="api=OpenAI 兼容 API；local=本地 Qwen3-0.6B；mock=离线假模型")
    p.add_argument("--mode", default="zero", choices=["zero", "few"],
                   help="zero=不给示例；few=给 6 个难类型示例")
    p.add_argument("--n", default=200, type=int, help="评测样本数（默认 200）")
    p.add_argument("--seed", default=SEED, type=int)
    p.add_argument("--data", default=None, help="金标 JSONL（默认 data/processed/dev_extract.jsonl）")
    p.add_argument("--tag", default="", help="输出文件后缀，默认 {backend}_{mode}_n{n}")
    p.add_argument("--out", default="", help="预测输出路径，默认 outputs/llm/pred_{tag}.jsonl")
    p.add_argument("--model", default=os.getenv("LLM_MODEL", DEFAULT_MODEL))
    p.add_argument("--model_path", default=MODEL_PATH_DEFAULT,
                   help="本地模型目录（默认自动找 pretrain_models/Qwen3-0.6B）")
    p.add_argument("--max_tokens", default=MAX_TOKENS_DEFAULT, type=int,
                   help="API 单次最大输出 token（推理模型别小于 256）")
    p.add_argument("--max_new_tokens", default=384, type=int, help="本地模型最大新 token 数")
    p.add_argument("--gen_batch", default=8, type=int,
                   help="本地模型批量解码的 batch（Windows 上逐条解码要被 kernel 启动开销拖死，"
                        "batch=8 实测快 4~6 倍；API 后端忽略此参数）")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                   help="本地模型精度；auto = 有 CUDA 用 bf16")
    p.add_argument("--max_length", default=1280, type=int, help="本地模型 prompt 长度上限")
    p.add_argument("--retries", default=3, type=int, help="API 失败重试次数")
    p.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true",
                   help="离线演练：不联网不占 GPU，用假模型把整条流水线跑通（数字无意义）")
    p.add_argument("--print_prompt", "--print-prompt", dest="print_prompt",
                   action="store_true", help="只打印提示词全文然后退出")
    p.add_argument("--no_eval", action="store_true", help="只生成预测，不跑统一评测")
    p.add_argument("--from_raw", default="",
                   help="不调模型，只把已有的 raw_*.jsonl 用**当前解析器**重新解析一遍"
                        "（改了容错逻辑后不必重新花 API 成本/GPU 时间，语义与重跑完全一致）")
    return p.parse_args()

def main():
    global SCHEMA
    args = parse_args()
    eval_logger = EvalLogger(method=args.tag, args=args, root=ROOT,
                             notes=f"LLM {args.backend}/{args.mode} 事件抽取")
    if args.dry_run or args.backend == "mock":
        args.backend = "mock"

    SCHEMA = load_schema()

    if args.print_prompt:
        print(build_system_prompt(SCHEMA))
        return

    tag = args.tag or f"{args.backend}_{args.mode}_n{args.n}"
    OUTPUT_LLM_DIR.mkdir(parents=True, exist_ok=True)

    gold_path_in = Path(args.data) if args.data else (ROOT / "data" / "processed" / "dev_extract.jsonl")
    all_rows = [json.loads(line) for line in open(gold_path_in, encoding="utf-8") if line.strip()]
    rows = sample_rows(all_rows, args.n, args.seed)     # 固定 seed + 全项目统一口径

    # 评测集三件套：样本 id 列表、金标子集、预测文件
    eval_ids_path = OUTPUT_LLM_DIR / "eval_ids.json"
    gold_subset = OUTPUT_LLM_DIR / f"gold_n{len(rows)}.jsonl"
    pred_path = Path(args.out) if args.out else OUTPUT_LLM_DIR / f"pred_{tag}.jsonl"
    raw_path = OUTPUT_LLM_DIR / f"raw_{tag}.jsonl"
    report_path = OUTPUT_LLM_DIR / f"report_{tag}.json"
    metrics_path = ROOT / "outputs" / "metrics" / "llm" / f"{tag}.json"

    write_eval_ids(eval_ids_path, rows, args.seed)
    write_gold_subset(gold_subset, rows)

    print(f"数据：{gold_path_in.name} → 采样 {len(rows)} 条（seed={args.seed}，"
          f"规则见 event_data.canonical_order）")
    print(f"输出：{pred_path}")

    # 只重解析模式：把已有 raw 输出按 (id, sent_id) 建索引
    raw_map = {}
    if args.from_raw:
        raw_src = Path(args.from_raw)
        for line in open(raw_src, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            raw_map[f'{r.get("id")}|{r.get("sent_id")}'] = r
        print(f"[from_raw] 复用 {raw_src.name} 的 {len(raw_map)} 条模型输出，"
              f"只重跑解析 + 评测（不调模型）")

    # ── 准备后端 ──────────────────────────────────────────────────────────────
    client = local_model = local_tok = None
    device = None
    if args.from_raw:
        pass
    elif args.backend == "api":
        client = build_api_client()
        base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
        print(f"[api] base_url={base_url} model={args.model} "
              f"max_tokens={args.max_tokens} few_shot={args.mode == 'few'}")
    elif args.backend == "local":
        import torch
        from model_paths import QWEN_MODEL_DIR, resolve_model_path
        check_gpu_idle()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = args.model_path or resolve_model_path(QWEN_MODEL_DIR)
        print(f"[local] 设备={device} 模型={model_path} "
              f"batch={args.gen_batch} dtype={args.dtype}")
        local_model, local_tok = load_causal_lm(
            str(model_path), device, resolve_torch_dtype(device, args.dtype))
    else:
        print("[mock] 离线假模型：不联网、不占 GPU —— 数字仅用于验证流水线，不可写进实验结论！")

    # ── 推理（本地模型按 batch 批量解码，API/mock 一条一条来）──────────────────
    stat_keys = ("empty_reply", "json_ok", "repaired", "noise_stripped", "type_invalid",
                 "role_invalid", "trigger_not_in_source", "arg_not_in_source")
    stats = {k: 0 for k in stat_keys}
    latencies, prompt_toks, completion_toks = [], 0, 0
    api_errors = []
    t_start = time.time()
    step = args.gen_batch if args.backend == "local" else 1

    with open(pred_path, "w", encoding="utf-8") as f_pred, \
            open(raw_path, "w", encoding="utf-8") as f_raw:
        for start in range(0, len(rows), step):
            chunk = rows[start:start + step]
            chunk_msgs = [build_messages(r["text"], SCHEMA, few_shot=(args.mode == "few"))
                          for r in chunk]
            if args.from_raw:
                # 复用已有 raw：只重跑解析（text 原样送进解析器）
                outs = []
                for r in chunk:
                    prev = raw_map.get(uid(r), {})
                    outs.append({"text": prev.get("raw", ""),
                                 "prompt_tokens": 0, "completion_tokens": 0,
                                 "latency_s": float(prev.get("latency_s", 0.0) or 0.0),
                                 "error": ""})
            elif args.backend == "api":
                outs = [call_api(client, m, args.model, args.max_tokens, args.retries)
                        for m in chunk_msgs]
            elif args.backend == "local":
                outs = generate_batch(local_model, local_tok, chunk_msgs, device,
                                      max_new_tokens=args.max_new_tokens,
                                      batch_size=args.gen_batch,
                                      max_length=args.max_length)
            else:
                outs = [{"text": mock_generate(r, start + k + 1, args.seed),
                         "prompt_tokens": 0, "completion_tokens": 0,
                         "latency_s": 0.0, "error": ""}
                        for k, r in enumerate(chunk)]

            for k, (row, out) in enumerate(zip(chunk, outs)):
                i = start + k + 1
                if out.get("error"):
                    api_errors.append({"i": i, "uid": uid(row), "error": out["error"]})
                events, info = parse_model_output(out["text"], row["text"], SCHEMA)
                for key in stat_keys:
                    stats[key] += int(info.get(key, 0) or 0)

                latencies.append(out["latency_s"])
                prompt_toks += out["prompt_tokens"]
                completion_toks += out["completion_tokens"]

                f_pred.write(json.dumps(make_pred_row(row, events), ensure_ascii=False) + "\n")
                f_raw.write(json.dumps({"id": row.get("id"), "sent_id": row.get("sent_id"),
                                        "raw": out["text"], "events": events,
                                        "latency_s": round(out["latency_s"], 3)},
                                       ensure_ascii=False) + "\n")

            if i % 10 < step or i == len(rows):
                print(f"  [{i:3d}/{len(rows)}] 合法 {stats['json_ok']} 条，"
                      f"平均 {(time.time() - t_start) / i:.2f}s/条", flush=True)

    total_s = time.time() - t_start
    n = len(rows)
    lat_sorted = sorted(latencies)
    # from_raw 模式不重新生成，所以生成侧统计（token/耗时/延迟）沿用上一次的报告
    prev_report = {}
    if args.from_raw and report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            prev_report = json.load(f)
    report = {
        "tag": tag, "backend": args.backend, "mode": args.mode, "n": n,
        "model": args.model if args.backend == "api" else (str(args.model_path) if args.backend == "local" else "mock"),
        "seed": args.seed,
        "few_shot_examples": len(FEW_SHOT_EXAMPLES) if args.mode == "few" else 0,
        "json_valid_rate": _pct(stats["json_ok"], n),
        "parse_fail": n - stats["json_ok"],
        "empty_reply": stats["empty_reply"],
        "repaired": stats["repaired"],
        "noise_stripped": stats["noise_stripped"],
        "type_invalid": stats["type_invalid"],
        "role_invalid": stats["role_invalid"],
        "trigger_not_in_source": stats["trigger_not_in_source"],
        "arg_not_in_source": stats["arg_not_in_source"],
        "latency_mean_s": round(sum(latencies) / n, 3) if n else 0,
        "latency_p50_s": round(lat_sorted[len(lat_sorted) // 2], 3) if n else 0,
        "total_s": round(total_s, 1),
        "prompt_tokens": prompt_toks, "completion_tokens": completion_toks,
        "total_tokens": prompt_toks + completion_toks,
        "api_errors": len(api_errors),
        "pred_path": str(pred_path), "gold_subset": str(gold_subset),
        "eval_ids": str(eval_ids_path),
        # 注明数据来源，防止把 mock 数字误当实验结果
        "is_mock": args.backend == "mock",
    }
    if args.from_raw:
        # 生成侧统计沿用上一份报告（延迟/token/总耗时），并标明这是"只重解析"
        for key in ("model", "backend", "latency_mean_s", "latency_p50_s",
                    "total_s", "prompt_tokens", "completion_tokens", "total_tokens"):
            if key in prev_report:
                report[key] = prev_report[key]
        report["source"] = f"from_raw::{Path(args.from_raw).name}"
        print(f"[from_raw] 解析统计已更新（合法率 {report['json_valid_rate']}%，"
              f"修复 {stats['repaired']} 条）；生成侧指标沿用原报告")
    # 可选成本估算：把官网单价填进环境变量即可，代码不写死任何厂商价目
    price_in = os.getenv("LLM_PRICE_IN_PER_M")
    price_out = os.getenv("LLM_PRICE_OUT_PER_M")
    if price_in and price_out:
        report["cost_estimate"] = round(prompt_toks / 1e6 * float(price_in)
                                        + completion_toks / 1e6 * float(price_out), 4)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n统计 → {report_path}（总耗时 {total_s:.0f}s）")
    if api_errors[:3]:
        print("API 错误示例：", json.dumps(api_errors[:3], ensure_ascii=False)[:400])

    # ── 统一评测 ──────────────────────────────────────────────────────────────
    if not args.no_eval:
        res = run_shared_evaluation(gold_subset, pred_path, metrics_path)
        _print_metrics(res, report)
        print(f"统一评测 → {metrics_path}")
        if report["is_mock"]:
            print("\n⚠ 本次是 mock/--dry-run：指标只证明流水线通了，不能当实验结果。")
        print("\n下一步：把这条预测和其它方案一起画进对比图（eval/plot_compare.py）")
        eval_logger.finalize(
            metrics={k: res[k]["f1"] for k in ("trigger", "argument", "event_exact")},
            type_macro_f1=res.get("type_macro_f1"), report=report)
    else:
        eval_logger.finalize(metrics={}, report=report)

if __name__ == "__main__":
    main()
