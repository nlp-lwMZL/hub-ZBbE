"""通过本地 Ollama API 调用 Qwen3 做事件抽取（免训练、不需要 torch/GPU）。

与 ``extract/llm_lora`` 的关系
-----------------------------
**提示词、输出格式、解析、对齐、指标全部复用 llm_lora 的那一套**，所以两种方式的
结果可以直接对比：

- ``extract.llm_lora.prompt``：system/user 提示词、JSON 解析、非法类型/槽位过滤
- ``extract.llm_lora.align``：把「角色 + 原文片段」回填成 start/end（含基金主代码锚点）
- ``common.metrics``：label 级 F1、事件类型准确率、event exact

唯一的区别是「模型怎么被调用」：这里走 HTTP 调 Ollama，不加载权重、不依赖 torch。

示例
----
# 1) 检查服务与模型
python -m extract.llm_api.run --check

# 2) 单条文本抽取（调试提示词最快的方式）
python -m extract.llm_api.run --text "某某基金管理有限公司关于...的公告"

# 3) 在验证集上评估（默认分层，保证稀有类进入评估）
python -m extract.llm_api.run --eval-per-class 15 --concurrency 2
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.data import load_records  # noqa: E402
from common.logging_utils import setup_logger  # noqa: E402
from common.metrics import (  # noqa: E402
    compute_event_exact_match,
    compute_metrics,
    compute_type_metrics,
)
from common.schema import EVENT_SCHEMA  # noqa: E402
from extract.llm_lora.align import (  # noqa: E402
    attach_spans,
    gold_event_to_metric,
    gold_spans,
    spans_of,
    to_metric_event,
)
from extract.llm_lora.prompt import (  # noqa: E402
    build_system_prompt,
    build_target,
    build_user_prompt,
    normalize_prediction,
    parse_model_output,
)

from .client import OllamaClient, OllamaError  # noqa: E402

DEFAULT_VAL = PROJECT_ROOT / "data" / "dataset" / "split" / "val.jsonl"
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "dataset" / "split" / "train.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "extract" / "output" / "llm_api"
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# API 模式追加的规则（不动共享 prompt.py，避免影响 llm_lora 的训练/评估口径）
EXTRA_RULES = (
    "\n\n额外要求（务必遵守）：\n"
    "1. trigger 必须是原文中**逐字出现**的词语，绝不能填事件类型名。"
    "例如原文写「解聘基金经理林心龙」时 trigger 应为「解聘」而不是「基金经理离任」。\n"
    "2. 判断事件类型的要点：只涉及原基金经理离任、没有新任 → 基金经理离任；"
    "同时出现原基金经理与新（任）基金经理 → 基金经理变更；"
    "只给出新基金经理、没有原基金经理 → 基金经理增聘。\n"
    "3. 基本信息表里的「基金名称」「基金主代码」「基金管理人」以及"
    "「离任原因」「离任日期」「生效日期」等字段值都要输出，不要只输出人名。\n"
    "4. 日期类字段（任职日期、离任日期、生效日期、聘任日期、恢复日期）**最容易漏**，"
    "只要原文出现对应的日期就必须输出，格式照抄原文（如「2025 年 03 月 01 日」或「2025-03-01」），"
    "不要改写、不要补零。\n"
    "5. 说明类字段（聘任说明、履职说明、变动内容）原文有内容时必须输出。\n"
    "6. trigger 只能是**一个连续词语**，不允许拼接两个词。"
    "例如不能输出「增聘兼解聘」，应输出公告主事件对应的单个词（如「变更」或「解聘」）。\n"
)

# 各类型在语料中的高频触发词（按 train 集频次排序），用于纠正模型「拿类型名当触发词」
TRIGGER_LEXICON: dict[str, list[str]] = {
    "基金经理增聘": ["增聘", "聘任"],
    "基金经理离任": ["解聘", "离任"],
    "基金经理变更": ["变更", "调整", "更换"],
    "基金经理助理聘任": ["聘任", "增聘"],
    "基金经理助理离任": ["解聘", "不再担任", "免去", "卸任", "离任", "免除"],
    "基金经理助理变更": ["变更", "调整", "任免"],
    "恢复履职": ["恢复履行", "恢复履职"],
    "代为履职": ["代为履行", "暂停履行", "代为履职", "代为管理"],
    "其他人事变动": ["聘任", "调整"],
}


def strip_think(raw: str) -> str:
    return THINK_RE.sub("", raw or "").strip()


def build_demos(train_file: str, per_class: int, max_chars: int):
    """每类挑 per_class 条训练样本做少样本示例，选取规则完全确定（无随机）。

    规则：类别按名称排序；每类内按 slice_id 升序，优先取原文长度 <= max_chars 的记录，
    不足则补最短的记录。这样同一份数据任何时候都选出同一批示例。
    """
    if per_class <= 0:
        return [], []
    buckets: dict[str, list[dict]] = {}
    for r in load_records(train_file):
        ev_type = (r.get("event") or {}).get("type")
        if ev_type:
            buckets.setdefault(ev_type, []).append(r)
    demos: list[tuple[str, str]] = []
    meta: list[dict] = []
    for ev_type in sorted(buckets):
        cands = sorted(buckets[ev_type], key=lambda r: r.get("slice_id", ""))
        short = [r for r in cands if len(r.get("text") or "") <= max_chars]
        picked = (short or sorted(cands, key=lambda r: len(r.get("text") or "")))[:per_class]
        for r in picked:
            demos.append((build_user_prompt(r.get("text") or ""), build_target(r.get("event"))))
            meta.append({"type": ev_type, "slice_id": r.get("slice_id", ""),
                         "chars": len(r.get("text") or "")})
    return demos, meta


def stratified_sample(records: list[dict], per_class: int, seed: int):
    """按事件类型分层采样，每类最多 per_class 条（与 llm_lora.train 一致）。"""
    buckets: dict[str, list[dict]] = {}
    for r in records:
        key = (r.get("event") or {}).get("type") or "无"
        buckets.setdefault(key, []).append(r)
    rng = random.Random(seed)
    out: list[dict] = []
    dist: dict[str, int] = {}
    for key in sorted(buckets):
        items = list(buckets[key])
        rng.shuffle(items)
        keep = items[:per_class] if per_class > 0 else items
        out.extend(keep)
        dist[key] = len(keep)
    rng.shuffle(out)
    return out, dist


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ollama API 事件抽取")
    p.add_argument("--model", default="qwen3:4b", help="Ollama 模型名")
    p.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama 服务地址")
    p.add_argument("--check", action="store_true", help="只检查服务与模型是否可用")
    p.add_argument("--text", default="", help="对单条文本抽取并打印结果（不走数据集）")

    p.add_argument("--val-file", default=str(DEFAULT_VAL))
    p.add_argument("--eval-limit", type=int, default=200, help="评估集取前 N 条（--eval-per-class 启用时忽略）")
    p.add_argument("--eval-per-class", type=int, default=0, help="按类型分层评估，每类最多 N 条")

    p.add_argument("--num-predict", type=int, default=256, help="最大生成 token 数")
    p.add_argument("--num-ctx", type=int, default=4096, help="上下文窗口（公告长文建议 >=4096）")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="固定为 0（贪心解码，与 llm_lora 的 do_sample=False 等价）；传非 0 会报错")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--retries", type=int, default=3,
                   help="单条调用失败的重试次数（Ollama runner 偶发崩溃/502）")
    p.add_argument("--resume", action="store_true",
                   help="断点续跑：跳过该 run 目录里已有可用结果的样本")
    p.add_argument("--concurrency", type=int, default=1,
                   help="并发请求数；需服务端 OLLAMA_NUM_PARALLEL>1 才能真正并行")
    p.add_argument("--think", action="store_true", help="允许模型思考（默认关闭）")
    p.add_argument("--no-extra-rules", action="store_true",
                   help="不追加 API 专用规则（触发词必须逐字摘录、类型判定要点、公共槽位别丢）")
    p.add_argument("--no-trigger-fix", action="store_true",
                   help="不纠正「把事件类型名当触发词」的输出")

    p.add_argument("--few-shot", type=int, default=0,
                   help="每类放几条训练样本作少样本示例（0=零样本）")
    p.add_argument("--few-shot-file", default=str(DEFAULT_TRAIN), help="示例来源（默认训练集）")
    p.add_argument("--few-shot-max-chars", type=int, default=800,
                   help="单条示例原文长度上限，控制上下文占用（不足时自动取最短的）")

    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    p.add_argument("--run-name", default="")
    p.add_argument("--log-every", type=int, default=10)
    return p


def system_prompt(args) -> str:
    """共享 system 提示 +（默认）API 模式专用规则。"""
    base = build_system_prompt()
    return base if args.no_extra_rules else base + EXTRA_RULES


def make_client(args) -> OllamaClient:
    return OllamaClient(
        model=args.model,
        host=args.host,
        num_predict=args.num_predict,
        num_ctx=args.num_ctx,
        temperature=args.temperature,
        seed=args.seed,
        timeout=args.timeout,
        think=args.think,
        retries=args.retries,
    )


def extract_one(client: OllamaClient, text: str, system: str,
                fix_trigger: bool,
                demos: list[tuple[str, str]] | None = None) -> tuple[dict, str, dict]:
    """调用一次模型 -> (规整后的 pred, 原始输出, 元信息)。"""
    raw, meta = client.chat(system, build_user_prompt(text), demos=demos)
    raw = strip_think(raw)
    obj = parse_model_output(raw)
    pred = normalize_prediction(obj) if obj else {"type": "", "trigger": "", "arguments": []}
    if fix_trigger:
        pred = repair_trigger(pred, text)
    return pred, raw, meta


def repair_trigger(pred: dict, text: str) -> dict:
    """模型常把事件类型名当触发词（如 trigger="基金经理变更"），这里纠正为原文触发词。

    判定：trigger 在原文里找不到、或本身就是事件类型/槽位名时，改用该类型的高频触发词。
    """
    ev_type = pred.get("type", "")
    trig = (pred.get("trigger") or "").strip()
    all_roles = {r for spec in EVENT_SCHEMA.values() for r in spec["roles"]}
    invalid = (not trig) or trig in EVENT_SCHEMA or trig in all_roles or trig not in text
    if not invalid:
        return pred
    for cand in TRIGGER_LEXICON.get(ev_type, []):
        if cand in text:
            return {**pred, "trigger": cand}
    return {**pred, "trigger": ""}


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("llm_api")
    if args.temperature != 0.0:
        logger.error("temperature 必须为 0（贪心解码），收到 %s；为保证与 llm_lora 可比且可复现，禁止采样。",
                     args.temperature)
        return 2
    client = make_client(args)
    demos, demo_meta = build_demos(args.few_shot_file, args.few_shot, args.few_shot_max_chars)

    if args.check:
        try:
            models = client.list_models()
        except OllamaError as e:
            logger.error("检查失败：%s", e)
            return 2
        logger.info("服务可用：%s | 已安装模型 %d 个", args.host, len(models))
        for m in models:
            logger.info("  - %s%s", m, "  <= 目标模型" if m == args.model else "")
        if args.model not in models:
            logger.error("未找到 %s，请先执行：ollama pull %s", args.model, args.model)
            return 3
        return 0

    if args.text:
        text = args.text.strip()
        system = system_prompt(args)
        pred, raw, meta = extract_one(client, text, system, not args.no_trigger_fix, demos)
        spanned = attach_spans(pred, text)
        logger.info("耗时 %.1fs | 生成 %d tokens | done_reason=%s",
                    meta["wall_sec"], meta["eval_count"], meta["done_reason"])
        logger.info("原始输出：%s", raw.replace("\n", " ")[:400])
        print(json.dumps(spanned, ensure_ascii=False, indent=2))
        return 0

    # ---------------- 评估 ----------------
    if args.eval_per_class > 0:
        # 与 llm_lora.train 一致：验证集用 seed+1 采样，保证两种方式的评估集可逐条对照
        records, dist = stratified_sample(load_records(args.val_file),
                                          args.eval_per_class, args.seed + 1)
        logger.info("评估集分层采样（每类<=%d, seed=%d）：%d 条 | %s",
                    args.eval_per_class, args.seed + 1, len(records),
                    json.dumps(dist, ensure_ascii=False))
    else:
        records = load_records(args.val_file, limit=args.eval_limit)
        logger.info("评估集：前 %d 条", len(records))

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S") + "_" + args.model.replace(":", "-")
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("run: %s | 模型 %s | 并发 %d | num_ctx=%d | temperature=0(贪心) | think=%s",
                run_name, args.model, args.concurrency, args.num_ctx, args.think)
    if demos:
        logger.info("少样本：%d 条示例（每类<=%d），原文合计 %d 字符", len(demos), args.few_shot,
                    sum(m["chars"] for m in demo_meta))
        logger.info("示例 slice_id: %s", ", ".join(m["slice_id"] for m in demo_meta))
        (run_dir / "few_shot.json").write_text(
            json.dumps(demo_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.num_ctx < 12288:
            logger.warning("少样本会显著加长上下文，建议 --num-ctx 16384（当前 %d，超出部分会被截断）",
                           args.num_ctx)

    t0 = time.time()
    system = system_prompt(args)

    # 断点续跑：复用该 run 目录里已有且成功的样本
    prev: dict[str, dict] = {}
    pred_path = run_dir / "val_predictions.jsonl"
    if args.resume and pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            raw = (row.get("raw") or "").strip()
            if raw and not raw.startswith("__ERROR__"):
                prev[row.get("slice_id") or row.get("source") or ""] = row
        logger.info("断点续跑：已有可用结果 %d 条，本次只需再跑 %d 条",
                    len(prev), max(0, len(records) - len(prev)))

    results: list[tuple[str, str, dict] | None] = [None] * len(records)
    n_done = n_err = 0
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {}
        for i, rec in enumerate(records):
            sid = rec.get("slice_id") or rec.get("source") or ""
            if sid in prev:
                continue
            futures[pool.submit(extract_one, client, rec.get("text") or "", system,
                                not args.no_trigger_fix, demos)] = i
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except OllamaError as e:
                n_err += 1
                logger.warning("第 %d 条调用失败：%s", i, e)
                results[i] = ({"type": "", "trigger": "", "arguments": []}, f"__ERROR__ {e}", {})
            n_done += 1
            if n_done % args.log_every == 0 or n_done == len(records):
                speed = n_done / max(1e-6, time.time() - t0)
                logger.info("  评估 %d/%d | %.2f 条/秒 | 失败 %d", n_done, len(futures), speed, n_err)

    # 续跑：把上次已成功的样本回填
    for i, rec in enumerate(records):
        if results[i] is not None:
            continue
        row = prev.get(rec.get("slice_id") or rec.get("source") or "")
        if row is None:
            results[i] = ({"type": "", "trigger": "", "arguments": []}, "__ERROR__ 缺少结果", {})
            continue
        raw = row.get("raw") or ""
        obj = parse_model_output(raw)
        pred = normalize_prediction(obj) if obj else {"type": "", "trigger": "", "arguments": []}
        if not args.no_trigger_fix:
            pred = repair_trigger(pred, rec.get("text") or "")
        results[i] = (pred, raw, {"eval_count": 0, "prompt_eval_count": 0,
                                  "wall_sec": 0.0, "done_reason": "cached"})

    gold_spans_list, pred_spans_list = [], []
    gold_types, pred_types = [], []
    gold_events, pred_events = [], []
    rows: list[dict] = []
    n_parse_fail = n_no_event = n_unmatched = 0
    total_gen = 0

    for rec, res in zip(records, results):
        text = rec.get("text") or ""
        pred, raw, meta = res
        if not pred["type"] and raw.startswith("__ERROR__"):
            n_parse_fail += 1
        elif parse_model_output(raw) is None:
            n_parse_fail += 1
        if pred["type"] == "无":
            n_no_event += 1
        total_gen += int(meta.get("eval_count") or 0)

        spanned = attach_spans(pred, text)
        n_unmatched += len(spanned["unmatched"])

        gold_spans_list.append(gold_spans(rec.get("event")))
        pred_spans_list.append(spans_of(spanned))
        gold_types.append((rec.get("event") or {}).get("type", ""))
        pred_types.append(spanned["type"])
        gold_events.append(gold_event_to_metric(rec.get("event")))
        pred_events.append(to_metric_event(spanned))
        rows.append({
            "source": rec.get("source", ""),
            "slice_id": rec.get("slice_id", ""),
            "gold": rec.get("event"),
            "pred": spanned,
            "raw": (raw or "")[:500],
        })

    report = compute_metrics(gold_spans_list, pred_spans_list)
    report.event_type = compute_type_metrics(gold_types, pred_types)
    report.event_match = compute_event_exact_match(gold_events, pred_events)
    d = report.as_dict()

    elapsed = round(time.time() - t0, 1)
    stats = {
        "n": len(records),
        "parse_fail": n_parse_fail,
        "no_event": n_no_event,
        "unmatched_spans": n_unmatched,
        "api_errors": n_err,
        "gen_tokens": total_gen,
        "elapsed_sec": elapsed,
        "sec_per_item": round(elapsed / max(1, len(records)), 1),
    }

    em = d.get("event_match") or {}
    per_type = em.get("per_type") or {}
    hit = ", ".join(f"{k} {v['exact']}/{v['gold']}" for k, v in per_type.items())
    logger.info("评估: P=%.4f R=%.4f F1=%.4f | trigger F1=%.4f | arg F1=%.4f | "
                "type acc=%.4f macroF1=%.4f microF1=%.4f | event exact=%d/%d=%.4f | "
                "解析失败 %d | 无事件 %d | 定位失败 %d | %.0fs",
                d["overall"]["precision"], d["overall"]["recall"], d["overall"]["f1"],
                d["trigger"]["f1"], d["arg"]["f1"],
                d["event_type"]["accuracy"], d["event_type"]["macro_f1"], d["event_type"]["micro_f1"],
                em.get("exact", 0), em.get("total", 0), em.get("accuracy", 0.0),
                n_parse_fail, n_no_event, n_unmatched, elapsed)
    logger.info("分类型 exact: %s", hit)

    (run_dir / "val_predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (run_dir / "val_report.json").write_text(json.dumps({
        "run": {"name": run_name, "kind": "ollama_api", "model": args.model},
        "args": vars(args),
        "stats": stats,
        "report": d,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = Path(args.output_root) / "experiments.csv"
    new = not csv_path.exists()
    if not args.resume:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["run", "model", "n", "event_exact", "overall_f1", "trigger_f1",
                            "arg_f1", "type_macro_f1", "parse_fail", "sec_per_item"])
            w.writerow([run_name, args.model, stats["n"], f"{em.get('accuracy', 0):.4f}",
                        f"{d['overall']['f1']:.4f}", f"{d['trigger']['f1']:.4f}",
                        f"{d['arg']['f1']:.4f}", f"{d['event_type']['macro_f1']:.4f}",
                        n_parse_fail, stats["sec_per_item"]])

    logger.info("报告: %s", run_dir / "val_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
