"""在相同样本上运行词典、CRF 与可替换的生成式 API。"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from time import perf_counter

from baselines import CRFExtractor, LexiconExtractor
from chat_api import ApiConfig, ChatExtractor, Prompts
from data_sources import COURSE, DUEE_ARCHIVE, ROOT, load_dataset

sys.path.insert(0, str(COURSE / "common"))
from evaluate_events import evaluate


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("duee-fin", "duee"), default=["duee-fin", "duee"])
    parser.add_argument("--methods", nargs="+", choices=("lexicon", "crf"), default=["lexicon", "crf"])
    parser.add_argument("--api-only", action="store_true", help="仅执行 --api 指定的配置")
    parser.add_argument("--train-n", type=int, default=2400, help="每个数据集的训练样本数，0 为全量")
    parser.add_argument("--eval-n", type=int, default=40, help="每个数据集的评测样本数，0 为全量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duee-archive", type=Path, default=DUEE_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--api", nargs=5, action="append", default=[],
                        metavar=("NAME", "BASE_URL", "MODEL", "KEY_ENV", "SHOTS"),
                        help="可重复指定 API。免鉴权填 -，zero-shot 填 0")
    parser.add_argument("--api-timeout", type=float, default=60)
    parser.add_argument("--max-tokens", type=int, default=384)
    args = parser.parse_args()

    if min(args.train_n, args.eval_n) < 0 or min(args.api_timeout, args.max_tokens) <= 0:
        parser.error("样本数必须非负，API 超时与最大生成 token 数必须为正")
    configs = []
    names = {"lexicon", "crf"}
    for name, base_url, model, key_env, shots in args.api:
        try:
            config = ApiConfig(name, base_url, model, key_env, int(shots))
            _ = config.endpoint
            if config.shots < 0:
                raise ValueError("SHOTS 必须是非负整数")
        except ValueError as exc:
            parser.error(str(exc))
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in names:
            parser.error("API NAME 必须唯一，且只能包含字母、数字、下划线或连字符")
        configs.append(config)
        names.add(name)
    if args.api_only and not configs:
        parser.error("--api-only 需要至少一个 --api")
    return args, configs


def api_stats(rows):
    stats = [row["diagnostics"] for row in rows]
    status = Counter(item["status"] for item in stats)
    report = {
        "valid_output_rate": status["ok"] / len(rows),
        "status": dict(status),
        "request_errors": dict(Counter(item["error"] for item in stats if "error" in item)),
        "finish_reasons": dict(Counter(item["finish_reason"] for item in stats if item.get("finish_reason"))),
        "response_models": sorted({item["response_model"] for item in stats if item.get("response_model")}),
    }
    for key in ("wrapped_json", "unknown_type", "unknown_role", "unmatched_spans"):
        report[key] = sum(item.get(key, 0) for item in stats)
    for key in ("prompt_tokens", "completion_tokens"):
        tokens = [item[key] for item in stats if isinstance(item.get(key), int)]
        report[key] = {"sum": sum(tokens), "samples_with_usage": len(tokens)} if tokens else None
    return report


def run_method(name, setup, dev, out_dir, config=None):
    print(f"  {name} · {len(dev)} 条", flush=True)
    started = perf_counter()
    model_info = {"model": config.model, "base_url": config.base_url, "shots": config.shots} if config else {}
    try:
        extractor = setup()
    except (ImportError, OSError, ValueError) as exc:
        return {**model_info, "status": "failed", "error": str(exc), "metrics": None}

    setup_seconds = perf_counter() - started
    predictions = []
    for row in dev:
        started = perf_counter()
        if config:
            events, diagnostics = extractor.predict(row["text"])
        else:
            events, diagnostics = extractor.predict(row["text"]), {"status": "ok"}
        predictions.append({
            "uid": row["uid"], "text": row["text"], "gold": row["event"], "events": events,
            "latency_ms": (perf_counter() - started) * 1000,
            "diagnostics": diagnostics,
        })

    with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    latencies = sorted(row["latency_ms"] for row in predictions)
    result = {
        **model_info,
        "status": "ok", "setup_seconds": setup_seconds,
        "latency_ms_mean": mean(latencies),
        "latency_ms_p95": latencies[(95 * len(latencies) + 99) // 100 - 1],
    }
    if config:
        result["api"] = api_stats(predictions)
        errors = result["api"]["status"].get("api_error", 0)
        if errors:
            result["status"] = "failed" if errors == len(dev) else "partial"
    result["metrics"] = evaluate(dev, predictions, verbose=False) if result["status"] != "failed" else None
    return result


def markdown(report):
    lines = ["# 事件抽取对比", "", f"采样 seed = {report['seed']}。F1 与合法输出率均为百分比。"]
    for dataset, experiment in report["datasets"].items():
        manifest = experiment["data"]
        lines.extend([
            "", f"## {dataset}", "",
            f"训练 {manifest['train_used']} 条／评测 {manifest['dev_used']} 条。",
            "",
            "| 方法 | 状态 | 触发词 F1 | 论元 F1 | 事件完全匹配 F1 | 类型 Macro-F1 | 平均 ms/条 | P95 ms/条 | JSON 结构合法率 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for name, result in experiment["methods"].items():
            metrics = result["metrics"]
            scores = ["—"] * 4
            if metrics:
                scores = [f"{100 * metrics[key]['f1']:.2f}" for key in ("trigger", "argument", "event_exact")]
                scores.append(f"{100 * metrics['type_macro_f1']:.2f}")
            api = result.get("api")
            valid = f"{100 * api['valid_output_rate']:.1f}%" if api else "—"
            latency = [f"{result[key]:.1f}" if key in result else "—"
                       for key in ("latency_ms_mean", "latency_ms_p95")]
            lines.append("| " + " | ".join([name, result["status"], *scores, *latency, valid]) + " |")
        for name, result in experiment["methods"].items():
            errors = result.get("error") or result.get("api", {}).get("request_errors")
            if errors:
                lines.extend(["", f"- {name} 请求或初始化失败原因为 {errors}。"])
    return "\n".join(lines) + "\n"


def main():
    args, configs = options()
    report = {
        "seed": args.seed, "train_n": args.train_n, "eval_n": args.eval_n,
        "api_max_tokens": args.max_tokens, "api_timeout": args.api_timeout, "datasets": {},
    }
    failed = False
    for dataset in dict.fromkeys(args.datasets):
        try:
            schema, train, dev, manifest = load_dataset(
                dataset, args.duee_archive, args.train_n, args.eval_n, args.seed,
            )
        except (OSError, ValueError) as exc:
            print(f"{dataset} 数据准备失败：{exc}", file=sys.stderr)
            return 1
        print(f"{dataset} · 训练 {len(train)} 条 / 评测 {len(dev)} 条", flush=True)
        out_dir = args.output_dir / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        if not args.api_only:
            for name in dict.fromkeys(args.methods):
                if name == "lexicon":
                    results[name] = run_method(name, lambda: LexiconExtractor(train, schema), dev, out_dir)
                else:
                    results[name] = run_method(name, lambda: CRFExtractor(train, schema, args.seed), dev, out_dir)
        if configs:
            prompts = Prompts(schema, train, max(config.shots for config in configs))
            for config in configs:
                results[config.name] = run_method(
                    config.name,
                    lambda: ChatExtractor(config, prompts, schema, args.api_timeout, args.max_tokens),
                    dev, out_dir, config,
                )
        failed |= any(result["status"] != "ok" for result in results.values())
        report["datasets"][dataset] = {"data": manifest, "methods": results}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    summary = markdown(report)
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
