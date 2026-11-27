"""
统一评测 + 计时/资源统计的公共封装（本项目的所有方法都从这里出数字）

为什么单独封一层：
  1. 评测口径必须唯一。直接 import common/evaluate_events.py（Lead 维护、只读），
     而不是每个脚本自己再算一遍 P/R/F1 —— 否则「谁的方法更好」不可信。
  2. 结果 JSON 里必须带上**溯源字段**（方法名、超参、训练时长、延迟、设备），
     否则做 PPT 图表时很容易出现「图注写 3 epoch、实际跑了 5 epoch」这种自相矛盾。
     （这是 项目材料准备/CLAUDE.md 踩坑表里明确记过的一类问题）
  3. 预测文件路径固定 norms：outputs/disc/pred_<方法>.jsonl，指标固定 outputs/metrics/disc/<方法>.json

使用方式：
  from eval_utils import evaluate_predictions, Timer
  with Timer() as t:
      preds = run_inference(...)
  res = evaluate_predictions("bert_crf", preds, meta={"epochs": 3})
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows: torch 与 numpy 各带 libiomp5md，不设会 OMP Error #15

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
COMMON_DIR = ROOT / "common"
DATA_DIR = ROOT / "data" / "processed"
PRED_DIR = ROOT / "outputs" / "disc"
METRIC_DIR = ROOT / "outputs" / "metrics" / "disc"
LOG_DIR = ROOT / "outputs" / "logs"
CKPT_DIR = ROOT / "outputs" / "checkpoints"

# 把 common/ 加进 import 路径，直接复用唯一口径的评测函数（只读，不修改）
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))


def _import_evaluator():
    """延迟导入：保证脚本在没有 common/ 时给出清楚报错，而不是 ImportError 堆栈。"""
    try:
        from evaluate_events import as_event_list, evaluate, load_jsonl  # noqa
        return as_event_list, evaluate, load_jsonl
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"找不到统一评测模块：{COMMON_DIR / 'evaluate_events.py'}\n原始错误：{exc}"
        ) from exc


def save_predictions(rows: list[dict], name: str) -> Path:
    """写预测文件到 outputs/disc/pred_<name>.jsonl。"""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"pred_{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def evaluate_nested_subset(name: str, pred_rows: list[dict], verbose: bool = False) -> dict | None:
    """
    把全量预测**投影到固定抽样集**上再评一次，用于和 LLM 链路同批对比。

    为什么需要：本项目有两条链路（判别式 / LLM），LLM 那边为了控成本只在固定随机抽的
    200 条上评测。如果判别式方法报 930 条全量、LLM 报 200 条抽样，两个数字并排放进 PPT
    会被质疑「样本不同不可比」。所以所有方法都额外报一份 n=200 子集指标。

    注意：抽样规则由 llm-extract 固定在 outputs/llm/eval_ids.json（dev 排序后 shuffle(seed=42) 取前 200），
    是**嵌套子集**（n=150/n=100 是同一个顺序的前缀），保证不同 n 之间可比。
    """
    n200 = ROOT / "outputs" / "llm" / "gold_n200.jsonl"
    if not n200.exists():
        return None
    from evaluate_events import evaluate, row_key
    as_event_list, _, load_jsonl = _import_evaluator()
    gold = load_jsonl(str(n200))
    keys = {row_key(g) for g in gold}
    sub = [p for p in pred_rows if row_key(p) in keys]
    res = evaluate(gold, sub, verbose=False)
    res["method"] = name
    res["gold"] = str(n200)
    res["note"] = "投影到 llm-extract 固定的 dev 200 条抽样（seed=42），用于跨链路同批对比"
    out = METRIC_DIR / f"{name}_n200.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"  n200 子集：trigger F1={res['trigger']['f1']:.4f} "
              f"arg F1={res['argument']['f1']:.4f} → {out.name}")
    return res


GOLD_BASELINE = METRIC_DIR / "bio_gold_baseline.json"


def attach_gold_baseline(meta: dict) -> None:
    """
    把「金标自身的非法 BIO 地板值」挂到指标的 meta 里。

    为什么必须一起报：非法序列数不能和 0 比 —— 金标因重叠论元投影到 BIO 天生就有约 1.9% 非法。
    有了这个字段，PPT 上才能写成「金标 18 处(1.9%) → linear 1516 处(67%) → CRF ? 处」，
    而不是把 67% 全归咎于模型、也不是误以为任务本身非法。
    """
    if not GOLD_BASELINE.exists():
        return
    try:
        with open(GOLD_BASELINE, "r", encoding="utf-8") as f:
            base = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    dev = (base.get("splits") or {}).get("dev") or {}
    if dev:
        meta["illegal_gold_dev"] = dev.get("illegal")
        meta["overlap_conflicts_gold_dev"] = dev.get("overlap_conflicts")


def evaluate_predictions(name: str, pred_rows: list[dict], meta: dict | None = None,
                         gold_path=None, with_detect: bool = False,
                         verbose: bool = True) -> dict:
    """
    对一批预测算全套指标并落盘 JSON。

    参数：
      name        方法名（同时用作 pred_<name>.jsonl / <name>.json 的文件名）
      pred_rows   预测行（每行必须带 id + sent_id + events）
      meta        溯源信息（超参/训练时长/延迟等），原样写进结果 JSON
      gold_path   gold 文件，默认 data/processed/dev_extract.jsonl
      with_detect 额外算句子级「有没有事件」的二分类指标（需要 dev_detect.jsonl）
    """
    as_event_list, evaluate, load_jsonl = _import_evaluator()

    gold_path = Path(gold_path) if gold_path else DATA_DIR / "dev_extract.jsonl"
    gold_rows = load_jsonl(str(gold_path))
    pred_path = save_predictions(pred_rows, name)

    res = evaluate(gold_rows, pred_rows, verbose=verbose)

    # 额外算 detection（两段式的第一段要用；和 CLI --detect 的口径完全一致）
    if with_detect:
        gold_detect = gold_path.with_name("dev_detect.jsonl")
        if gold_detect.exists():
            from evaluate_events import detection_metrics, row_key
            g_rows = load_jsonl(str(gold_detect))
            known = {row_key(r) for r in pred_rows}
            p_rows = list(pred_rows)
            # 预测文件里没有的句子 = 判成「无事件」，不删分母
            for g in g_rows:
                if row_key(g) not in known:
                    p_rows.append({"uid": row_key(g), "has_event": 0})
            res["detection"] = detection_metrics(g_rows, p_rows)
            if verbose:
                d = res["detection"]
                print(f"  detection    P={d['precision']:.4f} R={d['recall']:.4f} "
                      f"F1={d['f1']:.4f} acc={d['accuracy']:.4f}")

    # 溯源字段：方法名 + 预测文件 + 自定义 meta（不要硬编码超参到图里，读这里）
    if meta and "illegal_dev" in meta:
        attach_gold_baseline(meta)   # 顺带挂上金标非法地板值，供 P25 对比
    res["method"] = name
    res["gold"] = str(gold_path)
    res["pred"] = str(pred_path)
    if meta:
        res["meta"] = meta

    # 顺带产出 n=200 子集指标（跨链路同批对比用）；抽样集还没生成时静默跳过。
    # 主 JSON 里只存一个精简摘要，完整子集结果在单独的 <name>_n200.json 里。
    sub = evaluate_nested_subset(name, pred_rows, verbose=verbose)
    if sub:
        res["n200_summary"] = {
            "trigger_f1": sub["trigger"]["f1"],
            "argument_f1": sub["argument"]["f1"],
            "event_f1": sub["event_exact"]["f1"],
            "num_samples": sub["num_samples"],
        }

    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    out = METRIC_DIR / f"{name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"  指标已写入 → {out}")
    return res


class Timer:
    """
    计时 + 峰值显存统计（with 语句用）。

    为什么要在指标表里带延迟：事件抽取的下游是「公告实时监控」，
    一个慢 10 倍但只高 2 个点的模型在生产里通常不可接受。
    PPT 上的「延迟/成本」表就是靠这里测出来的真实数字。
    """

    def __enter__(self):
        self.t0 = time.perf_counter()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
        self.peak_gpu_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                self.peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        except Exception:
            pass
        return False

    def summary(self, n_samples: int) -> dict:
        """换算成「每句毫秒」和「每秒句数」，PPT 里直接引用。"""
        n = max(int(n_samples), 1)
        return {
            "total_s": round(self.elapsed, 2),
            "ms_per_sentence": round(self.elapsed * 1000.0 / n, 2),
            "sentences_per_s": round(n / self.elapsed, 1) if self.elapsed > 0 else 0.0,
            "peak_gpu_mb": round(self.peak_gpu_mb, 1),
        }


def write_method_table(names: list[str], out_md: Path | None = None) -> list[dict]:
    """
    汇总所有方法的中文对比表（读各自的结果 JSON），供 PPT / NOTES 直接引用。

    只收录**存在且字段完整**的结果：缺文件就跳过并提示，不编造数字。
    """
    rows = []
    for name in names:
        p = METRIC_DIR / f"{name}.json"
        if not p.exists():
            print(f"[warn] 缺少指标文件，跳过：{p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            r = json.load(f)
        meta = r.get("meta", {})
        rows.append({
            "方法": meta.get("display_name", name),
            "trigger_F1": r["trigger"]["f1"],
            "argument_F1": r["argument"]["f1"],
            "event_F1": r["event_exact"]["f1"],
            "type_macro_F1": r.get("type_macro_f1", 0.0),
            "训练时长(s)": meta.get("train_s", ""),
            "参数量(M)": meta.get("params_m", ""),
            "延迟(ms/句)": meta.get("ms_per_sentence", ""),
        })

    if out_md:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        header = "| " + " | ".join(rows[0].keys()) + " |" if rows else ""
        sep = "|" + "|".join(["---"] * len(rows[0])) + "|" if rows else ""
        body = "\n".join("| " + " | ".join(str(v) for v in r.values()) + " |" for r in rows)
        out_md.write_text(f"{header}\n{sep}\n{body}\n", encoding="utf-8")
        print(f"对比表已写入 → {out_md}")
    return rows


if __name__ == "__main__":
    # 自检：gold 转 pred 必须全 1.0（用来验证预测键带了 sent_id，没被同文档多句覆盖）
    as_event_list, evaluate, load_jsonl = _import_evaluator()
    gold = load_jsonl(str(DATA_DIR / "dev_extract.jsonl"))
    preds = []
    for r in gold:
        evs = [r["event"]] if r.get("event") else (r.get("events") or [])
        preds.append({"id": r["id"], "sent_id": r["sent_id"], "events": evs})
    res = evaluate(gold, preds, verbose=False)
    print("gold→pred 自检（必须全 1.0000）：")
    for k in ("trigger", "argument", "event_exact"):
        print(f"  {k:12s} F1={res[k]['f1']:.4f}")
    ok = all(res[k]["f1"] == 1.0 for k in ("trigger", "argument", "event_exact"))
    print("自检", "PASS" if ok else "FAIL —— 预测键没带 sent_id or 格式不对")
