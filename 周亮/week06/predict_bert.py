"""
方法三（推理/评测）：加载 BERT checkpoint，在独立测试集上报准确率与 Macro-F1

参照课程 code/bert/predict_bert.py，适配本作业的 ckpt 与 jsonl 结构。

教学重点：
  1. **评测配置必须从 checkpoint 里读，不能靠人手动传。**
     训练时用了 --use_title（拼 title）、--max_length 192、--pool mean，
     评测时少传一个，指标就会静默掉几个点且不报错——这是这类脚本最贵的坑。
     本脚本把 pool / max_length / use_title 全部从 ckpt["args"] 恢复并打印。
  2. **必须和训练用同一个指标函数**（metrics.classification_report）。
     主指标是 Macro-F1，Accuracy 只作参考：本任务「中」类占 55.3%，
     闭眼全猜「中」就是 0.553 准确率但 Macro-F1 只有 0.238。
  3. **对账机制**：如果 logs/{run_name}.json 里已经有训练时算出的 test 指标，
     本脚本用 ckpt 独立重算一遍并逐项比对。完全相等说明「ckpt + 数据 + 预处理」
     三者一致；不相等就是预处理漂移的哨兵，必须查清才能用这个数写报告。
  4. 顺手测延迟：tokenize 耗时 / 前向 ms 条（batch）/ 单条 ms。
     老师实测单条 vs batch=32 差 3 倍多——「模型多大」不是成本，「能不能批处理」才是。

用法：
  python predict_bert.py                                  # 批量评测 checkpoints/ 下所有跑，出汇总表
  python predict_bert.py --ckpt checkpoints/xxx/best.pt   # 只评一个
  python predict_bert.py --data_file data/val_2k.jsonl    # 换评测集
  python predict_bert.py --latency_single 20              # 额外测单条延迟（默认不测）

产出（默认 outputs/eval，Colab 上随 WEEK6_OUT_ROOT 落到 Drive）：
  {run_name}__{tag}.json          单跑完整指标（含对账结果）
  {run_name}__{tag}_preds.jsonl   逐条预测：id/title/text/true/pred/conf/probs，供错误分析
  summary_{tag}.md / .json        所有跑的横向汇总表（贴报告用）；
                                  md 的「方法」列是中文说明（如「CLS 池化（基线）」
                                  「Mean 池化 + Focal γ=2」，只列与基线的差异），
                                  表后附「中文说明 ←→ run_name」对照表保证可追溯；
                                  json 的键仍是 run_name（机器可读，调度器对账用）
"""

import argparse
import json
import logging
import re
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paths                                                    # noqa: E402
from dataset import (BertDataset, build_text, load_jsonl,        # noqa: E402
                     load_label_map, make_collate_fn, stratified_subset)
from metrics import (classification_report, describe_bert_run,     # noqa: E402
                     format_confusion, pred_distribution, to_markdown_table)
from model_bert import BertClassifier                            # noqa: E402
from train_bert import ensure_local_model                        # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# 推理
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, dl, device):
    """返回 (y_true, y_pred, probs, 纯前向耗时秒)。probs 用于 ECE 与低置信转人工。"""
    model.eval()
    ys, ps, probs = [], [], []
    fwd_s = 0.0
    for batch in dl:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        tt = batch.get("token_type_ids")
        tt = tt.to(device) if tt is not None else None
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(ids, mask, tt)
        if device == "cuda":
            torch.cuda.synchronize()          # GPU 异步执行，不同步计时只有几十微秒
        fwd_s += time.perf_counter() - t0
        p = torch.softmax(logits.float(), dim=-1)
        ps.extend(p.argmax(-1).cpu().tolist())
        probs.extend(p.cpu().tolist())
        ys.extend(batch["labels"].tolist())
    return ys, ps, probs, fwd_s


@torch.no_grad()
def latency_single(model, tok, texts, device, max_length, n):
    """逐条推理（含 tokenize）的中位耗时，用来复现老师「单条 vs batch」的对照"""
    model.eval()
    times = []
    for t in texts[:n]:
        t0 = time.perf_counter()
        enc = tok(t, truncation=True, max_length=max_length, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        tt = enc.get("token_type_ids")
        tt = tt.to(device) if tt is not None else None
        if device == "cuda":
            torch.cuda.synchronize()
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            model(ids, mask, tt)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 1) if times else None


# ─────────────────────────────────────────────────────────────────────────────
# 对账：与训练时写入 logs/{run_name}.json 的指标逐项比对
# ─────────────────────────────────────────────────────────────────────────────

def _opt_str(v):
    """ckpt 里的可选字符串参数：旧版训练脚本用 str(v) 序列化，会把 None 存成 'None'，
    字符串 'None' 是真值——不处理的话「是否指定了 test_file」会被误判为「指定了」。"""
    if v is None:
        return None
    s = str(v).strip()
    return None if s in ("", "None", "null", "False") else s


def _opt_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def reconcile(rep, run_name, split_tag, log_dir, data_file, cargs):
    """
    训练脚本已经在最佳 ckpt 上算过一次 test 指标并存进 logs/{run_name}.json。
    这里用 ckpt 独立重算，两者必须完全相等 —— 不等就说明有预处理漂移
    （最典型：训练拼了 title、评测没拼；或 max_length 不一致）。
    """
    log_json = Path(log_dir) / f"{run_name}.json"
    if not log_json.exists():
        return {"checked": False, "reason": f"未找到 {log_json.name}"}
    try:
        train_log = json.loads(log_json.read_text(encoding="utf-8"))
    except Exception as e:                                  # noqa: BLE001
        return {"checked": False, "reason": f"解析失败: {e}"}

    train_test_file = _opt_str(cargs.get("test_file"))
    max_eval = _opt_int(cargs.get("max_eval"))

    if "test" in split_tag and train_log.get("test"):
        ref = train_log["test"]
        ref_desc = "test"
        if train_test_file and Path(train_test_file).name != data_file.name:
            return {"checked": False,
                    "reason": f"训练用的是 {Path(train_test_file).name}，本次是 {data_file.name}"}
    elif "val" in split_tag and train_log.get("epochs"):
        if max_eval:
            return {"checked": False,
                    "reason": f"训练时用了 --max_eval {max_eval}（val 只评了子集），"
                              f"无法与全量 val 对账"}
        best = [e for e in train_log["epochs"] if e.get("is_best")]
        if not best:
            return {"checked": False, "reason": "logs 里没有 is_best 的 epoch"}
        ref, ref_desc = best[-1]["val"], f"epoch {best[-1]['epoch']} val"
    else:
        return {"checked": False, "reason": "logs 里没有可比对的指标"}

    diffs = {}
    for k in ("acc", "macro_f1"):
        d = round(float(rep.get(k, 0)) - float(ref.get(k, 0)), 6)
        if abs(d) > 1e-9:
            diffs[k] = {"ours": rep.get(k), "train_log": ref.get(k), "delta": d}
    return {"checked": True, "against": ref_desc, "train_log": log_json.name,
            "match": not diffs, "diffs": diffs}


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="BERT severity 三分类评测（准确率 + Macro-F1）")
    ap.add_argument("--ckpt", default=None,
                    help="单个 ckpt 文件或目录（目录取其中 best.pt）；不给则批量评测 ckpt_dir 下所有跑")
    ap.add_argument("--ckpt_dir", default=None, type=Path,
                    help=f"默认 {paths.CKPT_DIR}")
    ap.add_argument("--include_smoke", action="store_true",
                    help="批量模式默认跳过文件名带 smoke 的冒烟跑")
    ap.add_argument("--skip_done", action="store_true",
                    help="批量模式跳过已有 {run}__{tag}.json 的 ckpt（colab/run_predict.sh 默认开启）")
    ap.add_argument("--data_file", default=None, type=Path,
                    help="评测集，默认 data_dir/test_3k.jsonl")
    ap.add_argument("--data_dir", default=None, type=Path)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out_root", default=None, type=Path,
                    help="产物根目录（与训练保持一致；Colab 上指向 Drive）")
    ap.add_argument("--out_dir", default=None, type=Path, help="评测结果目录，默认 out_root/outputs/eval")
    ap.add_argument("--log_dir", default=None, type=Path, help="训练日志目录（用于对账）")
    ap.add_argument("--n", type=int, default=0,
                    help="只评前 N 条（等距抽样），0=全量；仅用于快速排查，报告必须用全量")
    ap.add_argument("--latency_single", type=int, default=0,
                    help="额外测多少条单条推理延迟，0=不测（CPU 上单条很慢）")
    ap.add_argument("--top_errors", type=int, default=5,
                    help="控制台展示几条「高置信但判错」的样本，0=不展示")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return ap.parse_args(argv)


def discover_ckpts(args, ckpt_dir: Path, out_dir: Path | None = None, tag: str | None = None):
    if args.ckpt:
        # 显式指定的 ckpt 永远评，不受 skip_done 影响
        p = Path(args.ckpt)
        if p.is_dir():
            p = p / "best.pt"
        if not p.exists():
            raise FileNotFoundError(f"找不到 checkpoint: {p}")
        return [p]
    found = sorted(ckpt_dir.glob("*/best.pt"))
    if not args.include_smoke:
        found = [p for p in found if "smoke" not in p.parent.name]
    if getattr(args, "skip_done", False) and out_dir is not None and tag:
        found = [p for p in found if not (out_dir / f"{p.parent.name}__{tag}.json").exists()]
    return found


def short_name(run_name: str) -> str:
    """汇总表里去掉各消融组共有的尾部参数（n/bs/lr/ep/seed），只留真正的消融维度。
    10+ 行的表格用全名会宽到读不了；完整名字始终保留在 json 里。"""
    s = re.sub(r"_n-\d+_bs-\d+_lr-[^_]+_ep-\d+_seed-\d+.*$", "", run_name)
    return s or run_name


def setup_logger(out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("predict_bert")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / f"predict_{tag}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def main(argv=None):
    args = parse_args(argv)
    from transformers import AutoTokenizer

    out_root = Path(args.out_root) if args.out_root else paths.OUT_ROOT
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else out_root / "checkpoints"
    log_dir = Path(args.log_dir) if args.log_dir else out_root / "logs"
    out_dir = Path(args.out_dir) if args.out_dir else out_root / "outputs" / "eval"
    data_dir = Path(args.data_dir) if args.data_dir else paths.DATA_DIR
    data_file = Path(args.data_file) if args.data_file else data_dir / "test_3k.jsonl"

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if not data_file.exists():
        raise FileNotFoundError(f"评测集不存在: {data_file}")
    tag = data_file.stem                                   # test_3k / val_2k / 自定义名

    name2id, id2name, _ = load_label_map(data_dir)

    ckpts = discover_ckpts(args, ckpt_dir, out_dir, tag)
    logger = setup_logger(out_dir, tag)
    logger.info(f"device={device} | 评测集={data_file}"
                f"（{sum(1 for _ in open(data_file, encoding='utf-8'))} 行）"
                f" | 待评测 ckpt {len(ckpts)} 个 | ckpt_dir={ckpt_dir}"
                + (" | 已评过的自动跳过（--skip_done）" if args.skip_done else ""))
    if not ckpts:
        all_ck = sorted(ckpt_dir.glob("*/best.pt"))
        if all_ck and args.skip_done:
            logger.warning(f"{ckpt_dir} 下 {len(all_ck)} 个 ckpt 都已有 {tag} 评测结果"
                           f"（或被冒烟过滤）；要强制重评直接跑 predict_bert.py 不加 --skip_done")
        elif all_ck and not args.include_smoke:
            logger.warning(f"{ckpt_dir} 下有 {len(all_ck)} 个 ckpt，但都是冒烟跑"
                           f"（文件名带 smoke）已被跳过；需要时加 --include_smoke")
        else:
            logger.warning(f"{ckpt_dir} 下没有 best.pt，先跑 train_bert.py（不要加 --no_save）")
        return 0

    rows_cache = load_jsonl(data_file)                      # 原始行，错误分析要 id/title
    if args.n and args.n < len(rows_cache):
        rows_cache = stratified_subset(rows_cache, args.n)
        logger.warning(f"--n {args.n}：等距抽样到 {len(rows_cache)} 条"
                       f"（此指标仅用于快速排查，写报告必须去掉 --n 跑全量）")

    tok_cache, model_cache, ds_cache = {}, {}, {}
    summary, results_md = {}, {}

    for idx, ckpt_path in enumerate(ckpts, 1):
        run_name = ckpt_path.parent.name
        logger.info("=" * 78)
        logger.info(f"[{idx}/{len(ckpts)}] {run_name}")

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        cargs = ck.get("args", {}) or {}
        # ── 关键：评测配置从 ckpt 恢复（见文件头教学重点 1）──
        pool = ck.get("pool") or cargs.get("pool", "cls")
        max_length = int(cargs.get("max_length", 192))
        use_title = str(cargs.get("use_title", "False")).lower() in ("true", "1")
        model_name = str(cargs.get("model_name_or_path",
                                   "iic/nlp_roberta_backbone_base_std"))
        logger.info(f"  从 ckpt 恢复配置: pool={pool} max_length={max_length} "
                    f"use_title={use_title} | 训练时 val_macro_f1={ck.get('val_macro_f1')}")

        if model_name not in model_cache:
            model_dir = ensure_local_model(model_name, logger)
            tok_cache[model_name] = AutoTokenizer.from_pretrained(str(model_dir))
            model_cache[model_name] = model_dir
        tok, model_dir = tok_cache[model_name], model_cache[model_name]

        ds_key = (data_file, use_title, max_length)
        if ds_key not in ds_cache:
            texts = [build_text(r, use_title) for r in rows_cache]
            labels = [r["label_id"] for r in rows_cache]
            bad = [r["id"] for r in rows_cache if r["label_id"] != name2id[r["label"]]]
            if bad:
                raise ValueError(f"{len(bad)} 行 label_id 与 label_map 不一致"
                                 f"（例：{bad[:3]}），拒绝评测")
            t_tok = time.perf_counter()
            ds = BertDataset(texts, labels, tok, max_length)
            ds_cache[ds_key] = (ds, texts, labels, time.perf_counter() - t_tok)
        ds, texts, _, tok_s = ds_cache[ds_key]

        model = BertClassifier(model_dir, num_labels=len(name2id), pool=pool).to(device)
        model.load_state_dict(ck["state_dict"], strict=True)   # strict：权重对不上直接报错
        model.eval()

        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=make_collate_fn(tok))
        n = len(ds)
        t0 = time.perf_counter()
        ys, yp, probs, fwd_s = run_inference(model, dl, device)
        e2e_s = time.perf_counter() - t0

        rep = classification_report(ys, yp, probs=probs, id2name=id2name)
        rep["latency"] = {
            "batch_size": args.batch_size,
            "tokenize_ms_per_sample": round(tok_s * 1000 / n, 3),
            "forward_ms_per_sample": round(fwd_s * 1000 / n, 3),
            "e2e_ms_per_sample": round(e2e_s * 1000 / n, 3),
        }
        if args.latency_single:
            single = latency_single(model, tok, texts, device, max_length, args.latency_single)
            rep["latency"]["single_ms_per_sample"] = single
            if single:
                rep["latency"]["batch_speedup"] = round(
                    single / max(1e-9, rep["latency"]["forward_ms_per_sample"]), 1)
        rep["pred_distribution"] = pred_distribution(yp, id2name)
        if args.n:
            # 抽样评测的指标本来就与全量对不上，不能拿去对账（否则每次抽样都误报漂移）
            rep["reconcile"] = {"checked": False, "reason": "--n 抽样评测，指标不与训练日志对账"}
        else:
            rep["reconcile"] = reconcile(rep, run_name, tag, log_dir, data_file, cargs)
        rep["ckpt"] = str(ckpt_path)
        rep["eval_file"] = str(data_file)
        rep["config"] = {"pool": pool, "max_length": max_length, "use_title": use_title,
                         "val_macro_f1": ck.get("val_macro_f1")}

        # ── 控制台：准确率与 Macro-F1 放最前面（本脚本的核心交付）──
        logger.info(f"  准确率(acc) = {rep['acc']:.4f}   Macro-F1 = {rep['macro_f1']:.4f}   "
                    f"weighted-F1 = {rep['weighted_f1']:.4f}   ECE = {rep.get('ece', '-')}")
        for name, v in rep["per_class"].items():
            if v:
                logger.info(f"    {name}: P={v['P']:.4f} R={v['R']:.4f} "
                            f"F1={v['F1']:.4f} (n={v['support']})")
        logger.info("  混淆矩阵（行=真实，列=预测）:\n" + format_confusion(rep, id2name))
        logger.info(f"  预测分布: {rep['pred_distribution']['counts']}")
        lat = rep["latency"]
        extra = (f" | 单条 {lat['single_ms_per_sample']} ms"
                 f"（batch 加速 {lat.get('batch_speedup')}x）"
                 if lat.get("single_ms_per_sample") else "")
        logger.info(f"  延迟: tokenize {lat['tokenize_ms_per_sample']} ms/条 | "
                    f"前向 {lat['forward_ms_per_sample']} ms/条（batch {args.batch_size}）{extra}")

        rc = rep["reconcile"]
        if rc.get("checked"):
            if rc["match"]:
                logger.info(f"  对账 ✓ 与训练日志 {rc['train_log']} 的 {rc['against']} 完全一致"
                            f"（acc={rep['acc']} macro_f1={rep['macro_f1']}）")
            else:
                logger.warning(f"  对账 ✗ 与训练日志 {rc['train_log']} 的 {rc['against']} 不一致："
                               f"{rc['diffs']} —— 检查 use_title / max_length / 数据文件是否漂移")
        else:
            logger.info(f"  对账 - 跳过（{rc.get('reason')}）")

        # ── 高置信错误：模型很有把握却判错，是最值得看的 bad case ──
        errors = []
        for i, (yt, yp_i, p) in enumerate(zip(ys, yp, probs)):
            if yt != yp_i:
                errors.append({"id": rows_cache[i]["id"], "title": rows_cache[i]["title"],
                               "true": id2name[yt], "pred": id2name[yp_i],
                               "conf": round(max(p), 4)})
        errors.sort(key=lambda e: -e["conf"])
        rep["n_errors"] = len(errors)
        if args.top_errors and errors:
            logger.info(f"  高置信错误 Top{min(args.top_errors, len(errors))}"
                        f"（共 {len(errors)} 条错分）:")
            for e in errors[:args.top_errors]:
                logger.info(f"    真={e['true']} 判={e['pred']} 置信={e['conf']:.3f} "
                            f"| {e['id']} | {e['title'][:34]}")

        # ── 落盘：单跑指标 + 逐条预测（供错误分析与三路方法对照）──
        out_json = out_dir / f"{run_name}__{tag}.json"
        out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        preds_path = out_dir / f"{run_name}__{tag}_preds.jsonl"
        with open(preds_path, "w", encoding="utf-8") as f:
            for i, (yt, yp_i, p) in enumerate(zip(ys, yp, probs)):
                f.write(json.dumps({"id": rows_cache[i]["id"], "title": rows_cache[i]["title"],
                                    "text": texts[i], "true": id2name[yt],
                                    "pred": id2name[yp_i], "correct": yt == yp_i,
                                    "conf": round(max(p), 4),
                                    "probs": [round(x, 4) for x in p]},
                                   ensure_ascii=False) + "\n")

        summary[run_name] = rep
        # 汇总表「方法」列用中文说明（如「CLS 池化（基线）」「Mean 池化 + Focal γ=2」），
        # 只列与基线的差异；对照表附在 summary 末尾，保证报告里的行可回溯到 run_name。
        # 只改 seed 这类描述不出来的差异时，退回 short_name 防止两行中文说明撞名丢行。
        display = describe_bert_run(run_name)
        if display in results_md:
            display = f"{display}（{short_name(run_name)}）"
        results_md[display] = rep

    # ── 横向汇总表（贴进作业报告）──
    # ECE 表头写明族：BERT 的 conf 是 3 类 softmax 的绝对概率，与 LLM 的
    # 「三选一条件概率」不是一个事件，两张表拼起来时不能当同一列读。
    md = to_markdown_table(results_md, ece_header="ECE·BERT")
    # 名称对照：中文说明 ←→ run_name（文件名即配置，机器可读）
    mapping = ["", "## 名称对照（中文说明 ←→ run_name）", "",
               "| 中文说明 | run_name（完整配置）|", "|---|---|"]
    for run_name in summary:
        mapping.append(f"| {describe_bert_run(run_name)} | `{run_name}` |")
    mapping += ["", "> ECE·BERT 只在本表内可比（同为 BERT 族、同为 3 类 softmax 的绝对概率）。"
                    "与 LLM 结果的 ECE 不可横向比较——那边是「三选一」条件下的重归一化概率，"
                    "词表外质量被丢弃，分母口径不同。", ""]
    md += "\n".join(mapping)
    logger.info("=" * 78)
    logger.info(f"汇总（{tag}，{len(summary)} 个跑）:\n{md}")
    (out_dir / f"summary_{tag}.md").write_text("# " + tag + " 评测汇总\n\n" + md + "\n",
                                               encoding="utf-8")
    (out_dir / f"summary_{tag}.json").write_text(
        json.dumps({k: {kk: v[kk] for kk in ("acc", "macro_f1", "weighted_f1", "ece",
                                             "per_class", "config", "n_errors")
                        if kk in v}
                    for k, v in summary.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    logger.info(f"→ 汇总写入 {out_dir}/summary_{tag}.md 与 .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
