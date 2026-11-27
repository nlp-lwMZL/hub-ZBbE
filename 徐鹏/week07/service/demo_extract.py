"""
篇章级事件抽取现场演示脚本（课堂上直接跑，打印一篇公告的事件 JSON）

演示的叙事线：
  一篇 A 股公告原文（几百字）→ 切句 → 逐句判断有没有事件 → 抽事件 → 校验/正则兜底
  → 合并成事件流 → 低置信度标 needs_review
  最后打印：① 原始公告 ② 逐句抽取过程 ③ 最终事件流 JSON ④ 复核标记

使用方式：
  python service/demo_extract.py --n 3                     # 抽 3 篇公告演示
  python service/demo_extract.py --n 1 --doc_id <公告id>   # 指定某篇
  python service/demo_extract.py --n 2 --backend pipeline  # 用两段式后端
  python service/demo_extract.py --n 1 --mock              # 用 20 条手造数据（无需真实数据）
  python service/demo_extract.py --n 3 --out outputs/demo/doc_events.json

注意：需要先训练出 checkpoint。
  联合后端：python bert/train.py --encoder crf      → outputs/checkpoints/bert_crf_256.pt
  流水线后端：python bert_pipeline/train.py             → outputs/checkpoints/pipe_*.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

from event_data import load_docs, load_jsonl  # noqa: E402
from extract_service import ExtractService  # noqa: E402

def describe(doc: dict, result: dict) -> str:
    """把一篇的抽取结果整理成人能读的报告（PPT 现场演示直接投屏）。"""
    lines = []
    lines.append("=" * 78)
    lines.append(f"公告 ID：{result['id']}")
    lines.append(f"标题：{result['title'][:60]}")
    lines.append(f"原文：{doc['text'][:180]}{'...' if len(doc['text']) > 180 else ''}")
    lines.append(f"切句数：{result['n_sentences']}")
    lines.append("-" * 78)
    lines.append("【逐句抽取过程】")
    for s in result["sentences"]:
        if s["events"]:
            brief = "；".join(f"{e['type']}(触发词={e['trigger']['text']})" for e in s["events"])
        else:
            brief = "（无事件）"
        flag = " [!]需复核:" + ",".join(s["review_reasons"]) if s["needs_review"] else ""
        lines.append(f"  [{s['sent_idx']}] conf={s['confidence']:.2f} {brief}{flag}")
    lines.append("-" * 78)
    lines.append("【最终事件流 JSON】")
    lines.append(json.dumps(result["events"], ensure_ascii=False, indent=2))
    if result["needs_review"]:
        lines.append("[!] 整篇标记 needs_review：建议人工复核")
    else:
        lines.append("✓ 全部句子置信度达标，可直接进下游")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="篇章级事件抽取演示")
    ap.add_argument("--n", type=int, default=3, help="演示几篇公告")
    ap.add_argument("--doc_id", default="", help="指定公告 id，可用逗号分隔多个（优先级高于 --n）")
    ap.add_argument("--backend", choices=["joint", "pipeline"], default="joint")
    ap.add_argument("--joint_ckpt", default="", help="联合模型 checkpoint 路径")
    ap.add_argument("--review_threshold", type=float, default=0.55)
    ap.add_argument("--no_fallback", action="store_true", help="关闭正则兜底（看模型原始能力）")
    ap.add_argument("--no_hard", action="store_true", help="关闭 CRF 解码期硬约束（默认开）")
    ap.add_argument("--device", default="", help="cuda / cpu；留空格子时建议 cpu 做冒烟")
    ap.add_argument("--out", default="", help="把结果写到这个 json 文件")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    data_dir = ROOT / "data" / "_mock" if args.mock else ROOT / "data" / "processed"
    docs = load_docs("dev", data_dir)
    if args.doc_id:
        # 指定 id 时按给定顺序取：现场演示要挑「有多个事件、效果好」的公告，
        # 而不是第一条（可能是负样本/效果差的）
        wanted = [x.strip() for x in args.doc_id.split(",") if x.strip()]
        by_id = {d.get("id"): d for d in docs}
        docs = [by_id[w] for w in wanted if w in by_id]
        missing = [w for w in wanted if w not in by_id]
        if missing:
            print(f"[warn] 找不到这些公告 id：{missing}")
        if not docs:
            print("没有可演示的公告")
            return
    else:
        docs = docs[:args.n]

    print(f"加载模型（backend={args.backend}）...")
    svc = ExtractService(backend=args.backend, joint_ckpt=args.joint_ckpt,
                         review_threshold=args.review_threshold,
                         use_fallback=not args.no_fallback, device=args.device,
                         hard_constraint=not args.no_hard)
    svc.load()

    all_out = []
    for doc in docs:
        result = svc.extract_document(doc)
        print(describe(doc, result))
        all_out.append({"doc_id": result["id"], "title": result["title"],
                        "events": result["events"],
                        "needs_review": result["needs_review"]})

    # 与金标对照（dev_docs 自带 events）：直接看「抽没抽到、抽对几条」
    print("\n" + "=" * 78)
    print("【与金标对照】")
    for doc, out in zip(docs, all_out):
        gold = doc.get("events") or []
        got = out["events"]
        gold_types = [e.get("type") for e in gold]
        got_types = [e.get("type") for e in got]
        hit = len(set(gold_types) & set(got_types))
        print(f"  {doc['id'][:12]}… 金标 {len(gold)} 个事件 {gold_types} | "
              f"抽出 {len(got)} 个 {got_types} | 类型命中 {hit}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_out, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入 → {args.out}")

if __name__ == "__main__":
    main()
