"""
LLM 评测：zero-shot / few-shot / SFT 三模式 ×（logprob 打分 | 生成式）

参照老师 llm_zero_shot/classify_zero_shot.py 与 llm_lora/eval_lora.py，教学点原样保留：
  1. zero-shot 把分类当「生成」做——不训练、冷启动最快；但生成式总有不听话的时候，
     **解析失败率（-1）必须单独统计**，预测塌缩看 pred_distribution。
  2. few-shot：shot 只能从 train 抽（固定 seed，禁止碰 val/test），shot 清单写进产物，
     保证任何人在任何时候能复现同一个 prompt。
  3. SFT 评测必须与训练**同一个 chat 模板 + 同一个 enable_thinking=False**——
     模板统一走 dataset.render_messages，且 use_title 自动从 ckpt 的 train_args.json
     恢复，评测侧想改配置都会被拦下。
  4. 默认 logprob 打分：把「生成三选一」退化为「三个标签 token 各算一次 logit」，
     快几十倍、无解析失败；--generate 才走逐 token 生成（用于失败率统计）。

用法：
  python predict_llm.py --model qwen3-1.7b --mode zero_shot               # 不训练的冷基线
  python predict_llm.py --model qwen3-1.7b --mode few_shot --k_shots 8   # few-shot 臂
  python predict_llm.py --ckpt checkpoints/{run_name}                    # SFT 臂（自动恢复配置）
  python predict_llm.py --model qwen3-1.7b --mode zero_shot --generate --n 300
      # 生成式：看解析失败率（老师 zero-shot 的原口径），慢，只抽 300 条
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import torch

import paths
from dataset import (build_user_prompt, load_label_map, load_jsonl, render_messages,
                     render_chat_prompt, stratified_subset)
from metrics import classification_report, format_confusion, pred_distribution, to_markdown_table
from model_llm import (REGISTRY, label_first_tokens, load_base_model, load_tokenizer,
                       logprob_classify, pick_amp)


# ─────────────────────────────────────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser():
    ap = argparse.ArgumentParser(description="LLM 评测：zero-shot / few-shot / SFT")
    ap.add_argument("--mode", choices=["zero_shot", "few_shot", "sft"], default=None,
                    help="缺省按 --ckpt 是否给出自动判断（给了=sft）")
    ap.add_argument("--model", default=None,
                    help="注册表键或本地目录；sft 模式缺省从 ckpt 的 train_args.json 恢复")
    ap.add_argument("--source", default="auto", choices=["auto", "hf", "ms"],
                    help="底座下载源：auto=Colab走HuggingFace/本机走ModelScope")
    ap.add_argument("--ckpt", default=None, help="SFT 产物目录（adapter 或全量权重）")
    ap.add_argument("--k_shots", type=int, default=8, help="few-shot 的示例条数")
    ap.add_argument("--shots_file", default=None, help="缺省 data/train_5k.jsonl（shot 只准从 train 来）")
    ap.add_argument("--data_file", default=None, help="缺省 data/test_3k.jsonl")
    ap.add_argument("--n", type=int, default=0, help=">0 时等距抽 n 条（会跳过与训练日志的对账）")
    ap.add_argument("--use_title", dest="use_title", action="store_true", default=True)
    ap.add_argument("--no_title", dest="use_title", action="store_false",
                    help="sft 模式下会被训练配置覆盖并告警")
    ap.add_argument("--generate", action="store_true",
                    help="生成式评测（统计解析失败率用），默认 logprob 打分")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def parse_label(text: str, label_names: list) -> int:
    """
    生成式结果的标签解析（老师的实现）：用「包含」而不是「等于」——
    模型经常多吐标点或一句解释（"等级：高。"），严格相等会误判为失败；
    取最长匹配避免单字标签被误覆盖。抽不到返回 -1（解析失败，单独统计）。
    """
    hits = [n for n in label_names if n in text]
    return label_names.index(max(hits, key=len)) if hits else -1


def sample_shots(train_rows: list, k: int, seed: int) -> list:
    """
    few-shot 示例：按类分组后各类内部 shuffle，再轮流取 k 条——
    k>=3 时三类全覆盖；固定 seed 保证 prompt 逐字节可复现。
    """
    rng = random.Random(seed)
    by_cls = {}
    for r in train_rows:
        by_cls.setdefault(r["label"], []).append(r)
    for v in by_cls.values():
        rng.shuffle(v)
    classes = sorted(by_cls)
    shots = []
    while len(shots) < k:
        for c in classes:
            if len(shots) >= k:
                break
            if by_cls[c]:
                shots.append(by_cls[c].pop())
    return shots


def build_few_shot_messages(row: dict, shots: list, use_title: bool) -> list:
    """few-shot = system + 若干 user/assistant 示例对 + 待判样本的 user"""
    from dataset import SYSTEM_PROMPT
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for s in shots:
        msgs.append({"role": "user", "content": build_user_prompt(s, use_title)})
        msgs.append({"role": "assistant", "content": s["label"]})
    msgs.append({"role": "user", "content": build_user_prompt(row, use_title)})
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = build_arg_parser().parse_args()
    paths.ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    name2id, id2name, label_names = load_label_map(paths.DATA_DIR)
    data_path = Path(args.data_file) if args.data_file else paths.DATA_DIR / "test_3k.jsonl"
    if args.mode is None:
        args.mode = "sft" if args.ckpt else "zero_shot"
    if args.mode == "sft" and not args.ckpt:
        raise SystemExit("sft 模式需要 --ckpt（adapter 或全量权重目录）")
    train_meta = {}
    if args.mode == "sft":
        # 模型名与训练数据量从训练配置恢复：评测与训练必须同一个底座、同一份数据的口径
        ta = json.loads((Path(args.ckpt) / "train_args.json").read_text(encoding="utf-8"))
        train_meta = {"train_n": ta.get("train_n"), "train_file": ta.get("train_file"),
                      "train_sha256": ta.get("train_sha256")}
        if not args.model:
            args.model = ta["model"]
        # 无论有没有显式传 --model，use_title 都必须跟训练一致。
        # 原来只在「未传 --model」时恢复，显式传模型名时会静默用错输入口径。
        if args.use_title != ta["use_title"]:
            print(f"!! 训练时 use_title={ta['use_title']}，评测必须同口径，"
                  f"忽略命令行的 --no_title/--use_title")
            args.use_title = ta["use_title"]
        train_mode = ta.get("mode")
    else:
        train_mode = None

    tag_parts = [str(args.model).split("/")[-1], args.mode]
    if args.mode == "sft" and args.ckpt:
        # 关键：同一底座的 5k / 10k / LoRA / full 各有独立 ckpt，
        # tag 必须带上 run_name，否则后一次评测会写前一个的 _r2，汇总时被去重丢掉
        tag_parts.append(Path(args.ckpt).name)
    if args.mode == "few_shot":
        tag_parts.append(f"k{args.k_shots}")
    if args.generate:
        tag_parts.append("gen")
    tag_parts.append(data_path.stem)
    tag = "_".join(tag_parts)
    if paths.EVAL_DIR.joinpath(f"{tag}.json").exists():
        k = 2
        while paths.EVAL_DIR.joinpath(f"{tag}_r{k}.json").exists():
            k += 1
        tag = f"{tag}_r{k}"

    logger = logging.getLogger("predict_llm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(paths.EVAL_DIR / f"{tag}.log", encoding="utf-8"),
              logging.StreamHandler()):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.info(f"tag = {tag}")
    logger.info(f"device={device} mode={args.mode} generate={args.generate} | {paths.summary()}")

    # ── 数据 ──
    rows = load_jsonl(data_path)
    if args.n:
        rows = stratified_subset(rows, args.n)
    truth = [name2id[r["label"]] for r in rows]
    logger.info(f"评测 {len(rows)} 条（{data_path.name}）use_title={args.use_title}")

    # ── 模型 ──
    if args.mode == "sft":
        # 与训练同口径：QLoRA 训的 adapter 在 4bit 底座上评，LoRA 在常精度底座上评，
        # 全量微调的目录本身就是完整权重，直接加载（不再白拉一遍底座）
        ck = Path(args.ckpt)
        if (ck / "adapter_config.json").exists():
            load_mode = "qlora" if train_mode == "qlora" else "lora"
            model, tok, _ = load_base_model(args.model, load_mode, device, logger,
                                            source=args.source)
            from peft import PeftModel
            # torch_device 显式传入：peft 缺省会 infer_device()，本机 MPS 误判直接崩
            model = PeftModel.from_pretrained(model, str(ck), torch_device=device)
            kind = f"lora_adapter（{'4bit' if load_mode == 'qlora' else '常精度'}底座）"
        else:
            from transformers import AutoModelForCausalLM
            # 全量 ckpt 只有 model.save_pretrained 写出的权重/配置，**没有 tokenizer 文件**。
            # 从 ckpt 目录 AutoTokenizer.from_pretrained 会拿到空/错 tokenizer，
            # 表现为「标签「高」被切成 0 个 token」后 IndexError。
            # 统一从底座模型目录加载 tokenizer，与训练时完全同口径。
            tok, tok_path = load_tokenizer(args.model, logger, source=args.source)
            logger.info(f"tokenizer 来自底座：{tok_path}（ckpt 目录只有权重）")
            dtype = torch.float16 if device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(str(ck), dtype=dtype)
            kind = "full_weights"
        logger.info(f"SFT ckpt 加载方式：{kind}（{ck}）")
    else:
        model, tok, _ = load_base_model(args.model, "plain", device, logger, source=args.source)
        kind = "base"
    if next(model.parameters()).device.type != device:
        model = model.to(device)
    model.eval()
    amp_dtype, _ = pick_amp(device)

    # ── prompt ──
    shots = []
    if args.mode == "few_shot":
        shot_file = Path(args.shots_file) if args.shots_file else paths.DATA_DIR / "train_5k.jsonl"
        shots = sample_shots(load_jsonl(shot_file), args.k_shots, args.seed)
        logger.info(f"few-shot k={args.k_shots}，shot 来自 {shot_file.name}（seed={args.seed}）"
                    f"：{[s['label'] for s in shots]}")
    if args.mode == "few_shot":
        prompts = [render_messages(tok, build_few_shot_messages(r, shots, args.use_title))
                   for r in rows]
    else:
        prompts = [render_chat_prompt(tok, r, args.use_title) for r in rows]

    t0 = time.time()
    raw_texts = None
    if args.generate:
        # 生成式：batched 需要左 padding，否则 pad 在结尾会污染「下一个 token」
        tok.padding_side = "left"
        preds, raw_texts = [], []
        for i in range(0, len(prompts), args.batch_size):
            enc = tok(prompts[i:i + args.batch_size], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
            for j in range(len(enc["input_ids"])):
                cont = tok.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                raw_texts.append(cont)
                preds.append(parse_label(cont, label_names))
            if i % (args.batch_size * 5) == 0:
                logger.info(f"  {i}/{len(prompts)}（{(time.time()-t0)/60:.1f} min）")
        probs = None
    else:
        label_ids = label_first_tokens(tok, label_names, logger)
        preds, probs = logprob_classify(model, tok, prompts, label_ids, device,
                                        batch_size=args.batch_size,
                                        amp_dtype=amp_dtype if device == "cuda" else None,
                                        logger=logger)

    rep = classification_report(truth, preds, probs=probs, id2name=id2name)
    rep.update({"method": f"llm_{args.mode}", "model": str(args.model), "ckpt_kind": kind,
               "generate": bool(args.generate),
               "sec_per_item": round((time.time() - t0) / max(1, len(rows)), 3),
               "pred_distribution": pred_distribution(preds, id2name)})

    # ── 对账：SFT + 打分 + 全量数据时，与训练日志的 test 指标必须逐位一致 ──
    reconcile = {"checked": False}
    if args.mode == "sft" and not args.generate and not args.n:
        run_name = Path(args.ckpt).name
        log_json = paths.LOG_DIR / f"{run_name}.json"
        if log_json.exists():
            tr = json.loads(log_json.read_text(encoding="utf-8"))
            ref = tr.get("test")
            if ref and ref.get("file") == data_path.name:
                diffs = [f"{k}: {ref[k]} vs {rep[k]}"
                         for k in ("acc", "macro_f1") if rep.get(k) != ref.get(k)]
                reconcile = {"checked": True, "against": f"logs/{run_name}.json",
                             "match": not diffs, "diffs": diffs}
                logger.info(("✓ 对账一致：" if not diffs else "✗ 对账不一致：")
                            + f"acc={rep['acc']} macro_f1={rep['macro_f1']}")
            else:
                reconcile = {"checked": False, "reason": "训练日志里没有可比对指标或数据文件不同"}
        else:
            reconcile = {"checked": False, "reason": f"未找到 logs/{run_name}.json"}

    # ── 产物 ──
    out = {"tag": tag, "mode": args.mode, "model": str(args.model), "ckpt": args.ckpt,
           "k_shots": args.k_shots, "use_title": args.use_title, "generate": args.generate,
           "data_file": data_path.name, "n": len(rows), "shots": shots and
           [{"id": s["id"], "label": s["label"]} for s in shots],
           **train_meta, "report": rep, "reconcile": reconcile}
    (paths.EVAL_DIR / f"{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    # 逐条预测：bad case 分析与三路方法对照用；生成式额外存 raw 原文（解析失败审计）
    with open(paths.EVAL_DIR / f"{tag}_preds.jsonl", "w", encoding="utf-8") as f:
        for i, (r, y, p) in enumerate(zip(rows, truth, preds)):
            item = {"id": r["id"], "true": id2name[y],
                    "pred": id2name[p] if p >= 0 else None}
            if raw_texts is not None:
                item["raw"] = raw_texts[i]
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"准确率(acc)={rep.get('acc')}  Macro-F1={rep.get('macro_f1')}"
                + (f"  解析失败率={rep['unparsed_rate']}" if rep.get("unparsed") else ""))
    if rep.get("confusion_matrix"):
        logger.info("混淆矩阵（行为真实，列为预测）:\n" + format_confusion(rep, id2name))
    logger.info(f"预测覆盖类别数：{rep['pred_distribution']['used_classes']}/{len(label_names)} "
                f"| {rep['pred_distribution']['counts']}")
    logger.info(f"产物 → {paths.EVAL_DIR / (tag + '.json')}")
    print("\n" + to_markdown_table({tag: rep}, ece_header="ECE·LLM"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
