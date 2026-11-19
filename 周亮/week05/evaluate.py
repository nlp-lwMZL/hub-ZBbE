"""evaluate.py — 模型评测：likelihood（PPL）+ 生成行为（停止率 / 平均生成长度 / 样例对照）

生成式模型的评测必须两条腿走路（week4 的教训：PPL 低 ≠ 会说话，week5 的对应版本：PPL 低 ≠ 会回答）：
  - likelihood 侧：train / val PPL，只在回答段（y != -100）计算，按 token 加权
  - 生成侧：贪心解码逐条生成回答，统计
      · 停止率 —— 生成出 <|end|> 的比例。「停止」的操作性定义：生成循环必有终止，
        要么 <|end|>（自停）要么跑满预算（截断），未停 = 截断
      · 平均生成长度
      · 抽 3 条样例与验证集标准答案并排对照（人工定性评测的入口）

生成协议与 week5作业.md 4.3 节完全一致（seed=42 抽 100 条 val、贪心、预算 48 token），
因此本脚本可直接复现报告里的停止率数字。

用法：
    python evaluate.py                          # best_sft.pt：PPL + 停止率 + 样例
    python evaluate.py --ckpt best_scratch.pt   # 评测对比模型
    python evaluate.py --n_questions 0          # 用全部 200 条验证问题（默认 100 条）

每次评测的结果自动写入 logs/<run_name>.json（--run_name 可命名，默认按时间戳），
内容与终端输出一致：PPL、停止率/平均生成长度、样例对照、ckpt 元信息、环境。
"""
import argparse
import json
import os
import random
import time

import torch
from torch.utils.data import DataLoader

from dataset import DEFAULT_TRAIN, DEFAULT_VAL, SFTDataset
from model import TransformerLM
from train import evaluate_ppl
from chat import encode_prompt

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    char2idx, special_ids = ck["char2idx"], ck["special_ids"]
    idx2char = {int(k): v for k, v in ck["idx2char"].items()}
    a = ck["args"]
    model = TransformerLM(len(char2idx), a["d_model"], a["heads"], a["layers"],
                          a["max_len"], 0.0).to(device)   # 推理关 dropout
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, char2idx, idx2char, special_ids, a


@torch.no_grad()
def gen_once(model, question, char2idx, idx2char, special_ids, max_len, max_new):
    """贪心生成一条回答。返回 (回答文本, 是否自停在<|end|>, 生成 token 数)"""
    ids = encode_prompt(question, char2idx, special_ids)
    end_id = special_ids["<|end|>"]
    cur, stopped = list(ids), False
    for _ in range(min(max_new, max_len - len(ids))):
        nxt = int(model(torch.tensor([cur]))[0, -1].argmax())
        cur.append(nxt)
        if nxt == end_id:
            stopped = True
            break
    ans = "".join(idx2char[i] for i in cur[len(ids):]
                  if not (idx2char[i].startswith("<|") or idx2char[i] == "<pad>"))
    return ans, stopped, len(cur) - len(ids)


def main():
    ap = argparse.ArgumentParser(description="SFT 模型评测：PPL + 停止率 + 样例对照")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "best_sft.pt"))
    ap.add_argument("--batch_size", type=int, default=64, help="算 PPL 的 batch 大小")
    ap.add_argument("--n_questions", type=int, default=100,
                    help="停止率统计用多少条验证问题；0 = 全部 200 条")
    ap.add_argument("--max_new", type=int, default=48,
                    help="生成预算（token）；未在预算内生成 <|end|> 即记为未停")
    ap.add_argument("--n_samples", type=int, default=3, help="展示多少条生成样例")
    ap.add_argument("--seed", type=int, default=42, help="抽样种子（与报告口径一致）")
    ap.add_argument("--log_dir", default="logs", help="评测日志目录（相对本脚本目录）")
    ap.add_argument("--run_name", default=None,
                    help="本次评测的日志文件名（不含扩展名）；默认按时间戳生成")
    args = ap.parse_args()

    device = torch.device("cpu")
    model, char2idx, idx2char, special_ids, ck_args = load_model(args.ckpt, device)
    print(f"ckpt: {os.path.basename(args.ckpt)}")

    # ── 评测日志：与 train.py 同一套模式，逐阶段落盘 ──
    log_dir = args.log_dir if os.path.isabs(args.log_dir) else os.path.join(HERE, args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    run_name = args.run_name or time.strftime("eval_%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{run_name}.json")
    log = {
        "run_name": run_name,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": {"torch": torch.__version__, "device": str(device)},
        "ckpt": os.path.basename(args.ckpt),
        "ckpt_args": {k: ck_args[k] for k in ("init_from", "lr", "epochs", "d_model",
                                              "heads", "layers", "max_len", "dropout", "seed")
                      if k in ck_args},
        "config": {"batch_size": args.batch_size, "n_questions": args.n_questions,
                   "max_new": args.max_new, "n_samples": args.n_samples, "seed": args.seed},
        "ppl": {},
        "generation": {},
        "samples": [],
        "finished_at": None,
    }

    def flush():
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)

    flush()
    print(f"日志: {log_path}")

    # ── likelihood 侧：train / val PPL ──
    print("\n[PPL]（仅回答段，按 token 加权）")
    for name, path in [("train", DEFAULT_TRAIN), ("val", DEFAULT_VAL)]:
        ds = SFTDataset(path, char2idx, special_ids)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        loss, ppl = evaluate_ppl(model, loader, device, special_ids["<pad>"])
        print(f"{name:>5} PPL: {ppl:8.2f}   (CE loss {loss:.4f}, {len(ds)} 条)")
        log["ppl"][name] = {"loss": loss, "ppl": ppl, "n": len(ds)}
    flush()

    # ── 生成侧：停止率 + 平均生成长度 + 样例 ──
    val_rows = [json.loads(l) for l in open(DEFAULT_VAL, encoding="utf-8") if l.strip()]
    n = min(args.n_questions, len(val_rows)) if args.n_questions > 0 else len(val_rows)
    random.seed(args.seed)
    qs = random.sample(val_rows, n)

    results = [gen_once(model, d["instruction"], char2idx, idx2char, special_ids,
                        ck_args["max_len"], args.max_new) for d in qs]
    n_stop = sum(1 for _, stopped, _ in results if stopped)
    avg_len = sum(L for _, _, L in results) / len(results)
    print(f"\n[生成]（{n} 条验证问题，贪心，预算 {args.max_new} token，seed={args.seed}）")
    print(f"停止率: {n_stop}/{n}   平均生成长度: {avg_len:.1f} token   "
          f"（未停 = 跑满预算被截断）")
    log["generation"] = {"n": n, "n_stop": n_stop, "stop_rate": n_stop / n,
                         "avg_gen_tokens": avg_len, "budget": args.max_new,
                         "decoding": "greedy", "seed": args.seed}
    flush()

    print(f"\n[样例对照] 前 {args.n_samples} 条（贪心，可复现）")
    for d, (ans, stopped, L) in list(zip(qs, results))[:args.n_samples]:
        print(f"Q: {d['instruction']}")
        print(f"标准答案: {d['output']}")
        print(f"模型回答: {ans}")
        print(f"（{'生成到<|end|>自行停止' if stopped else '未停，跑满预算'}，生成 {L} token）\n")
        log["samples"].append({"question": d["instruction"], "gold": d["output"],
                               "answer": ans, "stopped": stopped, "gen_tokens": L})
    flush()

    log["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    flush()
    print(f"日志已写入: {log_path}")


if __name__ == "__main__":
    main()
