"""在 week04 字符级因果语言模型上做回答区域的监督微调。"""

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
# 复用上一周的模型和按有效 token 汇总 NLL 的训练循环。
sys.path.insert(0, str(ROOT.parent / "week04"))
from transformer_lm import (  # noqa: E402
    IGNORE_INDEX, TransformerLM, choose_device, collate_batch, load_model, run_epoch,
)

BASE = ROOT.parent / "week04/output/transformer_lm.pt"
MODEL = ROOT / "output/sft_lm.pt"
SPECIALS = ("<PAD>", "<BOS>", "<EOS>")


def format_prompt(instruction, text=""):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction 必须是非空字符串")
    if not isinstance(text, str):
        raise ValueError("input 必须是字符串")
    return f"指令：{instruction}\n" + (f"输入：{text}\n" if text else "") + "回答："


def read_records(path):
    records = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("记录必须为 JSON 对象")
            format_prompt(row.get("instruction"), row.get("input", ""))
            if not isinstance(row.get("output"), str) or not row["output"].strip():
                raise ValueError("output 必须是非空字符串")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
        records.append(row)
    if not records:
        raise ValueError(f"数据集为空: {path}")
    return records


def check_splits(splits):
    """同一问题不得跨集合出现，避免答案泄漏或冲突。"""
    seen = {}
    for split, rows in splits.items():
        for row in rows:
            prompt = format_prompt(row["instruction"], row.get("input", ""))
            if prompt in seen:
                raise ValueError(f"重复问题出现在 {seen[prompt]} 和 {split}: {prompt!r}")
            seen[prompt] = split


def extend_vocab(vocab, rows):
    vocab = dict(vocab)
    chars = set()
    for row in rows:
        chars.update(format_prompt(row["instruction"], row.get("input", "")) + row["output"])
    for token in [*SPECIALS, *sorted(chars)]:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def initialize_model(base_path, train_rows):
    base, checkpoint = load_model(base_path, "cpu")
    vocab = extend_vocab(checkpoint["vocab"], train_rows)
    config = {**checkpoint["config"], "vocab_size": len(vocab)}
    model = TransformerLM(**config)
    state = model.state_dict()
    for name, weight in base.state_dict().items():
        if name in ("token_embedding.weight", "head.weight", "head.bias"):
            # 保留旧字符 ID 及其嵌入、输出权重；仅新增行随机初始化。
            state[name][:weight.shape[0]].copy_(weight)
        else:
            state[name].copy_(weight)
    model.load_state_dict(state)
    return model, vocab, config


class SFTDataset(Dataset):
    def __init__(self, rows, vocab, seq_len):
        self.samples = []
        self.unknown = 0
        self.characters = 0
        for index, row in enumerate(rows, 1):
            prompt = format_prompt(row["instruction"], row.get("input", ""))
            answer = row["output"]
            self.unknown += sum(c not in vocab for c in prompt + answer)
            self.characters += len(prompt + answer)
            prefix = [vocab["<BOS>"]] + [vocab.get(c, 0) for c in prompt]
            target = [vocab.get(c, 0) for c in answer] + [vocab["<EOS>"]]
            ids = prefix + target
            if len(ids) - 1 > seq_len:
                raise ValueError(f"样本 {index} 需要 {len(ids)-1} 个位置，超过上下文 {seq_len}；请缩短样本")
            # logits[t] 预测 ids[t+1]，首个答案恰好由提示最后一字预测。
            labels = [IGNORE_INDEX] * (len(prefix) - 1) + target
            self.samples.append((torch.tensor(ids[:-1]), torch.tensor(labels)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def loader(dataset, batch_size, shuffle=False):
    # 复用 week04 的右补齐：输入用 UNK=0，标签 -100；因果注意力保证
    # 真实位置看不到右侧补齐，因此无需额外 padding attention mask。
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)


@torch.no_grad()
def generate(model, vocab, instruction, text="", max_new_tokens=32,
             temperature=1.0, top_k=0, top_p=1.0, greedy=True):
    if (max_new_tokens < 0 or top_k < 0 or not math.isfinite(temperature)
            or temperature <= 0 or not 0 < top_p <= 1):
        raise ValueError("生成长度和 top_k 须非负，temperature 须为有限正数，0 < top_p <= 1")
    prompt = format_prompt(instruction, text)
    unknown = sorted(set(prompt) - set(vocab))
    if unknown:
        raise ValueError(f"提示含词表外字符: {unknown}")
    ids = [vocab["<BOS>"]] + [vocab[c] for c in prompt]
    if len(ids) > model.seq_len:
        raise ValueError("提示超过模型上下文长度")
    inverse = {i: c for c, i in vocab.items()}
    result = []
    model.eval()
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.seq_len:]], device=next(model.parameters()).device)
        logits = model(x)[0, -1] / temperature
        logits[[0, vocab["<PAD>"], vocab["<BOS>"]]] = -torch.inf
        if greedy:
            token = int(logits.argmax())
        else:
            if top_k:
                threshold = logits.topk(min(top_k, logits.numel())).values[-1]
                logits[logits < threshold] = -torch.inf
            sorted_logits, order = logits.sort(descending=True)
            probs = sorted_logits.softmax(-1)
            # 保留首次跨过 top-p 阈值的候选。
            remove = probs.cumsum(-1) - probs >= top_p
            sorted_logits[remove] = -torch.inf
            token = int(order[torch.multinomial(sorted_logits.softmax(-1), 1)].item())
        if token == vocab["<EOS>"]:
            break
        result.append(inverse[token])
        ids.append(token)
    return "".join(result)


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    splits = {name: read_records(getattr(args, name)) for name in ("train", "val", "test")}
    check_splits(splits)
    model, vocab, config = initialize_model(args.base_model, splits["train"])
    device = choose_device(args.device)
    model.to(device)
    datasets = {k: SFTDataset(v, vocab, model.seq_len) for k, v in splits.items()}
    loaders = {k: loader(v, args.batch_size, k == "train") for k, v in datasets.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    save = Path(args.save)
    if save.resolve() == Path(args.base_model).resolve():
        raise ValueError("SFT 输出路径不能覆盖基础模型")
    save.parent.mkdir(parents=True, exist_ok=True)
    report = {"base_model": str(Path(args.base_model).resolve()), "config": config,
              "seed": args.seed, "device": str(device), "torch_version": str(torch.__version__),
              "training": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr},
              "data": {k: {"path": str(Path(getattr(args, k)).resolve()), "samples": len(v),
                            "sha256": hashlib.sha256(Path(getattr(args, k)).read_bytes()).hexdigest(),
                            "unk_ratio": v.unknown / v.characters} for k, v in datasets.items()},
              "initial_val": run_epoch(model, loaders["val"], device), "history": []}
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, loaders["train"], device, optimizer)
        val = run_epoch(model, loaders["val"], device)
        report["history"].append({"epoch": epoch, "train": training, "val": val})
        print(f"epoch={epoch:03d} train_loss={training['loss']:.4f} val_loss={val['loss']:.4f}", flush=True)
        if val["loss"] < best:
            best = val["loss"]
            report["best_epoch"] = epoch
            torch.save({"state_dict": model.state_dict(), "config": config, "vocab": vocab,
                        "epoch": epoch, "val": val, "format": "instruction-sft-v1"}, save)
    model, _ = load_model(save, device)
    report["best_val"] = run_epoch(model, loaders["val"], device)
    report["test"] = run_epoch(model, loaders["test"], device)
    report["examples"] = [{"instruction": row["instruction"], "input": row.get("input", ""),
                           "expected": row["output"], "generated": generate(
                               model, vocab, row["instruction"], row.get("input", ""))}
                          for row in splits["test"]]
    report["exact_match"] = sum(r["expected"] == r["generated"] for r in report["examples"]) / len(splits["test"])
    save.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": report["best_epoch"], "test": report["test"],
                      "exact_match": report["exact_match"]}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--base-model", type=Path, default=BASE)
    for name in ("train", "val", "test"):
        p.add_argument(f"--{name}", type=Path, default=ROOT / f"data/{name}.jsonl")
    p.add_argument("--save", type=Path, default=MODEL)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    for command in ("evaluate", "generate"):
        p = sub.add_parser(command)
        p.add_argument("--model", type=Path, default=MODEL)
        p.add_argument("--device", default="cpu")
        if command == "evaluate":
            p.add_argument("--data", type=Path, default=ROOT / "data/test.jsonl")
        else:
            p.add_argument("--instruction", required=True)
            p.add_argument("--input", default="")
            p.add_argument("--max-new-tokens", type=int, default=32)
            p.add_argument("--sample", action="store_true")
            p.add_argument("--temperature", type=float, default=0.8)
            p.add_argument("--top-k", type=int, default=20)
            p.add_argument("--top-p", type=float, default=0.9)
            p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads 必须为正数")
    torch.set_num_threads(args.threads)
    if args.command == "train":
        if min(args.epochs, args.batch_size) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
            parser.error("epochs、batch-size 和 lr 必须为正数")
        train(args)
    else:
        model, checkpoint = load_model(args.model, choose_device(args.device))
        vocab = checkpoint["vocab"]
        if checkpoint.get("format") != "instruction-sft-v1":
            parser.error("请使用 SFT 训练产出的模型")
        if args.command == "evaluate":
            dataset = SFTDataset(read_records(args.data), vocab, model.seq_len)
            print(json.dumps(run_epoch(model, loader(dataset, 8), next(model.parameters()).device), indent=2))
        else:
            torch.manual_seed(args.seed)
            print(generate(model, vocab, args.instruction, args.input, args.max_new_tokens,
                           args.temperature, args.top_k, args.top_p, not args.sample))


if __name__ == "__main__":
    main()
