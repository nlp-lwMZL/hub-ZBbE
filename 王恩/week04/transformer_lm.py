"""字符级因果 Transformer：训练、PPL 评估与自回归生成。"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "corpus_general.txt"
DEFAULT_MODEL = ROOT / "output/transformer_lm.pt"
IGNORE_INDEX = -100


class CharDataset(Dataset):
    """每段输入和标签错开一位；相邻块的目标不重叠，保留最后的短块。"""

    def __init__(self, text, vocab, seq_len):
        if len(text) < 2 or seq_len < 1:
            raise ValueError("每个数据集至少需要 2 个字符，seq_len 必须为正数")
        self.ids = torch.tensor([vocab.get(c, 0) for c in text], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.ids) - 1 + self.seq_len - 1) // self.seq_len

    def __getitem__(self, index):
        start = index * self.seq_len
        end = min(start + self.seq_len, len(self.ids) - 1)
        return self.ids[start:end], self.ids[start + 1:end + 1]


def collate_batch(batch):
    # 末块右侧补齐；因果掩码阻止真实位置读取右侧补齐，标签补齐不计入 loss。
    xs, ys = zip(*batch)
    return (nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0),
            nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=IGNORE_INDEX))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, seq_len=64, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        if min(vocab_size, seq_len, d_model, nhead, num_layers, dim_feedforward) < 1:
            raise ValueError("模型维度与层数必须为正数")
        if d_model % nhead:
            raise ValueError("d_model 必须能被 nhead 整除")
        self.seq_len = seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        # 使用独立实例初始化各层，避免 TransformerEncoder 克隆导致相同初始参数。
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.register_buffer("causal_mask", torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1), persistent=False)

    def forward(self, x):
        length = x.size(1)
        if not 1 <= length <= self.seq_len:
            raise ValueError(f"输入长度必须在 1 到 {self.seq_len} 之间")
        positions = torch.arange(length, device=x.device)
        hidden = self.dropout(self.token_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            # True 表示不允许关注；只可读取当前位置及之前的字符。
            hidden = layer(hidden, src_mask=self.causal_mask[:length, :length])
        return self.head(self.norm(hidden))


def split_text(text, val_ratio, test_ratio):
    if not (0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError("验证、测试比例须大于 0，且两者之和小于 1")
    n_val, n_test = int(len(text) * val_ratio), int(len(text) * test_ratio)
    n_train = len(text) - n_val - n_test
    if min(n_train, n_val, n_test) < 2:
        raise ValueError("语料过短，每个划分至少需要 2 个字符")
    return {"train": text[:n_train], "val": text[n_train:n_train + n_val],
            "test": text[n_train + n_val:]}


def read_corpus(path, max_chars):
    if max_chars < 0:
        raise ValueError("max_chars 不能为负数；0 表示读取全部")
    with Path(path).open(encoding="utf-8") as f:
        return f.read(max_chars) if max_chars else f.read()


def make_loader(text, vocab, seq_len, batch_size, shuffle=False):
    return DataLoader(CharDataset(text, vocab, seq_len), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate_batch)


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_nll, total_tokens = 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            nll = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                                  ignore_index=IGNORE_INDEX, reduction="sum")
            count = int((y != IGNORE_INDEX).sum().item())
            if not torch.isfinite(nll):
                raise RuntimeError("loss 非有限值，请检查数据或降低学习率")
            if training:
                optimizer.zero_grad(set_to_none=True)
                (nll / count).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_nll += nll.item()
            total_tokens += count
    if not total_tokens:
        raise ValueError("没有可用于计算 PPL 的目标字符")
    loss = total_nll / total_tokens
    return {"loss": loss, "ppl": math.exp(loss) if loss < 709 else float("inf"),
            "tokens": total_tokens}


def choose_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TransformerLM(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def corpus_splits(metadata, corpus=None):
    text = read_corpus(corpus or metadata["path"], metadata["max_chars"])
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError("评估语料与训练记录不一致，请提供原始语料，避免划分发生变化")
    return split_text(text, metadata["val_ratio"], metadata["test_ratio"])


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    text = read_corpus(args.corpus, args.max_chars)
    splits = split_text(text, args.val_ratio, args.test_ratio)
    # 先分数据、再建表；0 是独立特殊项，未知字符不会被丢弃。
    vocab = {"<UNK>": 0, **{c: i + 1 for i, c in enumerate(sorted(set(splits["train"])))}}
    config = {"vocab_size": len(vocab), "seq_len": args.seq_len,
              "d_model": args.d_model, "nhead": args.nhead, "num_layers": args.num_layers,
              "dim_feedforward": args.dim_feedforward, "dropout": args.dropout}
    model = TransformerLM(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loaders = {name: make_loader(part, vocab, args.seq_len, args.batch_size, name == "train")
               for name, part in splits.items()}
    data_info = {"path": str(Path(args.corpus).resolve()), "max_chars": args.max_chars,
                 "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                 "val_ratio": args.val_ratio, "test_ratio": args.test_ratio}
    split_info = {name: {"chars": len(part), "targets": len(part) - 1,
                         "unk_ratio": sum(c not in vocab for c in part) / len(part)}
                  for name, part in splits.items()}
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    initial_val = run_epoch(model, loaders["val"], device)
    report = {"config": config, "seed": args.seed, "device": str(device),
              "torch_version": str(torch.__version__), "data": data_info,
              "splits": split_info, "initial_val": initial_val,
              "training": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr},
              "parameters": sum(p.numel() for p in model.parameters()), "history": []}
    print(f"device={device}, vocab={len(vocab)}, parameters={report['parameters']:,}", flush=True)
    print(f"字符划分: {split_info}\n初始验证 PPL={initial_val['ppl']:.3f}", flush=True)
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], device, optimizer)
        val_metrics = run_epoch(model, loaders["val"], device)
        report["history"].append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(f"epoch={epoch:02d} train_loss={train_metrics['loss']:.4f} "
              f"train_ppl={train_metrics['ppl']:.3f} val_loss={val_metrics['loss']:.4f} "
              f"val_ppl={val_metrics['ppl']:.3f}", flush=True)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            report["best_epoch"] = epoch
            torch.save({"state_dict": model.state_dict(), "config": config, "vocab": vocab,
                        "data": data_info, "epoch": epoch, "seed": args.seed,
                        "val": val_metrics}, save_path)
    model, _ = load_model(save_path, device)
    report["best_val"] = run_epoch(model, loaders["val"], device)
    report["test"] = run_epoch(model, loaders["test"], device)
    save_path.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best_epoch={report['best_epoch']} test_ppl={report['test']['ppl']:.3f}")
    print(f"模型: {save_path}\n报告: {save_path.with_suffix('.json')}")


def evaluate(args):
    device = choose_device(args.device)
    model, checkpoint = load_model(args.model, device)
    part = corpus_splits(checkpoint["data"], args.corpus)[args.split]
    metrics = run_epoch(model, make_loader(part, checkpoint["vocab"], model.seq_len,
                                         args.batch_size), device)
    metrics["unk_ratio"] = sum(c not in checkpoint["vocab"] for c in part) / len(part)
    print(json.dumps({"split": args.split, "epoch": checkpoint["epoch"], **metrics}, indent=2))


@torch.no_grad()
def generate_text(model, vocab, prompt, max_new_tokens=100, temperature=1.0, top_k=20):
    if not prompt:
        raise ValueError("prompt 不能为空")
    if not math.isfinite(temperature) or temperature <= 0 or max_new_tokens < 0 or top_k < 0:
        raise ValueError("temperature 须为有限正数，生成长度和 top_k 不能为负数")
    unknown = sorted(set(prompt) - set(vocab))
    if unknown:
        raise ValueError(f"提示包含词表外字符: {unknown}，请换用训练词表中的字符")
    model.eval()
    device = next(model.parameters()).device
    ids = [vocab[c] for c in prompt]
    idx_to_char = {i: c for c, i in vocab.items()}
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.seq_len:]], device=device)
        logits = model(x)[0, -1] / temperature
        logits[0] = -float("inf")  # <UNK> 只用于编码，不能生成一个真实字符。
        if top_k:
            k = min(top_k, len(vocab) - 1)
            values, indices = torch.topk(logits, k)
            next_id = indices[torch.multinomial(F.softmax(values, dim=-1), 1)].item()
        else:
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        ids.append(next_id)
    return "".join(idx_to_char[i] for i in ids)


def generate(args):
    model, checkpoint = load_model(args.model, choose_device(args.device))
    print(generate_text(model, checkpoint["vocab"], args.prompt,
                        args.max_new_tokens, args.temperature, args.top_k))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train", help="训练并按验证 PPL 保存最佳模型")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--save", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--max-chars", type=int, default=200_000, help="0 表示完整语料")
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--test-ratio", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dim-feedforward", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=train)
    p = commands.add_parser("evaluate", help="重新加载最优模型计算 PPL")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--corpus", type=Path, help="语料移动后指定新路径，内容必须相同")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=64)
    p.set_defaults(func=evaluate)
    p = commands.add_parser("generate", help="逐字采样生成文本")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--prompt", default="今天")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20, help="0 表示不限制候选字符数")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=generate)
    for subparser in commands.choices.values():
        subparser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
        subparser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "seq_len", "threads"):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(f"{name} 必须为正数")
    if hasattr(args, "lr") and (not math.isfinite(args.lr) or args.lr <= 0):
        parser.error("lr 必须为有限正数")
    if hasattr(args, "dropout") and not 0 <= args.dropout < 1:
        parser.error("dropout 必须在 [0, 1) 内")
    torch.set_num_threads(args.threads)
    if hasattr(args, "seed"):
        torch.manual_seed(args.seed)
    args.func(args)


if __name__ == "__main__":
    main()
