"""字符 CNN（从零训练）和多语种预训练 Encoder（冻结/全量微调）。"""
import random
import time
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, rows, labels, max_length, vocab=None, tokenizer=None):
        self.rows, self.labels = rows, labels
        self.max_length, self.vocab, self.tokenizer = max_length, vocab, tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.tokenizer is not None:
            encoded = self.tokenizer(row["text"], truncation=True, max_length=self.max_length,
                                     padding="max_length", return_tensors="pt")
            inputs = {key: value.squeeze(0) for key, value in encoded.items()}
        else:
            ids = [self.vocab.get(c, 1) for c in row["text"].casefold()[:self.max_length]]
            inputs = {"input_ids": torch.tensor(ids + [0] * (self.max_length - len(ids)))}
        inputs["labels"] = torch.tensor(self.labels.index(row["label"]))
        return inputs


class CharCNN(nn.Module):
    def __init__(self, vocab_size, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(64, 64, k, padding=k // 2) for k in (3, 5, 7))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(192, num_labels))

    def forward(self, input_ids):
        mask = input_ids.ne(0).unsqueeze(1)
        x = self.embedding(input_ids).transpose(1, 2)
        pooled = [conv(x).relu().masked_fill(~mask, -1e4).amax(dim=-1) for conv in self.convs]
        return self.head(torch.cat(pooled, dim=-1))


class EncoderClassifier(nn.Module):
    def __init__(self, model_path, num_labels, frozen=False):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_path)
        self.frozen = frozen
        self.encoder.requires_grad_(not frozen)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.encoder.config.hidden_size, num_labels))

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()  # 冻结时也关闭 encoder dropout，避免特征不断变化。
        return self

    def forward(self, **inputs):
        hs = self.encoder(**inputs, return_dict=True).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hs.dtype)
        vector = (hs * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(vector)


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def predict(model, loader, device, labels):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            predictions.extend(labels[i] for i in model(**inputs).argmax(-1).cpu().tolist())
    return predictions


def train_neural(method, splits, labels, args, output_dir):
    from sklearn.metrics import f1_score
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    vocab, tokenizer = None, None
    if method == "char_cnn":
        counts = Counter(c for row in splits["train"] for c in row["text"].casefold())
        vocab = {c: i + 2 for i, (c, _) in enumerate(sorted(counts.items(), key=lambda x: (-x[1], x[0])))}
        model = CharCNN(len(vocab) + 2, len(labels))
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.encoder)
        model = EncoderClassifier(args.encoder, len(labels), frozen=method == "encoder_frozen")
    model.to(args.device)
    loaders = {split: DataLoader(TextDataset(rows, labels, args.max_length, vocab, tokenizer),
                                batch_size=args.batch_size, shuffle=split == "train",
                                generator=torch.Generator().manual_seed(args.seed))
               for split, rows in splits.items()}
    if method == "encoder_finetune":
        parameters = [{"params": model.encoder.parameters(), "lr": args.encoder_lr},
                      {"params": model.head.parameters(), "lr": args.encoder_lr * 5}]
    else:
        parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    history, best_score, best_state, best_epoch = [], -1.0, None, 0
    synchronize(args.device)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, size = 0.0, 0
        for batch in loaders["train"]:
            target = batch.pop("labels").to(args.device)
            inputs = {k: v.to(args.device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(**inputs), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(target)
            size += len(target)
        dev_pred = predict(model, loaders["dev"], args.device, labels)
        score = f1_score([r["label"] for r in splits["dev"]], dev_pred,
                         labels=labels, average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": loss_sum / size, "dev_macro_f1": score})
        print(f"  {method} epoch={epoch} loss={loss_sum / size:.4f} dev_f1={score:.4f}", flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    synchronize(args.device)
    fit_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    synchronize(args.device)
    started = time.perf_counter()
    test_pred = predict(model, loaders["test"], args.device, labels)
    synchronize(args.device)
    predict_seconds = time.perf_counter() - started
    if args.save_models:
        torch.save({"state_dict": best_state, "vocab": vocab, "labels": labels,
                    "method": method, "args": vars(args), "best_epoch": best_epoch},
                   output_dir / f"{method}.pt")
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir / f"{method}_tokenizer")
            model.encoder.config.save_pretrained(output_dir / f"{method}_tokenizer")
    return test_pred, {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                       "history": history, "best_epoch": best_epoch,
                       "parameters": sum(p.numel() for p in model.parameters()),
                       "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
