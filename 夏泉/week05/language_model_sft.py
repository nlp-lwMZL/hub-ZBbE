import torch
import torch.nn as nn
from pathlib import Path
import json
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import argparse
import random
import math
from tqdm import tqdm


ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
MODEL_PATH  = ROOT / "best_model.pt"

# ─────────────────────────── 模型 ───────────────────────────
# 与 pinyin_ime.py 中的 LM 保持完全一致，以便正确加载 checkpoint
class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # 根据 model_type 参数在 RNN 和 LSTM 之间切换
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
        self.rnn = rnn_cls(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            # 单层时 PyTorch 不允许设置 dropout，需显式置 0
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (B, T) 字符 id 序列
        e = self.drop(self.embed(x))       # (B, T, embed_dim)
        out, _ = self.rnn(e)               # (B, T, hidden_dim)，丢弃 hidden state
        return self.fc(self.drop(out))     # (B, T, vocab_size)

# ─────────────────────────── Dataset ───────────────────────────
class SFTDataset(Dataset):
    def __init__(self, path, char2idx, max_len, sep="\n"):
        self.data = load_file(path)
        self.char2idx = char2idx
        self.max_len = max_len
        self.sep_id = encode(sep, char2idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt, response = item["instruction"], item["output"]
        p_ids = encode(prompt, self.char2idx)
        r_ids = encode(response, self.char2idx)
        budget = self.max_len - len(self.sep_id)
        if len(r_ids) > budget:
            r_ids = r_ids[:budget]
        remain = budget - len(r_ids)
        if len(p_ids) > remain:
            p_ids = p_ids[-remain:] if remain > 0 else []
        input_ids = p_ids + self.sep_id + r_ids
        labels = [-100] * (len(p_ids) + len(self.sep_id)) + r_ids
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }

def collate_fn(batch):
    # batch 是一个字典列表: [{'input_ids': ..., 'labels': ...}, {...}]
    max_len = max(len(item["input_ids"]) for item in batch)
    
    input_ids_list = []
    labels_list = []
    
    for item in batch:
        x = item["input_ids"]
        y = item["labels"]
        pad_len = max_len - len(x)
        
        # 在 __getitem__ 里已经转了 tensor，这里用 torch.cat 拼接 pad
        # input 补 0
        padded_x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
        # label 补 -100
        padded_y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])
        
        input_ids_list.append(padded_x)
        labels_list.append(padded_y)
        
    return {
        "input_ids": torch.stack(input_ids_list), # 形状 (B, max_len)
        "labels": torch.stack(labels_list)        # 形状 (B, max_len)
    }

# ─────────────────────────── 工具函数 ───────────────────────────
def encode(text, char2idx):
    return [char2idx[c] for c in text if c in char2idx]


# ─────────────────────────── 加载数据 ───────────────────────────
def load_file(path):
    data_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                data_list.append(data)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse line: {line}")

    return data_list
            

# ─────────────────────────── 加载预训练权重 ───────────────────────────
def load_pretrained_weights(ckpt_path, device):
    """
    加载预训练权重，返回模型参数字典。
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["args"]                       # vars(argparse.Namespace)
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]

    model = LM(
        vocab_size=len(char2idx),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        model_type=cfg["model"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])

    return model, char2idx, idx2char, cfg

# ─────────────────────────── 训练 / 评估 ───────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, epoch, epochs, train=True):
    model.train(train)
    total_loss, total_tok = 0.0, 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        if train:
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    
        n = (labels != -100).sum().item()      # 实际参与计算的 token 数
        total_loss += loss.item() * n
        total_tok += n

    avg = total_loss / max(total_tok, 1)
    return avg, math.exp(min(avg, 20))

# ─────────────────────────── 推理样例（训练后目测效果） ───────────────────────────
@torch.no_grad()
def generate(model, char2idx, idx2char, prompt,
             max_new=80, stop_ids=None, device="cpu"):
    model.eval()
    ids = encode(prompt, char2idx) + stop_ids
    x = torch.tensor([ids], device=device)
    out = list(ids)
    stop = set(stop_ids or [])
    for _ in range(max_new):
        nxt = model(x)[0, -1].argmax().item()
        out.append(nxt)
        if nxt in stop:
            break
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return "".join(idx2char.get(i, "") for i in out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        default=str(MODEL_PATH))
    parser.add_argument("--data_dir",    default=str(DATA_DIR))
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--max_len",     type=int,   default=128)
    parser.add_argument("--val_ratio",   type=float, default=0.05)
    parser.add_argument("--freeze_embed", action="store_true",
                        help="冻结字向量，只训练 RNN+FC（数据极少时更稳）")
    parser.add_argument("--eos",         default="\n",  help="response 末尾追加的结束符，如 \\n")
    parser.add_argument("--save",        default="sft_model.pt")
    parser.add_argument("--demo_prompt", default="请问北京奥运会的举办时间和地点分别是什么？",  help="训练后打印一条生成样例")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, char2idx, idx2char, cfg = load_pretrained_weights(args.ckpt, device=device)
    print(f"device: {device}  vocab: {len(char2idx)}  "
              f"base model: {cfg['model'].upper()}")

    tran_dataset = SFTDataset(Path(args.data_dir) / "sft_train.jsonl", char2idx, args.max_len)
    val_dataset = SFTDataset(Path(args.data_dir) / "sft_val.jsonl", char2idx, args.max_len)

    train_loader = DataLoader(tran_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)  

    best_val = float("inf")
    print(f"{"epoch":>6}  {"tr_loss":>11}  {"tr_ppl":>10}  "
                              f"{"va_loss":>10}  {"va_ppl":>10}")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, epoch, args.epochs, train=False)

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": cfg,
            }, args.save)
            print(f"{epoch:>6}  {tr_loss:>11.4f}  {tr_ppl:>10.2f}  "
                          f"{va_loss:>10.4f}  {va_ppl:>10.2f}")

    print(f"\nSFT 完成，最佳 val loss: {best_val:.4f}  已保存至 {args.save}")

    # 训练后目测
    if args.demo_prompt:
        stop_ids = encode(args.eos, char2idx) if args.eos else []
        text = generate(model, char2idx, idx2char, args.demo_prompt,
                        stop_ids=stop_ids, device=device)
        print("\n===== demo =====")
        print(text)

if __name__ == "__main__":
    main()

