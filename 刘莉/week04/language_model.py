# language_model.py
# Transformer Decoder 因果语言模型，训练语料 corpus_general.txt
# 修复mask类型警告，适配4G显存
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BertTokenizer
import random
import os

# ===================== 全局超参数配置【适配4G显存】 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORPUS_FILE = "corpus_general.txt"
MODEL_SAVE_PATH = "transformer_lm.pt"
SAVE_MODEL_PT = True   # True保存pt权重，False仅内存运行

# 4G显存调小参数
D_MODEL = 256
N_HEADS = 4
NUM_LAYERS = 4
DIM_FF = 512
DROPOUT = 0.1
SEQ_LEN = 64        # 序列长度缩小
BATCH_SIZE = 4      # 小batch，4G显存友好
EPOCHS = 10
LEARNING_RATE = 2e-4

# 中文分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
VOCAB_SIZE = tokenizer.vocab_size
PAD_ID = tokenizer.pad_token_id
SOS_ID = tokenizer.cls_token_id
EOS_ID = tokenizer.sep_token_id


# ===================== 位置编码 =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# ===================== 因果Transformer语言模型（纯Decoder） =====================
class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = D_MODEL
        self.pad_idx = PAD_ID

        self.token_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_encoding = PositionalEncoding(D_MODEL, DROPOUT)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=DIM_FF,
            dropout=DROPOUT,
            batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=NUM_LAYERS)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    @staticmethod
    def generate_causal_mask(sz: int, device):
        """因果掩码：看不到未来token，float类型"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def generate_pad_mask(self, seq):
        """【修复警告】返回float类型padding mask，padding位置填‑inf，有效位置0.0"""
        mask_bool = (seq == self.pad_idx).transpose(0, 1)
        mask_float = torch.zeros_like(mask_bool, dtype=torch.float)
        mask_float.masked_fill_(mask_bool, float("-inf"))
        return mask_float

    def forward(self, src_tokens):
        """
        src_tokens: [seq_len, batch]
        return logits: [seq_len, batch, vocab_size]
        """
        seq_len, batch_size = src_tokens.shape
        emb = self.token_embedding(src_tokens)
        emb = self.pos_encoding(emb)

        tgt_mask = self.generate_causal_mask(seq_len, src_tokens.device)
        pad_mask = self.generate_pad_mask(src_tokens).to(src_tokens.device)

        # nn.TransformerDecoder强制要求memory参数，传入极小全零tensor
        dummy_memory = torch.zeros((1, batch_size, self.d_model), device=src_tokens.device)
        out = self.decoder(
            tgt=emb,
            memory=dummy_memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=pad_mask
        )
        logits = self.head(out)
        return logits

    @torch.no_grad()
    def generate_text(self, start_text: str, max_gen_len=80):
        """自回归文本生成"""
        self.eval()
        encode_out = tokenizer(start_text, return_tensors="pt")
        input_ids = encode_out["input_ids"].to(DEVICE).transpose(0,1) # [T,1]

        for _ in range(max_gen_len):
            logits = self(input_ids)
            next_token = torch.argmax(logits[-1, 0, :], dim=-1, keepdim=True).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=0)
            if next_token.item() == EOS_ID:
                break
        output_text = tokenizer.decode(input_ids.squeeze(1).cpu(), skip_special_tokens=True)
        return output_text


# ===================== 文本数据集 =====================
class TextCorpusDataset(Dataset):
    def __init__(self, text_lines, seq_len):
        self.seq_len = seq_len
        self.samples = []
        for line in text_lines:
            line = line.strip()
            if len(line) < 5:
                continue
            token_result = tokenizer(
                line,
                max_length=seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            self.samples.append(token_result["input_ids"].squeeze(0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        # LM任务：输入[:-1],标签[1:]
        return tokens


# ===================== 训练循环 =====================
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for batch_tokens in dataloader:
        batch_tokens = batch_tokens.transpose(0,1).to(DEVICE) # [seq_len, batch]

        input_tokens = batch_tokens[:-1, :]
        label_tokens = batch_tokens[1:, :]

        logits = model(input_tokens)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), label_tokens.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    if not os.path.exists(CORPUS_FILE):
        raise FileNotFoundError(f"请确认 {CORPUS_FILE} 和代码在同一目录!")
    with open(CORPUS_FILE, encoding="utf-8") as f:
        all_lines = f.readlines()

    random.shuffle(all_lines)
    split_idx = int(len(all_lines)*0.9)
    train_lines = all_lines[:split_idx]
    val_lines = all_lines[split_idx:]

    train_ds = TextCorpusDataset(train_lines, SEQ_LEN)
    val_ds = TextCorpusDataset(val_lines, SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"训练样本数:{len(train_ds)} 验证样本:{len(val_ds)}")

    model = TransformerLanguageModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS+1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch:02d} | Train Loss: {loss:.4f}")

        if SAVE_MODEL_PT:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> 已保存模型权重 {MODEL_SAVE_PATH}")
        else:
            print("-> SAVE_MODEL_PT=False，跳过保存pt权重文件")

    if SAVE_MODEL_PT:
        print(f"训练完成，模型保存在 {MODEL_SAVE_PATH}")
    else:
        print("训练完成，权重仅保存在内存，程序退出后全部丢失！")

    # 文本生成测试
    prompt = "女朋友和我打冷战，怎么办啊"
    gen_result = model.generate_text(prompt)
    print(f"\n【输入prompt】{prompt}")
    print(f"【模型生成结果】{gen_result}")


if __name__ == "__main__":
    main()
