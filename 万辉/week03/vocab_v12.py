import csv
import os
from build_langid_dataset import build_char_vocab, save_vocab, OUT_ROOT

# 正确路径：OUT_ROOT 已经是 data/langid
train_csv = os.path.join(OUT_ROOT, "v12", "train.csv")
out_path = os.path.join(OUT_ROOT, "vocab", "vocab_v12.txt")

# 建议运行前先打印确认一下路径是否找对
print("读取路径：", train_csv)
print("路径存在：", os.path.exists(train_csv))

with open(train_csv, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

texts = [r["text"] for r in rows]
vocab, freq = build_char_vocab(texts)

save_vocab(vocab, freq, out_path)

print("vocab_v12.txt 已生成，字符数：", len(vocab))