"""L4 · LLM 指令微调：LoRA / 全量微调（用 USE_LORA 切换）。运行: python l4_sft.py

与其他方法共用 train.tsv / test.tsv，转换为指令后训练，只计算回答部分的 loss。
"""
import csv
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from common import DATA_DIR, report, LABEL2NAME, INSTRUCTION
from l3_llm_prompt import parse_answer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
USE_LORA = True  # True=LoRA / False=全量微调（全量更吃显存，lr 也要调低）
MAX_LEN = 192  # 给指令、标题和回答留足空间，避免回答被截断
BATCH_SIZE = 4  # 词表较大，使用较小 batch 控制内存占用
LR = 2e-4 if USE_LORA else 1e-5  # 全量微调使用更小的学习率

EPOCHS = 3
MAX_STEPS = 300  # 每个 SFT 方法最多更新 300 次
TRAIN_SECONDS = 240  # 每个方法最多约 4 分钟训练；不含加载和评测

LORA_R = 8
LORA_ALPHA = 16
LORA_TARGETS = ["q_proj", "v_proj"]  # LoRA 作用于 q_proj 和 v_proj


class SFTDataset(Dataset):
    def __init__(self, tsv_path, tokenizer):
        self.samples = []
        with open(tsv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for row in rows:
            user_msg = INSTRUCTION + "\n" + row["text"]
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True,
            )
            full_text = prompt_text + LABEL2NAME[row["label"]] + tokenizer.eos_token

            # 对话模板已有特殊标记，不再重复添加。
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:MAX_LEN]

            # 提问部分设为 -100，只计算回答部分的损失。
            labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
            if all(label == -100 for label in labels):
                raise ValueError("MAX_LEN 太小，回答被全部截断，请增大 MAX_LEN")
            self.samples.append((full_ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        input_ids, labels = self.samples[i]
        item = {}
        item["input_ids"] = input_ids
        item["labels"] = labels
        return item


PAD_ID = 0  # main() 里设置成 tokenizer.pad_token_id


def collate_fn(batch):
    # 右侧补齐；标签用 -100，补齐位置不计入损失。
    max_len = 0
    for b in batch:
        if len(b["input_ids"]) > max_len:
            max_len = len(b["input_ids"])

    input_ids = []
    labels = []
    attn = []
    for b in batch:
        ids = list(b["input_ids"])
        labs = list(b["labels"])
        n_pad = max_len - len(ids)
        ids = ids + [PAD_ID] * n_pad
        labs = labs + [-100] * n_pad
        mask = [1] * len(b["input_ids"]) + [0] * n_pad
        input_ids.append(ids)
        labels.append(labs)
        attn.append(mask)
    return {"input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn),
            "labels": torch.tensor(labels)}


def build_model():
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32,
                                                attn_implementation="eager")
    if not USE_LORA:
        return base
    # r 是 LoRA 矩阵的秩；alpha 控制缩放；target_modules 指定给哪些层加 LoRA。
    config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS)
    # 包装后冻结原权重，只训练新增参数；USE_LORA=False 则训练整个模型。
    model = get_peft_model(base, config)
    model.print_trainable_parameters()
    return model


def train(model, loader, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    started = time.monotonic()
    steps = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n = 0
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}", total=min(len(loader), MAX_STEPS - steps)):
            batch = {key: value.to(device) for key, value in batch.items()}
            # 先清空上一批梯度，减少前向计算时的内存占用。
            optimizer.zero_grad()
            # **batch 展开字典参数；传 labels 后模型自动计算下一 token 的交叉熵。
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
            steps += 1
            if steps >= MAX_STEPS or time.monotonic() - started >= TRAIN_SECONDS:
                break
        print(f"epoch {epoch + 1}  平均loss={total_loss / n:.4f}")

        if steps >= MAX_STEPS or time.monotonic() - started >= TRAIN_SECONDS:
            break
    print(f"实际训练 {steps} 步，耗时 {time.monotonic() - started:.1f} 秒")
    return steps


@torch.no_grad()
def evaluate_sft(model, tokenizer, device):
    with open(DATA_DIR / "test.tsv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    model.eval()
    y_true, y_pred, failed = [], [], 0
    for row in tqdm(rows, desc="评测"):
        messages = [{"role": "user", "content": INSTRUCTION + "\n" + row["text"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(device)
        # 分类只需要几个 token，不生成长回答。
        out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        answer = parse_answer(raw)
        if answer is None:
            failed += 1
            answer = "109"  # 解析失败时回退到科技类
        y_pred.append(answer)
        y_true.append(row["label"])

    print(f"解析失败: {failed}/{len(rows)}")
    return report(y_true, y_pred)


def main():
    global PAD_ID
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    PAD_ID = tokenizer.pad_token_id

    model = build_model().to(device)

    ds = SFTDataset(DATA_DIR / "train.tsv", tokenizer)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    steps = train(model, loader, device)
    result = evaluate_sft(model, tokenizer, device)
    result["train_steps"] = steps
    return result


if __name__ == "__main__":
    from common import prepare_quick_data
    prepare_quick_data()
    main()
