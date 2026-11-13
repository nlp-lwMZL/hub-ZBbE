"""
sft.py — 在基座上做 SFT：教一个只会「接龙」的字级模型「按格式回答」

【这个脚本就干三件事】
    1) 数据：把 (问题, 回答) 拼成一条对话模板
           <|user|>问题<|assistant|>回答<|end|>
    2) Loss mask：模板和问题那一段的标签全部设成 -100（不算 loss），
       模型只在「回答」上被监督 —— 这就是课件里那张「×不计算 / √计算」的图。
    3) 训练 + 评测：在留出的 200 条验证集上算「回答段 PPL」，并打印几条生成结果。

【为什么模板和问题不算 loss】
    推理时问题是**用户给的**，模型永远不需要会「生成问题」。
    把它算进 loss，等于花一部分容量去学一件推理时用不上的事（而容量正是我们最缺的）。
    所以标准做法就是：只在回答上算 loss —— 这也让验证集的数字只反映「回答好不好」，
    口径干净，才好和别的配置比。

【和 transformer_lm.py 的关系】
    模型结构、字表、特殊标记全部复用 transformer_lm.py —— 两边的字表必须一致，
    否则 ckpt 的 embedding 行数对不上、id 也对不上。所以本文件开头就 import 它。

用法:
    python sft.py                                  # 从 base_lm.pt 接着训，默认存 sft_mask.pt
    python sft.py --epochs 20 --lr 1e-3            # 调参跑
    python sft.py --eval-only                      # 只评测，不训练
    python sft.py --generate "列出三种健康的早餐食品。"   # 只生成一句话看看
    python sft.py --save-best last                 # 想复现「背下训练集」时用（见 README 第五节）
    # 另有 --no-loss-mask：关掉 loss mask 的对照实验，作业不要求，想探究可以用
"""

import argparse
import json
import math
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 复用基座的结构与字表逻辑：同一份字表 = 同一个 embedding 形状，ckpt 才能直接接着训
from transformer_lm import SPECIAL_TOKENS, TransformerLM, encode

HERE = os.path.dirname(os.path.abspath(__file__))

# 三个特殊标记的 id 在建字表时就固定成了 0/1/2（见 transformer_lm.build_vocab），
# 这里按同样的顺序取名字，方便后面拼模板时读代码
USER, ASSISTANT, END = SPECIAL_TOKENS


def build_prompt(question):
    """对话模板的前半段：用户说什么 + 轮到助手了。这段不算 loss。"""
    return f"{USER}{question}{ASSISTANT}"


def read_jsonl(path):
    """读 (问题, 回答) 对；jsonl 每行一个对象，字段是 instruction / output。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                rows.append((rec["instruction"], rec["output"]))
    return rows


class SFTDataset(Dataset):
    """
    一条样本 = 模板 + 回答，x 是输入 id，y 是「每个位置的预测目标」。

    关键在 y 怎么造：自回归模型在位置 t 预测的是「下一个字」S[t+1]，
    所以对回答段（从 len(prompt) 开始）来说，第一个回答字是在位置 len(prompt)-1
    预测出来的 —— 也就是 <|assistant|> 这个位置。因此：
        y[t] = S[t+1]     当 t 落在 [len(prompt)-1, len(S)-2]
        y[t] = -100       其余位置（模板、问题、padding）
    -100 是 PyTorch cross_entropy 的 ignore_index，表示「这个位置不算 loss、不回传梯度」。
    """

    def __init__(self, rows, char2idx, seq_len, loss_mask=True):
        self.samples = []
        self.dropped = 0
        for q, a in rows:
            prompt_ids = encode(build_prompt(q), char2idx)
            answer_ids = encode(a + END, char2idx)
            if not prompt_ids or not answer_ids:
                self.dropped += 1
                continue
            ids = prompt_ids + answer_ids
            # 超长直接砍回答的尾巴（模板和问题一定要保留，否则模型学不到「答在哪个位置开始」）
            if len(ids) > seq_len:
                keep = seq_len - len(prompt_ids)
                if keep < 4:            # 问题太长，砍完不够答，直接丢
                    self.dropped += 1
                    continue
                ids = prompt_ids + answer_ids[:keep]
                answer_ids = answer_ids[:keep]
            # 至少要有 2 个回答 token 才有学习价值
            if len(answer_ids) < 2:
                self.dropped += 1
                continue
            self.samples.append((ids, len(prompt_ids), loss_mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ids, n_prompt, loss_mask = self.samples[i]
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.full((len(ids),), -100, dtype=torch.long)
        # 只在回答段给标签；--no-loss-mask 时整条序列都给（含模板和问题）
        start = (n_prompt - 1) if loss_mask else 0
        end = len(ids) - 1
        for t in range(start, end):
            y[t] = ids[t + 1]
        return x, y


def collate(batch, pad_id):
    """
    把一个 batch 的变长样本 pad 成同样长度。

    返回四样东西：
        x    —— (B, T) 输入 id，短样本右边补 pad_id
        y    —— (B, T) 目标，padding 位置是 -100（不算 loss）
        mask —— (B, T) bool，True 表示「这是 padding」→ 传进模型的 key_padding_mask
        n_tok—— 这个 batch 里真正参与 loss 的 token 数（算平均 loss / PPL 用）
    为什么要 mask：padding 位置如果参与注意力，短样本会「看到」一堆填充字符，
    学到的分布会被污染；而且 -100 只挡住 loss，挡不住注意力，所以两个都要做。
    """
    T = max(x.size(0) for x, _ in batch)
    B = len(batch)
    x_pad = torch.full((B, T), pad_id, dtype=torch.long)
    y_pad = torch.full((B, T), -100, dtype=torch.long)
    mask = torch.ones((B, T), dtype=torch.bool)     # 默认全是 padding
    n_tok = 0
    for i, (x, y) in enumerate(batch):
        n = x.size(0)
        x_pad[i, :n] = x
        y_pad[i, :n] = y
        mask[i, :n] = False                          # 真实位置标成「不是 padding」
        n_tok += int((y != -100).sum())
    return x_pad, y_pad, mask, n_tok


def build_model(ck, device, seq_len):
    """按 ckpt 里记下的超参重建模型，并载入基座权重（这就是「接着训」）。"""
    c = ck["args"]
    # max_len 必须用 ckpt 里训练时的 seq_len：位置编码表就只有那么多行，
    # 想用更长的上下文得先扩位置表（真实 LLM 做长上下文也是这么改的）
    model = TransformerLM(len(ck["char2idx"]), c["d_model"], c["heads"], c["layers"],
                          c["seq_len"], c["dropout"],
                          use_pos=not c.get("no_pos", False),
                          causal=not c.get("no_mask", False)).to(device)
    model.load_state_dict(ck["model_state"])
    if seq_len > c["seq_len"]:
        raise SystemExit(f"--seq_len {seq_len} 超过基座的 {c['seq_len']}，位置编码表不够长")
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    """在验证集上算「回答段」的 PPL —— 只看回答，模板和问题不计入。"""
    model.eval()
    total, n_tok = 0.0, 0
    for x, y, mask, nt in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits = model(x, key_padding_mask=mask)
        # ignore_index=-100 会自动跳过模板/问题/padding 位置，正是我们要的口径
        total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       y.reshape(-1), ignore_index=-100, reduction="sum"))
        n_tok += nt
    loss = total / max(1, n_tok)
    return loss, math.exp(loss)


@torch.no_grad()
def generate(model, char2idx, idx2char, question, seq_len, n_chars=40, temperature=0.7):
    """给一个问题，从 <|assistant|> 往后生成，遇到 <|end|> 就停（这正是「答完就停」）。"""
    model.eval()
    device = next(model.parameters()).device
    ids = encode(build_prompt(question), char2idx)
    out = list(ids)
    end_id = char2idx[END]
    for _ in range(n_chars):
        logits = model(torch.tensor([out[-seq_len:]], device=device))[0, -1]
        if temperature <= 0.01:
            nxt = int(logits.argmax())
        else:
            nxt = int(torch.multinomial(F.softmax(logits / temperature, dim=-1), 1))
        out.append(nxt)
        if nxt == end_id:              # 模型自己决定说完了
            break
    # 只把回答部分翻译回文字，顺便去掉结尾的 <|end|>
    answer = "".join(idx2char[i] for i in out[len(ids):] if i != end_id)
    return answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(HERE, "base_lm.pt"), help="基座 ckpt")
    ap.add_argument("--train", default=os.path.join(HERE, "sft_train.jsonl"))
    ap.add_argument("--val", default=os.path.join(HERE, "sft_val.jsonl"))
    # 模板(约 22 字) + 问题(≤48) + 回答(≤64) 最长约 134，128 能覆盖绝大多数
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    # 10 个 epoch 在 4060 上约 40 秒，验证 PPL 大约在第 9 轮见底（再练就开始记住训练集了）
    ap.add_argument("--epochs", type=int, default=10)
    # SFT 的学习率要比预训练小一个数量级：基座已经会写字了，这里只是「调格式」
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    # ★ 消融开关：关掉 loss mask，让模板和问题也参与 loss（作业不要求，仅作探究）
    ap.add_argument("--no-loss-mask", dest="no_loss_mask", action="store_true",
                    help="对照实验：模板和问题也算 loss")
    ap.add_argument("--save", default=os.path.join(HERE, "sft_mask.pt"))
    # 按什么选 ckpt：
    #   val（默认）—— 存验证集最好的那轮，正经训练的做法（防过拟合）；
    #   last       —— 存最后一轮。想复现「把训练集背下来」的效果必须用这个：
    #                 过拟合之后验证集一路变差，只存 val 最优的话你拿到的是第 1~2 轮的模型，
    #                 看起来「训练 loss 都 0.004 了怎么还是不会答」，就是这个坑。
    ap.add_argument("--save-best", dest="save_best", choices=["val", "last"], default="val")
    ap.add_argument("--eval-only", dest="eval_only", action="store_true")
    ap.add_argument("--generate", default=None, help="只生成，不训练")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.base):
        raise SystemExit(f"找不到基座 {args.base}，先跑：python transformer_lm.py")

    ck = torch.load(args.base, map_location=device)
    char2idx, idx2char = ck["char2idx"], ck["idx2char"]
    model = build_model(ck, device, args.seq_len)

    # 只生成模式：拿基座或某个 SFT 后的 ckpt 直接说一句话
    if args.generate:
        for q in [args.generate]:
            print(f"问: {q}")
            print(f"答: {generate(model, char2idx, idx2char, q, args.seq_len)}")
        return

    loss_mask = not args.no_loss_mask
    # 提示一下是从哪个 ckpt 接着训的（默认是基座）
    if os.path.basename(args.base) != "base_lm.pt":
        print(f"（从 {os.path.basename(args.base)} 接着训）")
    print(f"模板 = {USER}问题{ASSISTANT}回答{END}   loss mask = {'开' if loss_mask else '**关（消融）**'}")

    N_SPECIAL = len(SPECIAL_TOKENS)
    pad_id = char2idx[END]        # 用 <|end|> 当 pad：它在标签里永远是 -100，不会被学到
    V = len(char2idx)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n_par/1e6:.2f}M   字表 {V}（其中 {N_SPECIAL} 个特殊标记）")

    tr = SFTDataset(read_jsonl(args.train), char2idx, args.seq_len, loss_mask)
    # ★ 验证集永远只算「回答段」：否则关掉 loss mask 的那次实验里，
    #   验证集的数字会把问题和模板也算进去，两次结果没法横向比
    va = SFTDataset(read_jsonl(args.val), char2idx, args.seq_len, loss_mask=True)
    print(f"训练样本 {len(tr)}（丢弃 {tr.dropped}）  验证样本 {len(va)}（丢弃 {va.dropped}）")

    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate(b, pad_id), drop_last=True)
    val_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate(b, pad_id))

    if args.eval_only:
        loss, ppl = evaluate(model, val_loader, device)
        print(f"[eval-only] 回答段 loss={loss:.4f}  PPL={ppl:.2f}")
        return

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    best_ep = 0
    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>9} {'ValPPL(回答段)':>15}")
    print("-" * 46)
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, seen = 0.0, 0
        for x, y, mask, nt in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x, key_padding_mask=mask)
            # ★ 整个 SFT 的关键就这一行的 ignore_index=-100：
            #   被 mask 掉的位置（模板、问题、padding）不产生梯度，
            #   模型只从「回答」上学 —— 这就是「只学怎么答，不背怎么问」。
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * nt
            seen += nt
        v_loss, v_ppl = evaluate(model, val_loader, device)
        mark = ""
        # --save-best val：只存验证集最好的那轮（防过拟合，正经训练该这么做）
        # --save-best last：每轮都覆盖，最后存下的是最后一轮（想「背下训练集」时用这个）
        if args.save_best == "last" or v_loss < best:
            if v_loss < best:
                best, best_ep = v_loss, ep
            mark = " *"
            torch.save({"model_state": model.state_dict(), "char2idx": char2idx,
                        "idx2char": idx2char, "args": ck["args"],
                        "sft": {"loss_mask": loss_mask, "seq_len": args.seq_len,
                                "epochs": args.epochs, "val_ppl": v_ppl,
                                "train_loss": tot / seen, "epoch": ep}}, args.save)
        print(f"{ep:>6} {tot/seen:>10.4f} {v_loss:>9.4f} {v_ppl:>15.2f}{mark}")

    if args.save_best == "last":
        print(f"\n已保存最后一轮（epoch {args.epochs}）→ {args.save}")
    else:
        print(f"\n最佳回答段 PPL: {math.exp(best):.2f}（epoch {best_ep}）已保存 {args.save}")
    # 训练完顺手生成两条，方便立刻确认「它会不会答、会不会停」
    print("\n生成示例：")
    for q, _ in read_jsonl(args.val)[:3]:
        print(f"  问: {q}")
        print(f"  答: {generate(model, char2idx, idx2char, q, args.seq_len)}")


if __name__ == "__main__":
    main()
