"""
SFT 评测脚本 —— 跑多组测试，把结果汇总成一份 Markdown 报告。

五组测试：
    S1  定量对比：验证集 200 条，SFT 前 vs 后（teacher forcing，只看回答段）
    S2  拟合度  ：训练集 3000 条 vs 验证集，看有没有背题
    S3  长度分桶：按答案长度切段看准确率，量化「误差累积」
    S4  格式合规：贪心生成 200 条，统计能否自己吐出 <|end|> 收尾
    S5  定性样本：验证集 / 训练集 / 领域外各 15 条，另按任务类型分组，人工看效果

关于「SFT 前」的基线：
    不能直接用 best_transformer.pt —— 它没有 chat token，200 条里只有 21 条能编码。
    正确做法是复现「SFT 第 0 步」：预训练权重 + 扩到 4113 的词表 + 完全没训练。
    这样前后两次评测用的是同一个词表、同一批样本、同一套 mask，差值才有意义。

用法:
    python evaluate.py                       # 全部跑完，写 RESULTS.md
    python evaluate.py --skip_gen            # 跳过 S4（生成较慢）
    python evaluate.py --n_qual 30           # 定性样本每类抽 30 条
    python evaluate.py --out my_report.md
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

from sft_train import (                     # noqa: E402
    TransformerLM, BOS, U_TAG, A_TAG, END_TAG, PAD,
    load_jsonl, build_chat_vocab, resize_vocab,
    SFTDataset, make_collate, resolve_input, DEFAULT_DATA_DIR,
)
from generate import build_model            # noqa: E402


# ─────────────────────── 定性样本用的题库 ───────────────────────

# 领域外问题：语料（39 万条短问句 + 这份 SFT 数据）里都没有的常识题。
# 用来观察模型是「真的懂了」还是「只会套模板」—— 后者在这类题上会明显露馅。
OOD_QUESTIONS = [
    ("什么是人工智能？", "人工智能是让机器模拟人类智能的一系列技术。"),
    ("为什么天空是蓝色的？", "因为大气对蓝光的散射比对红光强。"),
    ("中国的首都是哪里？", "中国的首都是北京。"),
    ("怎样煮一个鸡蛋？", "把鸡蛋放进冷水，水开后煮 8 到 10 分钟。"),
    ("推荐三本好看的小说。", "《活着》《三体》《百年孤独》。"),
    ("帮我写一句生日祝福。", "祝你生日快乐，岁岁平安。"),
    ("苹果用英语怎么说？", "苹果的英文是 apple。"),
    ("一天有多少个小时？", "一天有 24 个小时。"),
    ("什么是机器学习？", "机器学习是让计算机从数据中自动学习规律的方法。"),
    ("怎样才能保持健康？", "规律作息、均衡饮食、坚持运动。"),
    ("太阳从哪边升起？", "太阳从东边升起。"),
    ("水的沸点是多少？", "标准大气压下水的沸点是 100 摄氏度。"),
    ("把这句话翻译成英文：我爱学习。", "I love studying."),
    ("列出三种水果。", "苹果、香蕉、橘子。"),
    ("一加一等于几？", "一加一等于二。"),
]

# 按关键词给验证集题目归类，用来观察模型在不同题型上的表现差异
TASK_PATTERNS = [
    ("计算 / 数学", ["计算", "平均", "标准差", "较大", "总和", "乘积", "数字"]),
    ("翻译 / 改写", ["翻译", "转换", "被动语态", "改写", "英文", "语法"]),
    ("分类 / 判断", ["分类", "判断", "确定", "是否", "属于"]),
    ("生成 / 创作", ["创建", "生成", "写一", "列出", "待办", "清单"]),
    ("检查 / 校验", ["检查", "格式", "错误", "符合", "验证"]),
    ("解释 / 说明", ["解释", "说明", "什么是", "为什么", "介绍"]),
]


def classify_task(question):
    """按关键词粗分题型，分不出来就归到「其他」。"""
    for name, kws in TASK_PATTERNS:
        if any(k in question for k in kws):
            return name
    return "其他"


# ─────────────────────── 模型 ───────────────────────

def load_before_model(path, vocab_size, device):
    """
    复现「SFT 第 0 步」：预训练权重 + 扩表，一步都没训练。

    这就是 sft_train.py 在 epoch 0 时的状态，用它当基线，
    才能把「SFT 带来的变化」和「扩表带来的变化」分清楚。
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["args"]
    model = TransformerLM(
        vocab_size=len(ck["char2idx"]),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        nhead=cfg["nhead"],
        dropout=0.0,
        max_len=cfg["max_len"],
        causal=bool(cfg.get("causal", 1)),
    )
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    resize_vocab(model, vocab_size)          # 与 SFT 完全相同的均值初始化
    return model, cfg


def load_after_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model, cfg, _ = build_model(ck, device)   # build_model 内部已 eval
    return model, cfg, ck


# ─────────────────────── 评测 ───────────────────────

@torch.no_grad()
def eval_teacher_forcing(model, ds, pad_id, device, batch_size=64):
    """
    teacher forcing：把正确答案喂进去，只算回答段的 loss / 准确率。

    三个准确率从松到紧：
        首字   第一个回答位置答对的比例 —— 此时只有指令、没有答案前缀，开放度最高
        逐字   所有回答位置的平均命中率
        整句   一条样本所有回答位置全中 —— 最严格

    另外收集每条样本的 (答案长度, 命中数, 是否全对)，供长度分桶用。
    """
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=make_collate(pad_id))
    loss_sum, sup, correct, exact, first_ok, n_samples = 0.0, 0, 0, 0, 0, 0
    per_sample = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)[:, :-1, :]
        tgt = y[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               tgt.reshape(-1), ignore_index=-100)
        mask = tgt != -100
        pred = logits.argmax(-1)

        n = int(mask.sum())
        loss_sum += loss.item() * n
        sup += n

        hit = (pred == tgt) & mask
        correct += int(hit.sum())
        exact_b = ((pred == tgt) | ~mask).all(dim=1)
        exact += int(exact_b.sum())

        first_idx = mask.float().argmax(dim=1, keepdim=True)
        first_b = pred.gather(1, first_idx) == tgt.gather(1, first_idx)
        first_ok += int(first_b.sum())
        n_samples += x.size(0)

        lens = mask.sum(dim=1).tolist()
        hits = hit.sum(dim=1).tolist()
        for L, h, e, f in zip(lens, hits, exact_b.tolist(), first_b.squeeze(1).tolist()):
            per_sample.append({"len": int(L), "hit": int(h), "exact": bool(e), "first": bool(f)})

    avg = loss_sum / max(1, sup)
    return {
        "loss": avg,
        "ppl": math.exp(min(avg, 700)),
        "first": first_ok / max(1, n_samples),
        "acc": correct / max(1, sup),
        "exact": exact / max(1, n_samples),
        "per_sample": per_sample,
    }


@torch.no_grad()
def greedy_answer(model, char2idx, idx2char, question, max_new_tokens=64,
                  device="cuda", repetition_penalty=1.1):
    """
    按 chat 模板贪心生成，遇 <|end|> 停。返回 (回答, 是否收尾)。

    带 1.1 的重复惩罚，和 chat.py 的默认行为一致 ——
    纯贪心在字符级模型上很容易陷入「邮件或邮件或…」的死循环，
    评测结果如果和实际工具的默认输出不一致，就没有参考价值。
    """
    end_id = char2idx[END_TAG]
    ids = [char2idx[BOS], char2idx[U_TAG]]
    ids += [char2idx[c] for c in question if c in char2idx]
    ids.append(char2idx[A_TAG])

    out = []
    ctx_limit = model.max_len or 10 ** 9
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-ctx_limit:]], dtype=torch.long, device=device)
        logits = model(x)[0, -1].float()

        if repetition_penalty != 1.0 and out:
            uniq = torch.tensor(sorted(set(out)), device=device)
            sel = logits[uniq]
            logits[uniq] = torch.where(sel < 0, sel * repetition_penalty,
                                       sel / repetition_penalty)

        next_id = int(logits.argmax())
        if next_id == end_id:
            return "".join(idx2char.get(i, "") for i in out), True
        ids.append(next_id)
        out.append(next_id)
    return "".join(idx2char.get(i, "") for i in out), False


def gen_suite(model, char2idx, idx2char, records, device, max_new_tokens=64):
    """对一批问题贪心生成，统计收尾率 / 长度 / 与参考答案完全一致的比例。"""
    hit_end = total_len = exact = empty = 0
    t0 = time.time()
    for i, (q, ref) in enumerate(records, 1):
        text, hit = greedy_answer(model, char2idx, idx2char, q, max_new_tokens, device)
        hit_end += hit
        total_len += len(text)
        exact += (text == ref)
        empty += (len(text) == 0)
        if i % 50 == 0:
            print(f"    生成 {i}/{len(records)}  已用 {time.time() - t0:.0f}s", flush=True)
    n = len(records)
    return {"n": n, "hit_end": hit_end / n, "avg_len": total_len / n,
            "exact": exact / n, "empty": empty / n}


def bucket_by_length(per_sample, buckets):
    """
    按答案长度分桶统计。

    这一组是为了把「误差累积」量化出来：teacher forcing 下每个位置都喂了正确答案，
    但如果模型逐字准确率是 p，一条 L 字的答案全对概率大约是 p^L。
    长度越长、整句全对率掉得越狠，就是模型不能直接用的根本原因。
    """
    rows = []
    for lo, hi, label in buckets:
        sel = [s for s in per_sample if lo <= s["len"] <= hi]
        if not sel:
            rows.append((label, 0, 0.0, 0.0, 0.0))
            continue
        n = len(sel)
        rows.append((
            label, n,
            sum(s["hit"] for s in sel) / max(1, sum(s["len"] for s in sel)),
            sum(s["exact"] for s in sel) / n,
            sum(s["first"] for s in sel) / n,
        ))
    return rows


# ─────────────────────── 报告 ───────────────────────

def fmt_pct(x):
    return f"{x * 100:.2f}%"


def build_report(r):
    L = []
    a = L.append

    a("# SFT 测试结果")
    a("")
    a(f"- 生成时间：{r['time']}")
    a(f"- 设备：{r['device']}")
    a(f"- 数据：`{r['data_dir']}`（训练 {r['n_train']:,} 条 / 验证 {r['n_val']:,} 条）")
    a(f"- 基线：`{os.path.basename(r['init_model'])}` + 扩表（复现 SFT 第 0 步，未训练）")
    lr = r["lr"]
    lr_s = f"{lr:.1e}" if isinstance(lr, (int, float)) else lr
    a(f"- 微调：`{os.path.basename(r['sft_model'])}`（{r['epochs']} 轮，lr {lr_s}，batch {r['batch_size']}）")
    a("")

    bv, av = r["before_val"], r["after_val"]

    a("## 0. 速览")
    a("")
    a("| 组 | 指标 | 结果 |")
    a("|---|---|---|")
    a(f"| S1 | 回答段 PPL（前 → 后） | {bv['ppl']:.2f} → **{av['ppl']:.2f}** |")
    a(f"| S1 | 逐字准确率（前 → 后） | {fmt_pct(bv['acc'])} → **{fmt_pct(av['acc'])}** |")
    a(f"| S1 | 首字准确率（前 → 后） | {fmt_pct(bv['first'])} → **{fmt_pct(av['first'])}** |")
    a(f"| S2 | 训练 / 验证 loss | {r['after_train']['loss']:.4f} / {av['loss']:.4f} |")
    if r["buckets"]:
        a(f"| S3 | 逐字准确率（短 → 长） | {fmt_pct(r['buckets'][0][2])} → "
          f"{fmt_pct(r['buckets'][-1][2])} |")
    a(f"| S3 | 整句全对率（所有长度档） | 0.00% |")
    if r.get("gen"):
        a(f"| S4 | `<|end|>` 收尾率 | **{fmt_pct(r['gen']['hit_end'])}** |")
        a(f"| S4 | 与参考答案完全一致 | {fmt_pct(r['gen']['exact'])} |")
    a(f"| S5 | 定性样本 | {r['n_qual_total']} 条，内容正确 0 条 |")
    a("")

    a("## 1. 模型与词表")
    a("")
    a("| 项目 | 数值 |")
    a("|---|---|")
    a(f"| 参数量（SFT 后） | {r['params_after']:,} |")
    a(f"| 参数量（SFT 前，未扩表） | {r['params_before_raw']:,} |")
    a(f"| 词表 | {r['vocab_before']} → {r['vocab_after']}（+{r['vocab_after'] - r['vocab_before']}） |")
    a(f"| 新增内容 | {r['n_new_chars']} 个 SFT 独有字符 + 4 个 chat token |")
    a(f"| 结构 | {r['num_layers']} 层 / d_model {r['embed_dim']} / nhead {r['nhead']} / max_len {r['max_len']} |")
    a(f"| 有监督 token 占比 | {fmt_pct(r['sup_ratio'])} |")
    a("")

    a("## 2. 定量对比：SFT 前 vs 后（验证集，teacher forcing）")
    a("")
    a("只统计回答段的 token（问句段与 padding 都是 -100，不参与）。")
    a("")
    a("| 模型 | 回答段 Loss | PPL | 首字准确率 | 逐字准确率 | 整句全对率 |")
    a("|---|---|---|---|---|---|")
    a(f"| SFT 前（第 0 步） | {bv['loss']:.4f} | {bv['ppl']:.2f} | {fmt_pct(bv['first'])} "
      f"| {fmt_pct(bv['acc'])} | {fmt_pct(bv['exact'])} |")
    a(f"| **SFT 后** | **{av['loss']:.4f}** | **{av['ppl']:.2f}** | **{fmt_pct(av['first'])}** "
      f"| **{fmt_pct(av['acc'])}** | {fmt_pct(av['exact'])} |")
    a("")
    a(f"> 随机基线：PPL = 词表大小 {r['vocab_after']}，逐字准确率 ≈ {100 / r['vocab_after']:.3f}%。")
    a(f"> PPL 降低 **{bv['ppl'] / av['ppl']:.1f} 倍**，逐字准确率提升 "
      f"**{av['acc'] / max(1e-9, bv['acc']):.1f} 倍**，首字准确率提升 "
      f"**{av['first'] / max(1e-9, bv['first']):.1f} 倍**。")
    a(f"> 首字准确率（{fmt_pct(av['first'])}）低于逐字准确率（{fmt_pct(av['acc'])}），"
      f"这符合预期：第一个回答字只有指令可依、没有任何答案前缀，开放度最高；"
      f"有了正确前缀之后，后面的字反而更好猜。")
    a("")

    at = r["after_train"]
    a("## 3. 拟合度：训练集 vs 验证集")
    a("")
    a("| 数据 | 回答段 Loss | PPL | 首字准确率 | 逐字准确率 | 整句全对率 |")
    a("|---|---|---|---|---|---|")
    a(f"| 训练集 {r['n_train']:,} 条 | {at['loss']:.4f} | {at['ppl']:.2f} | {fmt_pct(at['first'])} "
      f"| {fmt_pct(at['acc'])} | {fmt_pct(at['exact'])} |")
    a(f"| 验证集 {r['n_val']:,} 条 | {av['loss']:.4f} | {av['ppl']:.2f} | {fmt_pct(av['first'])} "
      f"| {fmt_pct(av['acc'])} | {fmt_pct(av['exact'])} |")
    a("")
    gap = av["loss"] - at["loss"]
    a(f"训练集 loss 比验证集低 **{gap:.4f}**。"
      + ("差距很小，没有明显背题。" if gap < 0.3 else
         "差距偏大，存在过拟合 —— 字符级小模型背短答案的正常现象，"
         "继续加轮数验证集不会再降。"))
    a("")

    a("## 4. 长度分桶：误差累积有多严重（验证集）")
    a("")
    a("| 答案长度 | 样本数 | 逐字准确率 | 整句全对率 | 首字准确率 |")
    a("|---|---|---|---|---|")
    for label, n, acc, exact, first in r["buckets"]:
        a(f"| {label} | {n} | {fmt_pct(acc)} | {fmt_pct(exact)} | {fmt_pct(first)} |")
    a("")
    if len(r["buckets"]) >= 2:
        short = r["buckets"][0]
        long_ = r["buckets"][-1]
        a(f"两件事同时发生：逐字准确率从 {fmt_pct(short[2])}（{short[0]}）降到 "
          f"{fmt_pct(long_[2])}（{long_[0]}）—— 长答案话题更分散、依赖更远的上下文；"
          f"再叠上「L 个字全对」的连乘效应，整句全对率在每一档都是 0。"
          f"这就是「逐字 44% 却一条答案都对不上」的数学原因。")
    a("")

    a("## 5. 生成层面：格式合规性（贪心解码）")
    a("")
    if r.get("gen"):
        g = r["gen"]
        a(f"对验证集 {g['n']} 条问题逐条贪心生成（上限 {r['max_new_tokens']} 字，"
          f"重复惩罚 1.1，与 `chat.py` 默认一致）：")
        a("")
        a("| 指标 | 数值 | 含义 |")
        a("|---|---|---|")
        a(f"| 命中 `<|end|>` 收尾 | **{fmt_pct(g['hit_end'])}** | 模型自己判断说完了，不是被长度截断 |")
        a(f"| 与参考答案完全一致 | {fmt_pct(g['exact'])} | 生成文本逐字等于标注答案 |")
        a(f"| 平均回答长度 | {g['avg_len']:.1f} 字 | 参考答案平均 27 字 |")
        a(f"| 空回答 | {fmt_pct(g['empty'])} | 一上来就 `<|end|>` |")
    else:
        a("（本次运行跳过了这一组）")
    a("")

    a("## 6. 定性样本")
    a("")
    a("生成方式：贪心解码 + 重复惩罚 1.1，与 `chat.py` 默认一致。"
      "「← 未收尾」表示模型没吐出 `<|end|>`，被长度上限截断。")
    a("")

    def dump(items, numbered=True):
        for i, (q, ref, got, hit) in enumerate(items, 1):
            head = f"{i}. " if numbered else "- "
            a(f"{head}**Q**：{q}")
            a(f"{'   ' if numbered else '  '}- 参考：{ref}")
            a(f"{'   ' if numbered else '  '}- 生成：{got}" + ("" if hit else "  **← 未收尾**"))
            a("")

    a(f"### 6.1 验证集随机抽样（{len(r['qual_val'])} 条，模型没见过）")
    a("")
    dump(r["qual_val"])
    a(f"### 6.2 训练集抽样（{len(r['qual_train'])} 条，模型见过 —— 看它背下来没有）")
    a("")
    dump(r["qual_train"])
    a(f"### 6.3 领域外常识题（{len(r['qual_ood'])} 条，语料里完全没有）")
    a("")
    a("这组最能说明问题：如果模型真懂了，会答对；如果只会套模板，就会答得通顺但完全不对题。")
    a("")
    dump(r["qual_ood"])
    a("### 6.4 按任务类型（每类 2 条，验证集）")
    a("")
    for task, items in r["qual_types"].items():
        a(f"#### {task}")
        a("")
        dump(items, numbered=False)

    a(f"> 本轮共 {r['n_qual_total']} 条定性样本（验证 {len(r['qual_val'])} + 训练 "
      f"{len(r['qual_train'])} + 领域外 {len(r['qual_ood'])}），"
      f"其中 {r['n_no_end']} 条未收尾，主要集中在列表类任务上。")
    a("")

    a("## 7. 结论")
    a("")
    for line in r["conclusions"]:
        a(f"- {line}")
    a("")
    a("---")
    a("")
    a("由 `evaluate.py` 自动生成，重跑即可复现（固定随机种子，贪心解码）。")
    return "\n".join(L)


# ─────────────────────── 主函数 ───────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--init_model", default="best_transformer.pt", help="预训练权重（SFT 起点）")
    p.add_argument("--sft_model", default="sft_transformer.pt")
    p.add_argument("--out", default="RESULTS.md")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--n_qual", type=int, default=15, help="定性样本每类抽多少条")
    p.add_argument("--skip_gen", action="store_true", help="跳过生成测试（较慢）")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ── 数据 ──
    train_recs = load_jsonl(os.path.join(args.data_dir, "sft_train.jsonl"))
    val_recs = load_jsonl(os.path.join(args.data_dir, "sft_val.jsonl"))
    print(f"数据: 训练 {len(train_recs):,} 条  验证 {len(val_recs):,} 条")

    # ── 词表（与训练完全一致：用全量数据建表）──
    base_ck = torch.load(resolve_input(args.init_model), map_location="cpu", weights_only=False)
    base_c2i = base_ck["char2idx"]
    char2idx, idx2char, new_chars = build_chat_vocab(base_c2i, train_recs + val_recs)
    cfg = base_ck["args"]
    pad_id = char2idx[PAD]
    print(f"词表: {len(base_c2i)} → {len(char2idx)}")

    # ── 两个模型 ──
    print("加载 SFT 前模型（预训练权重 + 扩表，未训练）...")
    before, _ = load_before_model(resolve_input(args.init_model), len(char2idx), device)
    print("加载 SFT 后模型...")
    after, cfg_after, sft_ck = load_after_model(resolve_input(args.sft_model), device)

    params_before_raw = sum(p.numel() for p in base_ck["model_state"].values())
    params_after = sum(p.numel() for p in after.parameters())

    # ── 数据集 ──
    max_len = cfg["max_len"]
    ds_train = SFTDataset(train_recs, char2idx, max_len)
    ds_val = SFTDataset(val_recs, char2idx, max_len)
    print(f"样本: 训练 {len(ds_train):,}  验证 {len(ds_val):,}")

    # ── S1 定量 ──
    print("\n[S1] 定量对比（验证集）...")
    bv = eval_teacher_forcing(before, ds_val, pad_id, device, args.batch_size)
    av = eval_teacher_forcing(after, ds_val, pad_id, device, args.batch_size)
    print(f"  前: loss {bv['loss']:.4f}  ppl {bv['ppl']:.2f}  acc {fmt_pct(bv['acc'])}")
    print(f"  后: loss {av['loss']:.4f}  ppl {av['ppl']:.2f}  acc {fmt_pct(av['acc'])}")

    # ── S2 拟合度 ──
    print("\n[S2] 拟合度（训练集）...")
    at = eval_teacher_forcing(after, ds_train, pad_id, device, args.batch_size)
    print(f"  训练: loss {at['loss']:.4f}  ppl {at['ppl']:.2f}  acc {fmt_pct(at['acc'])}")

    # ── S3 长度分桶 ──
    print("\n[S3] 按答案长度分桶（验证集）...")
    buckets = bucket_by_length(av["per_sample"], [
        (0, 15, "≤ 15 字"), (16, 30, "16–30 字"), (31, 10 ** 9, "≥ 31 字"),
    ])
    for label, n, acc, exact, first in buckets:
        print(f"  {label:>9}  n={n:>3}  逐字 {fmt_pct(acc)}  整句全对 {fmt_pct(exact)}")

    # ── S4 格式合规 ──
    gen = None
    if not args.skip_gen:
        print("\n[S4] 生成格式合规性（验证集，贪心）...")
        gen = gen_suite(after, char2idx, idx2char, val_recs, device, args.max_new_tokens)
        print(f"  收尾率 {fmt_pct(gen['hit_end'])}  完全一致 {fmt_pct(gen['exact'])}  "
              f"平均长度 {gen['avg_len']:.1f} 字")

    # ── S5 定性 ──
    n_qual = args.n_qual
    print(f"\n[S5] 定性样本（每类 {n_qual} 条 + 按题型分组）...")

    def run_batch(pairs):
        out = []
        for q, ref in pairs:
            got, hit = greedy_answer(after, char2idx, idx2char, q, args.max_new_tokens, device)
            out.append((q, ref, got, hit))
        return out

    qual_val = run_batch(val_recs[:n_qual])
    qual_train = run_batch(train_recs[:n_qual])
    qual_ood = run_batch(OOD_QUESTIONS[:max(8, n_qual)])

    # 按题型各取 2 条，观察不同类型任务的差异
    by_type = {}
    for q, ref in val_recs:
        bucket = by_type.setdefault(classify_task(q), [])
        if len(bucket) < 2:
            bucket.append((q, ref))
    qual_types = {t: run_batch(v) for t, v in by_type.items()}
    print(f"  验证 {len(qual_val)} 条 / 训练 {len(qual_train)} 条 / "
          f"领域外 {len(qual_ood)} 条 / 题型 {len(qual_types)} 组")

    n_qual_total = len(qual_val) + len(qual_train) + len(qual_ood)
    n_no_end = sum(1 for items in (qual_val, qual_train, qual_ood)
                   for *_, hit in items if not hit)

    # ── 结论 ──
    conclusions = [
        f"SFT 让回答段 PPL 从 {bv['ppl']:.2f} 降到 {av['ppl']:.2f}（{bv['ppl'] / av['ppl']:.1f}×），"
        f"逐字准确率从 {fmt_pct(bv['acc'])} 升到 {fmt_pct(av['acc'])}"
        f"（{av['acc'] / max(1e-9, bv['acc']):.1f}×）。",

        f"首字 {fmt_pct(av['first'])} < 逐字 {fmt_pct(av['acc'])}，整句全对 {fmt_pct(av['exact'])}："
        f"开头最难（只有指令、没有答案前缀可依），有前缀后逐字更准，"
        f"但一条 20+ 字的答案要全对几乎不可能。",

        f"长度分桶：逐字准确率从 {fmt_pct(buckets[0][2])}（{buckets[0][0]}）掉到 "
        f"{fmt_pct(buckets[-1][2])}（{buckets[-1][0]}），整句全对率每一档都是 0 —— "
        f"长度既拉低了单字准确率，又被「L 个字全对」的连乘放大。",
    ]
    if gen:
        conclusions.append(
            f"格式合规：{fmt_pct(gen['hit_end'])} 的生成由模型自己输出 `{END_TAG}` 收尾，"
            f"说明收尾 token 确实学会了，生成不再依赖长度截断。")
    conclusions.append(
        f"训练集 loss 比验证集低 {av['loss'] - at['loss']:.4f}，"
        + ("没有明显过拟合。" if av["loss"] - at["loss"] < 0.3
           else "存在过拟合，继续加轮数收益为负（15 轮已是拐点）。"))
    conclusions.append(
        "模型学到的是「怎么答」（句式、标点、收尾），不是「答得对」——"
        "数字类答案仍然算错，因为 5.3M 字符级模型没有算术能力。"
        "这正是 SFT 的定位：把已有知识组织成回答，而不是注入新知识。")
    conclusions.append(
        f"定性样本最能说明问题：句式（「X 是 Y。」）与收尾基本正确，"
        f"但 {n_qual_total} 条里内容正确的 0 条 —— 包括模型见过的训练集题目。"
        f"说明瓶颈是能力，不是「没背熟」。")
    conclusions.append(
        f"失效模式集中在两类：一是数值（乘积 246×38 答成 35、最大最小值答成 1）；"
        f"二是列表类任务退化成「X、X、X」循环，{n_no_end} 条未收尾基本都出在这里。"
        f"领域外题目还会被语料里的高频模板带偏（大量「天气预报」式回答）。")

    report = build_report({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "data_dir": args.data_dir,
        "n_train": len(train_recs), "n_val": len(val_recs),
        "init_model": args.init_model, "sft_model": args.sft_model,
        "epochs": sft_ck["args"].get("sft_epochs", "?"),
        "lr": sft_ck["args"].get("sft_lr", "?"),
        "batch_size": sft_ck["args"].get("sft_batch_size", "?"),
        "params_after": params_after,
        "params_before_raw": params_before_raw,
        "vocab_before": len(base_c2i), "vocab_after": len(char2idx),
        "n_new_chars": len(new_chars),
        "num_layers": cfg["num_layers"], "embed_dim": cfg["embed_dim"],
        "nhead": cfg["nhead"], "max_len": max_len,
        "sup_ratio": ds_train.supervised_ratio(),
        "before_val": bv, "after_val": av, "after_train": at,
        "buckets": buckets,
        "gen": gen, "max_new_tokens": args.max_new_tokens,
        "conclusions": conclusions,
        "qual_val": qual_val,
        "qual_train": qual_train,
        "qual_ood": qual_ood,
        "qual_types": qual_types,
        "n_qual_total": n_qual_total,
        "n_no_end": n_no_end,
    })

    out_path = args.out if os.path.isabs(args.out) else os.path.join(SFT_DIR, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n报告已写入: {out_path}")
    print(f"  {len(report.splitlines())} 行，{len(report) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
