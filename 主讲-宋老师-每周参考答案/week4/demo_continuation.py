"""
demo_continuation.py — 课堂演示：同一段开头，让「有因果掩码」和「无因果掩码」两个模型分别续写

依赖同目录的 transformer_lm.py（模型定义与数据口径完全复用，不另起一套）。

用法：
    # 1) 先用训练脚本各训一个（默认语料是 80 万字样例；想效果好换全量语料）
    python transformer_lm.py --epochs 6 --save tf_mask.pt
    python transformer_lm.py --epochs 6 --no-mask --save tf_nomask.pt

    # 2) 课堂演示：一组 prompt 并排看
    python demo_continuation.py

    # 3) 课上一句一句输入（回车结束）
    python demo_continuation.py --interactive

    # 4) 想固定输出、方便反复讲同一个例子
    python demo_continuation.py --prompt "今天天气" --temperature 0 --n_chars 40
"""

# ═══════════════════ 这个演示想说明什么（讲「BERT 为什么不适合生成」时用） ═══════════════════
# 两个模型是「同一份语料、同一套代码、同样的结构」训出来的，唯一差别就是训练时有没有加
# 因果掩码（--no-mask）。所以它们续写风格的差异，只能归因于这一件事 —— 这就是控制变量。
#
# 跑完会看到两组很反直觉的现象，正好是本演示的核心：
#   1) 无 mask 的模型「验证集 PPL 只有 1.07」，看起来强得离谱；
#      但让它续写，它只会把开头的最后一个字复读 40 遍（「们们们们…」「天天天天…」）。
#   2) 同一个无 mask 模型，只看「每段最后一个位置」的 PPL 是 88.50，比有 mask 的 13.19 差得多。
#
# 原因一句话：无掩码时位置 t 的输出能直接看到 x_t（也就是要预测的标签的一部分），
# 「把当前字抄到输出」就能拿满分 —— 全位置 PPL 1.07 是抄答案抄出来的假象；
# 生成时喂进去的是上文、要预测的字根本不在输入里，没得抄，模型立刻现原形。
# BERT 就是这种「双向、能看到未来」的结构：它天生适合做填空（MLM），不适合从左到右写文章。
# ↑ 演示代码本身不改模型，只负责「并排跑出来、并把复读程度量化」——
#   让你不用盯着屏幕数「们」有多少个，也能一眼看出谁在退化。

import argparse
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 直接 import 训练脚本里的模型/数据集/评测函数，保证演示和作业代码是「同一套口径」：
# 一旦评测函数被改，演示里的 PPL 会跟着变，不会出现「两份代码算出两个数」的情况
from transformer_lm import CharDataset, TransformerLM, evaluate, load_corpus

HERE = os.path.dirname(os.path.abspath(__file__))
# 算 PPL 的兜底语料：ckpt 里没记语料路径时用它（80 万字样例，和训练默认值一致）
DEFAULT_CORPUS = os.path.join(HERE, "corpus_sample.txt")


def load_model(path, device):
    """
    从 ckpt 还原一个可推理的模型。

    关键是「结构超参从 ckpt 里读、不写死」：
        ckpt 里存了训练时的 args，用它们重建模型才能保证各层形状和 state_dict 对得上；
        如果这里把 d_model/layers 写成常量，换个 ckpt 就会 load_state_dict 报形状不匹配。
    dropout 传 0.0：反正 eval() 已经关掉了 dropout，传 0 是双保险（也不希望推理结果随机漂）。
    no_pos / no_mask 用 .get(..., False)：向后兼容早期训练存下的 ckpt —— 那时 args 里可能
    还没有这两个键，直接用 a["no_mask"] 会 KeyError。
    """
    ck = torch.load(path, map_location=device)
    a = ck["args"]
    model = TransformerLM(
        len(ck["char2idx"]), a["d_model"], a["heads"], a["layers"], a["seq_len"], 0.0,
        use_pos=not a.get("no_pos", False),
        causal=not a.get("no_mask", False),
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


@torch.no_grad()
def generate(model, ck, prompt, n_chars, temperature, top_k, seed, ban=""):
    """
    逐字续写 n_chars 个字。推理必须一个字一个字来（第 t+1 个字要先采样出来才能当第 t+2 个的前文），
    这也是「自回归」和「训练时可并行」的分水岭 —— 详见 transformer_lm.generate 的说明。
    """
    c2i, i2c = ck["char2idx"], ck["idx2char"]
    seq_len = ck["args"]["seq_len"]
    device = next(model.parameters()).device
    # 开头里字表没有的字符直接跳过；一个都没有就返回提示，否则后面 out[-seq_len:] 会取到空列表
    ids = [c2i[ch] for ch in prompt if ch in c2i]
    if not ids:
        return "（开头里的字都不在字表里）"
    # ban 是把「禁用字符」的 logit 压成 -inf，让 softmax 之后它的概率严格为 0 ——
    # 这样它是「从来没被采样到」，而不是「生成了再过滤」（后者会让输出变短、字数不对）。
    # 默认屏蔽换行：语料是按行拼起来的，不屏蔽的话模型很容易生成回车跳到另一条无关语料上，
    # 看起来像在列问题清单，课堂上会分散注意力。
    ban_ids = [c2i[ch] for ch in ban if ch in c2i]
    # 固定采样种子：两个模型用同一个 seed 走采样，随机性来源一致，对比才公平、课堂上也可复现
    torch.manual_seed(seed)
    out = list(ids)
    for _ in range(n_chars):
        # 只喂最近 seq_len 个 token：位置编码只有 seq_len 行，训练时模型也只见过这么长的上下文。
        # [0, -1] = batch 里第一条、最后一个位置的 logits，即对「下一个字」的预测
        logits = model(torch.tensor([out[-seq_len:]], device=device))[0, -1]
        if temperature <= 0.01:
            # 温度≈0 就当贪心解码：取概率最大的那个字，结果完全确定。
            # 课上想反复讲同一个例子时用这个（--temperature 0），输出每次都一样
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature      # 除以温度：<1 更尖锐（保守）、>1 更平坦（发散）
            for b in ban_ids:
                logits[b] = float("-inf")
            if top_k and top_k > 0:
                # top-k 截断：只保留概率最大的 k 个候选，其余压成 -inf。
                # 目的不是「更准」，而是砍掉长尾 —— 4000 个字里总有一堆极低概率的怪字，
                # 不截断就容易蹦出莫名其妙的一个，看起来像坏掉
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = float("-inf")
            # 从（温度缩放 + 屏蔽 + top-k 之后的）分布里按概率采样一个，而不是永远取最大：
            # 贪心解码在语言模型上很容易陷入「复读循环」，采样能缓解这个问题
            nxt = int(torch.multinomial(F.softmax(logits, dim=-1), 1))
        out.append(nxt)
    # 只返回新生成的部分（out 前面是用户给的开头）
    return "".join(i2c[i] for i in out[len(ids):])


def repeat_stats(text):
    """
    返回（相邻重复率, 最长连续重复片段），用来量化「气气气气」这种退化

    为什么要量化：肉眼数「们」有几个既费劲又容易看错，而且课上要能一句话说清
    「无 mask 的模型重复率 0.98、最长连续 40 个字」—— 这就是「它在复读、不是在写话」的证据。
    """
    if not text:
        return 0.0, "—"
    # 相邻重复率 = 相邻两个字相同的比例。正常中文大约 0.0x（叠词「人人/看看」有限），
    # 退化时能到 0.9 以上，是一个非常好认的指标
    rep = sum(1 for a, b in zip(text, text[1:]) if a == b) / max(1, len(text) - 1)
    # 最长连续重复片段：顺着扫一遍，维护「当前连续长度 cur」和「历史最长 best」。
    # 阈值取 3 而不是 2：中文里两个相同的字连用太常见（「人人」「常常」），
    # 连续 3 个以上才像「模型卡住了」，才值得在演示里标出来
    best, cur, best_ch = 1, 1, text[0]
    for a, b in zip(text, text[1:]):
        if a == b:
            cur += 1
            if cur > best:
                best, best_ch = cur, b
        else:
            cur = 1
    return rep, (f"{best_ch}×{best}" if best >= 3 else "—")


@torch.no_grad()
def val_ppls(model, ck, corpus, device):
    """按训练时的同一套划分（seed / val_ratio 都在 ckpt 里）复算 PPL"""
    # ★ 这里必须「复现训练时的那次切分」，否则 PPL 没有可比性：
    #   训练时是 random.shuffle(lines) 之后按 95% / 5% 切，而 shuffle 的结果由 seed 决定。
    #   所以要先 random.seed(ckpt 里的 seed) 再打乱同样的行，才能拿到「当初那个验证集」。
    #   这解释了为什么 ckpt 里要连 seed / val_ratio / max_chars 一起存下来。
    a = ck["args"]
    text = load_corpus(corpus)
    if a.get("max_chars"):
        # 训练时若截断过语料，这里也要截断，否则切出来的验证集和当初不是同一份
        text = text[: a["max_chars"]]
    lines = text.splitlines()
    random.seed(a.get("seed", 42))
    random.shuffle(lines)
    split = int(len(lines) * (1 - a.get("val_ratio", 0.05)))
    va = CharDataset("\n".join(lines[split:]), ck["char2idx"], a["seq_len"], a["stride"])
    # batch_size 写死 128 不影响 PPL：evaluate 内部是按 token 求和再除，和 batch 怎么切无关
    # （前提是 drop_last=True 丢掉的尾巴在两次计算里丢的是同一段，这里 shuffle=False 保证了这一点）
    dl = DataLoader(va, batch_size=128, shuffle=False, drop_last=True)
    _, full = evaluate(model, dl, device, last_only=False)
    _, last = evaluate(model, dl, device, last_only=True)
    return full, last


def model_tag(model, ck):
    """把「有没有 mask / 多大参数 / 什么结构」直接打到输出上，避免课上把两个模型念串"""
    a = ck["args"]
    n = sum(p.numel() for p in model.parameters())
    mask = "无 mask（看得到未来）" if a.get("no_mask") else "有因果 mask"
    return f"{mask}｜{n/1e6:.2f}M 参数｜d={a['d_model']} L={a['layers']}"


def show(prompt, outs, n_chars):
    """并排打印多个模型的续写，并在每行下面附上复读统计"""
    print("\n" + "─" * 72)
    print(f"输入：{prompt}")
    # 用最长的标签宽度做对齐，两个模型的输出左边缘才能对齐，一眼看得出差别
    width = max(len(v["label"]) for v in outs)
    for o in outs:
        # 换行替换成空格：续写里偶尔会带换行，直接打印会把表格对齐搞乱、看起来像在列条目
        body = o["text"].replace("\n", " ")
        rep, run = repeat_stats(body)
        print(f"  {o['label']:<{width}} │ {body}")
        print(f"  {'':<{width}} │ 相邻重复率 {rep:.2f}　最长连续重复 {run}")
    print("─" * 72)


def main():
    # 优先用课堂演示版的 12.21M 模型（tf_mask_big.pt / tf_nomask_big.pt），
    # 删了的话自动回落到作业口径的 2.56M 那对（tf_mask.pt / tf_nomask.pt）——
    # 这样学生在自己机器上只训了小模型也能直接跑这个演示
    big_mask = os.path.join(HERE, "tf_mask_big.pt")
    big_nomask = os.path.join(HERE, "tf_nomask_big.pt")
    use_big = os.path.exists(big_mask) and os.path.exists(big_nomask)

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_mask", default=big_mask if use_big else os.path.join(HERE, "tf_mask.pt"))
    ap.add_argument("--ckpt_nomask", default=big_nomask if use_big else os.path.join(HERE, "tf_nomask.pt"))
    ap.add_argument("--label_mask", default="有 mask")
    ap.add_argument("--label_nomask", default="无 mask")
    ap.add_argument("--prompt", nargs="*", default=["我们来到", "春天来了", "从前有座山", "他走进"])
    ap.add_argument("--n_chars", type=int, default=40)
    # 默认温度 0.65 偏低：既保留一点采样变化、又不至于太发散，课堂上最好看。
    # 想「结果完全确定」就传 0（走贪心分支），同一段开头永远得到同一段续写
    ap.add_argument("--temperature", type=float, default=0.65, help="0 = 贪心（完全确定）")
    ap.add_argument("--top_k", type=int, default=40, help=">0 时只从 top-k 里采样")
    # 默认屏蔽换行：语料按行拼起来，模型很容易「回车」跳到另一条无关语料上，
    # 看起来像在列 FAQ，课堂上容易跑题
    ap.add_argument("--ban", default="\n", help="生成时屏蔽这些字符（默认屏蔽换行，避免跳去另一条语料行）")
    # 同一个 seed 下两个模型走同样的采样路径，对比才公平
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", default=None, help="算 PPL 用的语料，默认取 ckpt 里记录的")
    ap.add_argument("--no_ppl", action="store_true", help="跳过 PPL 计算（更快）")
    ap.add_argument("--interactive", action="store_true", help="进入交互模式，逐句输入")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 两个 ckpt 缺一不可（演示的核心就是「对照」），缺了就给出可直接复制的训练命令，
    # 而不是让后面的 torch.load 抛一个不好读的文件错误
    if not (os.path.exists(args.ckpt_mask) and os.path.exists(args.ckpt_nomask)):
        raise FileNotFoundError(
            "找不到模型。先训练：\n"
            "  python transformer_lm.py --epochs 6 --save tf_mask.pt\n"
            "  python transformer_lm.py --epochs 6 --no-mask --save tf_nomask.pt"
        )

    # 两个模型都常驻显存：12M × 2 很小，换来的是「同一个开头立刻左右对比」的演示节奏
    m_mask, ck_mask = load_model(args.ckpt_mask, device)
    m_nomask, ck_nomask = load_model(args.ckpt_nomask, device)

    print("\n模型")
    print(f"  {args.label_mask:<8}：{model_tag(m_mask, ck_mask)}")
    print(f"  {args.label_nomask:<8}：{model_tag(m_nomask, ck_nomask)}")

    if not args.no_ppl:
        # 语料优先用 --corpus，其次用 ckpt 里记的训练语料，最后兜底样例语料：
        # PPL 只在「和训练同一份语料、同一套切分」上才有意义，所以默认不让你随便换
        corpus = args.corpus or ck_mask["args"].get("corpus") or DEFAULT_CORPUS
        if os.path.exists(corpus):
            # 用同一个 corpus / 同一划分分别复算两遍，两个数才是可比的
            f1, l1 = val_ppls(m_mask, ck_mask, corpus, device)
            f2, l2 = val_ppls(m_nomask, ck_nomask, corpus, device)
            print(f"\n验证集 PPL（同一划分）")
            print(f"  {args.label_mask:<8}：全位置 {f1:.2f}　末位 {l1:.2f}")
            print(f"  {args.label_nomask:<8}：全位置 {f2:.2f}　末位 {l2:.2f}")
            # 这句是给课堂上「念出来」的结论：没有它，学生很容易把 1.07 当成模型更强
            print("  说明：无 mask 的「全位置 PPL」极低是假象（它能抄到答案）；末位 PPL 才是真实水平。")

    print(f"\n生成长度 {args.n_chars} 字｜temperature={args.temperature}｜top_k={args.top_k or '不限'}｜屏蔽字符 {args.ban!r}｜seed={args.seed}")

    # 把「同一个 prompt 喂给两个模型」封装成一次 run，交互模式和批量模式共用，
    # 保证两种用法下的打印格式、采样参数完全一致（课上现敲和预置 list 看到的是同一套东西）
    def run(prompt):
        outs = [
            {"label": args.label_mask,
             "text": generate(m_mask, ck_mask, prompt, args.n_chars, args.temperature, args.top_k, args.seed, args.ban)},
            {"label": args.label_nomask,
             "text": generate(m_nomask, ck_nomask, prompt, args.n_chars, args.temperature, args.top_k, args.seed, args.ban)},
        ]
        show(prompt, outs, args.n_chars)

    if args.interactive:
        print("\n交互模式：输入开头（回车看两边续写），直接回车退出。")
        while True:
            try:
                p = input("\n开头> ").strip()
            except (EOFError, KeyboardInterrupt):
                # Ctrl+C / 管道输入结束时不甩 traceback，安静退出（课堂上按 Ctrl+C 是常规操作）
                break
            if not p:
                # 空回车 = 退出，避免课上一时想不出开头就卡住
                break
            run(p)
    else:
        for p in args.prompt:
            run(p)


if __name__ == "__main__":
    main()
