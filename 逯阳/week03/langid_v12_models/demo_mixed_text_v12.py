# -*- coding: utf-8 -*-
"""
demo_mixed_text_v12.py —— 12 语种混排文本试玩：模型 vs 纯规则（作业版）

和 8 类的 demo_mixed_text.py 功能一致，区别：
  1. 默认加载 12 类模型 langid_lstm_v12.pt
  2. ko / vi / tr 三条样例不再被跳过，模型能真正参与打分

【为什么要和纯规则比】
    语种识别最容易想到的做法是"纯规则"：看这句话里哪个 Unicode 区段的字符最多。
    这个规则有两个绕不过去的死穴：
      1. 同一个书写系统里的语言分不开：西语/葡语/英语/越南语/土耳其语都用拉丁字母
      2. 中英混排按字符数投票会投错："建议用 pytorch..." 拉丁字母更多，规则判英语

【运行】
    python demo_mixed_text_v12.py                         # 用 langid_lstm_v12.pt
    python demo_mixed_text_v12.py --model 别的模型.pt
"""

import os
import sys
import csv
import argparse

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from eval_langid_v12 import load_model, score_batch          # noqa: E402
from train_langid_cls_v12 import normalize                   # noqa: E402


# ─── 手写的混排样例：主体语言 + 混入的英文/其它语言 ────────────────────
MIXED_CASES = [
    # 中英混：英文只占一两个词
    ("我在用 chatgpt 写这份周报，写得有点快", "zh", "中文主体 + 英文产品名"),
    ("这个 bug 我查了一下午还没找到原因", "zh", "中文 + bug"),
    ("今天的 meeting 推迟到下午三点", "zh", "中文 + meeting"),
    ("刚 deploy 完，服务已经恢复正常了", "zh", "中文 + deploy"),
    # 中英混：英文占字符多数（规则必然投错）
    ("建议用 pytorch 而不是 tensorflow 来跑这个 demo", "zh", "拉丁字母比汉字还多"),
    ("把 config 里的 learning rate 调小一点再跑", "zh", "技术口语，英文词更多"),
    ("这个 pull request 麻烦你明天 review 一下", "zh", "工作场景混排"),
    ("先 git commit 再 push 上去，别直接改 master", "zh", "命令行流水话，英文占多数"),
    ("Python 的 list 和 tuple 有什么区别", "zh", "技术名词多"),
    ("这家 coffee 店的 wifi 密码是多少", "zh", "生活场景混排"),
    # 其它语种 + 英文
    ("この bug はまだ直っていないので、もう一度見てください", "ja", "日语 + 英文单词"),
    ("회의는 3 시부터 시작합니다", "ko", "韩语（英文只出现在数字上）"),
    ("Я скачал новый update вчера вечером", "ru", "俄语 + update"),
    ("أحتاج إلى download هذا الملف قبل الاجتماع", "ar", "阿拉伯语 + download"),
    ("मुझे यह file कल तक चाहिए", "hi", "印地语 + file"),
    ("Mañana tenemos una meeting con el cliente", "es", "西语 + meeting"),
    ("Preciso fazer um upgrade no sistema amanhã", "pt", "葡语 + upgrade"),
    ("Tôi đang viết code cho dự án mới", "vi", "越南语 + code"),
    # 同一书写系统内部：规则根本无从下手（这一组是规则的第二个死穴）
    ("Este sistema é muito bom mesmo", "pt", "纯拉丁字母，规则只能猜英语"),
    ("Bu sistemi çok beğendim", "tr", "土耳其语，同样只有拉丁字母"),
]


# ─── 纯规则：按 Unicode 区段数票，拉丁字母一律算英语 ──────────────────
SCRIPT_RULES = [
    ("\u4e00-\u9fff", "zh"),                  # 汉字 → 中文
    ("\u3040-\u30ff", "ja"),                  # 假名 → 日语
    ("\uac00-\ud7af", "ko"),                  # 韩文音节 → 韩语
    ("\u0400-\u04ff", "ru"),                  # 西里尔 → 俄语
    ("\u0600-\u06ff", "ar"),                  # 阿拉伯 → 阿拉伯语
    ("\u0900-\u097f", "hi"),                  # 天城文 → 印地语
    ("\u0e00-\u0e7f", "th"),                  # 泰文 → 泰语
    ("a-z", "en"),                            # 拉丁字母 → 一律猜英语
]


def script_of(ch):
    """单个字符属于哪个区段（这就是"纯规则"能看到的全部信息）。"""
    cp = ord(ch)
    for rng, lang in SCRIPT_RULES:
        lo, hi = rng.split("-")
        lo = ord(lo) if len(lo) == 1 else int(lo, 16)
        hi = ord(hi) if len(hi) == 1 else int(hi, 16)
        if lo <= cp <= hi:
            return lang
    return None


def rule_predict(text):
    """规则版：数各区段的字符数，取最多的那个。"""
    votes = {}
    for ch in normalize(text):
        lang = script_of(ch)
        if lang:
            votes[lang] = votes.get(lang, 0) + 1
    if not votes:
        return "?", {}
    return max(votes, key=votes.get), votes


def model_predict(model, vocab, labels, text):
    probs = score_batch(model, [text], vocab)[0]
    idx = int(probs.argmax())
    return labels[idx], float(probs[idx])


def rule_on_testset(data_dir, labels):
    """把纯规则直接拿去跑整张测试集，看它能打多少分。"""
    rows = []
    with open(os.path.join(data_dir, "test.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["text"], r["label"]))
    right = sum(rule_predict(t)[0] == g for t, g in rows)
    per = {}
    for t, g in rows:
        per.setdefault(g, [0, 0])
        per[g][1] += 1
        if rule_predict(t)[0] == g:
            per[g][0] += 1
    return right / len(rows), per, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "langid_lstm_v12.pt"))
    ap.add_argument("--data-dir", default=r"D:\AI\learning\work3\语种识别实战\data\langid\v12")
    args = ap.parse_args()

    torch.set_num_threads(1)
    model, vocab, labels = load_model(args.model)

    print("=" * 96)
    print("一、手写的混排句子（主体语言 + 混入的英文/其它语言）")
    print("=" * 96)
    print(f"{'句子':<44}{'期望':>5}{'模型':>7}{'置信':>7}{'规则':>6}  结果")
    skipped = []
    win_model, win_both, lose_both = [], [], []
    for text, expect, note in MIXED_CASES:
        if expect not in labels:
            skipped.append((text, expect))
            continue
        mp, conf = model_predict(model, vocab, labels, text)
        rp, _ = rule_predict(text)
        m_ok, r_ok = mp == expect, rp == expect
        flag = ("模型✓" if m_ok else "模型✗") + " " + ("规则✓" if r_ok else "规则✗")
        show = text if len(text) <= 28 else text[:27] + "…"
        print(f"{show:<44}{expect:>5}{mp:>7}{conf:>7.2f}{rp:>6}  {flag}")
        if m_ok and not r_ok:
            win_model.append((text, expect, note))
        elif m_ok and r_ok:
            win_both.append(text)
        else:
            lose_both.append((text, expect, mp, rp, conf))

    total = len(MIXED_CASES) - len(skipped)
    print(f"\n  结果：模型 {len(win_model) + len(win_both)}/{total}，"
          f"纯规则 {len(win_both)}/{total}")
    if skipped:
        print(f"  跳过 {len(skipped)} 条（当前模型不认识 "
              f"{'/'.join(sorted({e for _, e in skipped}))}）")

    print("\n  ── 规则判错、模型判对的（规则的死穴）──")
    for text, expect, note in win_model:
        print(f"    [{expect}] {text}   （{note}）")

    if lose_both:
        print("\n  ── 模型也判错的（诚实展示：模型不是万能的）──")
        for text, expect, mp, rp, conf in lose_both:
            print(f"    [{expect}] {text}   → 模型 {mp}({conf:.2f}) / 规则 {rp}")

    print("\n" + "=" * 96)
    print("二、把纯规则直接拿去跑整张测试集")
    print("=" * 96)
    rule_acc, per, n = rule_on_testset(args.data_dir, labels)
    print(f"{'语种':<8}{'规则准确率':>12}   说明")
    for lab in sorted(per):
        hit, tot = per[lab]
        if lab in ("es", "pt", "vi", "tr"):
            note = "拉丁字母 → 规则只会判英语，这几类直接 0 分"
        elif lab == "en":
            note = "拉丁字母 → 规则默认就是它"
        elif lab == "ja":
            note = "含大量汉字，规则会把一部分判成中文"
        elif lab == "ko":
            note = "含少量汉字，规则可能判成中文"
        else:
            note = ""
        print(f"{lab:<8}{hit/tot:>12.3f}   {note}")
    print(f"\n  纯规则在 {n} 条测试集上的准确率：{rule_acc:.4f}"
          f"（12 类 LSTM 模型同一份测试集：0.9672）")
    print("  → 规则只认得\"字符属于哪个书写系统\"，同系统内的语言它无从下手；")
    print("    模型学的是字符组合的统计规律，所以 es / pt / tr / vi 也能分开，混排时还能抓住主体语言。")


if __name__ == "__main__":
    main()
