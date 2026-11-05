# -*- coding: utf-8 -*-
"""
build_langid_dataset.py —— 多语种「语种识别」数据集构建脚本

【谁用】老师侧离线跑一次，产出 data/langid/ 下的 csv；学生不需要运行本脚本。

【数据来源】Tatoeba 逐语种句子导出（https://tatoeba.org，CC-BY 2.0 FR，需署名）

【产出】
  data/langid/v8/train.csv  dev.csv  test.csv      课堂版：8 个语种
  data/langid/v12/train.csv dev.csv  test.csv      作业版：12 个语种
  data/langid/test_hard.csv                        难度诊断集（用 slice 字段分组）
  data/langid/test_openset.csv                     4 个训练里从未出现过的语种
  data/langid/vocab/vocab_v8.txt  vocab_v12.txt   字符表（字符 \t id \t 出现次数）
  data/langid/DATA_CARD.md                         数据卡（来源/许可/规模/已知坑）

【清洗流程（全部离线做完，课堂不讲）】
  1. 下载 Tatoeba 逐语种句子
  2. 繁 → 简（opencc t2s）
  3. 丢弃：URL、字母占比 < 60% 的行、Tatoeba 高频人名（Tom / Mary 系列，见 REPORT）
  4. NFKC 归一化 + 转小写 + 只保留「字母(L)」与「组合符号(M)」
     —— 必须保留 M：印地语、泰语的元音是组合标记，只留字母会把印地语砍成裸辅音
  5. 长度窗口 8~40 个字符，按归一化文本去重
  6. 按长度分桶对齐采样：四个长度桶各取相同数量，避免模型走「短 = 中文」的捷径

【字符表怎么建】见 build_char_vocab()：
  把训练集所有句子按字拆开，统计每个字出现次数，出现次数排序后编号。
  第 0 号固定 <PAD>，第 1 号固定 <UNK>。字符级下 <UNK> 几乎用不到。
"""

import os
import re
import csv
import bz2
import random
import unicodedata
import urllib.request
from collections import Counter, OrderedDict

# ─── 配置 ──────────────────────────────────────────────────────────────
SEED = 42

TATOEBA = "https://downloads.tatoeba.org/exports/per_language/{code}/{code}_sentences.tsv.bz2"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "langid_raw")

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "langid")

# (Tatoeba 语种代码, 标签)
L8 = [("cmn", "zh"), ("jpn", "ja"), ("eng", "en"), ("spa", "es"),
      ("por", "pt"), ("rus", "ru"), ("ara", "ar"), ("hin", "hi")]
L12 = L8 + [("kor", "ko"), ("tha", "th"), ("vie", "vi"), ("tur", "tr")]
OPEN_SET = [("ell", "el"), ("bul", "bg"), ("swh", "sw"), ("urd", "ur")]

TRAIN_PER_CLASS = 800
DEV_PER_CLASS = 100
TEST_PER_CLASS = 150
HARD_PER_CLASS = 50
OPENSET_PER_CLASS = 100

LEN_MIN, LEN_MAX = 8, 40
SHORT_MIN, SHORT_MAX = 2, 7          # 难度集用的「短文本」窗口
BUCKETS = [(8, 12), (12, 20), (20, 30), (30, LEN_MAX + 1)]

URL_RE = re.compile(r"https?://|www\.|@[A-Za-z0-9_]{2,}", re.I)
# Tatoeba 里 Tom / Mary 两个名字的翻译句占了极大比例（英文里 23% 含 Tom），必须过滤
NAME_RE = re.compile(r"(?<![a-z])(tom|mary)(?![a-z])|トム|メアリー|汤姆|湯姆|玛丽|瑪麗", re.I)

_OPENCC = None


def to_simplified(text):
    """繁 → 简。只对含汉字的文本生效，其它语种原样返回。"""
    global _OPENCC
    if not re.search(r"[\u4e00-\u9fff]", text):
        return text
    if _OPENCC is None:
        from opencc import OpenCC
        _OPENCC = OpenCC("t2s")
    return _OPENCC.convert(text)


def normalize(text):
    """NFKC + 小写 + 只保留字母(L)与组合符号(M)。这就是喂给模型的形式。"""
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


def strip_marks(text):
    """去掉重音符号（只用于拉丁语种的难度切片，如 café → cafe）。"""
    d = unicodedata.normalize("NFD", text)
    return "".join(c for c in d if unicodedata.category(c) != "Mn")


def strip_kana(text):
    """去掉日语假名，只留汉字（难度切片：zh / ja 的极限混淆）。"""
    return "".join(c for c in text if not ("\u3040" <= c <= "\u30ff"))


# ─── 1. 下载 & 缓存 ────────────────────────────────────────────────────
def fetch_sentences(code):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, code + ".tsv")
    if not os.path.exists(path):
        print(f"  下载 {code} ...", flush=True)
        raw = urllib.request.urlopen(TATOEBA.format(code=code), timeout=180).read()
        text = bz2.decompress(raw).decode("utf-8", "replace")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[2].strip():
                out.append(parts[2].strip())
    return out


# ─── 2. 清洗 & 分池 ────────────────────────────────────────────────────
def build_pools(code):
    """返回 (主池, 短池)。主池按长度桶分组，短池用于难度集。"""
    raw = fetch_sentences(code)
    main = {i: [] for i in range(len(BUCKETS))}
    short = []
    seen = set()
    dropped = Counter()

    for text in raw:
        if URL_RE.search(text):
            dropped["url"] += 1
            continue
        if NAME_RE.search(text):
            dropped["name"] += 1
            continue
        text = to_simplified(text)
        norm = normalize(text)
        if len(norm) < 0.6 * len(text):
            dropped["symbol"] += 1
            continue
        if norm in seen:
            dropped["dup"] += 1
            continue

        if SHORT_MIN <= len(norm) <= SHORT_MAX:
            seen.add(norm)
            short.append(text)
            continue
        if not (LEN_MIN <= len(norm) <= LEN_MAX):
            dropped["len"] += 1
            continue

        seen.add(norm)
        for i, (lo, hi) in enumerate(BUCKETS):
            if lo <= len(norm) < hi:
                main[i].append(text)
                break

    return main, short, dropped


_POOL_CACHE = {}


def build_pools_cached(code):
    """v8 和 v12 会共用同一批语种，读一次就够（大语种文件上百万行）。"""
    if code not in _POOL_CACHE:
        _POOL_CACHE[code] = build_pools(code)
    main, short, dropped = _POOL_CACHE[code]
    return {i: list(v) for i, v in main.items()}, list(short), dropped


def take_balanced(buckets, n, rng):
    """从各长度桶等量取 n 条，桶不够就从其它桶补。"""
    for v in buckets.values():
        rng.shuffle(v)
    quota = n // len(buckets) + 1
    picked, rest = [], []
    for i in sorted(buckets):
        chunk = buckets[i][:quota]
        picked.extend(chunk)
        rest.extend(buckets[i][quota:])
        buckets[i] = buckets[i][quota:]
    if len(picked) < n:                       # 桶不够，从剩余里补
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    elif len(picked) > n:                     # 桶够了，多的还回去
        rng.shuffle(picked)
        leftover = picked[n:]
        for j, item in enumerate(leftover):
            buckets[j % len(buckets)].append(item)
        picked = picked[:n]
    rng.shuffle(picked)
    return picked


def take_from(pool, n, rng):
    rng.shuffle(pool)
    picked = pool[:n]
    del pool[:n]
    return picked


# ─── 3. 字符表 ─────────────────────────────────────────────────────────
def build_char_vocab(sentences, min_freq=1):
    """
    字符表构建：统计训练集每个字的出现次数 → 排序 → 编号。

    第 0 号 <PAD>（补齐用，Embedding 里 padding_idx=0）
    第 1 号 <UNK>（字符级下极少用到）
    之后按出现次数从高到低编号，出现次数相同时按字符编码排序（保证可复现）。
    """
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab, freq


def save_vocab(vocab, freq, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def coverage(vocab, freq, ratio=0.95):
    """前多少个字符覆盖 ratio 的字符出现次数。"""
    total = sum(freq.values())
    acc, n = 0, 0
    for ch in vocab:
        if ch in ("<PAD>", "<UNK>"):
            continue
        n += 1
        acc += freq.get(ch, 0)
        if acc >= ratio * total:
            return n
    return n


# ─── 4. 写出 ───────────────────────────────────────────────────────────
def write_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def build_profile(name, lang_pairs, rng):
    """构建一个 profile（v8 / v12），返回 (统计信息, 各语种剩余池)。"""
    print(f"\n=== 构建 {name}（{len(lang_pairs)} 语种）===")
    stats, leftovers = [], {}
    train_rows, dev_rows, test_rows, hard_rows = [], [], [], []
    charsets_in_order, train_by_label = [], {}

    for code, label in lang_pairs:
        main, short, dropped = build_pools_cached(code)
        buckets = {i: list(v) for i, v in main.items()}
        n = TRAIN_PER_CLASS + DEV_PER_CLASS + TEST_PER_CLASS
        picked = take_balanced(buckets, n, rng)
        if len(picked) < n:
            print(f"  [警告] {label} 可用样本只有 {len(picked)}，少于 {n}")
        tr = picked[:TRAIN_PER_CLASS]
        dv = picked[TRAIN_PER_CLASS:TRAIN_PER_CLASS + DEV_PER_CLASS]
        te = picked[TRAIN_PER_CLASS + DEV_PER_CLASS:]
        train_rows += [(t, label) for t in tr]
        dev_rows += [(t, label) for t in dv]
        test_rows += [(t, label) for t in te]

        # 难度切片 1：短文本（2~7 字符）
        rng.shuffle(short)
        hard_rows += [(t, label, "short") for t in short[:HARD_PER_CLASS]]

        # 难度切片 2：去重音（拉丁语种）—— 存派生后的文本，不是原文
        if label in ("es", "pt", "en", "vi", "tr"):
            for t in take_balanced(buckets, 120, rng):
                s = strip_marks(normalize(t))
                if len(s) >= 5:
                    hard_rows.append((s, label, "nodeaccent"))

        # 难度切片 3：去假名（日语）—— 剩下的就是一堆汉字，和中文几乎无法区分
        if label == "ja":
            got = 0
            for t in take_balanced(buckets, 400, rng):
                s = strip_kana(normalize(t))
                if len(s) >= 5:
                    hard_rows.append((s, label, "nokana"))
                    got += 1
                    if got >= 100:
                        break

        stats.append({
            "label": label, "code": code, "total": len(main[0]) + len(main[1]) + len(main[2]) + len(main[3]),
            "short": len(short), "drops": sum(dropped.values()),
        })
        charsets_in_order.append(set("".join(normalize(t) for t in tr)))
        stats[-1]["chars"] = len(charsets_in_order[-1])
        train_by_label[label] = tr
        leftovers[label] = buckets
        print(f"  {label}: 主池 {stats[-1]['total']:6d}  短句池 {len(short):5d}  "
              f"丢弃 {stats[-1]['drops']:6d} → 取 {len(tr)}/{len(dv)}/{len(te)}")

    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    rng.shuffle(test_rows)
    rng.shuffle(hard_rows)

    base = os.path.join(OUT_ROOT, name)
    write_csv(os.path.join(base, "train.csv"), train_rows, ["text", "label"])
    write_csv(os.path.join(base, "dev.csv"), dev_rows, ["text", "label"])
    write_csv(os.path.join(base, "test.csv"), test_rows, ["text", "label"])

    vocab, freq = build_char_vocab([t for t, _ in train_rows])
    # 只随课件发 8 类那张表；12 类的词表留给作业，由学生自己按课堂代码生成
    if name == "v8":
        save_vocab(vocab, freq, os.path.join(OUT_ROOT, "vocab", f"vocab_{name}.txt"))

    print(f"  字符表：{len(vocab)} 个（含 <PAD>/<UNK>），前 {coverage(vocab, freq)} 个覆盖 95%"
          + ("（已写出文件）" if name == "v8" else "（不写文件，留给作业生成）"))

    # 语种间的字符集关系（讲 zh/ja 混淆、讲"字符级为什么能通用"都用得上）
    facts = {}
    charsets = {s["label"]: chars for s, chars in zip(stats, charsets_in_order)}
    if "zh" in charsets and "ja" in charsets:
        kana = {c for c in charsets["ja"] if "\u3040" <= c <= "\u30ff"}
        ja_freq = Counter("".join(normalize(t) for t in train_by_label["ja"]))
        facts = {
            "zh_chars": len(charsets["zh"]),
            "ja_chars": len(charsets["ja"]),
            "overlap": len(charsets["zh"] & charsets["ja"]),
            "kana_chars": len(kana),
            "kana_share": sum(ja_freq[c] for c in kana) / max(1, sum(ja_freq.values())),
        }
    return stats, vocab, freq, hard_rows, facts


def build_openset(rng):
    print(f"\n=== 开放集（训练里没出现过的 {len(OPEN_SET)} 个语种）===")
    rows = []
    for code, label in OPEN_SET:
        main, _, _ = build_pools_cached(code)
        buckets = {i: list(v) for i, v in main.items()}
        picked = take_balanced(buckets, OPENSET_PER_CLASS, rng)
        rows += [(t, label) for t in picked]
        print(f"  {label}: {len(picked)} 条")
    rng.shuffle(rows)
    write_csv(os.path.join(OUT_ROOT, "test_openset.csv"), rows, ["text", "label"])
    return rows


# ─── 5. 主流程 ─────────────────────────────────────────────────────────
def main():
    rng = random.Random(SEED)
    os.makedirs(OUT_ROOT, exist_ok=True)

    stats8, vocab8, freq8, hard8, facts8 = build_profile("v8", L8, rng)
    stats12, vocab12, freq12, hard12, facts12 = build_profile("v12", L12, rng)
    openset = build_openset(rng)

    # 难度集用 v8 的那一份（作业版的额外语种由 v12 自带的 test 覆盖）
    write_csv(os.path.join(OUT_ROOT, "test_hard.csv"), hard12, ["text", "label", "slice"])
    slices = Counter(s for _, _, s in hard12)
    print(f"\n难度集 test_hard.csv：{len(hard12)} 条  {dict(slices)}")

    write_data_card(stats8, vocab8, freq8, stats12, vocab12, freq12,
                    openset, hard12, slices, facts8)
    print(f"\n全部产出在：{OUT_ROOT}")


def write_data_card(stats8, vocab8, freq8, stats12, vocab12, freq12,
                    openset, hard, slices, facts):
    def table(stats):
        return "\n".join(f"| {s['label']} | `{s['code']}` | {s['chars']} |" for s in stats)

    doc = f"""# 数据卡：多语种语种识别数据集

## 这是什么

供「深度学习常用组件」一课使用的**多语种文本分类**训练数据。任务是让模型看完一句话，
判断它是哪种语言。标签就是句子的来源语种，不需要人工标注。

## 来源与许可

- 全部句子来自 **[Tatoeba](https://tatoeba.org)** 的逐语种句子导出，许可为
  **[CC-BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/)**。
- 使用时请保留本数据卡与上述署名。本项目仅作教学用途。

## 规模

| 文件 | 语种数 | 行数 | 用途 |
|---|---|---|---|
| `v8/train.csv` | 8 | {TRAIN_PER_CLASS * 8} | 课堂训练 |
| `v8/dev.csv` | 8 | {DEV_PER_CLASS * 8} | 课堂验证 |
| `v8/test.csv` | 8 | {TEST_PER_CLASS * 8} | 课堂测试 |
| `v12/train.csv` | 12 | {TRAIN_PER_CLASS * 12} | 作业训练 |
| `v12/dev.csv` | 12 | {DEV_PER_CLASS * 12} | 作业验证 |
| `v12/test.csv` | 12 | {TEST_PER_CLASS * 12} | 作业测试 |
| `test_hard.csv` | {len(set(l for _, l, _ in hard))} | {len(hard)} | 难度诊断（`slice` 字段分组） |
| `test_openset.csv` | {len(OPEN_SET)} | {len(openset)} | 训练中从未出现的语种 |

### 课内 8 语种

| 标签 | Tatoeba 代码 | 本数据集字符数 |
|---|---|---|
{table(stats8)}

### 作业 12 语种（= 课内 8 个 + 4 个）

| 标签 | Tatoeba 代码 | 本数据集字符数 |
|---|---|---|
{table(stats12)}

### 开放集（只出现在测试集里，模型没见过）

`el` 希腊语 / `bg` 保加利亚语 / `sw` 斯瓦希里语 / `ur` 乌尔都语

## 字段

| 文件 | 字段 | 说明 |
|---|---|---|
| `v8/*.csv`、`v12/*.csv` | `text`, `label` | `label` 是 ISO 639-1 两字码 |
| `test_hard.csv` | `text`, `label`, `slice` | `slice` ∈ short / nodeaccent / nokana（合计 {len(hard)} 条） |
| `test_openset.csv` | `text`, `label` | 训练集里没有这些 `label` |

`slice` 含义：

- `short`：只有 2~7 个字符的短句（真实场景：搜索词、商品标题）
- `nodeaccent`：去掉重音符号（`café` → `cafe`），拉美语族内互相混淆会变严重
- `nokana`：日语句子去掉全部假名，只剩汉字，与中文几乎无法区分

## 已经做过的清洗（使用者不需要再处理）

1. **繁 → 简**：中文全部转成简体字（opencc t2s）。
2. **过滤 Tatoeba 高频人名**：Tatoeba 里 "Tom is …" 的翻译句极多（英文原始语料里
   23.3% 含 Tom），已连同 Mary 一起剔除，避免把「有没有 Tom」当成语种特征。
3. **归一化**：NFKC + 转小写 + 只保留字母与组合符号（标点、数字、emoji 已去掉）。
4. **长度窗口 8~40 字符**，按归一化文本去重。
5. **长度分桶对齐**：四个长度桶等量采样，避免模型走「句子短 = 中文」的捷径。

## 字符表

`vocab/vocab_v8.txt`：每行 `字符 \\t id \\t 出现次数`，
由 `build_langid_dataset.py` 的 `build_char_vocab()` 从训练集统计生成。
第 0 号 `<PAD>`、第 1 号 `<UNK>`，其余按出现次数从高到低编号。

**这张表是课堂 8 类用的，随课件一起发。**
12 类的词表**没有**提供 —— 作业要求学生按课堂代码自己从 `v12/train.csv` 生成
（`build_char_vocab()` → 写成同样格式的文件），本目录下不会有 `vocab_v12.txt`。

实测规模：

- v8 字符表 **{len(vocab8)}** 个字符，前 **{coverage(vocab8, freq8)}** 个覆盖 95% 的字符出现次数
- v12 字符表 **{len(vocab12)}** 个字符，前 **{coverage(vocab12, freq12)}** 个覆盖 95%
  （作业参考答案里的数字；文件需学生自己生成）

## 已知特点（讲课可以直接用）

- 中文与日文的字符集高度重叠：本数据集里中文出现 {facts.get('zh_chars', 0)} 个字符、日语出现
  {facts.get('ja_chars', 0)} 个，交集 {facts.get('overlap', 0)} 个
  —— 占中文的 {100 * facts.get('overlap', 0) / max(1, facts.get('zh_chars', 1)):.0f}%。
- 日语里假名只有 {facts.get('kana_chars', 0)} 个字符，却占日语字符出现次数的
  {100 * facts.get('kana_share', 0):.0f}%。→ 这就是 `nokana` 切片会崩掉的依据：
  把假名去掉，日语剩下的就是一堆汉字，和中文没有区别。
- 拉丁字母语种（en/es/pt/tr/vi）各自的字符集都只有几十个字符，
  却和有一千多个字符的中文表达同样多的信息量 —— 这正是后面讲子词/BPE 的动机。

## 重建方法

```bash
python build_langid_dataset.py
```

脚本会把 Tatoeba 原始文件缓存到 `~/.cache/langid_raw/`，只下载一次。
需要 `opencc`（繁简转换）：
`pip install opencc-python-reimplemented` 或 `pip install opencc`
"""
    path = os.path.join(OUT_ROOT, "DATA_CARD.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


if __name__ == "__main__":
    main()
