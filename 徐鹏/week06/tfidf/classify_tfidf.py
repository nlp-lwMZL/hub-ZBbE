"""
方法二：TF-IDF + 线性分类器（L1 基线）

教学重点：
  1. 中文短文本上，**字 n-gram 比词更稳**：不用分词，直接 char 1-2gram，
     就能抓住「股/盘/楼/球/手机」这类信号，20 秒训完。
  1b. 但它不是「强基线」：通用短文本上与微调差 5~7 个点（0.472 vs 0.544）。
     它的主场是**数据封闭**——不能引入预训练模型（内网 / 无权重 / 合规）、
     语料是自有黑话、只有几百到几千条样本、还要能秒级重训。
  2. 同一个脚本可通过 --ngram / --C 复现调参对比（实测干净集上最好 0.513）。
  3. 它最怕「数据泄漏」：验证集里只要混进 11% 训练样本，
     靠词面匹配就能把 F1 从 0.509 抬到 0.541 —— 见 --leak 开关的对照实验。

用法：
  python tfidf/classify_tfidf.py                  # 干净集，默认 char(1,2) + C=1.0
  python tfidf/classify_tfidf.py --leak           # 在全量 val（含重复）上评测，演示泄漏虚高
  python tfidf/classify_tfidf.py --n 500 --C 0.5 --ngram 1,3
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import clean_val, load_clean_val, load_splits
from common.metrics import classification_report, to_markdown_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="抽样条数，0=全部")
    ap.add_argument("--ngram", default="1,2", help="字符 n-gram 范围，如 1,2 或 1,3")
    ap.add_argument("--C", type=float, default=1.0, help="LinearSVC 正则强度")
    ap.add_argument("--leak", action="store_true", help="在全量 val（不去重）上评测，对照泄漏影响")
    ap.add_argument("--out", default="outputs/tfidf_pred.json")
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    train, val = load_splits()
    if args.leak:
        # 故意不剔除重复：用来量化「数据泄漏」到底能虚高多少
        samples = val[: args.n] if args.n else val
        meta = {"mode": "leaky_full_val", "n_eval": len(samples)}
    else:
        samples, meta = load_clean_val(args.n or None)

    lo, hi = (int(x) for x in args.ngram.split(","))
    vec = TfidfVectorizer(analyzer="char", ngram_range=(lo, hi), max_features=300000)

    t0 = time.time()
    Xtr = vec.fit_transform([d["sentence"] for d in train])          # 稀疏矩阵，内存友好
    Xte = vec.transform([d["sentence"] for d in samples])
    clf = LinearSVC(C=args.C).fit(Xtr, [d["label"] for d in train])  # 线性 SVM：高维稀疏文本上的经典线性基线
    y_pred = clf.predict(Xte).tolist()
    fit_seconds = round(time.time() - t0, 1)

    y_true = [d["label"] for d in samples]
    report = classification_report(y_true, y_pred)
    report.update({"method": "tfidf_svm", "fit_seconds": fit_seconds,
                   "ngram": args.ngram, "C": args.C})

    out = Path(__file__).resolve().parent.parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "report": report}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"评测 {meta['n_eval']} 条 | 训练+推理 {fit_seconds}s | ngram={args.ngram} C={args.C}")
    print(to_markdown_table({f"TF-IDF+SVM (ngram={args.ngram}, C={args.C})": report}))
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()
