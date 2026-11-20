"""
分流服务（CLI 版）：输入一条标题，输出对接下游的 JSON —— 就是课件 P43 那张「接口契约」的样子。

教学重点：
  1. **接口契约比准确率更重要**：下游要的是 {topic, relevant, conf, queue}。
     有 conf 才能做「低置信转人工」，只有 label 下游没法决策。
  2. **服务要能自己选后端**：有 BERT 权重就走 BERT，没有就退回 TF-IDF 线性模型，
     两者都没有再退规则 —— 而且**降级必须显式**（结果里的 model 字段会写清楚是哪条路径）。
  3. **置信度要真**：所以 TF-IDF 这档故意用「带 predict_proba 的逻辑回归」，
     而不是课件对比里用的 LinearSVC（SVC 没有概率，拿 decision_function 硬凑会误导下游）。

返回字段（与课件 P43 完全一致）：
  title     原始标题
  topic     4 组主题（金融 / 产业 / 宏观 / 噪音），给业务看的中文名
  sub_topic 15 类细标签（证券 / 科技 / 教育 …），给统计用
  relevant  是否与投研相关（由主题组派生）
  conf      置信度 0~1；低于阈值时 queue=review
  queue     research / review / drop（枚举，业务方按这个路由，不用自由文本）
  model     哪条路径出的结果：bert-cls-v1 / tfidf-lr-v1 / rule-v1 / fallback
  version   接口版本（出问题能追溯到哪天）

用法：
  python service/classify_cli.py                          # 交互模式（输入标题回车，q 退出）
  python service/classify_cli.py --text "贵州茅台：2023 年年度业绩预告"
  python service/classify_cli.py --file inputs.txt        # 每行一条
  type inputs.txt | python service/classify_cli.py -      # 从管道读
  python service/classify_cli.py --text "..." --json      # 只输出一行 JSON（给脚本/Agent 用）
  python service/classify_cli.py --model tfidf            # 强制指定后端
  python service/classify_cli.py --use-rules              # 打开规则快筛，演示 L5 级联

也被谁调用：第 9 周会把这里的 classify() 包成 Agent 的第一个工具（Function Calling / MCP / CLI 三接法）。
"""

import argparse
import json
import sys
import time

import numpy as np
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common.labels import LABELS, is_relevant, queue_name, to_group  # noqa: E402

VERSION = "2026-11"
BACKEND_NAME = {"bert": "bert-cls-v1", "tfidf": "tfidf-sgd-v1", "rule": "rule-v1", "fallback": "fallback-v0"}

# 规则快筛：公告标题套路很强，一条正则就能挡掉一批调用（生产里最常见的省钱手段）。
# 默认关闭，保证演示看到的是「模型」的结果；加 --use-rules 才是完整的 L5 级联。
FAST_RULES = [
    ("业绩预告", "财经"), ("年度报告", "财经"), ("季度报告", "财经"),
    ("回购", "证券"), ("减持", "证券"), ("增持", "证券"),
]


class ShuntService:
    """加载一次后端，之后每次调用复用（服务化写法：别在请求里现加载模型）"""

    def __init__(self, model="auto", ckpt=None, conf_threshold=0.8, use_rules=False, verbose=False):
        self.conf_threshold = conf_threshold
        self.use_rules = use_rules
        self.verbose = verbose
        self.backend = None            # bert / tfidf / rule / fallback
        self.tok = self.model = None
        self.vec = self.clf = None
        self.temperature = 1.0          # 置信度校准用的温度（1.0 = 不校准）

        self.ckpt_path = Path(ckpt) if ckpt else CODE_ROOT / "outputs" / "bert_ckpt.pt"
        if model in ("auto", "bert"):
            self._try_init_bert()
            if model == "bert" and self.backend is None:
                raise SystemExit(f"指定了 --model bert，但没加载成功：{self.ckpt_path} 不存在或读取失败")
        if self.backend is None and model in ("auto", "tfidf"):
            if model == "auto":
                # 降级要显式：让别人知道这次没用 BERT，以及怎么才能用上
                print("[info] 没有 BERT 权重，改用 TF-IDF 后端（想用 BERT 先跑 bert/train_bert.py）",
                      file=sys.stderr)
            self._init_tfidf()
        if self.backend is None:       # 兜底：只剩规则
            self.backend = "fallback"
            print("[warn] 没有任何可用的模型，进入 fallback（全部判 review）", file=sys.stderr)

    # ---------- 置信度校准：温度缩放 ----------
    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _fit_temperature(self, proba, y):
        """
        在一批「没参与训练」的样本上，找一个 T 让 NLL 最小。
        proba 是模型原始概率；返回 T（<1 表示原概率偏保守，需要放大）。
        """
        P = np.clip(proba, 1e-12, 1.0)
        logits = np.log(P)
        rows = np.arange(len(y))
        best_T, best_nll = 1.0, float("inf")
        for T in np.arange(0.2, 3.01, 0.05):
            q = self._softmax(logits / T)
            nll = -float(np.log(q[rows, y] + 1e-12).mean())
            if nll < best_nll:
                best_nll, best_T = nll, float(T)
        return round(best_T, 2)

    def _calibrate(self, proba):
        """把原始概率按温度缩放后再取最大类（T 由 _fit_temperature 得到）"""
        if self.temperature == 1.0:
            proba = proba.reshape(1, -1) if proba.ndim == 1 else proba
            return proba
        proba = proba.reshape(1, -1) if proba.ndim == 1 else proba
        return self._softmax(np.log(np.clip(proba, 1e-12, 1.0)) / self.temperature)

    # ---------- 后端 1：BERT 微调（效果最好，需要 ckpt） ----------
    def _try_init_bert(self):
        if not self.ckpt_path.exists():
            if self.verbose:
                print(f"[info] 没有 BERT 权重（{self.ckpt_path.name}），跳过", file=sys.stderr)
            return
        try:
            t0 = time.time()
            import torch
            from transformers import AutoTokenizer

            sys.path.insert(0, str(CODE_ROOT / "bert"))
            from train_bert import BertClassifier
            from common.model_paths import bert_path

            ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            self.tok = AutoTokenizer.from_pretrained(str(bert_path()))
            self.model = BertClassifier(bert_path(), num_labels=len(LABELS), pool=ckpt.get("pool", "cls"))
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self.backend = "bert"
            self._calibrate_bert()
            print(f"[init] BERT 后端就绪（{time.time() - t0:.1f}s，温度 T={self.temperature}）",
                  file=sys.stderr)
        except Exception as e:          # 权重坏了也不能让服务整体挂掉
            print(f"[warn] BERT 加载失败（{e}），尝试降级", file=sys.stderr)
            self.backend, self.model = None, None


    def _calibrate_bert(self, n=200):
        """用干净验证集里的一小批样本拟合温度（这批样本没参与训练；200 条足够，再多只是拖慢启动）"""
        import torch

        from common.data import clean_val, load_splits

        train, val = load_splits()
        clean, _ = clean_val(train, val)
        clean = clean[:n]
        probs, ys = [], []
        with torch.no_grad():
            for i in range(0, len(clean), 64):
                batch = clean[i:i + 64]
                enc = self.tok([d["sentence"] for d in batch], truncation=True, max_length=64,
                               padding="longest", return_tensors="pt")
                logits = self.model(enc["input_ids"], enc["attention_mask"])
                probs.append(torch.softmax(logits.float(), -1).numpy())
                ys.extend(d["label"] for d in batch)
        self.temperature = self._fit_temperature(np.vstack(probs), np.array(ys))

    # ---------- 后端 2：TF-IDF + 逻辑回归（20 秒内可训练，永远可用） ----------
    def _init_tfidf(self):
        t0 = time.time()
        import warnings

        from sklearn.exceptions import ConvergenceWarning
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier

        from common.data import load_splits

        train, _ = load_splits()
        # 切出 10% 专门做置信度校准（这些样本不参与训练，否则温度会被拉向过拟合的自信）
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(train))
        n_cal = max(1000, len(train) // 10)
        cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]
        fit_set = [train[i] for i in fit_idx]
        cal_set = [train[i] for i in cal_idx]

        self.vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2),
                                   max_features=300000, min_df=2, sublinear_tf=True)
        X = self.vec.fit_transform([d["sentence"] for d in fit_set])
        # 为什么用 SGD(log_loss) 而不是 LogisticRegression：
        #   111k 维 × 5.3 万条，lbfgs 求解要 80 秒才收敛；SGD 十秒内到同一水平
        #   （实测 macro-F1：SGD 0.518 / LinearSVC 0.510），服务启动才等得起。
        # class_weight=balanced 与课件里的「类别加权」同一个思路，否则长尾类几乎不会出现。
        self.clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=30,
                                 class_weight="balanced", random_state=42)
        with warnings.catch_warnings():      # 迭代上限的 warning 不影响结果，别刷屏
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.clf.fit(X, [d["label"] for d in fit_set])

        # 温度缩放：原始 SGD 概率明显偏保守，不校准的话阈值 0.8 几乎筛不出东西
        Xc = self.vec.transform([d["sentence"] for d in cal_set])
        self.temperature = self._fit_temperature(self.clf.predict_proba(Xc),
                                                np.array([d["label"] for d in cal_set]))
        self.backend = "tfidf"
        print(f"[init] TF-IDF 后端就绪（{X.shape[1]} 维特征，{X.shape[0]} 条样本，{time.time() - t0:.1f}s，温度 T={self.temperature}）",
              file=sys.stderr)

    # ---------- 单条预测 ----------
    def _rule_hit(self, text):
        for kw, label in FAST_RULES:
            if kw in text:
                return LABELS.index(label)
        return None

    def _predict(self, title):
        """返回 (label_id, conf, backend)"""
        if self.use_rules:
            idx = self._rule_hit(title)
            if idx is not None:
                # 规则命中的「置信度」是人为设定的，不是模型输出 —— 这点必须在代码里写清楚
                return idx, 0.99, "rule"

        if self.backend == "bert":
            import torch
            enc = self.tok(title, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
            with torch.no_grad():
                logits = self.model(enc["input_ids"], enc["attention_mask"])
            prob = self._calibrate(torch.softmax(logits.float(), -1)[0].numpy())[0]
            idx = int(prob.argmax())
            return idx, float(prob[idx]), "bert"

        if self.backend == "tfidf":
            raw = self.clf.predict_proba(self.vec.transform([title]))[0]
            prob = self._calibrate(raw)[0]          # 温度缩放后的置信度
            idx = int(prob.argmax())
            return idx, float(prob[idx]), "tfidf"

        return 0, 0.0, "fallback"

    def classify(self, title):
        """
        对外接口：一条标题 → 一个 dict（字段与课件 P43 的接口契约一致）。
        下游拿到 conf 和 queue 就能直接做路由，不需要理解模型内部。
        """
        title = title.strip()
        t0 = time.time()
        idx, conf, used = self._predict(title)
        out = {
            "title": title,
            "topic": to_group(idx),
            "sub_topic": LABELS[idx],
            "relevant": is_relevant(idx),
            "conf": round(conf, 4),
            "queue": queue_name(idx, conf, self.conf_threshold),
            "model": BACKEND_NAME[used],
            "version": VERSION,
        }
        if self.verbose:
            out["_debug"] = {"backend": used, "label_id": idx,
                             "latency_ms": round((time.time() - t0) * 1000, 1),
                             "threshold": self.conf_threshold}
        return out


# ---------- 便捷函数：给别的脚本/Agent 直接 import ----------
_DEFAULT = None


def classify(title, **kwargs):
    """from service.classify_cli import classify; classify("贵州茅台：2023 年年度业绩预告")"""
    global _DEFAULT
    if _DEFAULT is None or kwargs:
        _DEFAULT = ShuntService(**kwargs)
    return _DEFAULT.classify(title)


def run_oneshot(svc, texts, as_json=False):
    for t in texts:
        if not t.strip():
            continue
        res = svc.classify(t)
        if as_json:
            print(json.dumps(res, ensure_ascii=False))
        else:
            print(json.dumps(res, ensure_ascii=False, indent=2))


def run_repl(svc):
    print(f"分流服务已就绪：后端={BACKEND_NAME[svc.backend]}，置信度阈值={svc.conf_threshold}，接口版本={VERSION}")
    print("输入一条标题回车看结果；q / exit 退出。")
    while True:
        try:
            title = input("\n标题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if title.lower() in ("q", "quit", "exit"):
            print("再见")
            break
        if not title:
            continue
        print(json.dumps(svc.classify(title), ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description="分流服务 CLI（输入标题 → 返回 P43 契约的 JSON）")
    ap.add_argument("--text", help="单条标题")
    ap.add_argument("--file", help="每行一条标题的文件；传 - 表示从标准输入读")
    ap.add_argument("--model", default="auto", choices=["auto", "bert", "tfidf"], help="指定后端")
    ap.add_argument("--ckpt", default=None, help="BERT 权重路径（默认 outputs/bert_ckpt.pt）")
    ap.add_argument("--conf", type=float, default=0.8, help="置信度阈值，低于它进 review")
    ap.add_argument("--use-rules", action="store_true", help="打开规则快筛（演示 L5 级联）")
    ap.add_argument("--json", action="store_true", help="紧凑输出：一条一行 JSON")
    ap.add_argument("--verbose", action="store_true", help="额外输出 _debug 字段（后端/延迟）")
    args = ap.parse_args()

    svc = ShuntService(args.model, args.ckpt, args.conf, args.use_rules, args.verbose)

    if args.text:
        run_oneshot(svc, [args.text], args.json)
    elif args.file:
        if args.file == "-":
            run_oneshot(svc, sys.stdin.read().splitlines(), args.json)
        else:
            run_oneshot(svc, Path(args.file).read_text(encoding="utf-8").splitlines(), args.json)
    elif not sys.stdin.isatty():       # 管道输入但没给 -：也当作 stdin
        run_oneshot(svc, sys.stdin.read().splitlines(), args.json)
    else:
        run_repl(svc)


if __name__ == "__main__":
    main()
