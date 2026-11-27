"""事件抽取评估指标：span 级精确匹配（(base_label, start, end) 完全一致才算对）。

分组统计
--------
- trigger：基标签 == TRIGGER
- arg    ：基标签 ∈ ENTITY_ROLES
- overall：全部实体

两套口径
--------
- strict：字符区间严格相等（主指标，保持既有口径）
- norm  ：先把文本的空白全部剥离，再比较归一化区间（辅助指标）
          用途是排除"2017 年 6 月 30 日"与"2017年6月30日"这类写法差异的干扰，
          避免把真实能力判成 0；只用于诊断与跨实现对比，不替代 strict。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .label import BIO_ID2LABEL, OUTSIDE_LABEL, PAD_LABEL_ID, TRIGGER_LABEL, split_bio_label

Span = tuple[int, int, str]  # (start, end, base_label)


def spans_from_bio(label_ids: list[int]) -> list[Span]:
    """BIO id 序列 -> [(start, end, base)]。

    - 遇到 PAD_LABEL_ID(-100) 提前结束；
    - 孤立 I-X（前一个不是 B-X/I-X）按 B-X 容错处理；
    - base 变化时切分。
    """
    spans: list[Span] = []
    cur: list | None = None  # [start, end, base]

    def flush() -> None:
        nonlocal cur
        if cur is not None:
            spans.append((cur[0], cur[1], cur[2]))
            cur = None

    for i, lid in enumerate(label_ids):
        if lid == PAD_LABEL_ID:
            break
        label = BIO_ID2LABEL.get(int(lid), OUTSIDE_LABEL)
        if label == OUTSIDE_LABEL:
            flush()
            continue
        prefix, base = split_bio_label(label)
        if prefix == "B":
            flush()
            cur = [i, i + 1, base]
        else:  # I
            if cur is not None and cur[2] == base:
                cur[1] = i + 1
            else:
                flush()
                cur = [i, i + 1, base]
    flush()
    return spans


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def add(self, other: "PRF") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass
class TypeReport:
    """事件类型推断指标（由槽位推断类型，与金标类型比较）。"""

    total: int = 0
    correct: int = 0
    per_class: dict[str, PRF] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def macro_f1(self) -> float:
        vals = [v.f1 for v in self.per_class.values() if (v.tp + v.fn) > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def micro_f1(self) -> float:
        tp = sum(v.tp for v in self.per_class.values())
        fp = sum(v.fp for v in self.per_class.values())
        fn = sum(v.fn for v in self.per_class.values())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def summary(self) -> str:
        return (
            f"type acc={self.accuracy:.4f} macroF1={self.macro_f1:.4f} "
            f"microF1={self.micro_f1:.4f}"
        )

    def as_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "micro_f1": round(self.micro_f1, 4),
            "per_class_f1": {
                k: round(v.f1, 4) for k, v in sorted(self.per_class.items())
            },
        }


def compute_type_metrics(
    gold_types: list[str], pred_types: list[str | None]
) -> TypeReport:
    """计算事件类型推断的 accuracy / macro-F1 / micro-F1。"""
    assert len(gold_types) == len(pred_types)
    report = TypeReport()
    labels = sorted(set(gold_types) | {p for p in pred_types if p})
    counter: dict[str, Counter] = {lab: Counter() for lab in labels}
    for gold, pred in zip(gold_types, pred_types):
        report.total += 1
        if gold == pred:
            report.correct += 1
        # 对每个类做 one-vs-rest
        for lab in labels:
            g = gold == lab
            p = pred == lab
            if g and p:
                counter[lab]["tp"] += 1
            elif p and not g:
                counter[lab]["fp"] += 1
            elif g and not p:
                counter[lab]["fn"] += 1
    for lab in labels:
        c = counter[lab]
        report.per_class[lab] = PRF(tp=c["tp"], fp=c["fp"], fn=c["fn"])
    return report


@dataclass
class EventMatchReport:
    """事件级完全匹配率：trigger 与全部槽位 span 完全一致才算抽对一个事件。

    per_type 按 gold 事件类型统计 {gold: 总数, exact: 完全命中数}。
    稀有类 F1 在样本极少时没有统计意义（错 1 条即为 0），因此额外报告绝对命中数。
    """

    total: int = 0
    exact: int = 0
    per_type: dict[str, dict] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.exact / self.total if self.total else 0.0

    def summary(self) -> str:
        base = f"event exact={self.exact}/{self.total}={self.accuracy:.4f}"
        rare = {k: v for k, v in self.per_type.items() if v["gold"] <= 200}
        if rare:
            base += " | 稀有类命中: " + ", ".join(
                f"{k} {v['exact']}/{v['gold']}" for k, v in rare.items()
            )
        return base

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "exact": self.exact,
            "accuracy": round(self.accuracy, 4),
            "per_type": {
                k: {**v, "accuracy": round(v["exact"] / v["gold"], 4) if v["gold"] else 0.0}
                for k, v in self.per_type.items()
            },
        }


def compute_event_exact_match(
    gold_events: list[dict], pred_events: list[dict]
) -> EventMatchReport:
    """逐个文档比较事件：type + trigger + args 集合完全相同才算对。

    event 结构：{"type": str|None, "trigger": Span|None, "args": frozenset[Span]}
    """
    assert len(gold_events) == len(pred_events)
    report = EventMatchReport()
    for gold, pred in zip(gold_events, pred_events):
        report.total += 1
        gt = gold.get("type") or "未知"
        cell = report.per_type.setdefault(gt, {"gold": 0, "exact": 0})
        cell["gold"] += 1
        if (
            gold.get("type") == pred.get("type")
            and gold.get("trigger") == pred.get("trigger")
            and set(gold.get("args") or ()) == set(pred.get("args") or ())
        ):
            report.exact += 1
            cell["exact"] += 1
    return report


@dataclass
class MetricReport:
    overall: PRF = field(default_factory=PRF)
    trigger: PRF = field(default_factory=PRF)
    arg: PRF = field(default_factory=PRF)
    per_type: dict[str, PRF] = field(default_factory=dict)
    event_type: TypeReport | None = None
    event_match: EventMatchReport | None = None
    # 空白归一化口径（可选，只有传了 texts 才有）
    overall_norm: PRF | None = None
    arg_norm: PRF | None = None

    @property
    def f1(self) -> float:
        return self.overall.f1

    def summary(self) -> str:
        o, t, a = self.overall, self.trigger, self.arg
        base = (
            f"P={o.precision:.4f} R={o.recall:.4f} F1={o.f1:.4f} | "
            f"trigger F1={t.f1:.4f} | arg F1={a.f1:.4f}"
        )
        if self.arg_norm is not None:
            base += f" | arg F1(norm={self.arg_norm.f1:.4f})"
        if self.event_type is not None:
            base += f" | {self.event_type.summary()}"
        if self.event_match is not None:
            base += f" | {self.event_match.summary()}"
        return base

    def as_dict(self) -> dict:
        out = {
            "overall": self.overall.as_dict(),
            "trigger": self.trigger.as_dict(),
            "arg": self.arg.as_dict(),
            "per_type_f1": {k: round(v.f1, 4) for k, v in self.per_type.items()},
        }
        if self.arg_norm is not None:
            out["arg_norm"] = self.arg_norm.as_dict()
            out["overall_norm"] = self.overall_norm.as_dict() if self.overall_norm else None
        if self.event_type is not None:
            out["event_type"] = self.event_type.as_dict()
        if self.event_match is not None:
            out["event_match"] = self.event_match.as_dict()
        return out


def _prf_pair(gold: set[Span], pred: set[Span]) -> tuple[PRF, Counter]:
    tp = gold & pred
    fp = pred - gold
    fn = gold - pred
    stat = PRF(tp=len(tp), fp=len(fp), fn=len(fn))
    hit_by_base = Counter(base for _, _, base in tp)
    return stat, hit_by_base


def compute_metrics(
    gold_spans_list: list[list[Span]],
    pred_spans_list: list[list[Span]],
) -> MetricReport:
    """逐样本比较 gold / pred span 集合（span 精确匹配）。"""
    assert len(gold_spans_list) == len(pred_spans_list)
    report = MetricReport()

    for gold_spans, pred_spans in zip(gold_spans_list, pred_spans_list):
        gold = set(gold_spans)
        pred = set(pred_spans)

        # 分组：trigger / arg
        gold_trig = {s for s in gold if s[2] == TRIGGER_LABEL}
        pred_trig = {s for s in pred if s[2] == TRIGGER_LABEL}
        gold_arg = gold - gold_trig
        pred_arg = pred - pred_trig

        report.overall.add(_prf_pair(gold, pred)[0])
        report.trigger.add(_prf_pair(gold_trig, pred_trig)[0])
        arg_stat, _ = _prf_pair(gold_arg, pred_arg)
        report.arg.add(arg_stat)

        # 按槽位/trigger 类型细分
        bases = {s[2] for s in gold} | {s[2] for s in pred}
        for base in bases:
            g = {s for s in gold if s[2] == base}
            p = {s for s in pred if s[2] == base}
            report.per_type.setdefault(base, PRF()).add(_prf_pair(g, p)[0])

    return report


def _norm_prefix(text: str) -> list[int]:
    """返回长度 len(text)+1 的映射：pref[i] = text[:i] 中非空白字符的个数。

    用它可以把任意字符区间投影到"剥离空白后"的坐标系，
    从而比较 "2017 年 6 月 30 日" 与 "2017年6月30日" 这类写法差异。
    """
    pref = [0] * (len(text) + 1)
    n = 0
    for i, ch in enumerate(text):
        if not ch.isspace():
            n += 1
        pref[i + 1] = n
    return pref


def compute_norm_metrics(
    gold_spans_list: list[list[Span]],
    pred_spans_list: list[list[Span]],
    texts: list[str],
) -> MetricReport:
    """空白归一化口径的指标：剥离文本空白后比较区间。

    入参与 compute_metrics 一致，只多一个逐样本的原文 texts。
    归一化后长度为 0 的 span（全空白）直接丢弃。
    """
    assert len(gold_spans_list) == len(pred_spans_list) == len(texts)
    gold_norm: list[list[Span]] = []
    pred_norm: list[list[Span]] = []
    for gold, pred, text in zip(gold_spans_list, pred_spans_list, texts):
        pref = _norm_prefix(text)
        gold_norm.append(
            [
                (pref[s], pref[e], base)
                for s, e, base in gold
                if 0 <= s <= e <= len(text) and pref[e] > pref[s]
            ]
        )
        pred_norm.append(
            [
                (pref[s], pref[e], base)
                for s, e, base in pred
                if 0 <= s <= e <= len(text) and pref[e] > pref[s]
            ]
        )
    return compute_metrics(gold_norm, pred_norm)

