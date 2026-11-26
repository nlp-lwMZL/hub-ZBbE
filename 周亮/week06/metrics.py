"""
统一评测指标（BERT / TF-IDF / LLM 三路方法共用）

教学重点（沿用课程 common/metrics.py 的结论）：
  1. 分类任务只看 Accuracy 会被长尾类骗：头部类预测对就够高，尾部类全错也看不出来。
     本任务「中」占 55.3%，闭眼全猜「中」就有 0.553 的准确率，但 Macro-F1 只有 0.238。
     所以**主指标是 Macro-F1（每类等权）**，Accuracy 只当参考。
  2. Macro-F1 与 Accuracy 的差距本身就是诊断信号：差得多 → 尾部类（「低」）出问题。
  3. 置信度校准（ECE）回答「模型说 0.9 时实际命中率是不是 0.9」，
     决定了置信度能不能拿来做「低置信转人工」。

为什么单独成文件：
  四路方法必须用**同一个函数**算指标。这节课最容易出的错是「BERT 脚本报 macro-F1、
  TF-IDF 脚本报 weighted-F1」，两个数放在一张表里就失去可比性——而且不报错、看不出来。
"""

import re

import numpy as np


def expected_calibration_error(y_true, probs, n_bins=10):
    """
    ECE：把置信度分桶，比较「桶内平均置信度」和「桶内实际准确率」的差，按样本数加权。
    值越小越可信；> 0.1 基本不能拿置信度当阈值用。

    probs: shape [N, C] 的 softmax 概率；y_true: [N] 的整数标签

    ⚠️ 只吃 probs 绝对值，所以**只能同族内比**：
      BERT 的 conf 是 3 类 softmax 的绝对概率（假设空间天然封闭，P(必是三类之一)=1）；
      LLM 的 conf 是「模型愿意只输出这三个词之一」条件下的重归一化概率，
      词表外那 80%~90% 的质量被丢掉了（见 model_llm.logprob_classify），
      归一化操作把它抬成 1，conf 系统性虚高。
      两个数分别是两个不同事件的概率，放一列里比是伪比较。汇总表因此拆成
      ECE·BERT / ECE·LLM 两列，各自纵向可比，横向不可比。
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # 第一桶含 0，其余左开右闭，保证 conf=1.0 不会漏
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        gap = abs(float(correct[mask].mean()) - float(conf[mask].mean()))
        ece += gap * float(mask.sum()) / float(len(conf))
    return round(ece, 4)


def classification_report(y_true, y_pred, probs=None, id2name=None, n_classes=None):
    """
    统一口径的指标字典。

    y_pred 里允许出现 -1 表示「解析失败」（生成式方法专用）：
      acc                —— 只在解析成功的样本上算（反映模型答对的能力）
      acc_incl_unparsed  —— 解析失败算答错，分母是全部样本（对外张榜用这个数）
      macro_f1           —— 同上，只在解析成功的样本上算
    判别式模型（BERT/TF-IDF）不会有 -1，两个 acc 恒相等。

    per_class 的 key 用中文标签名（id2name），报告里直接可读；
    id2name 缺省时退化为整数 id。
    """
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    n_total = int(len(yt))

    if n_classes is None:
        n_classes = len(id2name) if id2name else int(max(yt.max(), yp.max())) + 1
    name_of = (lambda c: id2name.get(c, str(c))) if id2name else (lambda c: str(c))

    valid = yp >= 0
    n_unparsed = int((~valid).sum())
    rep = {
        "n": n_total,
        "unparsed": n_unparsed,
        "unparsed_rate": round(n_unparsed / n_total, 4) if n_total else 0.0,
        "labels": [name_of(c) for c in range(n_classes)],
    }
    if valid.sum() == 0:
        return rep

    yv, pv = yt[valid], yp[valid]
    rep["acc"] = round(float(accuracy_score(yv, pv)), 4)
    rep["acc_incl_unparsed"] = round(float((yv == pv).sum()) / n_total, 4) if n_total else 0.0
    rep["macro_f1"] = round(float(f1_score(yv, pv, average="macro", labels=list(range(n_classes)),
                                           zero_division=0)), 4)
    rep["weighted_f1"] = round(float(f1_score(yv, pv, average="weighted", zero_division=0)), 4)

    rep["per_class"] = {}
    for c in range(n_classes):
        mask = yv == c
        if mask.sum() == 0:
            rep["per_class"][name_of(c)] = None
            continue
        hit = int((pv[mask] == c).sum())
        recall = hit / float(mask.sum())
        pred_cnt = int((pv == c).sum())
        precision = hit / float(pred_cnt) if pred_cnt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rep["per_class"][name_of(c)] = {
            "P": round(precision, 4), "R": round(recall, 4), "F1": round(f1, 4),
            "support": int(mask.sum()),
        }
    rep["confusion_matrix"] = confusion_matrix(yv, pv, labels=list(range(n_classes))).tolist()

    if probs is not None:
        pv_probs = np.asarray(probs, dtype=np.float64)[valid]
        rep["ece"] = expected_calibration_error(yv, pv_probs)
        rep["mean_conf"] = round(float(pv_probs.max(axis=1).mean()), 4)
    return rep


def pred_distribution(y_pred, id2name=None):
    """预测分布：模型塌缩到少数几类时（尤其 LLM「全猜中」）一眼就能看出来"""
    counts = {}
    for p in y_pred:
        p = int(p)
        key = id2name.get(p, str(p)) if id2name else str(p)
        counts[key] = counts.get(key, 0) + 1
    used = sum(1 for k in counts if not k.startswith("-"))
    return {"used_classes": used, "counts": counts}


def brief(rep, id2name=None):
    """每个 epoch 一行的紧凑摘要：低类 recall 是本任务的观测重点，必须一眼可见"""
    per = " ".join(f"{k}R={v['R']:.3f}" for k, v in (rep.get("per_class") or {}).items() if v)
    return (f"acc={rep.get('acc', 0):.4f} macro_f1={rep.get('macro_f1', 0):.4f}"
            + (f" | {per}" if per else ""))


def to_markdown_table(results: dict, low_label="低", ece_header="ECE"):
    """
    {方法名: 指标字典} → Markdown 表，直接贴进作业报告。
    方法名建议先用 describe_bert_run / describe_llm_run 转成中文说明再进来。

    额外带上「低类 F1 / 低类 R」两列：本任务 6.2x 不均衡，消融的真实差异几乎
    都体现在尾部类上，总指标经常被「中」类淹掉看不出来。

    ece_header：ECE 列的表头。BERT 与 LLM 的置信度口径不同（见
    expected_calibration_error 的说明），**跨族不可比**，所以各脚本把自己的族
    写进表头（ECE·BERT / ECE·LLM）。两张表拼在一起时，列名不同就不会被误读成
    同一列数字。
    """
    has_low = any((r.get("per_class") or {}).get(low_label) for r in results.values())
    if has_low:
        head = [f"| 方法 | 准确率 | Macro-F1 | 低类 F1 | 低类 R | {ece_header} |",
                "|---|---|---|---|---|---|"]
    else:
        head = [f"| 方法 | 准确率 | Macro-F1 | {ece_header} |", "|---|---|---|---|"]
    lines = list(head)
    for name, r in results.items():
        acc = r.get("acc", "-")
        f1 = r.get("macro_f1", "-")
        ece = r.get("ece", "-")
        if has_low:
            low = (r.get("per_class") or {}).get(low_label) or {}
            lines.append(f"| {name} | {acc} | {f1} | {low.get('F1', '-')} | "
                         f"{low.get('R', '-')} | {ece} |")
        else:
            lines.append(f"| {name} | {acc} | {f1} | {ece} |")
    return "\n".join(lines)


def _display_width(s):
    """中文/全角字符占 2 列，用 len() 算列宽会让等宽表格错位"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in str(s))


def _rjust(s, width):
    return " " * max(0, width - _display_width(s)) + str(s)


def format_confusion(rep, id2name):
    """混淆矩阵转成带中文行列表头的等宽表格（日志里能直接看懂哪类被误判到哪类）"""
    names = [id2name[c] for c in sorted(id2name)]
    cm = rep["confusion_matrix"]
    colw = max(6, max(_display_width("预" + n) for n in names),
               max(_display_width(v) for row in cm for v in row))
    roww = max(_display_width("真" + n) for n in names) + 2
    lines = [" " * roww + "".join(_rjust("预" + n, colw) for n in names)]
    for i, n in enumerate(names):
        lines.append(_rjust("真" + n, roww) + "".join(_rjust(v, colw) for v in cm[i]))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# run_name → 可读中文说明：汇总表的「方法」列给人看，run_name 给机器看
#
# run_name 是「文件名即配置」（pool-mean_cw-none_focal-0_...），调度器靠它断点续跑、
# 评测靠它找 ckpt——但对人来说可读性差。转换规则：**只列与基线的差异**，
# 全默认的组标注「（基线）」，改了什么就写什么：
#   pool-mean_cw-none_focal-0_... → "Mean 池化"
#   pool-cls_cw-none_focal-2.0... → "CLS 池化 + Focal γ=2"
#   pool-cls_..._lr-3e-05_...     → "CLS 池化 + 学习率 3e-5"
# 中文说明与 run_name 的对照表会一并写进 summary，保证报告里的行可追溯。
# ─────────────────────────────────────────────────────────────────────────────

# 基线口径（train_bert.py 的默认参数）：与它不同的旋钮才进描述
_BERT_BASELINE = {"cw": "none", "focal": "0", "ls": "0", "fgm": "0",
                  "len": "192", "text": "desc", "n": "5000",
                  "bs": "32", "lr": "2e-05", "ep": "3"}

_POOL_CN = {"cls": "CLS 池化", "mean": "Mean 池化", "max": "Max 池化"}
_CW_CN = {"sqrt": "类别权重 √freq", "balanced": "类别权重 balanced"}


def _parse_run_tokens(run_name: str) -> dict:
    """run_name → {旋钮: 值}。fgm-1-eps1 这类多段值取第一段做键。"""
    toks = {}
    for part in str(run_name).split("_"):
        if "-" in part:
            k, v = part.split("-", 1)
            toks[k] = v
    return toks


def _fmt_lr(v: str) -> str:
    """2e-05 → 2e-5（Python 的 :g 会给两位指数，人读着别扭）"""
    s = str(v).replace("e-0", "e-").replace("e+0", "e+")
    return s[:-1] if s.endswith("+") else s


def describe_bert_run(run_name: str) -> str:
    """
    BERT 消融 run_name → 中文说明（只列与基线的差异）。
    汇总表用它做「方法」列；对照表里保留 run_name 保证可追溯。
    """
    t = _parse_run_tokens(run_name)
    is_llm = str(run_name).startswith("llm-")
    pool = t.get("pool", "")
    if is_llm or not pool:
        # 不是 BERT 命名（比如 llm- 开头）：原样返回，绝不瞎猜
        return str(run_name)

    items = [_POOL_CN.get(pool, f"{pool.upper()} 池化")]

    cw = t.get("cw", "none")
    if cw != _BERT_BASELINE["cw"]:
        items.append(_CW_CN.get(cw, f"类别权重 {cw}"))
    try:
        focal = float(t.get("focal", 0))
    except ValueError:
        focal = 0.0
    if focal > 0:
        items.append(f"Focal γ={focal:g}")
    try:
        ls = float(t.get("ls", 0))
    except ValueError:
        ls = 0.0
    if ls > 0:
        items.append(f"标签平滑 {ls:g}")
    fgm = t.get("fgm", "0")
    if fgm.startswith("1"):
        m = re.search(r"eps([\d.]+)", fgm)
        items.append(f"FGM 对抗（eps={m.group(1) if m else '1'}）")
    if t.get("len") != _BERT_BASELINE["len"]:
        items.append(f"截断长度 {t['len']}")
    if t.get("text") == "titledesc":
        items.append("拼接标题")
    if t.get("n") != _BERT_BASELINE["n"]:
        n = t.get("n", "")
        items.append(f"数据量 {int(n) / 1000:g}k" if n.isdigit() else f"数据量 {n}")
    if t.get("bs") != _BERT_BASELINE["bs"]:
        items.append(f"batch {t['bs']}")
    if t.get("lr") != _BERT_BASELINE["lr"]:
        items.append(f"学习率 {_fmt_lr(t['lr'])}")
    if t.get("ep") != _BERT_BASELINE["ep"]:
        items.append(f"{t['ep']} epoch")

    desc = " + ".join(items)
    # 全默认 = 基线组：标出来，报告里一眼知道谁是参照系
    if len(items) == 1 and pool == "cls":
        desc += "（基线）"
    if "smoke" in str(run_name):
        desc += "（冒烟）"
    return desc


_LLM_MODEL_CN = {"qwen3-0.6b": "Qwen3-0.6B", "qwen3-1.7b": "Qwen3-1.7B",
                 "qwen3-4b-instruct-2507": "Qwen3-4B"}


def describe_llm_run(run_name: str) -> str:
    """
    LLM 训练 run_name（llm-qwen3-1.7b_mode-lora_r8a16tgt-qkvo_len-384_...）
    → 中文说明，如 "Qwen3-1.7B · LoRA 微调（r=8）"。
    """
    s = str(run_name)
    if not s.startswith("llm-"):
        return s
    t = _parse_run_tokens(s)
    model = s.split("_")[0].replace("llm-", "")
    model_cn = _LLM_MODEL_CN.get(model, model)
    mode = {"lora": "LoRA 微调", "full": "全量微调", "qlora": "QLoRA 微调"}.get(
        t.get("mode", ""), t.get("mode", "微调"))
    extra = ""
    if t.get("mode") in ("lora", "qlora"):
        mr = re.search(r"r(\d+)a(\d+)", s)
        if mr:
            extra = f"（r={mr.group(1)}）"
    if "smoke" in s:
        extra += "（冒烟）"
    return f"{model_cn} · {mode}{extra}"
