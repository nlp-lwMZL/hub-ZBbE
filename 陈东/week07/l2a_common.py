"""
L2a 预训练模型替换对比实验 —— 公共配置

对比对象（唯一变量 = backbone 预训练权重，其余全部固定）：
  baseline : bert-base-chinese         （本地 week4 目录，原 L2a 用的就是它）
  roberta  : chinese-roberta-wwm-ext   （hfl，全词掩码，同规模 12L/768）

只服务 homework/ 下的脚本，不改动课程原始代码。
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../week7-.../code/homework
ROOT = HERE.parent                               # .../week7-.../code  （项目根）
COURSE = ROOT.parents[2]                         # .../badou/course

# 两个 backbone 的本地路径（绝对路径，直接传给课程脚本的 --bert_path）
MODELS = {
    "baseline": {
        "name": "bert-base-chinese",
        "path": COURSE / "week4-大语言模型" / "bert-base-chinese",
        "desc": "bert-base-chinese（原 L2a）",
    },
    "roberta": {
        "name": "chinese-roberta-wwm-ext",
        "path": COURSE / "pretrain_models" / "chinese-roberta-wwm-ext",
        "desc": "hfl/chinese-roberta-wwm-ext（新 L2a）",
    },
}

# 课程脚本写入的"中转站"路径（bert/train.py / evaluate_hard_constraint.py 硬编码 ROOT/outputs/）
COURSE_CKPT = ROOT / "outputs" / "checkpoints"
COURSE_METRIC = ROOT / "outputs" / "metrics" / "disc"
COURSE_PRED = ROOT / "outputs" / "disc"
COURSE_LOG = ROOT / "outputs" / "logs"

# 作业自己的产物目录（week6 作业在 homework/ckpts、homework/results，这里对齐）
HW_CKPT = HERE / "ckpts"
HW_METRIC = HERE / "metrics"
HW_PRED = HERE / "preds"
HW_LOG = HERE / "logs"
for d in (HW_CKPT, HW_METRIC, HW_PRED, HW_LOG):
    d.mkdir(parents=True, exist_ok=True)

# 兼容旧名字：analysis.py 里用了 CKPT_DIR / METRIC_DIR / PRED_DIR，统一指向 homework
CKPT_DIR = HW_CKPT
METRIC_DIR = HW_METRIC
PRED_DIR = HW_PRED

GOLD_PATH = ROOT / "data" / "processed" / "dev_extract.jsonl"
HW_OUT = HERE
COMPARISON_MD = HERE / "comparison.md"

# 实验矩阵：decoder × backbone，以及 CRF 的硬约束解码变体
# (decoder, run_name 后缀, 是否需要硬约束复评)
RUNS = [
    ("linear", "hw7_linear_baseline", "baseline", False),
    ("linear", "hw7_linear_roberta", "roberta", False),
    ("crf", "hw7_crf_baseline", "baseline", True),
    ("crf", "hw7_crf_roberta", "roberta", True),
]
HARD_PAIRS = [
    ("hw7_crf_baseline", "hw7_crf_baseline_hard"),
    ("hw7_crf_roberta", "hw7_crf_roberta_hard"),
]

# 历史参考行（原项目产出，仅作对照；主对比用本次同环境重训的 baseline）
HISTORICAL_REF = {
    "hw7_ref_linear": ("bert_linear_256", "BERT-12L+Linear（历史，异环境）"),
    "hw7_ref_crf": ("bert_crf_256", "BERT-12L+CRF（历史，异环境）"),
    "hw7_ref_crf_hard": ("bert_crf_256_hard", "BERT-12L+CRF 硬约束（历史，异环境）"),
}


def model_path(key: str) -> Path:
    return Path(MODELS[key]["path"])


def has_weights(p: Path) -> bool:
    p = Path(p)
    return p.is_dir() and ((p / "pytorch_model.bin").exists() or any(p.glob("*.safetensors")))


def rsync_hw7_from_course():
    """把课程脚本写进 outputs/ 的 hw7_* 文件全部同步到 homework/ 子目录。

    课程脚本（bert/train.py / evaluate_hard_constraint.py）硬编码写到
    ROOT/outputs/{checkpoints,metrics/disc,disc,logs}，我们不碰它们的源码，
    这里把 hw7_* 前缀的文件整体搬到 homework/ckpts、homework/metrics、
    homework/preds、homework/logs，方便作业独立归档。
    """
    import shutil

    mapping = [
        (COURSE_CKPT, HW_CKPT, ["hw7_*.pt"]),
        (COURSE_METRIC, HW_METRIC, ["hw7_*.json"]),
        (COURSE_PRED, HW_PRED, ["pred_hw7_*.jsonl"]),
        (COURSE_LOG, HW_LOG, ["hw7_*"]),
    ]
    for src_dir, dst_dir, patterns in mapping:
        if not src_dir.exists():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for pat in patterns:
            for f in src_dir.glob(pat):
                dst = dst_dir / f.name
                shutil.copy2(f, dst)
    print(f"[rsync] hw7_* → homework/ 已同步 "
          f"({len(list(HW_CKPT.glob('hw7_*.pt')))} ckpts, "
          f"{len(list(HW_METRIC.glob('hw7_*.json')))} metrics, "
          f"{len(list(HW_PRED.glob('pred_hw7_*.jsonl')))} preds, "
          f"{len(list(HW_LOG.glob('hw7_*')))} logs)")
