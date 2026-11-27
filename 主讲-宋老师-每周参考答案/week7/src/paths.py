# -*- coding: utf-8 -*-
"""
路径解析：数据目录、预训练模型目录、产物目录。

为什么单独写一个文件：
  学生把作业目录放到不同位置（课程根目录下、单独解压到桌面、或者老师本机的
  `课后作业/week7/` 下），写死 `../../pretrain_models` 一定会有人跑不通。
  这里统一做「向上逐级找 + 环境变量 + 命令行显式指定」三级兜底，
  找不到时把**尝试过的所有路径**打印出来，避免只报一句 FileNotFoundError。
"""

import os
from pathlib import Path

# 本文件在 <项目根>/src/paths.py，parents[1] 就是项目根（bio_ner_exp/）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "peoples_daily"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
PRED_DIR = OUTPUT_DIR / "predictions"
FIG_DIR = OUTPUT_DIR / "figures"

# 本作业统一用 bert-base-chinese：它和课上 week7 的 BERT 基线同源，
# 换数据集时只改数据、不改模型，才能把「格式处理」这一个变量单独拎出来看。
BERT_NAME = "bert-base-chinese"


def ensure_output_dirs() -> None:
    """一次性建好所有产物目录，后面脚本不用反复 mkdir。"""
    for d in (LOG_DIR, CKPT_DIR, PRED_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _candidate_roots():
    """
    可能藏着 `pretrain_models/` 的目录，按优先级从近到远。

    覆盖两种常见摆放：
      1. 作业目录在课程根下：<课程根>/week7-xxx/课后作业/bio_ner_exp/
         → 向上 3 级就是 <课程根>，那里有 pretrain_models/
      2. 学生从课程根目录启动脚本（cwd 在 <课程根>）
         → 直接用 cwd 下的 pretrain_models/
    """
    seen = set()
    for base in (PROJECT_ROOT, Path.cwd().resolve()):
        for p in (base, *base.parents):
            if p not in seen:
                seen.add(p)
                yield p


def resolve_bert_path(cli_value: str | Path | None = None, name: str = BERT_NAME) -> Path:
    """
    找到本地 BERT 权重目录（必须含 config.json）。

    参数：
      cli_value：`--bert_path` 传入的路径，优先级最高
      name：模型目录名，默认 bert-base-chinese

    返回：
      绝对路径 Path

    异常：
      FileNotFoundError，消息里列出所有尝试过的路径
    """
    # 1) 命令行显式指定：最可靠，老师本机复现时用这个
    if cli_value:
        p = Path(cli_value).expanduser()
        if p.is_dir() and (p / "config.json").exists():
            return p.resolve()
        raise FileNotFoundError(
            f"--bert_path 指向的目录不是有效的 HuggingFace 模型目录：{p}\n"
            f"  需要包含 config.json（以及 pytorch_model.bin / model.safetensors）"
        )

    # 2) 环境变量：学生可以把 pretrain_models 放在任意位置
    env_dir = os.environ.get("BADOU_PRETRAIN_DIR")
    if env_dir:
        p = Path(env_dir).expanduser() / name
        if p.is_dir() and (p / "config.json").exists():
            return p.resolve()

    # 3) 向上逐级找 pretrain_models/<name>
    tried: list[Path] = []
    for root in _candidate_roots():
        p = root / "pretrain_models" / name
        if p in tried:
            continue
        tried.append(p)
        if p.is_dir() and (p / "config.json").exists():
            return p.resolve()

    listed = "\n".join(f"  - {p}" for p in tried)
    raise FileNotFoundError(
        f"未找到本地模型 {name}。\n"
        f"请用 --bert_path 指定权重目录，或设置环境变量 BADOU_PRETRAIN_DIR"
        f"（指向 pretrain_models 的父目录）。\n已尝试：\n{listed}"
    )


def resolve_data_dir(cli_value: str | Path | None = None) -> Path:
    """数据目录解析：默认本项目 data/peoples_daily，可用 --data_dir 覆盖。"""
    if cli_value:
        p = Path(cli_value).expanduser()
        if not (p / "train.json").exists():
            raise FileNotFoundError(f"数据目录里没有 train.json：{p}")
        return p.resolve()

    if not (DATA_DIR / "train.json").exists():
        raise FileNotFoundError(
            f"默认数据目录不完整：{DATA_DIR}\n"
            f"请确认 data/peoples_daily/ 下有 train.json / validation.json / test.json"
        )
    return DATA_DIR
