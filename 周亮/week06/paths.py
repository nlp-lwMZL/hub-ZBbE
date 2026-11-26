"""
统一路径解析 —— 同一份代码，本机与 Colab 各自落到正确的目录

为什么要单独一层：
  Colab 断线/重启会清空 /content（包括 checkpoints 和 logs），所以所有产物
  必须写到挂载的 Google Drive 上；但本机没有 Drive，也不该硬编码任何 /content 路径。
  把「输出根目录」收成一处配置，避免每个脚本各自拼路径——那是最典型的
  「本机跑得通、Colab 上白跑一晚」的原因。

三个环境变量（都有默认值，本机不设也能直接跑）：
  WEEK6_OUT_ROOT    训练日志 / checkpoints / outputs 的根     本机默认 homework/week6，
                    **Colab 上默认自动指向 Drive 的 week6_runs**（见 _default_out_root）
  WEEK6_DATA_DIR    数据目录（train_5k.jsonl 等）            默认 homework/week6/data
  WEEK6_MODEL_ROOT  底座模型缓存（BERT / LLM 共用）           **代码目录同级的 models/**：
                    Colab = /content/models，本机 = homework/models

「产物进 Drive，权重留本地」是两条不同的规则，别混：
  产物  不可再生（跑一晚的 ckpt / 日志 / 逐条预测）→ 必须扛住 /content 被清空 → 放 Drive
  底座  可再生（几十秒到几分钟就能重下）          → 只求读得快、不占配额 → 放本地磁盘
把 3~11GB 权重塞进 Drive 的代价：吃掉 15GB 免费额度的一大半，且每次读盘都走网络挂载。

Colab 上的默认布局（由 colab/_paths.sh 自动设置，与直接 python 调用一致）：
  /content/models/                              ← 底座缓存（本地盘，与 week6 同级）
      ├── nlp_roberta_backbone_base_std/        BERT 底座 412MB
      └── qwen3-0.6b/ qwen3-1.7b/ qwen3-4b-instruct-2507/
  /content/drive/MyDrive/week6_runs/            ← 只有产物
      logs/  checkpoints/  outputs/  state/

目录结构（OUT_ROOT 下，全部是「不可再生」的东西）：
  logs/            训练与评测日志（.log + 同名 .json）
  checkpoints/     {run_name}/best.pt（BERT）或 SFT adapter/权重（LLM）
  outputs/eval/    评测指标、逐条预测、汇总表
  outputs/figures/ 图表
  state/           调度器运行状态（断点续跑用）
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 底座缓存目录名 —— 代码目录的**同级**，不是里面（详见 _default_model_root）
MODEL_DIRNAME = "models"
# 旧版落点：代码目录内部的 pretrain_models。只读复用，不再往这里写新的
LEGACY_MODEL_DIRS = (HERE / "pretrain_models",)


def _env_path(name: str, default: Path) -> Path:
    """读环境变量；空字符串视为未设置（shell 里 export X= 是常见笔误）"""
    v = os.environ.get(name, "").strip()
    return Path(v).expanduser() if v else default


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:                                   # noqa: BLE001
        return False


def _drive_run_root() -> Path | None:
    """挂载了 Drive 就返回 <MyDrive>/week6_runs，否则 None"""
    for cand in (Path("/content/drive/MyDrive"), Path("/content/drive/My Drive")):
        if cand.is_dir():
            return cand / "week6_runs"
    return None


def _default_out_root() -> Path:
    """
    OUT_ROOT 的默认值 —— 「本机跑得通、Colab 上白跑」最隐蔽的一个坑就在这里：
    直接 `python predict_llm.py`（不经过 colab/*.sh）时，若默认落在代码目录，
    Colab 上产物会写进 /content，运行时一释放就全没了。所以 Colab 上默认指
    Drive 的 week6_runs；显式 WEEK6_OUT_ROOT 永远优先，本机行为完全不变。
    """
    v = os.environ.get("WEEK6_OUT_ROOT", "").strip()
    if v:
        return Path(v).expanduser()
    if not _is_colab():
        return HERE                                     # 本机：写到代码目录
    return _drive_run_root() or HERE                    # Colab 没挂 Drive：只能先跑本地


def _default_model_root() -> Path:
    """
    底座模型缓存的默认值 = **代码目录同级的 models/**（与 BERT 训练的做法一致）：
      本机  homework/week6/paths.py → homework/models
      Colab /content/week6/paths.py → /content/models

    刻意不放 Drive：
      1) 权重可再生 —— 断线重下几分钟的事，不值得为它占 15GB 免费额度；
      2) Drive 是网络挂载，加载 3~8GB safetensors 比本地盘慢一截；
      3) 只有「重跑一晚就没了」的产物才需要跨会话保命，那才是 OUT_ROOT 的职责。

    想换个地方（比如挂第二块盘）：export WEEK6_MODEL_ROOT=/data/models
    旧版留在 pretrain_models/ 的缓存不会浪费，见 legacy_model_roots()。
    """
    v = os.environ.get("WEEK6_MODEL_ROOT", "").strip()
    if v:
        return Path(v).expanduser()
    return HERE.parent / MODEL_DIRNAME


def legacy_model_roots() -> list[Path]:
    """
    历史缓存位置（旧版默认值），存在才返回：
      <代码目录>/pretrain_models        —— 改默认值之前本机的落点
      <Drive>/week6_runs/pretrain_models —— 上一版 Colab 的落点
    ensure_local_model / ensure_local_llm 会先在这两处找已经下好的权重，
    找到就复用：换默认位置不等于要把 3~11GB 重新下一遍。
    """
    cands = list(LEGACY_MODEL_DIRS)
    drive = _drive_run_root()
    if drive is not None:
        cands.append(drive / "pretrain_models")
    out = []
    for p in cands:
        try:
            if p.is_dir() and p.resolve() != MODEL_ROOT.resolve():
                out.append(p)
        except OSError:
            continue
    return out


# ── 四个根目录（模块级默认值，命令行/参数可再覆盖）──
OUT_ROOT = _default_out_root()
DATA_DIR = _env_path("WEEK6_DATA_DIR", HERE / "data")
MODEL_ROOT = _default_model_root()

LOG_DIR = OUT_ROOT / "logs"
CKPT_DIR = OUT_ROOT / "checkpoints"
EVAL_DIR = OUT_ROOT / "outputs" / "eval"
FIG_DIR = OUT_ROOT / "outputs" / "figures"
STATE_DIR = OUT_ROOT / "state"


def is_colab() -> bool:
    """是否运行在 Colab 里（用于选择下载源 / 给出针对性提示）"""
    return _is_colab()


def detect_drive_root(subdir: str = "week6_runs") -> Path | None:
    """
    探测 Google Drive 挂载点。挂载成功后通常是 /content/drive/MyDrive。
    没挂载返回 None（setup_env.sh 会提示怎么挂），调用方据此决定是否退回本地目录。
    """
    for cand in (Path("/content/drive/MyDrive"), Path("/content/drive/My Drive")):
        if cand.is_dir() and any(cand.iterdir()):
            return cand / subdir
    # 挂载了但目录为空也认（新账号第一次用）
    d = Path("/content/drive/MyDrive")
    return (d / subdir) if d.is_dir() else None


def ensure_dirs() -> dict:
    """建齐所有输出/缓存目录。返回 {名字: Path}，方便调用方打日志。"""
    dirs = {"out_root": OUT_ROOT, "logs": LOG_DIR, "checkpoints": CKPT_DIR,
            "eval": EVAL_DIR, "figures": FIG_DIR, "state": STATE_DIR,
            "models": MODEL_ROOT}
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def model_root_in_drive() -> bool:
    """MODEL_ROOT 是否落在 Drive 挂载点里（不该：吃配额、读盘还慢）"""
    return "/drive/" in str(MODEL_ROOT).replace("\\", "/")


def summary() -> str:
    """一行式路径概览，每个脚本启动时打印，出问题第一眼就能看到写去哪了"""
    s = f"OUT_ROOT={OUT_ROOT} | DATA_DIR={DATA_DIR} | MODEL_ROOT={MODEL_ROOT}"
    if model_root_in_drive():
        s += "\n!! 模型缓存落在 Drive 上（占配额且读盘慢），建议挪到本地盘：" \
             "export WEEK6_MODEL_ROOT=/content/models"
    return s
