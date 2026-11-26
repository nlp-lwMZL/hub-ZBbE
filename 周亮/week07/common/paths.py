"""
统一路径解析 —— 同一份代码，本机与 Colab 各自落到正确的目录。

为什么要单独一层（week6 的成熟经验，week7 直接复用）：
  Colab 断线/重启会清空 /content（含 checkpoints / logs / 逐条预测），
  所以「不可再生产物」必须写到挂载的 Google Drive；而底座权重是「可再生」的，
  放本地盘读得快、不占 Drive 15GB 配额。两条规则别混：
    产物  -> WEEK7_OUT_ROOT    本机默认 code/；Colab 默认 Drive 的 week7_runs
    数据  -> WEEK7_DATA_DIR    默认 code/data/cdtier/clean
    底座  -> WEEK7_MODEL_ROOT  代码目录同级 models/（Colab = /content/models）

环境变量（都有默认值，本机不设也能直接跑）：
  WEEK7_OUT_ROOT    训练日志 / checkpoints / eval 结果的根
  WEEK7_DATA_DIR    cdtier clean 数据目录
  WEEK7_MODEL_ROOT  底座模型缓存（RoBERTa / Qwen3 共用）

Colab 上推荐在 notebook 里先这样设（colab/README.md 有完整流程）：
    from google.colab import drive; drive.mount('/content/drive')
    import os
    os.environ['WEEK7_OUT_ROOT'] = '/content/drive/MyDrive/week7_runs'
    os.environ['WEEK7_MODEL_ROOT'] = '/content/models'
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent                      # code/ 目录
DEFAULT_DATA_DIR = CODE_DIR / "data" / "cdtier" / "clean"
LEGACY_MODEL_DIR = CODE_DIR / "pretrain_models"   # 旧缓存，只读复用


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name, "").strip()
    return Path(v).expanduser() if v else default


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:                                   # noqa: BLE001
        return False


def _drive_run_root() -> Path | None:
    for cand in (Path("/content/drive/MyDrive"), Path("/content/drive/My Drive")):
        if cand.is_dir():
            return cand / "week7_runs"
    return None


def get_out_root() -> Path:
    """产物根目录：显式环境变量 > Colab 上的 Drive > 本机 code/。"""
    root = _env_path("WEEK7_OUT_ROOT", CODE_DIR)
    # 只有当用户没有显式设置、且确实在 Colab 时，才自动改道到 Drive
    if not os.environ.get("WEEK7_OUT_ROOT", "").strip() and _is_colab():
        drive = _drive_run_root()
        if drive is not None:
            return drive
    return root


def get_data_dir() -> Path:
    return _env_path("WEEK7_DATA_DIR", DEFAULT_DATA_DIR)


def get_model_root() -> Path:
    """底座缓存目录：显式环境变量 > 本机/Colab 都在 code 同级的 models/。"""
    if os.environ.get("WEEK7_MODEL_ROOT", "").strip():
        return _env_path("WEEK7_MODEL_ROOT", CODE_DIR.parent / "models")
    if _is_colab():
        return Path("/content/models")
    return CODE_DIR.parent / "models"


# ── 产物子目录（都从 OUT_ROOT 派生，避免每个脚本各自拼路径） ──
def log_dir() -> Path:
    d = get_out_root() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ckpt_dir() -> Path:
    d = get_out_root() / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def eval_dir() -> Path:
    d = get_out_root() / "outputs" / "eval"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pred_dir() -> Path:
    d = get_out_root() / "outputs" / "preds"
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_dir() -> Path:
    d = get_out_root() / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def describe() -> str:
    return (
        f"OUT_ROOT   = {get_out_root()}\n"
        f"DATA_DIR   = {get_data_dir()}\n"
        f"MODEL_ROOT = {get_model_root()}\n"
        f"Colab      = {_is_colab()}"
    )


if __name__ == "__main__":
    print(describe())
