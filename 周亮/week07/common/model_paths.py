"""
底座模型路径解析与下载 —— RoBERTa（判别式）与 Qwen3-0.6B（生成式）。

设计（沿用 week6）：
  - 权重放 WEEK7_MODEL_ROOT（本机 code 同级 models/，Colab /content/models），不进 Drive；
  - 已存在就跳过，断线重跑只补缺的文件；HF 与 ModelScope 共用缓存；
  - 三个入口：
      resolve_roberta()  —— 返回本地路径，没下过就明确报错并提示下载命令
      resolve_qwen()     —— 同上
      ensure_model(name) —— 真正下载（colab/download_models.py 调用）

模型 id：
  roberta-wwm-ext : hfl/chinese-roberta-wwm-ext   （HF 主源，ModelScope 也有镜像）
  qwen3-0.6b      : Qwen/Qwen3-0.6B               （HF；MS 镜像 Qwen/Qwen3-0.6B）

【本机现状】只缓存了 bert-base-chinese，没有上面两个；本机跑判别式冒烟可用
  --tokenizer / --bert_path 传 bert-base-chinese 覆盖。正式实验在 Colab 下载后跑。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
from common.paths import LEGACY_MODEL_DIR, get_model_root

MODEL_IDS = {
    "roberta-wwm-ext": "hfl/chinese-roberta-wwm-ext",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
}
# 本地目录名（下载后落盘的名字）
LOCAL_DIRS = {
    "roberta-wwm-ext": "chinese-roberta-wwm-ext",
    "qwen3-0.6b": "Qwen3-0.6B",
}
# 判定「下载完整」的哨兵文件
SENTINELS = {
    "roberta-wwm-ext": ("config.json", "vocab.txt"),
    "qwen3-0.6b": ("config.json", "tokenizer_config.json"),
}


def _candidate_dirs(name: str) -> list[Path]:
    root = get_model_root()
    lname = LOCAL_DIRS[name]
    cands = [root / lname, root / MODEL_IDS[name].replace("/", "--"), LEGACY_MODEL_DIR / lname]
    # 允许环境变量直接指到某个具体目录（Colab 上复用 week6 已下的同名底座）
    env = os.environ.get(f"WEEK7_{name.upper().replace('-', '_')}_DIR", "").strip()
    if env:
        cands.insert(0, Path(env).expanduser())
    return cands


def _is_complete(d: Path, name: str) -> bool:
    return d.is_dir() and all((d / s).exists() for s in SENTINELS[name])


def resolve_model(name: str, required: bool = True) -> Path | None:
    """返回本地权重目录。优先本地缓存，其次让 transformers 从 hub 在线拉取。"""
    if name not in MODEL_IDS:
        raise KeyError(f"未知模型 {name}，可选：{list(MODEL_IDS)}")
    for d in _candidate_dirs(name):
        if _is_complete(d, name):
            return d
    # 本地没有：返回 hub id，交给 transformers 在线加载（Colab 直连 HF 通常可用）
    if not required:
        return None
    print(f"[model_paths] 本地未找到 {name} 的完整权重，将尝试从 HuggingFace "
          f"在线加载：{MODEL_IDS[name]}")
    print(f"              想预先下载到本地：python colab/download_models.py --model {name}")
    return Path(MODEL_IDS[name])   # 字符串路径，AutoTokenizer/AutoModel 可识别


def resolve_roberta() -> Path:
    return resolve_model("roberta-wwm-ext")


def resolve_qwen() -> Path:
    return resolve_model("qwen3-0.6b")


def ensure_model(name: str, source: str = "auto") -> Path:
    """
    真正下载到本地目录（幂等）。source: hf / ms / auto。
    auto：Colab 优先 HF，本机优先 ModelScope（本机直连 HF 常不通）。
    """
    from huggingface_hub import snapshot_download

    if name not in MODEL_IDS:
        raise KeyError(f"未知模型 {name}")
    model_id = MODEL_IDS[name]
    target = get_model_root() / LOCAL_DIRS[name]
    if _is_complete(target, name):
        print(f"[model_paths] 已存在，跳过下载：{target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)

    use_hf = source == "hf"
    if source == "auto":
        try:
            import google.colab  # noqa: F401
            use_hf = True
        except Exception:                                   # noqa: BLE001
            use_hf = False

    if use_hf:
        print(f"[model_paths] 从 HuggingFace 下载 {model_id} -> {target}")
        snapshot_download(repo_id=model_id, local_dir=str(target))
    else:
        print(f"[model_paths] 从 ModelScope 下载 {model_id} -> {target}")
        try:
            from modelscope import snapshot_download as ms_download
        except ImportError as exc:
            raise ImportError(
                "ModelScope 未安装：pip install modelscope；或改用 --source hf"
            ) from exc
        got = ms_download(model_id)
        # modelscope 下载到它自己的缓存目录，拷到统一位置
        got = Path(got)
        if got != target:
            import shutil
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(got, target)

    if not _is_complete(target, name):
        raise RuntimeError(f"下载后仍缺少哨兵文件 {SENTINELS[name]}，目录：{target}")
    print(f"[model_paths] ✓ 就绪：{target}")
    return target


def describe() -> str:
    lines = [f"MODEL_ROOT = {get_model_root()}"]
    for name in MODEL_IDS:
        local = next((str(d) for d in _candidate_dirs(name) if _is_complete(d, name)), "未下载")
        lines.append(f"  {name:<16} {MODEL_IDS[name]:<32} -> {local}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
