"""
本地预训练模型定位（离线课堂必备）

为什么不能写死路径：
  同一个项目会存在于两套目录树里（母版 `项目材料准备/` 与交付副本 `八斗课程/weekN/`），
  脚本所在层级不同，`../pretrain_models` 在其中一套里必然失效。
  做法：从脚本所在目录**向上逐级查找** `pretrain_models/<模型名>`，
  并且要求目录里真的有权重文件——否则会拿到半下载的空目录（离线课堂最常见的坑）。

用法：
  from common.model_paths import qwen3_path, bert_path
"""

from pathlib import Path

QWEN_DIR = "Qwen3-0.6B"
BERT_DIR = "bert-base-chinese"
PRETRAIN = "pretrain_models"


def _has_weights(p: Path) -> bool:
    if not p.is_dir():
        return False
    if any(p.glob("*.safetensors")):
        return True
    return (p / "pytorch_model.bin").exists()


def resolve(dir_name: str, extra_roots=None) -> Path:
    """
    从当前文件向上逐级找 pretrain_models/<dir_name>。
    找不到就抛错并列出尝试过的路径——**不静默回退到联网下载**，保证断网也能用。
    """
    here = Path(__file__).resolve()
    candidates = []
    for root in (extra_roots or []):
        root = Path(root)
        candidates.append(root if root.name == dir_name else root / dir_name)
    for parent in here.parents:
        candidates.append(parent / PRETRAIN / dir_name)
    for c in candidates:
        if _has_weights(c):
            return c
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"未找到本地模型 {dir_name}，尝试过：\n  {tried}")


def qwen3_path() -> Path:
    return resolve(QWEN_DIR)


def bert_path() -> Path:
    return resolve(BERT_DIR)


if __name__ == "__main__":
    # 自检：课堂排查「模型找不到」时先跑这个
    print(f"{QWEN_DIR:<20} -> {qwen3_path()}")
    print(f"{BERT_DIR:<20} -> {bert_path()}")
