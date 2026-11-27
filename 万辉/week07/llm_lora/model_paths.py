# -*- coding: utf-8 -*-
"""
预训练模型路径解析（同时兼容「母版」与「交付副本」两套目录树）

教学重点 / 为什么需要这个模块：
  同一个项目会存在于两套目录树里，而且层级深度不同：
    母版：    项目材料准备/event_extraction/llm_zero_shot/xxx.py
    交付副本：八斗课程/week7 信息抽取/event_extraction/llm_zero_shot/xxx.py
  从脚本所在目录向上到 pretrain_models/ 的层数不一样（一个差 2 级、一个差 3 级），
  所以任何写死的 "../../pretrain_models" 都必然在其中一套里失效。
  正确做法：从当前文件开始**向上逐级查找** pretrain_models/<模型名>，
  找到「真有权重文件」的那个目录才算命中。

  顺带解决另一个课堂高频事故：只下载了一半的模型目录（有 config.json、
  没有 model.safetensors）会让 from_pretrained 报一堆看不懂的错。
  这里用「必须有 *.safetensors 或 pytorch_model.bin」来判定，提前给出人话报错。

使用方式：
  from model_paths import resolve_model_path, QWEN_MODEL_DIR
  MODEL_PATH = resolve_model_path(QWEN_MODEL_DIR)   # → .../pretrain_models/Qwen3-0.6B
"""

from pathlib import Path

# ── 模型目录名（与 pretrain_models/ 下的实际目录一致）──────────────────────────
QWEN_MODEL_DIR      = "Qwen3-0.6B"         # llm_zero_shot/ 全部脚本使用的生成式基座
QWEN_DISPLAY_NAME   = "Qwen3-0.6B"         # 打印 / 文档里展示用
BERT_MODEL_DIR      = "bert-base-chinese"  # bert/ 等判别式脚本使用（本目录不用）

PRETRAIN_DIR_NAME = "pretrain_models"


def _is_model_dir(path: Path) -> bool:
    """目录里必须有真实权重文件才算模型（避免拿到半下载的空目录）"""
    if not path.is_dir():
        return False
    if any(path.glob("*.safetensors")):
        return True
    return (path / "pytorch_model.bin").exists()


def candidate_paths(dir_name: str, extra_roots: list[Path] | None = None) -> list[Path]:
    """列出所有候选路径（供 resolve_model_path 使用，也便于报错时展示）"""
    here = Path(__file__).resolve()
    candidates: list[Path] = []

    # 1. 调用方显式指定的根目录（可给模型目录本身，也可给它的父目录）
    for root in (extra_roots or []):
        root = Path(root)
        candidates.append(root if root.name == dir_name else root / dir_name)

    # 2. 从本文件所在目录向上逐级查找 pretrain_models/<dir_name>
    #    母版：命中 项目材料准备/pretrain_models/
    #    副本：命中 八斗课程/pretrain_models/（若副本目录里也放了权重）
    for parent in here.parents:
        candidates.append(parent / PRETRAIN_DIR_NAME / dir_name)

    return candidates


def resolve_model_path(dir_name: str = QWEN_MODEL_DIR,
                       extra_roots: list[Path] | None = None) -> Path:
    """
    定位本地预训练模型目录，返回绝对路径。

    优先级：
      1. extra_roots 里显式给出的目录
      2. 逐级向上找到的第一个含真实权重的 pretrain_models/<dir_name>

    找不到时抛 FileNotFoundError 并打印全部已尝试路径 —— 故意**不**静默回退到联网下载，
    保证课堂离线环境行为可预期（本课程约定：复用已有权重，不重新下载）。
    """
    candidates = candidate_paths(dir_name, extra_roots)
    for path in candidates:
        if _is_model_dir(path):
            return path

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"未找到本地模型目录：{dir_name}\n"
        f"已尝试的路径：\n  {tried}\n\n"
        f"请确认 pretrain_models/{dir_name}/ 下存在 model.safetensors。"
    )


if __name__ == "__main__":
    # 自检：直接打印解析结果，便于课堂排查「模型找不到」问题
    print(f"{QWEN_MODEL_DIR:<18} -> {resolve_model_path(QWEN_MODEL_DIR)}")
