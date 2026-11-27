"""
作业准备：校验两个 backbone 权重在本地可用。

  baseline : badou/course/week4-大语言模型/bert-base-chinese     （复用 week4 本地权重）
  roberta  : badou/course/pretrain_models/chinese-roberta-wwm-ext（week6 已从 hf-mirror 下载）

不下载任何东西——两个目录应已存在；缺失时只报错提示，不自动联网。

用法：python homework/prepare_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from l2a_common import MODELS, has_weights


def main():
    ok = True
    for key, info in MODELS.items():
        p = Path(info["path"])
        good = has_weights(p)
        print(f"[{'ok' if good else 'MISSING'}] {key:<8} {info['name']} -> {p}")
        ok = ok and good
    if not ok:
        print("\n缺失权重。roberta 可运行 week6 的 prepare_models.py 下载；"
              "baseline 需确认 week4-大语言模型/bert-base-chinese 目录存在。")
        sys.exit(1)
    print("两个 backbone 均可用。")


if __name__ == "__main__":
    main()
