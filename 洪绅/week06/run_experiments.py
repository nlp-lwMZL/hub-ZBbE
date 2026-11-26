"""依次运行各方法，把效果写入 results/对比表.md。"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import gc
import importlib
import os
from pathlib import Path
import random
import sys
import time
import traceback
import common

ROOT = Path(__file__).resolve().parent
EPOCHS = 3  # 轮数上限；同时受各算法文件中的步数和训练时间上限约束
SEED = 42

# 方法名对应模块和需要覆盖的配置。
METHODS = {
    "rules": ("l0_rules", {}),
    "tfidf_lr": ("l1_tfidf", {"MODEL": "lr"}),
    "tfidf_svm": ("l1_tfidf", {"MODEL": "svm"}),
    "bert": ("l2_bert", {}),
    "weighted": ("l2_bert_tricks", {"TRICK": "weighted", "USE_FGM": False}),
    "focal": ("l2_bert_tricks", {"TRICK": "focal", "USE_FGM": False}),
    "smooth": ("l2_bert_tricks", {"TRICK": "smooth", "USE_FGM": False}),
    "fgm": ("l2_bert_tricks", {"TRICK": "fgm", "USE_FGM": True}),
    "zero_shot": ("l3_llm_prompt", {"N_SHOT": 0}),
    "few_shot": ("l3_llm_prompt", {"N_SHOT": 15}),
    "lora": ("l4_sft", {"USE_LORA": True, "LR": 2e-4}),
    "full_sft": ("l4_sft", {"USE_LORA": False, "LR": 1e-5}),
}


def run_method(name):
    module_name, settings = METHODS[name]
    # 各方法使用同一种子，减少随机初始化带来的差异。
    random.seed(SEED)
    if module_name != "l0_rules":
        import numpy as np
        np.random.seed(SEED)
    if module_name.startswith(("l2", "l3", "l4")):
        import torch
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
    module = importlib.import_module(module_name)
    module.DATA_DIR = common.DATA_DIR  # 确保各方法读取同一份练习子集
    for key, value in settings.items():
        setattr(module, key, value)
    if module_name.startswith(("l2", "l4")):
        module.EPOCHS = EPOCHS
    return module.main()


def write_table(rows):
    lines = ["# 方法效果对比", "", f"共用 tnews_quick 测试子集；最多 {EPOCHS} 轮，随机种子 {SEED}。",
             "短训练习分数，不代表收敛效果；实际步数、逐类召回和错误详情见 train.log。", "",
             "| 方法 | accuracy | macro-F1 | 更新步数 | 总耗时(秒) |", "|---|---:|---:|---:|---:|"]
    for row in sorted(rows, key=lambda r: r.get("macro_f1", -1), reverse=True):
        accuracy = f'{row["accuracy"]:.4f}' if "accuracy" in row else "失败"
        f1 = f'{row["macro_f1"]:.4f}' if "macro_f1" in row else "—"
        lines.append(f'| {row["method"]} | {accuracy} | {f1} | {row.get("train_steps", "—")} | {row["seconds"]:.1f} |')
    path = ROOT / "results/对比表.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS), help="只跑指定方法，默认全部")
    parser.add_argument("--dry-run", action="store_true", help="只列出方法，不加载模型、不训练")
    args = parser.parse_args(argv)
    methods = list(dict.fromkeys(args.methods))
    if args.dry_run:
        print(" → ".join(methods))
        print("数据：训练 3000 / 验证 150 / 测试 150；目标约 30 分钟（不含首次下载）")
        return 0

    common.prepare_quick_data()
    previous = Path.cwd()
    os.chdir(ROOT)
    rows = []
    (ROOT / "results").mkdir(exist_ok=True)
    # 整轮共用一个日志句柄，避免外部库的日志处理器引用已关闭的文件。
    log = (ROOT / "results/train.log").open("w", encoding="utf-8")
    try:
        for name in methods:
            print(f"运行 {name}，日志：results/train.log", flush=True)
            started = time.monotonic()
            row = {"method": name}
            with redirect_stdout(log), redirect_stderr(log):
                print(f"\n===== {name} =====", flush=True)
                try:
                    row.update(run_method(name))
                except Exception:
                    # 保留报错信息，继续运行其余方法。
                    traceback.print_exc()
            row["seconds"] = time.monotonic() - started
            rows.append(row)
            write_table(rows)
            # 清理上一轮模型的内存和设备缓存。
            gc.collect()
            if "torch" in sys.modules:
                torch = sys.modules["torch"]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        print(f"对比结果：{ROOT / 'results/对比表.md'}")
    finally:
        log.close()
        os.chdir(previous)
    return int(any("macro_f1" not in row for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
