# -*- coding: utf-8 -*-
"""
统一训练日志（TrainLogger / EvalLogger）—— 让「训练过程」可被学生事后复看

为什么需要这个模块（老师的核心诉求）：
  课上没时间现场演示训练，学生想自己看「loss 怎么降、每个 epoch 多久、验证集什么时候最好」，
  所以每个方法跑完都要留下**完整可追溯**的训练痕迹，而不是只有一行最终 F1。
  更实际的理由：出了问题（比如 loss 不降、NaN、显存爆了）第一件事就是翻日志，
  没有结构化日志就只能猜。

产出三类文件（都在 `outputs/logs/` 下，按方法名区分，不会互相覆盖）：
  <method>_train.json     结构化总结：args、环境、每个 epoch 的 loss/验证/耗时、总耗时、参数量、峰值显存
  <method>_steps.jsonl    **每 N 步一行**（默认 50）：epoch/step/loss/lr/elapsed —— 学生画 loss 曲线就读这个
  <method>_console.log    该次运行的全部 stdout（同时仍然打印到屏幕，不影响课堂演示）

纯评测方法（规则基线 / zero-shot / evaluate_*.py）不训练，用 `EvalLogger` 写：
  <method>_eval.json + <method>_console.log

设计上的两个小决定：
  1. `_train.json` 里一定带 `argv` 与 `env`。换个超参重跑时，"这份结果到底是用哪组参数跑出来的"
     是第一个会搞混的问题；把 argv 原样存下来比让后人猜要可靠得多。
  2. steps.jsonl 用**追加写 + 立即 flush**：训练崩了也保留崩之前的曲线，这正是最需要看的时刻。

使用方式：
  from log_utils import TrainLogger
  logger = TrainLogger(method="bert_crf_256", args=args, root=ROOT, params=n_params)
  for epoch in ...:
      for step, batch in enumerate(loader):
          ...
          logger.log_step(epoch=epoch, step=step, loss=loss.item(), lr=lr)
      logger.log_epoch(epoch=epoch, train_loss=..., val_metrics={...}, elapsed=...)
  logger.finalize(total_s=..., best_epoch=..., best_val=..., peak_gpu_mb=...)
"""

import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _env_info() -> dict:
    """环境快照：复现实验时"我这台机器是不是同一个环境"必须先能回答。"""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    return info


def _jsonable(obj):
    """把 argparse.Namespace / Path / 其它奇怪类型转成能 json.dump 的东西。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class _ConsoleTee:
    """
    把 stdout 同时写到屏幕和文件。

    为什么要 tee 而不是把 print 全改成 logger.info：
      本项目所有脚本都用 print 汇报进度，改 print 会动到大量既有代码；
      包一层 sys.stdout 是最小侵入，且学生看到的行为完全不变。
    """

    def __init__(self, stream, file_path: Path):
        self.stream = stream
        self.path = file_path
        self.fh = open(file_path, "a", encoding="utf-8", buffering=1)

    def write(self, data):
        self.stream.write(data)
        try:
            self.fh.write(data)
        except Exception:
            pass
        return len(data)

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        try:
            self.fh.flush()
        except Exception:
            pass

    def isatty(self):
        return False

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


class TrainLogger:
    """训练日志器：steps.jsonl（每步）+ train.json（每 epoch/总结）+ console.log（stdout）。"""

    def __init__(self, method: str, args=None, root: Path | str | None = None,
                 params: int | None = None, log_every: int | None = None,
                 notes: str = ""):
        self.method = method
        self.root = Path(root) if root else Path(__file__).resolve().parents[1]
        self.log_dir = self.root / "outputs" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.train_json = self.log_dir / f"{method}_train.json"
        self.steps_jsonl = self.log_dir / f"{method}_steps.jsonl"
        self.console_log = self.log_dir / f"{method}_console.log"

        # log_every 优先取命令行 --log_every，其次构造参数，最后默认 50
        if log_every is None:
            log_every = getattr(args, "log_every", 50) if args is not None else 50
        self.log_every = max(int(log_every), 1)

        self.args = _jsonable(vars(args)) if hasattr(args, "args") is False and hasattr(args, "__dict__") else _jsonable(args)
        self.params = params
        self.notes = notes
        self.epochs: list[dict] = []
        self.t_start = time.perf_counter()
        self._step_t0 = self.t_start
        self._tee = None
        self._summary: dict = {}

        # 每次正式训练都重写这两个文件，避免上一次的曲线混进来
        self.steps_jsonl.write_text("", encoding="utf-8")
        self.console_log.write_text("", encoding="utf-8")
        self._install_tee()

    # ── stdout 同步 ──────────────────────────────────────────────
    def _install_tee(self):
        self._tee = _ConsoleTee(sys.stdout, self.console_log)
        sys.stdout = self._tee

    def _remove_tee(self):
        if self._tee is not None and sys.stdout is self._tee:
            sys.stdout = self._tee.stream
            self._tee.close()
            self._tee = None

    # ── 每步 ────────────────────────────────────────────────────
    def log_step(self, epoch: int, step: int, loss: float, lr: float | None = None,
                 extra: dict | None = None) -> None:
        """
        每 `log_every` 步写一行到 steps.jsonl。

        刻意**不每步都写**：一次训练几万步的话，每步一行会让 JSONL 变成几十 MB，
        学生反而打不开。默认每 50 步足够画出光滑曲线。
        """
        now = time.perf_counter()
        if step % self.log_every != 0 and step != 0:
            return
        rec = {
            "epoch": int(epoch),
            "step": int(step),
            "loss": round(float(loss), 6),
            "lr": (round(float(lr), 8) if lr is not None else None),
            "elapsed_s": round(now - self.t_start, 2),
            "step_elapsed_s": round(now - self._step_t0, 3),
        }
        if extra:
            rec.update(_jsonable(extra))
        with open(self.steps_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")   # 立即落盘：崩了也留曲线
        self._step_t0 = now

    # ── 每 epoch ────────────────────────────────────────────────
    def log_epoch(self, epoch: int, train_loss: float, elapsed: float,
                  val_metrics: dict | None = None, extra: dict | None = None) -> dict:
        rec = {
            "epoch": int(epoch),
            "train_loss": round(float(train_loss), 6),
            "elapsed_s": round(float(elapsed), 1),
        }
        if val_metrics:
            rec["val"] = _jsonable(val_metrics)
        if extra:
            rec.update(_jsonable(extra))
        self.epochs.append(rec)
        self._dump_train_json()      # 每个 epoch 都刷一次：中途崩了也能看到已完成的部分
        return rec

    # ── 收尾 ────────────────────────────────────────────────────
    def finalize(self, total_s: float | None = None, **summary) -> None:
        """
        写最终总结。`summary` 里建议放：best_epoch / best_val / test_metrics / peak_gpu_mb 等。

        峰值显存用 torch.cuda.max_memory_allocated()，调用方负责在该调用前不要 reset。
        """
        if total_s is None:
            total_s = time.perf_counter() - self.t_start
        self._summary = _jsonable(summary)
        self._dump_train_json(total_s=total_s, final=True)
        self._remove_tee()
        print(f"[log] 训练日志已写入 → {self.train_json.relative_to(self.root)}")
        print(f"[log] 逐步曲线 → {self.steps_jsonl.relative_to(self.root)}")
        print(f"[log] 控制台日志 → {self.console_log.relative_to(self.root)}")

    def _dump_train_json(self, total_s: float | None = None, final: bool = False) -> None:
        doc = {
            "method": self.method,
            "argv": sys.argv,
            "args": self.args,
            "env": _env_info(),
            "params": self.params,
            "log_every": self.log_every,
            "epochs": self.epochs,
            "total_elapsed_s": (round(total_s, 1) if total_s is not None else
                                round(time.perf_counter() - self.t_start, 1)),
            "finished": bool(final),
        }
        if self._summary:
            doc.update(self._summary)
        if self.notes:
            doc["notes"] = self.notes
        self.train_json.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is not None:      # 异常退出也要把已有内容落盘 + 恢复 stdout
            self._remove_tee()
        return False


class EvalLogger:
    """
    纯评测方法（规则 / zero-shot / few-shot / evaluate_*.py）用的日志器：不训练，只记录结果。

    产出 `<method>_eval.json` + `<method>_console.log`，字段与 TrainLogger 尽量对齐
    （method/argv/args/env/metrics），这样 `eval/compare_methods.py` 能统一读取。
    """

    def __init__(self, method: str, args=None, root: Path | str | None = None,
                 notes: str = ""):
        self.method = method
        self.root = Path(root) if root else Path(__file__).resolve().parents[1]
        self.log_dir = self.root / "outputs" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.eval_json = self.log_dir / f"{method}_eval.json"
        self.console_log = self.log_dir / f"{method}_console.log"
        self.args = _jsonable(args)
        self.notes = notes
        self.console_log.write_text("", encoding="utf-8")
        self._tee = _ConsoleTee(sys.stdout, self.console_log)
        sys.stdout = self._tee
        self.t0 = time.perf_counter()

    def finalize(self, metrics: dict | None = None, runtime_s: float | None = None,
                 **extra) -> None:
        doc = {
            "method": self.method,
            "argv": sys.argv,
            "args": self.args,
            "env": _env_info(),
            "runtime_s": round(runtime_s if runtime_s is not None
                               else time.perf_counter() - self.t0, 2),
            "metrics": _jsonable(metrics or {}),
        }
        doc.update(_jsonable(extra))
        if self.notes:
            doc["notes"] = self.notes
        if sys.stdout is self._tee:
            sys.stdout = self._tee.stream
        self._tee.close()
        self.eval_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"[log] 评测日志已写入 → {self.eval_json.relative_to(self.root)}")
        print(f"[log] 控制台日志 → {self.console_log.relative_to(self.root)}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is not None and sys.stdout is self._tee:
            sys.stdout = self._tee.stream
            self._tee.close()
        return False


def peak_gpu_mb() -> float:
    """当前进程的峰值显存（MiB）。没 GPU 或没装 torch 时返回 0。"""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
    except Exception:
        pass
    return 0.0


if __name__ == "__main__":
    # 自检：在临时目录里跑一遍，确认三类文件都能生成、且能读回来
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    class _A:
        log_every = 2
        lr = 1e-5
        epochs = 2
    lg = TrainLogger("selftest", args=_A(), root=tmp, params=123)
    print("训练中：这是一行会被同时写进 console.log 的输出")
    for ep in range(1, 3):
        lg.log_step(ep, 0, loss=1.0 / ep, lr=1e-5)
        lg.log_step(ep, 1, loss=0.9 / ep, lr=1e-5)
        lg.log_epoch(ep, train_loss=1.0 / ep, elapsed=1.5,
                     val_metrics={"span_f1": 0.3 * ep})
    lg.finalize(best_epoch=2, peak_gpu_mb=peak_gpu_mb())
    steps = (tmp / "outputs/logs/selftest_steps.jsonl").read_text(encoding="utf-8").strip().splitlines()
    doc = json.loads((tmp / "outputs/logs/selftest_train.json").read_text(encoding="utf-8"))
    print(f"steps 行数={len(steps)}  epochs={len(doc['epochs'])}  finished={doc['finished']}")
    print("console 非空:", len((tmp / "outputs/logs/selftest_console.log").read_text(encoding="utf-8")) > 0)
    print("自检通过")
