"""模型无关的训练 / 评估循环。

统一模型接口（BiLSTM+CRF 与后续 BERT 系列都遵循）
--------------------------------------------------
    loss, pred_ids = model(input_ids, attention_mask, labels=labels)
        input_ids      : LongTensor (B, T)
        attention_mask : BoolTensor (B, T)，True 表示有效字符
        labels         : LongTensor (B, T)，padding 为 -100（可为 None）
        loss           : 标量 Tensor（labels 为 None 时可返回 None）
        pred_ids       : list[list[int]]，已解码的标签 id（不含 padding）

这样 trainer 不关心内部是否为 CRF / 是否带约束，便于两种模型复用。
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from .label import PAD_LABEL_ID, TRIGGER_LABEL
from .metrics import (
    MetricReport,
    compute_event_exact_match,
    compute_metrics,
    compute_type_metrics,
    spans_from_bio,
)
from .schema import infer_event_type


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        "input_ids": batch["input_ids"].to(device),
        "labels": batch["labels"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "meta": batch["meta"],
    }


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 5.0,
    log_every: int = 50,
    logger: logging.Logger | None = None,
    warmup_steps: int = 0,
    base_lr: float | None = None,
    start_step: int = 0,
) -> tuple[float, int]:
    """跑一个 epoch，返回 (平均 loss, 累计 step 数)。

    warmup_steps > 0 时，前 warmup_steps 个全局 step 把学习率从约 0 线性升到
    base_lr（step 级线性 warmup），用于避免训练初期发散。
    """
    model.train()
    total_loss = 0.0
    total_steps = 0
    global_step = start_step
    t0 = time.time()

    for step, batch in enumerate(loader, start=1):
        if warmup_steps > 0 and global_step < warmup_steps and base_lr:
            lr = base_lr * (global_step + 1) / warmup_steps
            for group in optimizer.param_groups:
                group["lr"] = lr
        global_step += 1

        batch = move_batch_to_device(batch, device)
        loss, _ = model(
            batch["input_ids"],
            batch["attention_mask"],
            labels=batch["labels"],
        )
        if loss is None:
            continue

        optimizer.zero_grad()
        loss.backward()
        if gradient_clip and gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += float(loss.item())
        total_steps += 1
        if logger and log_every and step % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "    step %d/%d | batch_loss=%.4f | lr=%.2e | %.1fs",
                step, len(loader), float(loss.item()), lr, time.time() - t0,
            )

    return total_loss / max(total_steps, 1), global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[float, MetricReport]:
    """验证集：返回平均 loss 与 span 级指标。

    长文本被切成多个重叠 chunk，这里把每个 chunk 的 span 转回**原文绝对 offset**，
    按 source 聚合去重后再计算指标，避免重叠窗口导致同一实体重复计数。
    """
    model.eval()
    total_loss = 0.0
    total_steps = 0
    gold_by_doc: dict[str, set[tuple[int, int, str]]] = defaultdict(set)
    pred_by_doc: dict[str, set[tuple[int, int, str]]] = defaultdict(set)
    gold_type_by_doc: dict[str, str] = {}

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        loss, pred_ids = model(
            batch["input_ids"],
            batch["attention_mask"],
            labels=batch["labels"],
        )
        if loss is not None:
            total_loss += float(loss.item())
            total_steps += 1

        labels = batch["labels"].tolist()
        for b, meta in enumerate(batch["meta"]):
            src = meta["source"]
            off = int(meta["offset"])
            gold_type_by_doc.setdefault(src, meta.get("event_type", ""))
            gold_ids = [x for x in labels[b] if x != PAD_LABEL_ID]
            for s, e, base in spans_from_bio(gold_ids):
                gold_by_doc[src].add((s + off, e + off, base))
            for s, e, base in spans_from_bio(pred_ids[b]):
                pred_by_doc[src].add((s + off, e + off, base))

    docs = sorted(set(gold_by_doc) | set(pred_by_doc))
    gold_spans_list = [sorted(gold_by_doc[d]) for d in docs]
    pred_spans_list = [sorted(pred_by_doc[d]) for d in docs]
    report = compute_metrics(gold_spans_list, pred_spans_list)

    # 事件类型推断指标：由预测槽位集合推断类型，与金标类型比较
    gold_types = [gold_type_by_doc.get(d, "") for d in docs]
    pred_types = [
        infer_event_type({base for _, _, base in pred_by_doc[d]}) for d in docs
    ]
    report.event_type = compute_type_metrics(gold_types, pred_types)

    # 事件级完全匹配：type + trigger + 全部槽位 span 完全一致
    def _to_event(spans: set[tuple[int, int, str]], ev_type: str | None) -> dict:
        trigger = None
        args: set[tuple[int, int, str]] = set()
        for s, e, base in sorted(spans):
            if base == TRIGGER_LABEL:
                if trigger is None:
                    trigger = (s, e, base)
            else:
                args.add((s, e, base))
        return {"type": ev_type, "trigger": trigger, "args": frozenset(args)}

    gold_events = [_to_event(gold_by_doc[d], gold_type_by_doc.get(d)) for d in docs]
    pred_events = [_to_event(pred_by_doc[d], pred_types[i]) for i, d in enumerate(docs)]
    report.event_match = compute_event_exact_match(gold_events, pred_events)

    return total_loss / max(total_steps, 1), report


@dataclass
class TrainerConfig:
    epochs: int = 20
    patience: int = 5
    ckpt_dir: str = "checkpoints"
    ckpt_prefix: str = "model"
    monitor: str = "f1"  # 选优/早停依据：f1 | event_exact | type_macro_f1


class Trainer:
    """封装 epoch 循环、最优模型保存、early stopping 与日志输出。"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: TrainerConfig,
        logger: logging.Logger | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        warmup_steps: int = 0,
        log_every: int = 50,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.logger = logger
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.log_every = log_every
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.global_step = 0
        self.best_score = -1.0
        self.best_f1 = -1.0
        self.best_epoch = -1
        self.history: list[dict] = []

    def _score(self, report) -> float:
        """按 monitor 取本轮选优分数。"""
        monitor = self.config.monitor
        if monitor == "f1":
            return report.f1
        d = report.as_dict()
        if monitor == "event_exact":
            return (d.get("event_match") or {}).get("accuracy", 0.0)
        if monitor == "type_macro_f1":
            return (d.get("event_type") or {}).get("macro_f1", 0.0)
        raise ValueError(f"未知 monitor: {monitor}")

    def fit(self, train_loader, val_loader) -> dict:
        ckpt_dir = Path(self.config.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = ckpt_dir / f"{self.config.ckpt_prefix}_best.pt"

        epochs_without_improve = 0
        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss, self.global_step = train_one_epoch(
                self.model, train_loader, self.optimizer, self.device,
                log_every=self.log_every, logger=self.logger,
                warmup_steps=self.warmup_steps, base_lr=self.base_lr,
                start_step=self.global_step,
            )
            val_loss, report = evaluate(self.model, val_loader, self.device)
            score = self._score(report)
            # warmup 结束前不让调度器提前消耗学习率
            warmup_done = self.warmup_steps == 0 or self.global_step >= self.warmup_steps
            if self.scheduler is not None and warmup_done:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(score)
                else:
                    self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_f1": report.f1,
                    "monitor": self.config.monitor,
                    "monitor_score": score,
                    "report": report,
                }
            )

            if self.logger:
                self.logger.info(
                    "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | lr=%.2e | %.1fs | %s",
                    epoch, self.config.epochs, train_loss, val_loss, current_lr,
                    elapsed, report.summary(),
                )
                # 每轮输出完整指标明细：各槽位 F1 + 事件类型各分类 F1
                d = report.as_dict()
                slot_f1 = {k: v for k, v in d["per_type_f1"].items() if k != "TRIGGER"}
                self.logger.info("    槽位 F1: %s", json.dumps(slot_f1, ensure_ascii=False))
                et = d.get("event_type")
                if et:
                    self.logger.info(
                        "    事件类型: acc=%.4f macroF1=%.4f microF1=%.4f | 各类型 F1: %s",
                        et["accuracy"], et["macro_f1"], et["micro_f1"],
                        json.dumps(et["per_class_f1"], ensure_ascii=False),
                    )

            if score > self.best_score:
                self.best_score = score
                self.best_f1 = report.f1
                self.best_epoch = epoch
                epochs_without_improve = 0
                self._save(best_path)
                if self.logger:
                    self.logger.info(
                        "    ★ 新的最优模型 epoch=%d, %s=%.4f (val_F1=%.4f) -> %s",
                        epoch, self.config.monitor, score, report.f1, best_path,
                    )
            else:
                epochs_without_improve += 1
                if self.logger:
                    self.logger.info(
                        "    %s 未提升（%d/%d 次），当前最优 epoch=%d %s=%.4f",
                        self.config.monitor, epochs_without_improve, self.config.patience,
                        self.best_epoch, self.config.monitor, self.best_score,
                    )
                if epochs_without_improve >= self.config.patience:
                    if self.logger:
                        self.logger.info("    触发 early stopping，停止训练。")
                    break

        return {
            "best_epoch": self.best_epoch,
            "best_f1": self.best_f1,
            "best_score": self.best_score,
            "monitor": self.config.monitor,
            "best_ckpt": str(best_path),
            "history": self.history,
        }

    def _save(self, path: Path) -> None:
        """保存权重 + 模型结构超参 + 训练配置，供推理时无损重建模型。"""
        model_config = (
            self.model.get_config() if hasattr(self.model, "get_config") else {}
        )
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "model_config": model_config,
                "trainer_config": self.config.__dict__,
                "best_f1": self.best_f1,
                "best_score": self.best_score,
                "monitor": self.config.monitor,
                "best_epoch": self.best_epoch,
            },
            path,
        )


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> dict:
    """加载权重到已构建的模型，返回完整 checkpoint（含 model_config）。"""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return ckpt
