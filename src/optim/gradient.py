"""
Gradient-based optimisation with multiple restarts and warm-start support.

Returns the best monitor (deep-copied) plus per-restart statistics so that
restart-spread confidence intervals can be reported.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Literal

import torch

from src.core.loss import evaluate, majority_loss
from src.monitors.base import MonitorBase


@dataclass
class RestartResult:
    seed: int
    final_loss: float
    report_loss: float | None = None
    best_loss_curve: list[float] = field(default_factory=list)
    wall_clock_s: float = 0.0
    converged: bool = False


@dataclass
class TrainResult:
    best_monitor: MonitorBase
    best_loss: float
    best_train_loss: float
    restarts: List[RestartResult]

    @property
    def mean_loss(self) -> float:
        return sum(self.report_losses) / len(self.restarts)

    @property
    def std_loss(self) -> float:
        m = self.mean_loss
        var = sum((loss - m) ** 2 for loss in self.report_losses) / max(1, len(self.restarts) - 1)
        return var ** 0.5

    @property
    def report_losses(self) -> list[float]:
        return [r.report_loss if r.report_loss is not None else r.final_loss for r in self.restarts]

    @property
    def mean_train_loss(self) -> float:
        return sum(r.final_loss for r in self.restarts) / len(self.restarts)

    @property
    def std_train_loss(self) -> float:
        m = self.mean_train_loss
        var = sum((r.final_loss - m) ** 2 for r in self.restarts) / max(1, len(self.restarts) - 1)
        return var ** 0.5

    @property
    def ci95(self) -> tuple[float, float]:
        if len(self.restarts) <= 1:
            return self.mean_loss, self.mean_loss
        half_width = 1.96 * self.std_loss / math.sqrt(len(self.restarts))
        return self.mean_loss - half_width, self.mean_loss + half_width


def train_one(
    monitor: MonitorBase,
    n: int,
    lr: float = 1e-2,
    steps: int = 3000,
    patience: int = 500,
    log_every: int = 0,
) -> RestartResult:
    """Train a single monitor instance with Adam and early stopping."""
    opt = torch.optim.Adam(monitor.parameters(), lr=lr)
    best = float("inf")
    curve: list[float] = []
    stale = 0
    t0 = time.perf_counter()

    for step in range(steps):
        opt.zero_grad()
        loss = majority_loss(monitor, n)
        loss.backward()
        opt.step()

        lv = float(loss.item())
        curve.append(lv)
        if lv < best - 1e-10:
            best = lv
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
        if log_every and step % log_every == 0:
            print(f"[step {step:>5d}]  loss = {lv:.6f}  (best = {best:.6f})")

    return RestartResult(
        seed=-1,
        final_loss=best,
        best_loss_curve=curve,
        wall_clock_s=time.perf_counter() - t0,
        converged=stale >= patience,
    )


def train_with_restarts(
    monitor_factory: Callable[[int], MonitorBase],
    n: int,
    restarts: int = 20,
    lr: float = 1e-2,
    steps: int = 3000,
    warm_start_fn: Optional[Callable[[MonitorBase], None]] = None,
    report_eval_fn: Optional[Callable[[MonitorBase, int], float]] = None,
    select_by: Literal["report", "train"] = "report",
    verbose: bool = True,
) -> TrainResult:
    """
    Run `restarts` independent optimisations and return the best monitor.

    Parameters
    ----------
    monitor_factory : callable (seed: int) -> MonitorBase.  Must build a FRESH
                      monitor for each seed so gradients don't bleed across runs.
    warm_start_fn   : optional function applied to each fresh monitor before
                      training — e.g. from a smaller (n, k) solution or a
                      counting-restricted class.
    """
    best_metric = float("inf")
    best_report_loss = float("inf")
    best_train_loss = float("inf")
    best_monitor: Optional[MonitorBase] = None
    records: list[RestartResult] = []
    report_eval = report_eval_fn or evaluate

    for r in range(restarts):
        seed = 12345 + r
        m = monitor_factory(seed)
        if warm_start_fn is not None:
            warm_start_fn(m)
        res = train_one(m, n, lr=lr, steps=steps)
        res.seed = seed
        res.report_loss = report_eval(m, n)
        records.append(res)
        if verbose:
            print(
                f"  restart {r + 1:>2d}/{restarts}: "
                f"train={res.final_loss:.6f} report={res.report_loss:.6f} "
                f"({res.wall_clock_s:.1f}s)"
            )
        metric = res.report_loss if select_by == "report" else res.final_loss
        if metric < best_metric:
            best_metric = metric
            best_report_loss = res.report_loss
            best_train_loss = res.final_loss
            best_monitor = copy.deepcopy(m)

    assert best_monitor is not None
    return TrainResult(
        best_monitor=best_monitor,
        best_loss=best_report_loss,
        best_train_loss=best_train_loss,
        restarts=records,
    )
