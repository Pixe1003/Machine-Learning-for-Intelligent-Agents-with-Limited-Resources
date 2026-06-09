"""
Loss wrappers that call into `src.core.recursion`.

These are the functions the optimiser sees: monitor-object in, scalar out.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.core.recursion import compute_error

if TYPE_CHECKING:
    from src.monitors.base import MonitorBase


def majority_loss(monitor: "MonitorBase", n: int) -> Tensor:
    """
    f(n, k) for a monitor object.  The monitor must expose:
        - transitions(n) -> (H, T) of shape (k,k) or (n,k,k)
        - acceptance()   -> (k,) in [0, 1]
        - initial()      -> (k,) in Δ^k, optional (defaults to δ_0)
    """
    H, T = monitor.transitions(n)
    p = monitor.acceptance()
    pi0 = monitor.initial() if hasattr(monitor, "initial") else None
    return compute_error(H, T, p, n, pi0=pi0)


def evaluate(monitor: "MonitorBase", n: int) -> float:
    """Non-differentiable, detached evaluation (for logging / reporting)."""
    with torch.no_grad():
        return float(majority_loss(monitor, n).item())


def report_evaluate(monitor: "MonitorBase", n: int) -> float:
    """
    Evaluation path used for reported experiment results.

    M1-hard is trained with a soft acceptance relaxation but must be reported
    with a hard acceptance set. Other monitor classes keep their native
    acceptance.
    """
    report_monitor = copy.deepcopy(monitor)
    if hasattr(report_monitor, "hard_accept"):
        report_monitor.hard_accept = True
    return evaluate(report_monitor, n)


def evaluate_hard(monitor: "MonitorBase", n: int) -> float:
    """
    Evaluate the monitor under HARD (binary) acceptance regardless of class.

    For monitors that natively support a `hard_accept` flag (e.g. M1HardMonitor)
    we set the flag on a deep copy. For monitors without the flag (M1, M2, M3
    in the supervisor's renamed hierarchy), we monkey-patch their `acceptance`
    method so it thresholds the soft sigmoid output at 0.5. The original
    monitor object is left untouched.

    This is the canonical "hard eval" used to report f_M1-hard and to compute
    a hardened version of every other class for the dual-eval CSV columns.
    """
    report_monitor = copy.deepcopy(monitor)
    if hasattr(report_monitor, "hard_accept"):
        report_monitor.hard_accept = True
        return evaluate(report_monitor, n)

    original_acceptance = report_monitor.acceptance

    def _hard_acceptance() -> Tensor:
        p = original_acceptance()
        return (p >= 0.5).to(p.dtype)

    report_monitor.acceptance = _hard_acceptance  # type: ignore[method-assign]
    return evaluate(report_monitor, n)


def evaluate_soft(monitor: "MonitorBase", n: int) -> float:
    """
    Evaluate the monitor under SOFT (continuous sigmoid) acceptance.

    Inverse companion of `evaluate_hard`: for monitors carrying a
    `hard_accept` flag set to True, we temporarily disable it to recover the
    underlying sigmoid. For all other monitors this collapses to plain
    `evaluate(...)`.
    """
    if hasattr(monitor, "hard_accept") and monitor.hard_accept:
        report_monitor = copy.deepcopy(monitor)
        report_monitor.hard_accept = False
        return evaluate(report_monitor, n)
    return evaluate(monitor, n)
