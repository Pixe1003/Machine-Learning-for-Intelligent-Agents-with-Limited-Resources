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

    M1 is trained with a soft acceptance relaxation but must be reported with a
    hard acceptance set.  Other monitor classes keep their native acceptance.
    """
    report_monitor = copy.deepcopy(monitor)
    if hasattr(report_monitor, "hard_accept"):
        report_monitor.hard_accept = True
    return evaluate(report_monitor, n)
