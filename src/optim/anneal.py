"""
Softmax-temperature annealing via Concrete / Gumbel-Softmax relaxation.

Used to probe Hypothesis H2: if the optimal monitor is really randomised,
annealing τ → 0 should drive f(n, k) strictly upward.  If it is deterministic,
f is stable under annealing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

from src.core.loss import majority_loss
from src.monitors.base import MonitorBase


def annealed_softmax(logits: Tensor, tau: float, hard: bool = False) -> Tensor:
    """
    Concrete / Gumbel-Softmax relaxation.

    Called in place of torch.softmax(·, dim=-1) inside a monitor's transitions().
    At tau=1 this matches standard softmax; at tau→0 it approaches a hard argmax.
    """
    if hard:
        g = -torch.empty_like(logits).exponential_().log()
        y = torch.softmax((logits + g) / max(tau, 1e-8), dim=-1)
        y_hard = torch.zeros_like(y).scatter_(-1, y.argmax(-1, keepdim=True), 1.0)
        return (y_hard - y).detach() + y
    return torch.softmax(logits / max(tau, 1e-8), dim=-1)


@dataclass
class AnnealingTrace:
    taus: List[float]
    losses: List[float]


@torch.no_grad()
def anneal_trace(monitor: MonitorBase, n: int, taus=(1.0, 0.5, 0.1, 0.01, 0.001)) -> AnnealingTrace:
    """
    Evaluate f(n, k) at a schedule of softmax temperatures.

    The monitor's parametrisation is NOT re-optimised; we merely rescale
    the logits of any `W_*` parameters by 1/τ and compute the resulting loss.
    This is the empirical test of Hypothesis H2.
    """
    trace = AnnealingTrace(taus=list(taus), losses=[])

    # Save-and-restore any softmax weights by scaling them.
    scaled_params = {n: p.data.clone() for n, p in monitor.named_parameters() if n.startswith("W_")}
    try:
        for tau in taus:
            for pname, original in scaled_params.items():
                p = dict(monitor.named_parameters())[pname]
                p.data.copy_(original / max(tau, 1e-8))
            loss = float(majority_loss(monitor, n).item())
            trace.losses.append(loss)
    finally:
        for pname, original in scaled_params.items():
            dict(monitor.named_parameters())[pname].data.copy_(original)

    return trace
