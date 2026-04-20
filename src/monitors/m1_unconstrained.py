"""
M1: unconstrained row-stochastic (H, T) with hard acceptance F ⊆ {0,...,k-1}.

Parametrisation
---------------
    H = softmax(W_H, dim=-1)     W_H ∈ R^{k×k}
    T = softmax(W_T, dim=-1)     W_T ∈ R^{k×k}
    p(s) = σ(ω_s)                during training
    p(s) ∈ {0, 1} (thresholded at 0.5)   at evaluation

The softmax naturally enforces row-stochasticity; the hard-F discretisation
is applied only in eval mode so the optimiser sees a smooth objective.

Free parameters: 2 k(k-1) + 1  (hard F reduces to the accepting-subset size).
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from src.monitors.base import MonitorBase


class M1Monitor(MonitorBase):
    def __init__(self, k: int, init_scale: float = 0.1, dtype=None, device=None, seed: int | None = None):
        super().__init__(k=k, dtype=dtype if dtype is not None else torch.float64, device=device)
        if seed is not None:
            gen = torch.Generator(device=device if device is not None else "cpu")
            gen.manual_seed(seed)
        else:
            gen = None

        def randn(*shape):
            t = torch.empty(*shape, dtype=self._dtype_, device=device)
            if gen is not None:
                t.normal_(0.0, init_scale, generator=gen)
            else:
                t.normal_(0.0, init_scale)
            return t

        self.W_H = nn.Parameter(randn(k, k))
        self.W_T = nn.Parameter(randn(k, k))
        # ω ∈ R^k; during training we return σ(ω).  At eval, we threshold.
        self.omega = nn.Parameter(randn(k))
        self.hard_accept: bool = False  # flip to True at evaluation

    def transitions(self, n: int) -> Tuple[Tensor, Tensor]:
        H = torch.softmax(self.W_H, dim=-1)
        T = torch.softmax(self.W_T, dim=-1)
        return H, T

    def acceptance(self) -> Tensor:
        p = torch.sigmoid(self.omega)
        if self.hard_accept:
            p = (p >= 0.5).to(p.dtype)
        return p

    def load_from_m2(self, m2: "M1Monitor") -> None:
        """Warm-start from an M2 monitor (shares this parametrisation)."""
        self.W_H.data.copy_(m2.W_H.data)
        self.W_T.data.copy_(m2.W_T.data)
        self.omega.data.copy_(m2.omega.data)
