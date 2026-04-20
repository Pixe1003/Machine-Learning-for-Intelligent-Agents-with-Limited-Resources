"""
M4: minimal random walk with *state-independent* move probabilities.

Two scalar parameters q_H, q_T — state s transitions to s+1 (on H) w.p. q_H
*regardless of s*, and similarly for q_T.  This is the simplest non-trivial
monitor and serves as the reference point for the counting axis.

Free parameters: 2 (transitions) + k (acceptance) = k + 2.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from src.monitors.base import MonitorBase


class M4Monitor(MonitorBase):
    def __init__(self, k: int, init_scale: float = 0.1, dtype=None, device=None, seed: int | None = None):
        super().__init__(k=k, dtype=dtype if dtype is not None else torch.float64, device=device)
        gen = torch.Generator(device=device if device is not None else "cpu")
        if seed is not None:
            gen.manual_seed(seed)

        def randn(*shape):
            t = torch.empty(*shape, dtype=self._dtype_, device=device)
            t.normal_(0.0, init_scale, generator=gen if seed is not None else None)
            return t

        # Two scalar logits: q_H = σ(logit_qH), q_T = σ(logit_qT).
        self.logit_qH = nn.Parameter(randn(1))
        self.logit_qT = nn.Parameter(randn(1))
        self.omega = nn.Parameter(randn(k))

    def transitions(self, n: int) -> Tuple[Tensor, Tensor]:
        k = self.k
        qH = torch.sigmoid(self.logit_qH).expand(k)
        qT = torch.sigmoid(self.logit_qT).expand(k)

        H = torch.zeros(k, k, dtype=self._dtype_, device=self._device_)
        T = torch.zeros(k, k, dtype=self._dtype_, device=self._device_)

        for s in range(k):
            if s < k - 1:
                H[s, s + 1] = qH[s]
                H[s, s] = 1.0 - qH[s]
            else:
                H[s, s] = 1.0
            if s > 0:
                T[s, s - 1] = qT[s]
                T[s, s] = 1.0 - qT[s]
            else:
                T[s, s] = 1.0
        return H, T

    def acceptance(self) -> Tensor:
        return torch.sigmoid(self.omega)

    @torch.no_grad()
    def to_m3_warm_start(self, m3) -> None:
        """Broadcast M4's two logits into M3's per-state logits."""
        m3.logits_qH.data.fill_(self.logit_qH.item())
        m3.logits_qT.data.fill_(self.logit_qT.item())
        m3.omega.data.copy_(self.omega.data.clone())
