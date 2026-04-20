"""
M2: unconstrained (H, T) with *soft* acceptance p(s) ∈ [0, 1] learned jointly.

The transition parametrisation is identical to M1; only the acceptance rule
differs (no hard-threshold discretisation).  This isolates Δ^acc = f_M2 - f_M1
for hypothesis H5.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from src.monitors.base import MonitorBase


class M2Monitor(MonitorBase):
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
        self.omega = nn.Parameter(randn(k))

    def transitions(self, n: int) -> Tuple[Tensor, Tensor]:
        H = torch.softmax(self.W_H, dim=-1)
        T = torch.softmax(self.W_T, dim=-1)
        return H, T

    def acceptance(self) -> Tensor:
        return torch.sigmoid(self.omega)
