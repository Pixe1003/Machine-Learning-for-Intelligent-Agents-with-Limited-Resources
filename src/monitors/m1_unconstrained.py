"""M1-hard: unconstrained static monitor with hard report-time acceptance.

The training objective keeps a soft sigmoid acceptance relaxation for
optimisation.  `report_evaluate()` flips `hard_accept` on a deepcopy so reported
results use a hard accepting set.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from src.monitors.base import MonitorBase


class M1HardMonitor(MonitorBase):
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
        self.hard_accept: bool = False

    def transitions(self, n: int) -> Tuple[Tensor, Tensor]:
        H = torch.softmax(self.W_H, dim=-1)
        T = torch.softmax(self.W_T, dim=-1)
        return H, T

    def acceptance(self) -> Tensor:
        p = torch.sigmoid(self.omega)
        if self.hard_accept:
            p = (p >= 0.5).to(p.dtype)
        return p

    def load_from_m1(self, m1: "M1Monitor") -> None:
        """Warm-start from an M1 monitor with the same unconstrained transitions."""
        self.W_H.data.copy_(m1.W_H.data)
        self.W_T.data.copy_(m1.W_T.data)
        self.omega.data.copy_(m1.omega.data)
