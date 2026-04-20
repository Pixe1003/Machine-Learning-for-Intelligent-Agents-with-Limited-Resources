"""
Kontorovich M(k) baseline: a fixed, analytically-specified random-walk monitor.

Reference
---------
Leonid (Aryeh) Kontorovich. Statistical estimation with bounded memory.
Statistics and Computing, 22(5):1155-1164, 2012.

The M(k) automaton has states {0,...,k-1} and on each observation moves one
state towards the side of the observed symbol with a fixed probability q.
Acceptance is the upper half of states.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from src.monitors.base import MonitorBase


class MkBaseline(MonitorBase):
    """Non-trainable baseline with a single shared move probability q."""

    def __init__(self, k: int, q: float = 0.5, dtype=torch.float64, device=None):
        super().__init__(k=k, dtype=dtype, device=device)
        self.q = float(q)

    def transitions(self, n: int) -> Tuple[Tensor, Tensor]:
        k = self.k
        H = torch.zeros(k, k, dtype=self._dtype_, device=self._device_)
        T = torch.zeros(k, k, dtype=self._dtype_, device=self._device_)
        q = self.q
        for s in range(k):
            if s < k - 1:
                H[s, s + 1] = q
                H[s, s] = 1.0 - q
            else:
                H[s, s] = 1.0
            if s > 0:
                T[s, s - 1] = q
                T[s, s] = 1.0 - q
            else:
                T[s, s] = 1.0
        return H, T

    def acceptance(self) -> Tensor:
        """Accept if state ≥ ⌈k/2⌉."""
        k = self.k
        p = torch.zeros(k, dtype=self._dtype_, device=self._device_)
        half = (k + 1) // 2
        p[half:] = 1.0
        return p

    def initial(self) -> Tensor:
        k = self.k
        pi0 = torch.zeros(k, dtype=self._dtype_, device=self._device_)
        pi0[k // 2] = 1.0  # start at the middle
        return pi0
