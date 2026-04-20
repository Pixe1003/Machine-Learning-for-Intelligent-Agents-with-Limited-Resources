"""
Dynamic (time-varying) monitor: step-indexed (H_i, T_i) for i = 1..n.

Two variants:
    - DynamicMonitor (full): 2n·k(k-1) free parameters.  Use for n ≤ 51.
    - SharedDynamicMonitor:  W_{H,i} = g_θ(i/n) with g_θ a small MLP — fixed
      parameter count regardless of n.  Use for Phase 3 (n = 201).

Acceptance p(s_n) is time-invariant (it only fires at the terminal state).
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from src.monitors.base import MonitorBase


class DynamicMonitor(MonitorBase):
    """Full step-indexed parametrisation.  One (W_H, W_T) pair per step."""

    def __init__(self, k: int, n: int, init_scale: float = 0.1, dtype=None, device=None, seed: int | None = None):
        super().__init__(k=k, dtype=dtype if dtype is not None else torch.float64, device=device)
        self.n = n
        gen = torch.Generator(device=device if device is not None else "cpu")
        if seed is not None:
            gen.manual_seed(seed)

        def randn(*shape):
            t = torch.empty(*shape, dtype=self._dtype_, device=device)
            t.normal_(0.0, init_scale, generator=gen if seed is not None else None)
            return t

        self.W_H = nn.Parameter(randn(n, k, k))
        self.W_T = nn.Parameter(randn(n, k, k))
        self.omega = nn.Parameter(randn(k))

    def transitions(self, n: int) -> Tuple[Tensor, Tensor]:
        if n != self.n:
            raise ValueError(f"DynamicMonitor was built for n={self.n}, got {n}")
        H = torch.softmax(self.W_H, dim=-1)  # (n, k, k)
        T = torch.softmax(self.W_T, dim=-1)
        return H, T

    def acceptance(self) -> Tensor:
        return torch.sigmoid(self.omega)


class SharedDynamicMonitor(MonitorBase):
    """
    Shared-parameter variant: W_H(i) = MLP(i/n), W_T(i) = MLP(i/n).

    Keeps parameter count O(hidden²) independent of n — essential for Phase 3.
    """

    def __init__(
        self,
        k: int,
        n: int,
        hidden: int = 32,
        dtype=None,
        device=None,
        seed: int | None = None,
    ):
        super().__init__(k=k, dtype=dtype if dtype is not None else torch.float64, device=device)
        self.n = n
        if seed is not None:
            torch.manual_seed(seed)

        out_dim = 2 * k * k  # both W_H and W_T flat
        self.trunk = nn.Sequential(
            nn.Linear(1, hidden, dtype=self._dtype_),
            nn.Tanh(),
            nn.Linear(hidden, hidden, dtype=self._dtype_),
            nn.Tanh(),
            nn.Linear(hidden, out_dim, dtype=self._dtype_),
        )
        self.omega = nn.Parameter(torch.zeros(k, dtype=self._dtype_, device=device))

    def transitions(self, n: int) -> Tuple[Tensor, Tensor]:
        if n != self.n:
            raise ValueError(f"SharedDynamicMonitor was built for n={self.n}, got {n}")
        t = torch.arange(n, dtype=self._dtype_, device=self._device_).unsqueeze(-1) / n
        out = self.trunk(t)  # (n, 2 k²)
        out = out.view(n, 2, self.k, self.k)
        H = torch.softmax(out[:, 0], dim=-1)
        T = torch.softmax(out[:, 1], dim=-1)
        return H, T

    def acceptance(self) -> Tensor:
        return torch.sigmoid(self.omega)
