"""
M3: counting-restricted, state-dependent {stay, move} transitions.

States are linearly ordered 0, 1, ..., k-1.
On H observation, state s moves to s+1 with probability q_H(s), else stays.
On T observation, state s moves to s-1 with probability q_T(s), else stays.
Boundary states (0 on T, k-1 on H) always stay when the "move" would leave.

This encodes a stochastic counter of the H-T difference, testing hypothesis H1.
Free parameters: 2k (for q_H, q_T logits) + k (for acceptance) = 3k.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from src.monitors.base import MonitorBase


class M3Monitor(MonitorBase):
    def __init__(self, k: int, init_scale: float = 0.1, dtype=None, device=None, seed: int | None = None):
        super().__init__(k=k, dtype=dtype if dtype is not None else torch.float64, device=device)
        gen = torch.Generator(device=device if device is not None else "cpu")
        if seed is not None:
            gen.manual_seed(seed)

        def randn(*shape):
            t = torch.empty(*shape, dtype=self._dtype_, device=device)
            t.normal_(0.0, init_scale, generator=gen if seed is not None else None)
            return t

        # Logits for q_H(s), q_T(s), both of length k.
        self.logits_qH = nn.Parameter(randn(k))
        self.logits_qT = nn.Parameter(randn(k))
        self.omega = nn.Parameter(randn(k))

    def _build_transition(self, qH: Tensor, qT: Tensor) -> Tuple[Tensor, Tensor]:
        """Build (k,k) row-stochastic tridiagonal H, T from {stay, move} probs."""
        k = self.k
        H = torch.zeros(k, k, dtype=self._dtype_, device=self._device_)
        T = torch.zeros(k, k, dtype=self._dtype_, device=self._device_)

        # On Head: state s moves to s+1 w.p. qH(s); else stays.
        # Boundary: at s = k-1, the "move" probability is absorbed into staying.
        for s in range(k):
            if s < k - 1:
                H[s, s + 1] = qH[s]
                H[s, s] = 1.0 - qH[s]
            else:
                H[s, s] = 1.0  # absorb at top

        # On Tail: state s moves to s-1 w.p. qT(s); else stays.  Boundary: s=0 stays.
        for s in range(k):
            if s > 0:
                T[s, s - 1] = qT[s]
                T[s, s] = 1.0 - qT[s]
            else:
                T[s, s] = 1.0  # absorb at bottom

        return H, T

    def transitions(self, n: int) -> Tuple[Tensor, Tensor]:
        qH = torch.sigmoid(self.logits_qH)
        qT = torch.sigmoid(self.logits_qT)
        return self._build_transition(qH, qT)

    def acceptance(self) -> Tensor:
        return torch.sigmoid(self.omega)

    @torch.no_grad()
    def to_m1_warm_start(self, m1) -> None:
        """
        Warm-start an M1 (or M2) monitor from this M3.
        Embeds the tridiagonal (H, T) into full k×k logits.
        """
        H, T = self.transitions(n=1)
        eps = 1e-6
        # Convert to log-probabilities; the softmax will recover H, T up to ε.
        H_clamped = torch.clamp(H, min=eps)
        T_clamped = torch.clamp(T, min=eps)
        m1.W_H.data.copy_(torch.log(H_clamped))
        m1.W_T.data.copy_(torch.log(T_clamped))
        m1.omega.data.copy_(self.omega.data.clone())
