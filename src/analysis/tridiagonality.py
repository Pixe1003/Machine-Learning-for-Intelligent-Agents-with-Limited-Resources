"""
Tridiagonality score τ(H) = Σ_{|s-s'|≤1} H_{s,s'} / k.

Near-1 means the monitor is behaviourally a bounded random walk (Strategy A);
near 1/3 means transitions are nearly uniform; anything in between is a hybrid.
"""
from __future__ import annotations

import torch
from torch import Tensor


def tridiagonality_score(H: Tensor) -> float:
    """
    Return τ(H) ∈ [0, 1].  Works for (k, k) or (n, k, k) tensors — the latter
    returns the mean tridiagonality across time.
    """
    if H.dim() == 2:
        k = H.shape[0]
        mass = 0.0
        for s in range(k):
            for s_prime in range(k):
                if abs(s - s_prime) <= 1:
                    mass += float(H[s, s_prime].item())
        return mass / k

    if H.dim() == 3:
        scores = [tridiagonality_score(H[i]) for i in range(H.shape[0])]
        return sum(scores) / len(scores)

    raise ValueError(f"H must be 2D or 3D; got {H.shape}")
