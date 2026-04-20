"""
Core differentiable recursion for the Majority Problem's total error.

Let π^(h)_i ∈ Δ^k be the state distribution after i votes, of which h are
Heads (under the uniform prior on which votes are Heads).  The recursion is

    π^(h)_i = (h/i) · π^(h-1)_{i-1} · H  +  ((i-h)/i) · π^(h)_{i-1} · T,

with π^(0)_0 = e_{s0}.  This is a forward probability recursion whose total
cost is O(n² k²) and which is fully differentiable in (H, T, π_0, p(·)).

The final error probability is
    f(n, k) = Σ_{h<⌈n/2⌉} P(h) · E[p(s_n) | h]         (false positive)
            + Σ_{h>n/2}   P(h) · (1 - E[p(s_n) | h])   (false negative)

with P(h) = C(n, h) · 2^{-n}.

All tensors default to float64 — f(n, k) can be O(1e-3), so float32's rounding
noise pollutes the nesting checks f_M1 ≤ f_M2 ≤ f_M3 ≤ f_M4.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

DEFAULT_DTYPE = torch.float64


def initial_distribution(k: int, s0: int = 0, dtype=DEFAULT_DTYPE, device=None) -> Tensor:
    """Point-mass initial distribution on state ``s0``."""
    pi0 = torch.zeros(k, dtype=dtype, device=device)
    pi0[s0] = 1.0
    return pi0


def _log_binomial_coeffs(n: int, dtype=DEFAULT_DTYPE, device=None) -> Tensor:
    """Return log C(n, h) for h = 0..n using lgamma (numerically stable)."""
    h = torch.arange(n + 1, dtype=dtype, device=device)
    log_n_fact = torch.lgamma(torch.tensor(n + 1.0, dtype=dtype, device=device))
    log_h_fact = torch.lgamma(h + 1.0)
    log_nh_fact = torch.lgamma(n - h + 1.0)
    return log_n_fact - log_h_fact - log_nh_fact


def state_distribution(
    H: Tensor,
    T: Tensor,
    n: int,
    pi0: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute π^(h)_n for all h = 0..n using the forward recursion.

    Parameters
    ----------
    H, T : (k, k) row-stochastic tensors.  May be `(n, k, k)` for a dynamic
           monitor, in which case `H[i], T[i]` are used at step i+1.
    n    : number of votes.
    pi0  : optional initial distribution of shape (k,).  Defaults to δ_0.

    Returns
    -------
    pi_final : (n+1, k) tensor where row h is π^(h)_n.

    Notes
    -----
    * We carry the full table π^(h)_i at step i, of shape (i+1, k).
    * Time O(n² k²), memory O(n k).  Fine for n ≤ 201, k ≤ 20 on CPU.
    """
    if H.dim() == 2:
        H_static, T_static = True, True
        k = H.shape[0]
    elif H.dim() == 3:
        H_static = T_static = False
        if H.shape[0] != n or T.shape[0] != n:
            raise ValueError(
                f"Dynamic H, T must have leading dim = n = {n}, got {H.shape} / {T.shape}"
            )
        k = H.shape[1]
    else:
        raise ValueError(f"H must be 2D (static) or 3D (dynamic); got {H.shape}")

    dtype = H.dtype
    device = H.device
    if pi0 is None:
        pi0 = initial_distribution(k, 0, dtype=dtype, device=device)

    # pi_table: list of (i+1, k) tensors; pi_table[h] is π^(h)_i for current i.
    # We maintain it as a single (i+1, k) tensor and grow it by one row per step.
    pi = pi0.unsqueeze(0).clone()  # shape (1, k) representing π^(0)_0

    for i in range(1, n + 1):
        Hi = H if H_static else H[i - 1]
        Ti = T if T_static else T[i - 1]

        # pi has shape (i, k) representing π^(h)_{i-1} for h = 0..i-1.
        # New table has shape (i+1, k) for h = 0..i.
        # Recursion:
        #   π^(h)_i = (h/i) π^(h-1)_{i-1} H + ((i-h)/i) π^(h)_{i-1} T
        h_idx = torch.arange(i + 1, dtype=dtype, device=device)
        w_head = h_idx / i  # (i+1,)
        w_tail = (i - h_idx) / i

        # head_contrib[h] = π^(h-1)_{i-1} @ H      (undefined at h=0 -> set to 0)
        head_contrib = torch.zeros(i + 1, k, dtype=dtype, device=device)
        head_contrib[1:] = pi @ Hi  # pi[h-1] @ H for h = 1..i

        # tail_contrib[h] = π^(h)_{i-1} @ T        (undefined at h=i -> set to 0)
        tail_contrib = torch.zeros(i + 1, k, dtype=dtype, device=device)
        tail_contrib[:i] = pi @ Ti

        pi = w_head.unsqueeze(-1) * head_contrib + w_tail.unsqueeze(-1) * tail_contrib

    return pi  # shape (n+1, k), row h = π^(h)_n


def compute_error(
    H: Tensor,
    T: Tensor,
    p: Tensor,
    n: int,
    pi0: Optional[Tensor] = None,
) -> Tensor:
    """
    Total error probability f(H, T, p; n, k), fully differentiable.

    Parameters
    ----------
    H, T : (k,k) or (n,k,k) row-stochastic transition tensors.
    p    : (k,) per-state acceptance probabilities in [0, 1].
    n    : odd number of votes.
    pi0  : optional initial distribution.

    Returns
    -------
    f : scalar tensor, the sum of false-positive and false-negative terms.
    """
    if n % 2 == 0:
        raise ValueError(f"n must be odd to avoid ties; got n={n}")

    pi_final = state_distribution(H, T, n, pi0=pi0)  # (n+1, k)
    accept_prob = pi_final @ p  # (n+1,) : E[p(s_n) | h]

    log_binom = _log_binomial_coeffs(n, dtype=H.dtype, device=H.device)
    log_px = log_binom - n * math.log(2.0)
    px = torch.exp(log_px)  # P(h) for h = 0..n

    half = n / 2.0  # majority threshold (n odd so never hit)
    h_idx = torch.arange(n + 1, dtype=H.dtype, device=H.device)
    tail_majority = (h_idx < half).to(H.dtype)  # < ⌈n/2⌉ ⇔ Tail majority
    head_majority = (h_idx > half).to(H.dtype)

    false_positive = (px * tail_majority * accept_prob).sum()
    false_negative = (px * head_majority * (1.0 - accept_prob)).sum()
    return false_positive + false_negative
