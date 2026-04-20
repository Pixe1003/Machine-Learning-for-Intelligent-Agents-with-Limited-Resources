"""Derived diagnostics for trained monitors."""
from __future__ import annotations

import math

import numpy as np

from src.analysis.posterior_scatter import simulate_state_posterior
from src.analysis.tridiagonality import tridiagonality_score


def _normalised_row_entropy(matrix: np.ndarray) -> np.ndarray:
    clipped = np.clip(matrix, 1e-12, 1.0)
    ent = -(clipped * np.log(clipped)).sum(axis=-1)
    denom = math.log(matrix.shape[-1]) if matrix.shape[-1] > 1 else 1.0
    return ent / max(denom, 1e-12)


def transition_entropy_stats(H, T) -> dict[str, float]:
    """
    Return simple summary statistics for row-wise transition entropy.

    The values are normalised into [0, 1], where 0 is deterministic and 1 is
    maximally diffuse over the available states.
    """
    H_np = H.detach().cpu().numpy()
    T_np = T.detach().cpu().numpy()
    entropies = np.concatenate([
        _normalised_row_entropy(H_np.reshape(-1, H_np.shape[-1])),
        _normalised_row_entropy(T_np.reshape(-1, T_np.shape[-1])),
    ])
    return {
        "transition_entropy_mean": float(entropies.mean()),
        "transition_entropy_min": float(entropies.min()),
        "transition_entropy_max": float(entropies.max()),
    }


def posterior_monotonicity_stats(monitor, n: int, samples: int = 2048, seed: int = 0) -> dict[str, float]:
    """
    Approximate how monotone the state-to-posterior map is at each step.

    We report Spearman-style rank correlation between the discrete state index
    and the true posterior on each step, then aggregate across steps.
    """
    steps, states, post = simulate_state_posterior(monitor, n, M=samples, seed=seed)
    scores: list[float] = []
    for step in range(1, n + 1):
        mask = steps == step
        state_slice = states[mask]
        post_slice = post[mask]
        if len(state_slice) < 2 or np.all(state_slice == state_slice[0]) or np.allclose(post_slice, post_slice[0]):
            scores.append(0.0)
            continue
        state_ranks = np.argsort(np.argsort(state_slice))
        post_ranks = np.argsort(np.argsort(post_slice))
        corr = np.corrcoef(state_ranks, post_ranks)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        scores.append(float(corr))

    scores_np = np.asarray(scores, dtype=float)
    return {
        "posterior_monotonicity_mean": float(scores_np.mean()),
        "posterior_monotonicity_min": float(scores_np.min()),
    }


def monitor_diagnostics(monitor, n: int, samples: int = 2048, seed: int = 0) -> dict[str, float]:
    H, T = monitor.transitions(n)
    diagnostics = {"tridiagonality": float(tridiagonality_score(H))}
    diagnostics.update(transition_entropy_stats(H, T))
    diagnostics.update(posterior_monotonicity_stats(monitor, n=n, samples=samples, seed=seed))
    return diagnostics
