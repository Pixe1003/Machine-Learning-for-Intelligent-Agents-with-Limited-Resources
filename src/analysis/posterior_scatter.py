"""
State-to-posterior scatter: simulate M vote sequences, record (p_i, s_i) at
each step, and visualise whether the monitor partitions [0, 1] into k
intervals (the signature of Strategy B — posterior quantisation).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import binom


def _true_posterior(n: int, i: int, h_count: int) -> float:
    """p_i = P(Bin(n - i, 1/2) ≥ ⌈n/2⌉ - h_count)."""
    remaining = n - i
    threshold = int(np.ceil(n / 2) - h_count)
    if threshold <= 0:
        return 1.0
    if threshold > remaining:
        return 0.0
    return float(binom.sf(threshold - 1, remaining, 0.5))


def simulate_state_posterior(monitor, n: int, M: int = 50_000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run `M` simulations through the (static) monitor and return arrays of
    shape (M * n,): step index, state, true posterior p_i.
    """
    rng = np.random.default_rng(seed)
    H, T = monitor.transitions(n)
    H_np = H.detach().cpu().numpy()
    T_np = T.detach().cpu().numpy()
    k = H_np.shape[-1] if H_np.ndim == 2 else H_np.shape[1]

    steps_out = np.empty(M * n, dtype=int)
    states_out = np.empty(M * n, dtype=int)
    post_out = np.empty(M * n, dtype=float)

    idx = 0
    for m in range(M):
        s = 0
        h_count = 0
        for i in range(1, n + 1):
            x = int(rng.random() < 0.5)
            mat = (H_np[i - 1] if H_np.ndim == 3 else H_np) if x == 1 else \
                  (T_np[i - 1] if T_np.ndim == 3 else T_np)
            s = int(rng.choice(k, p=mat[s]))
            if x == 1:
                h_count += 1
            steps_out[idx] = i
            states_out[idx] = s
            post_out[idx] = _true_posterior(n, i, h_count)
            idx += 1

    return steps_out, states_out, post_out


def plot_posterior_scatter(monitor, n: int, out_path: str | Path, M: int = 50_000, seed: int = 0) -> None:
    steps, states, post = simulate_state_posterior(monitor, n, M=M, seed=seed)
    k = int(states.max()) + 1

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for s in range(k):
        mask = states == s
        ax.scatter(steps[mask], post[mask], s=1, alpha=0.05, label=f"state {s}")

    ax.set_xlabel("step i")
    ax.set_ylabel("true posterior p_i")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"State–posterior scatter (n={n}, k={k})")
    leg = ax.legend(loc="center right", markerscale=5)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
