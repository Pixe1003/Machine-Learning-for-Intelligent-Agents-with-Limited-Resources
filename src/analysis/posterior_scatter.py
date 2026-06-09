"""
State-to-posterior scatter plots: simulate vote sequences, record (p_i, s_i)
at each step, and visualise whether the monitor partitions [0, 1] into k
intervals (the signature of Strategy B -- posterior quantisation).

Two entry points:
    plot_posterior_scatter           -> single-class figure
    plot_class_comparison_scatter    -> 2x2 subplot with all four classes on
                                         one figure (preferred for reports).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _true_posterior(n: int, i: int, h_count: int) -> float:
    """p_i = P(Bin(n - i, 1/2) >= ceil(n/2) - h_count)."""
    remaining = n - i
    threshold = int(np.ceil(n / 2) - h_count)
    if threshold <= 0:
        return 1.0
    if threshold > remaining:
        return 0.0
    return float(binom.sf(threshold - 1, remaining, 0.5))


def simulate_state_posterior(
    monitor, n: int, M: int = 10_000, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run `M` vote sequences through the monitor; returns flat
    (step, state, true_posterior) arrays of length M * n."""
    rng = np.random.default_rng(seed)
    H, T = monitor.transitions(n)
    H_np = H.detach().cpu().numpy()
    T_np = T.detach().cpu().numpy()
    k = H_np.shape[-1] if H_np.ndim == 2 else H_np.shape[1]

    steps_out = np.empty(M * n, dtype=int)
    states_out = np.empty(M * n, dtype=int)
    post_out = np.empty(M * n, dtype=float)

    idx = 0
    for _ in range(M):
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


def _draw_scatter_panel(ax, steps, states, post, *, k: int, title: str,
                        point_size: float = 2.0, alpha: float = 0.08) -> None:
    cmap = plt.get_cmap("tab10")
    for s in range(k):
        mask = states == s
        ax.scatter(
            steps[mask], post[mask],
            s=point_size, alpha=alpha,
            color=cmap(s % 10),
            label=f"state {s}",
        )
    ax.set_xlabel("step i")
    ax.set_ylabel("true posterior p_i")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=10)


# ---------------------------------------------------------------------------
# single-class entry point (kept for backwards compatibility)
# ---------------------------------------------------------------------------

def plot_posterior_scatter(monitor, n: int, out_path: str | Path,
                           M: int = 10_000, seed: int = 0) -> None:
    steps, states, post = simulate_state_posterior(monitor, n, M=M, seed=seed)
    k = int(states.max()) + 1

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    _draw_scatter_panel(ax, steps, states, post, k=k,
                        title=f"State-posterior scatter (n={n}, k={k})")
    leg = ax.legend(loc="center right", markerscale=4)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# consolidated class-comparison scatter
# ---------------------------------------------------------------------------

def plot_class_comparison_scatter(
    monitors: Mapping[str, object],
    n: int,
    out_path: str | Path,
    *,
    M: int = 8_000,
    seed: int = 0,
    title: str | None = None,
) -> None:
    """
    Grid of state-posterior scatter plots, one subplot per monitor class.

    Layout is auto-chosen: 1xN for <=2 classes, 2x2 for 3-4, 2x3 for 5-6.
    """
    classes = list(monitors.keys())
    n_plots = len(classes)
    if n_plots <= 2:
        nrows, ncols = 1, max(n_plots, 1)
    elif n_plots <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 2, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    axes_flat = axes.flatten()

    last_k = 0
    for idx, name in enumerate(classes):
        ax = axes_flat[idx]
        steps, states, post = simulate_state_posterior(
            monitors[name], n, M=M, seed=seed + idx
        )
        k = int(states.max()) + 1
        last_k = max(last_k, k)
        _draw_scatter_panel(ax, steps, states, post, k=k, title=name)

    # hide any unused cells
    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # shared legend on the right
    cmap = plt.get_cmap("tab10")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          color=cmap(s % 10), label=f"state {s}")
               for s in range(last_k)]
    fig.legend(handles=handles, loc="center right",
               bbox_to_anchor=(1.02, 0.5),
               fontsize=9, markerscale=1.2, frameon=False)

    fig.suptitle(title or f"State-posterior scatter (n={n})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.96, 0.96))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
