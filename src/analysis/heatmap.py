"""Transition-matrix heat-maps (H*, T*) for the structural analysis section."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


def plot_heatmap(H: Tensor, T: Tensor, title: str, out_path: str | Path) -> None:
    """Side-by-side heat-maps of H and T with cell values annotated."""
    H_np = H.detach().cpu().numpy()
    T_np = T.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for ax, mat, name in zip(axes, (H_np, T_np), ("H (Head)", "T (Tail)")):
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="equal")
        ax.set_title(f"{name}")
        ax.set_xlabel("next state s'")
        ax.set_ylabel("state s")
        for s in range(mat.shape[0]):
            for s_prime in range(mat.shape[1]):
                if mat[s, s_prime] > 0.01:
                    ax.text(s_prime, s, f"{mat[s, s_prime]:.2f}", ha="center", va="center",
                            color="white" if mat[s, s_prime] < 0.5 else "black", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dynamic_heatmap(H: Tensor, T: Tensor, title: str, out_path: str | Path, step_indices=None) -> None:
    """For dynamic monitors: show H_i, T_i at a handful of representative steps."""
    n = H.shape[0]
    if step_indices is None:
        step_indices = np.linspace(0, n - 1, 4, dtype=int)

    fig, axes = plt.subplots(2, len(step_indices), figsize=(3 * len(step_indices), 6))
    for col, i in enumerate(step_indices):
        for row, (mat, name) in enumerate([(H[i], "H"), (T[i], "T")]):
            ax = axes[row, col]
            im = ax.imshow(mat.detach().cpu().numpy(), vmin=0, vmax=1, cmap="viridis")
            ax.set_title(f"{name}  i={int(i)}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
