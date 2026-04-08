"""
Plotting functions for the Poisson LOB simulator.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from lob import SimulationResult

COLORS = {
    "mid":       "steelblue",
    "cash":      "#9C27B0",
    "wealth":    "#4CAF50",
    "inventory": "#FF9800",
}


def plot_single_run(result: SimulationResult, p_fill: float, gamma: float) -> plt.Figure:
    """Four-panel figure for a single simulation run."""
    times       = np.array([s.time      for s in result.snapshots])
    mids        = np.array([s.mid_price for s in result.snapshots])
    inventories = np.array([s.inventory for s in result.snapshots])
    cashes      = np.array([s.cash      for s in result.snapshots])
    wealths     = np.array([s.wealth    for s in result.snapshots])
    rel_mids    = mids - mids[0]

    fig = plt.figure(figsize=(11, 13))
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.50)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, rel_mids, color=COLORS["mid"], lw=1, label="mid price − mid₀")
    ax1.axhline(0, color="gray", lw=0.5, ls="--")
    ax1.set_title("Mid-price change from start", fontsize=12)
    ax1.set_xlabel("time")
    ax1.set_ylabel("price change (ticks)")
    ax1.legend(fontsize=9)
    ax1.grid(True, ls="--", alpha=0.4)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(times, cashes, color=COLORS["cash"], lw=1, label="cash")
    ax2.axhline(0, color="gray", lw=0.5, ls="--")
    ax2.set_title("Cumulative cash from executions", fontsize=12)
    ax2.set_xlabel("time")
    ax2.set_ylabel("cash")
    ax2.legend(fontsize=9)
    ax2.grid(True, ls="--", alpha=0.4)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(times, wealths, color=COLORS["wealth"], lw=1, label="wealth (cash + inv × mid)")
    ax3.axhline(0, color="gray", lw=0.5, ls="--")
    ax3.set_title("Market maker wealth (cash + inventory × mid)", fontsize=12)
    ax3.set_xlabel("time")
    ax3.set_ylabel("wealth")
    ax3.legend(fontsize=9)
    ax3.grid(True, ls="--", alpha=0.4)

    ax4 = fig.add_subplot(gs[3])
    ax4.plot(times, inventories, color=COLORS["inventory"], lw=0.8, alpha=0.85, label="inventory")
    ax4.axhline(0, color="black", lw=0.5, ls="--")
    ax4.set_title("Market maker inventory over time", fontsize=12)
    ax4.set_xlabel("time")
    ax4.set_ylabel("inventory (units)")
    ax4.legend(fontsize=9)
    ax4.grid(True, ls="--", alpha=0.4)

    fig.suptitle(f"Poisson Limit Order Book  (p_fill={p_fill}, γ={gamma})", fontsize=13)
    return fig


def plot_gamma_comparison(
    gamma_results: list,
    seed: int,
    p_fill: float,
) -> plt.Figure:
    """2×3 grid comparing wealth and inventory across gamma values."""
    n = len(gamma_results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), sharey=False)
    axes = axes_grid.flatten()

    for ax, (g, label, res) in zip(axes, gamma_results):
        t   = np.array([s.time      for s in res.snapshots])
        w   = np.array([s.wealth    for s in res.snapshots])
        inv = np.array([s.inventory for s in res.snapshots])

        ax.plot(t, w, color=COLORS["wealth"], lw=1, label="wealth")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_title(f"γ = {g}  —  {label}", fontsize=11)
        ax.set_xlabel("time")
        ax.set_ylabel("wealth")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, ls="--", alpha=0.4)

        ax_inv = ax.twinx()
        ax_inv.plot(t, inv, color=COLORS["inventory"], lw=0.8, alpha=0.7, label="inventory")
        ax_inv.axhline(0, color="black", lw=0.4, ls=":")
        ax_inv.set_ylabel("inventory", color=COLORS["inventory"])
        ax_inv.tick_params(axis="y", labelcolor=COLORS["inventory"])
        ax_inv.legend(fontsize=8, loc="upper right")

    # Hide any unused axes in the grid
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Inventory-Aware MM  —  gamma comparison  (seed={seed}, p_fill={p_fill})",
        fontsize=12,
    )
    plt.tight_layout()
    return fig
