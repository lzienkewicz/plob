"""
Research experiments: gamma sweeps, multi-seed runs, etc.

Run:
    python experiments.py
"""

import random
import numpy as np

from lob import simulate
from config import LAMBDAS, DT, N_STEPS, P_FILL, LOB_INIT, GAMMA_CONFIGS
from plotting import plot_gamma_comparison

SEED = random.randint(0, 2**31 - 1)

SIM_KWARGS = dict(
    lambdas=LAMBDAS,
    dt=DT,
    n_steps=N_STEPS,
    p_fill=P_FILL,
    seed=SEED,
    **LOB_INIT,
)


def run_gamma_sweep() -> None:
    print(f"Running gamma comparison  (seed={SEED})...")
    gamma_results = []
    for g, label in GAMMA_CONFIGS:
        r   = simulate(**SIM_KWARGS, gamma=g)
        inv = np.array([s.inventory for s in r.snapshots])
        w   = np.array([s.wealth    for s in r.snapshots])
        fills = r.bid_fills + r.ask_fills
        print(
            f"  gamma={g:<4}  ({label:<18})  "
            f"fills={fills:>5}  inv_std={inv.std():.2f}  "
            f"final_wealth={w[-1]:.1f}"
        )
        gamma_results.append((g, label, r))

    import matplotlib.pyplot as plt
    fig = plot_gamma_comparison(gamma_results, seed=SEED, p_fill=P_FILL)
    fig.savefig("lob_gamma_comparison.png", dpi=150, bbox_inches="tight")
    print("Figure saved -> lob_gamma_comparison.png")
    plt.show()


if __name__ == "__main__":
    run_gamma_sweep()
