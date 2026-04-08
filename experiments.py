"""
Research experiments: gamma sweeps, fill model comparisons, etc.

Run:
    python experiments.py
"""

import random
import numpy as np
import matplotlib.pyplot as plt

from lob import simulate
from config import LAMBDAS, DT, N_STEPS, P_FILL, K_FILL, LOB_INIT, GAMMA_CONFIGS
from plotting import plot_gamma_comparison, plot_fill_comparison

SEED = random.randint(0, 2**31 - 1)

SIM_KWARGS = dict(
    lambdas=LAMBDAS,
    dt=DT,
    n_steps=N_STEPS,
    p_fill=P_FILL,
    seed=SEED,
    **LOB_INIT,
)

# Subset of gamma values used for the fill model comparison
FILL_COMPARE_GAMMAS = [(g, l) for g, l in GAMMA_CONFIGS if g in (0.0, 0.05, 0.10, 0.25)]


def run_gamma_sweep() -> None:
    print(f"Running gamma sweep  (seed={SEED}, fill_mode=logistic, k={K_FILL})...")
    gamma_results = []
    for g, label in GAMMA_CONFIGS:
        r     = simulate(**SIM_KWARGS, gamma=g, fill_mode="logistic", k=K_FILL)
        inv   = np.array([s.inventory for s in r.snapshots])
        w     = np.array([s.wealth    for s in r.snapshots])
        fills = r.bid_fills + r.ask_fills
        print(
            f"  gamma={g:<4}  ({label:<18})  "
            f"fills={fills:>5}  inv_std={inv.std():.2f}  "
            f"final_wealth={w[-1]:.1f}"
        )
        gamma_results.append((g, label, r))

    fig = plot_gamma_comparison(gamma_results, seed=SEED, p_fill=P_FILL)
    fig.savefig("lob_gamma_comparison.png", dpi=150, bbox_inches="tight")
    print("Figure saved -> lob_gamma_comparison.png")
    plt.show(block=False)


def run_fill_model_comparison() -> None:
    print(f"\nRunning fill model comparison  (seed={SEED})...")
    constant_results = []
    logistic_results = []

    for g, label in FILL_COMPARE_GAMMAS:
        rc = simulate(**SIM_KWARGS, gamma=g, fill_mode="constant")
        rl = simulate(**SIM_KWARGS, gamma=g, fill_mode="logistic", k=K_FILL)
        constant_results.append(rc)
        logistic_results.append(rl)
        print(
            f"  gamma={g:<4}  constant fills={rc.bid_fills + rc.ask_fills:>5}  "
            f"logistic fills={rl.bid_fills + rl.ask_fills:>5}"
        )

    fig = plot_fill_comparison(
        FILL_COMPARE_GAMMAS, constant_results, logistic_results,
        seed=SEED, p_fill=P_FILL, k=K_FILL,
    )
    fig.savefig("lob_fill_comparison.png", dpi=150, bbox_inches="tight")
    print("Figure saved -> lob_fill_comparison.png")
    plt.show()


if __name__ == "__main__":
    run_gamma_sweep()
    run_fill_model_comparison()
