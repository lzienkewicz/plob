"""
Poisson Limit Order Book — single simulation run.

Run:
    python main.py
"""

import random
import numpy as np
import matplotlib.pyplot as plt

from lob import simulate
from config import LAMBDAS, DT, N_STEPS, P_FILL, GAMMA, LOB_INIT, FILL_MODE, K_FILL
from plotting import plot_single_run

SEED = random.randint(0, 2**31 - 1)

print(f"Running Poisson LOB simulation  (seed={SEED}, gamma={GAMMA}, fill_mode={FILL_MODE})...")
result = simulate(
    lambdas=LAMBDAS,
    dt=DT,
    n_steps=N_STEPS,
    p_fill=P_FILL,
    gamma=GAMMA,
    fill_mode=FILL_MODE,
    k=K_FILL,
    seed=SEED,
    **LOB_INIT,
)

# ------------------------------------------------------------------
# Summary + diagnostics
# ------------------------------------------------------------------

times       = np.array([s.time      for s in result.snapshots])
mids        = np.array([s.mid_price for s in result.snapshots])
inventories = np.array([s.inventory for s in result.snapshots])
cashes      = np.array([s.cash      for s in result.snapshots])
wealths     = np.array([s.wealth    for s in result.snapshots])

price_changes = int(np.sum(np.diff(mids) != 0))
print(f"  Steps simulated : {N_STEPS:,}")
print(f"  Total time      : {times[-1]:.1f}")
print(f"  Mid start / end : {mids[0]:.1f} / {mids[-1]:.1f}")
print(f"  Mid std         : {mids.std():.3f}")
print(f"  Price moves     : {price_changes}")
print(f"  Spread          : always {result.snapshots[0].spread} (constant)")
print(f"  p_fill          : {P_FILL}")
print(f"\nMM summary:")
print(f"  Final inventory : {inventories[-1]}")
print(f"  Min  inventory  : {inventories.min()}")
print(f"  Max  inventory  : {inventories.max()}")
print(f"  Final cash      : {cashes[-1]:.1f}")
print(f"  Final wealth    : {wealths[-1]:.1f}")
print(f"  Bid fills       : {result.bid_fills}")
print(f"  Ask fills       : {result.ask_fills}")
print(f"  Fill imbalance  : {result.bid_fills - result.ask_fills}")
print(f"  Inventory/fills : {inventories[-1]} / ({result.bid_fills + result.ask_fills})")

result.print_diagnostics(LAMBDAS, DT)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------

fig = plot_single_run(result, p_fill=P_FILL, gamma=GAMMA, fill_mode=FILL_MODE, k=K_FILL)
fig.savefig("lob_simulation.png", dpi=150, bbox_inches="tight")
print("\nFigure saved -> lob_simulation.png")
plt.show()
