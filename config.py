"""
Shared simulation parameters.
"""

LAMBDAS = {
    "buy_market_order":  1.5,
    "sell_market_order": 1.5,
    "buy_limit_order":   2.0,
    "sell_limit_order":  2.0,
    "cancel_bid":        1.8,
    "cancel_ask":        1.8,
}

DT      = 0.005   # time step  (Λ * dt = 10.6 * 0.005 = 0.053 << 1)
N_STEPS = 10_000
P_FILL  = 0.5     # probability MM is filled on each market order
GAMMA   = 0.01    # default gamma for single-run

LOB_INIT = dict(
    bid_price=9999,
    ask_price=10001,
    bid_size=5,
    ask_size=5,
    tick_size=1,
    default_depth=5,
)

GAMMA_CONFIGS = [
    (0.00, "No Control"),
    (0.01, "Light Control"),
    (0.05, "Moderate Control"),
    (0.10, "Strong Control"),
    (0.25, "Very Conservative"),
    (0.50, "Extreme Aversion"),
]
