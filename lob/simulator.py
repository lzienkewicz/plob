"""
Poisson event simulator for the OrderBook.

Event model
-----------
Six independent Poisson processes, one per event type, with total rate:

    Λ = Σ λᵢ

Each time step dt:
  1. Fire one event with probability Λ * dt (Bernoulli).
  2. If an event fires, choose which type with probability λᵢ / Λ.
  3. Apply the event to the book.
  4. Record the book state.

This gives at most one book transition per time step, which is the
standard discrete-time approximation to a continuous-time Markov chain.
Valid when Λ * dt << 1.

Parameters
----------
lambdas : dict[str, float]
    Arrival rate for each of the six event types.
dt : float
    Time step size (keep Λ * dt << 1).
n_steps : int
    Number of time steps to simulate.
seed : int | None
    RNG seed for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .book import OrderBook
from .market_maker import MarketMaker


EVENTS = [
    "buy_market_order",
    "sell_market_order",
    "buy_limit_order",
    "sell_limit_order",
    "cancel_bid",
    "cancel_ask",
]


@dataclass
class Snapshot:
    step: int
    time: float
    bid_price: int
    ask_price: int
    bid_size: int
    ask_size: int
    mid_price: float
    spread: int
    event: Optional[str]   # event that fired this step, or None
    inventory: int
    cash: float
    wealth: float


@dataclass
class SimulationResult:
    snapshots: List[Snapshot]
    event_counts: Dict[str, int]   # total times each event fired
    ask_depletions: int             # times ask queue hit zero → price up
    bid_depletions: int             # times bid queue hit zero → price down
    bid_fills: int                  # times MM bought (sell MO hit)
    ask_fills: int                  # times MM sold (buy MO hit)

    @property
    def total_events(self) -> int:
        return sum(self.event_counts.values())

    def print_diagnostics(self, lambdas: Dict[str, float], dt: float) -> None:
        T = len(self.snapshots) * dt
        print(f"\nEvent diagnostics (T = {T:.1f}):")
        print(f"  {'event':<22}  {'fired':>6}  {'expected':>8}  {'ratio':>6}")
        print(f"  {'-'*22}  {'-'*6}  {'-'*8}  {'-'*6}")
        for e in EVENTS:
            fired    = self.event_counts.get(e, 0)
            expected = lambdas.get(e, 0.0) * T
            ratio    = fired / expected if expected > 0 else float("nan")
            print(f"  {e:<22}  {fired:>6}  {expected:>8.1f}  {ratio:>6.3f}")
        print(f"\nQueue depletions:")
        print(f"  ask depleted (price up):   {self.ask_depletions}")
        print(f"  bid depleted (price down): {self.bid_depletions}")
        print(f"\nMM fill counts:")
        print(f"  bid fills (MM bought): {self.bid_fills}")
        print(f"  ask fills (MM sold):   {self.ask_fills}")


def simulate(
    lambdas: Dict[str, float],
    dt: float = 0.01,
    n_steps: int = 5000,
    bid_price: int = 9999,
    ask_price: int = 10001,
    bid_size: int = 10,
    ask_size: int = 10,
    tick_size: int = 1,
    default_depth: int = 10,
    p_fill: float = 1.0,
    gamma: float = 0.0,
    seed: Optional[int] = None,
) -> SimulationResult:
    """
    Run the Poisson simulation.

    p_fill : probability the MM is filled on each market order.
             1.0 = always filled (original); 0.5 = queue competition.
    gamma  : inventory-aversion parameter.  0 = symmetric quoting (Step 2).
             The MM quotes around r = mid - gamma * inventory, so quotes
             lean against accumulated inventory.

    Returns a SimulationResult with snapshots and diagnostics.
    """
    rng = np.random.default_rng(seed)
    book = OrderBook(
        bid_price=bid_price,
        ask_price=ask_price,
        bid_size=bid_size,
        ask_size=ask_size,
        tick_size=tick_size,
        default_depth=default_depth,
    )

    rates = np.array([lambdas.get(e, 0.0) for e in EVENTS])
    total_rate = rates.sum()
    probs = rates / total_rate          # selection weights
    fire_prob = total_rate * dt         # probability any event fires this step

    assert fire_prob < 1, (
        f"Λ * dt = {fire_prob:.3f} must be < 1. Reduce dt or lambda."
    )

    mm = MarketMaker()
    snapshots: List[Snapshot] = []
    event_counts: Dict[str, int] = {e: 0 for e in EVENTS}
    ask_depletions = 0
    bid_depletions = 0
    bid_fills = 0
    ask_fills = 0
    t = 0.0

    for step in range(n_steps):
        fired_event: Optional[str] = None

        if rng.random() < fire_prob:
            idx = rng.choice(len(EVENTS), p=probs)
            fired_event = EVENTS[idx]
            event_counts[fired_event] += 1

            # 1. Compute inventory-adjusted MM quotes (Step 3A)
            #    r = mid - gamma * inventory  → lean against accumulated position
            mid = book.mid_price
            r = mid - gamma * mm.inventory
            half_spread = (book.ask_price - book.bid_price) / 2.0
            mm_bid = int(round(r - half_spread))
            mm_ask = int(round(r + half_spread))

            # 2. Fill MM only when quote crosses book price (Step 3B)
            #    sell MO hits bids → MM buys at mm_bid (only if mm_bid >= book.bid_price)
            #    buy  MO hits asks → MM sells at mm_ask (only if mm_ask <= book.ask_price)
            if fired_event == "sell_market_order":
                if mm_bid >= book.bid_price and rng.random() < p_fill:
                    mm.inventory += 1
                    mm.cash -= mm_bid
                    bid_fills += 1
            elif fired_event == "buy_market_order":
                if mm_ask <= book.ask_price and rng.random() < p_fill:
                    mm.inventory -= 1
                    mm.cash += mm_ask
                    ask_fills += 1

            # 2. Update book, track depletions via price moves
            pre_ask_price = book.ask_price
            pre_bid_price = book.bid_price
            getattr(book, fired_event)()
            if book.ask_price > pre_ask_price:
                ask_depletions += 1
            if book.bid_price < pre_bid_price:
                bid_depletions += 1

        # 3. Record state (wealth marked to post-event mid)
        snapshots.append(Snapshot(
            step=step,
            time=t,
            bid_price=book.bid_price,
            ask_price=book.ask_price,
            bid_size=book.bid_size,
            ask_size=book.ask_size,
            mid_price=book.mid_price,
            spread=book.spread,
            event=fired_event,
            inventory=mm.inventory,
            cash=mm.cash,
            wealth=mm.wealth(book.mid_price),
        ))
        t += dt

    return SimulationResult(
        snapshots=snapshots,
        event_counts=event_counts,
        ask_depletions=ask_depletions,
        bid_depletions=bid_depletions,
        bid_fills=bid_fills,
        ask_fills=ask_fills,
    )
