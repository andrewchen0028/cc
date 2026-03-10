"""Branch A: Protocol + Position Diffing.

Strategies declare *where they want to be* via `get_target_position()`.
The Backtester diffs target vs current holdings and emits orders automatically.

Pros:
  - Easy to write strategies (declarative, no order management).
  - Natural netting across strategies (targets are additive).
Cons:
  - Less control over order timing/execution (backtester decides).
  - Hard to express conditional/complex order types (TWAP, iceberg).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal, Mapping, Protocol

from dateutil.rrule import DAILY, rrule, rruleset
from tqdm import tqdm

import polars as pl

from backtester.dtypes import (
    Fill,
    Instrument,
    MarketOrder,
    OptionInstrument,
    Side,
    SpotInstrument,
)
from backtester.io import get_target_option
from utils import schemas


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------


class Strategy(Protocol):
    """A strategy declares its target position at each evaluation time.

    The backtester calls `get_target_position` at every scheduled time and
    diffs the result against the current portfolio to derive orders.
    """

    name: str
    rule: rrule

    def get_target_position(
        self,
        t: datetime,
        bars_priced: pl.LazyFrame,
        current_position: Mapping[Instrument, float],
    ) -> Mapping[Instrument, float]:
        """Return desired holdings as {Instrument: signed_quantity}.

        Returning an empty mapping means "flatten everything this strategy owns".
        Returning the same mapping as `current_position` means "no trade".
        """
        ...


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SingleOption:
    """Roll into the option closest to target delta/tenor each period."""

    name: str
    rule: rrule

    option_exchange: str
    option_base: str
    option_quote: str
    option_kind: Literal["c", "p"]

    spot_instrument: SpotInstrument

    target_delta: float
    target_tenor: timedelta
    qty: float  # signed: +1 = long, -1 = short

    def get_target_position(
        self,
        t: datetime,
        bars_priced: pl.LazyFrame,
        current_position: Mapping[Instrument, float],
    ) -> Mapping[Instrument, float]:
        # Select the option instrument closest to our targets.
        lf_rate = bars_priced.select("time_start", "time_end", "rate").unique()
        lf_spot = bars_priced.select(
            "time_start", "time_end",
            pl.lit(self.spot_instrument.exchange).alias("exchange"),
            pl.lit(self.spot_instrument.base).alias("base"),
            pl.lit(self.spot_instrument.quote).alias("quote"),
            pl.col("spot").alias("px_bid"),
            pl.col("spot").alias("px_ask"),
            pl.col("spot").alias("px_mark"),
        ).unique(["time_start", "time_end"])  # fmt: off
        lf_option = bars_priced.select(schemas.BARS_OPTION.names()).unique()

        target = get_target_option(
            lf_rate=lf_rate,
            lf_spot=lf_spot,
            lf_option=lf_option,
            option_exchange=self.option_exchange,
            option_base=self.option_base,
            option_quote=self.option_quote,
            option_kind=self.option_kind,
            spot_instrument=self.spot_instrument,
            target_time=t,
            target_delta=self.target_delta,
            target_tenor=self.target_tenor,
        )

        return {target: self.qty}


@dataclass(frozen=True, slots=True)
class Straddle:
    """Maintain a straddle (call + put at same strike/expiry)."""

    name: str
    rule: rrule

    option_exchange: str
    option_base: str
    option_quote: str

    spot_instrument: SpotInstrument

    target_delta: float  # used to find the ATM strike (delta ≈ 0.5)
    target_tenor: timedelta
    qty: float

    def get_target_position(
        self,
        t: datetime,
        bars_priced: pl.LazyFrame,
        current_position: Mapping[Instrument, float],
    ) -> Mapping[Instrument, float]:
        # Find the call leg, then construct the put with the same strike/expiry.
        lf_rate = bars_priced.select("time_start", "time_end", "rate").unique()
        lf_spot = bars_priced.select(
            "time_start", "time_end",
            pl.lit(self.spot_instrument.exchange).alias("exchange"),
            pl.lit(self.spot_instrument.base).alias("base"),
            pl.lit(self.spot_instrument.quote).alias("quote"),
            pl.col("spot").alias("px_bid"),
            pl.col("spot").alias("px_ask"),
            pl.col("spot").alias("px_mark"),
        ).unique(["time_start", "time_end"])  # fmt: off
        lf_option = bars_priced.select(schemas.BARS_OPTION.names()).unique()

        call = get_target_option(
            lf_rate=lf_rate,
            lf_spot=lf_spot,
            lf_option=lf_option,
            option_exchange=self.option_exchange,
            option_base=self.option_base,
            option_quote=self.option_quote,
            option_kind="c",
            spot_instrument=self.spot_instrument,
            target_time=t,
            target_delta=self.target_delta,
            target_tenor=self.target_tenor,
        )

        put = OptionInstrument(
            exchange=call.exchange,
            base=call.base,
            quote=call.quote,
            strike=call.strike,
            listing=call.listing,
            expiry=call.expiry,
            kind="p",
        )

        return {call: self.qty, put: self.qty}


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------

# @CLAUDE NOTE: Current positions are a simple dict (latest snapshot).
# History is accumulated as LazyFrames for analysis/PnL after the run.
# We track per-strategy positions separately so we can attribute PnL,
# then net across strategies only when emitting orders.


@dataclass
class PositionBook:
    """Tracks current holdings and accumulates position history."""

    current: dict[Instrument, float] = field(default_factory=dict)
    history: list[pl.LazyFrame] = field(default_factory=list)

    def net_quantity(self, instrument: Instrument) -> float:
        return self.current.get(instrument, 0.0)

    def update(self, instrument: Instrument, new_qty: float) -> None:
        if new_qty == 0.0:
            self.current.pop(instrument, None)
        else:
            self.current[instrument] = new_qty

    # @CLAUDE NOTE: History snapshots are appended per-timestep as LazyFrames
    # and concatenated at the end. This mirrors base.py's accumulation pattern.
    # An alternative is a single mutable DataFrame — but LazyFrame concat keeps
    # everything lazy until final collection.


# ---------------------------------------------------------------------------
# Order diffing
# ---------------------------------------------------------------------------


def diff_positions(
    target: dict[Instrument, float],
    current: dict[Instrument, float],
) -> list[MarketOrder]:
    """Diff target vs current holdings, emit MarketOrders for the delta.

    @CLAUDE NOTE: This always emits MarketOrders (instant fill at best bid/ask).
    To support limit orders or TWAP, we'd need the Strategy to return richer
    targets — but that breaks the simplicity of this architecture. See hybrid.py
    for an approach that keeps simple defaults but allows order overrides.
    """
    orders: list[MarketOrder] = []
    all_instruments = set(target) | set(current)

    t_now = datetime.now(tz=timezone.utc)  # placeholder; real impl uses bar time

    for inst in all_instruments:
        qty_target = target.get(inst, 0.0)
        qty_current = current.get(inst, 0.0)
        delta = qty_target - qty_current

        if delta == 0.0:
            continue

        orders.append(
            MarketOrder(
                t=t_now,
                i=inst,
                q=abs(delta),
                side=Side.BUY if delta > 0 else Side.SELL,
            )
        )

    return orders


# ---------------------------------------------------------------------------
# Fill model (instant only in this branch)
# ---------------------------------------------------------------------------


def instant_fill(order: MarketOrder, bars_priced: pl.LazyFrame) -> Fill:
    """Fill a MarketOrder at the best bid (sell) or ask (buy).

    @CLAUDE NOTE: This is the only fill model in Branch A. For extensible fill
    models (TWAP/VWAP), see hybrid.py's FillModel Protocol.
    """
    # In a real implementation, we'd query bars_priced for the bid/ask at
    # order.t for instrument order.i. Pseudocode:
    #   px = bars_priced.filter(instrument == order.i, time == order.t)
    #        .select("px_ask" if order.side == BUY else "px_bid").item()
    px = 0.0  # placeholder

    return Fill(
        t=order.t,
        i=order.i,
        o=order,
        q=order.q,
        side=order.side,
        px=px,
    )


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

SCHEMA_POSITION_SPOT = schemas.BARS_SPOT | pl.Schema({"qty": pl.Float64()})
SCHEMA_POSITION_OPTION = schemas.BARS_OPTION | pl.Schema({"qty": pl.Float64()})


class Backtest:
    """Position-centric backtest engine.

    Flow per timestep:
      1. Ask each Strategy for target positions.
      2. Aggregate targets across strategies (sum quantities per instrument).
      3. Diff aggregated target vs current portfolio → MarketOrders.
      4. Fill orders instantly at best bid/ask.
      5. Update position book; snapshot history.
    """

    def __init__(
        self,
        bars_priced: pl.LazyFrame,
        strategies: list[Strategy],
    ) -> None:
        self.bars_priced = bars_priced
        self.strategies = strategies

        # Per-strategy position books (for PnL attribution).
        self.books: dict[str, PositionBook] = {
            s.name: PositionBook() for s in strategies
        }
        self.orders: list[MarketOrder] = []
        self.fills: list[Fill] = []

    def run(self, t0: datetime, tf: datetime) -> None:
        ruleset = rruleset()
        for strategy in self.strategies:
            ruleset.rrule(strategy.rule)

        # @CLAUDE NOTE on expiry events: In this branch, contract expiries are
        # NOT handled as explicit events. Instead, the strategy should detect
        # that a held instrument has expired (its expiry <= t) and include a
        # replacement in its next target. This is the simplest model but means
        # expiry-triggered logic only fires at the next scheduled time, not at
        # the exact expiry moment. See event_driven.py for an alternative.

        for t in tqdm(ruleset.between(t0, tf, inc=True)):
            # --- 1. Collect per-strategy targets ---
            strategy_targets: dict[str, Mapping[Instrument, float]] = {}
            for strategy in self.strategies:
                strategy_targets[strategy.name] = strategy.get_target_position(
                    t,
                    self.bars_priced,
                    self.books[strategy.name].current,
                )

            # --- 2. Aggregate across strategies ---
            agg_target: dict[Instrument, float] = {}
            for targets in strategy_targets.values():
                for inst, qty in targets.items():
                    agg_target[inst] = agg_target.get(inst, 0.0) + qty

            agg_current: dict[Instrument, float] = {}
            for book in self.books.values():
                for inst, qty in book.current.items():
                    agg_current[inst] = agg_current.get(inst, 0.0) + qty

            # --- 3. Diff → orders ---
            new_orders = diff_positions(agg_target, agg_current)
            self.orders.extend(new_orders)

            # --- 4. Fill orders ---
            for order in new_orders:
                fill = instant_fill(order, self.bars_priced)
                self.fills.append(fill)

            # --- 5. Update per-strategy books ---
            for strategy in self.strategies:
                book = self.books[strategy.name]
                for inst, qty in strategy_targets[strategy.name].items():
                    book.update(inst, qty)
                # Zero out instruments no longer in strategy's target.
                for inst in list(book.current):
                    if inst not in strategy_targets[strategy.name]:
                        book.update(inst, 0.0)

    # @CLAUDE NOTE: Post-run, the user can inspect self.orders, self.fills,
    # and self.books[strategy_name].current / .history for analysis.


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------


def main() -> None:
    """Example Backtest configuration.

    @CLAUDE NOTE on user interface: Compared to base.py, this is cleaner because
    the user doesn't need to pass bars_rate/bars_spot/bars_option separately —
    only the pre-joined bars_priced LazyFrame. The downside is the caller must
    run `_build_lf_priced` upfront. A builder/factory pattern could wrap this.
    """
    spot = SpotInstrument(exchange="deribit", base="BTC", quote="USD")

    Backtest(
        bars_priced=pl.LazyFrame(schema=schemas.BARS_PRICED),
        strategies=[
            SingleOption(
                name="short_put_25d",
                rule=rrule(DAILY),
                option_exchange="deribit",
                option_base="BTC",
                option_quote="USD",
                option_kind="p",
                spot_instrument=spot,
                target_delta=-0.25,
                target_tenor=timedelta(days=30),
                qty=-1.0,
            ),
            Straddle(
                name="atm_straddle",
                rule=rrule(DAILY),
                option_exchange="deribit",
                option_base="BTC",
                option_quote="USD",
                spot_instrument=spot,
                target_delta=0.50,
                target_tenor=timedelta(days=30),
                qty=1.0,
            ),
        ],
    ).run(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
