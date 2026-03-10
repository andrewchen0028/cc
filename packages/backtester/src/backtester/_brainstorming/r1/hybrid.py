"""Branch C: Position Targets + Order Overrides.

Combines the declarative simplicity of Branch A with the flexibility of Branch B.
Strategies declare target positions (default path) but can also emit explicit
orders (override path). Execution model is pluggable via a FillModel Protocol.

Two-phase per timestep:
  Phase 1: Collect targets, net across strategies, diff → auto-orders.
  Phase 2: Allow strategies to override/add orders.

Pros:
  - Simple default path (just declare targets, like Branch A).
  - Extensible: explicit orders for complex strategies, pluggable fill models.
  - Contract expiries handled via optional on_expiry hook (like Branch B).
Cons:
  - Two mental models (targets vs explicit orders).
  - Strategies must be careful not to conflict between phases.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal, Mapping, Protocol

from dateutil.rrule import DAILY, rrule, rruleset
from tqdm import tqdm

import polars as pl

from backtester.dtypes import (
    Fill,
    Instrument,
    LimitOrder,
    MarketOrder,
    OptionInstrument,
    Order,
    Side,
    SpotInstrument,
)
from backtester.io import get_target_option
from utils import schemas


# ---------------------------------------------------------------------------
# Fill model Protocol
# ---------------------------------------------------------------------------


class FillModel(Protocol):
    """Pluggable execution model.

    @CLAUDE NOTE: This addresses base.py's NOTE about extensible order
    execution beyond instant fill. Each FillModel takes an order + market data
    and returns zero or more fills. Zero fills = order not (yet) executed.
    """

    def execute(
        self,
        order: Order,
        bars_priced: pl.LazyFrame,
    ) -> list[Fill]:
        """Execute an order against market data, returning fills."""
        ...


@dataclass(frozen=True, slots=True)
class InstantFill:
    """Fill immediately at best bid (sell) or ask (buy)."""

    def execute(self, order: Order, bars_priced: pl.LazyFrame) -> list[Fill]:
        # Pseudocode: query bars_priced for price at order.t for order.i
        #   if isinstance(order, MarketOrder):
        #       px = bars[side_price].item()
        #   elif isinstance(order, LimitOrder):
        #       market_px = bars[side_price].item()
        #       if order.side == BUY and market_px <= order.px_limit: px = market_px
        #       elif order.side == SELL and market_px >= order.px_limit: px = market_px
        #       else: return []  # limit not met
        px = 0.0  # placeholder

        return [
            Fill(
                t=order.t,
                i=order.i,
                o=order,
                q=order.q,
                side=order.side,
                px=px,
            )
        ]


@dataclass(frozen=True, slots=True)
class TWAPFill:
    """Time-weighted average price fill over n slices.

    @CLAUDE NOTE: This is a sketch showing how the FillModel Protocol enables
    TWAP without changing the core loop. In production, the engine would need
    to call execute() at each sub-slice time, which means the fill model needs
    to be stateful or the engine needs a sub-loop. One option:
    - TWAPFill.execute() returns a *partial* fill for the current slice.
    - The engine tracks unfilled remainder and re-calls at the next slice.
    For now, we simplify by returning a single fill at the average price.
    """

    n_slices: int = 10

    def execute(self, order: Order, bars_priced: pl.LazyFrame) -> list[Fill]:
        # Pseudocode:
        #   prices = bars_priced.filter(instrument, time in slice_times).select(px)
        #   avg_px = prices.mean()
        avg_px = 0.0  # placeholder

        return [
            Fill(
                t=order.t,
                i=order.i,
                o=order,
                q=order.q,
                side=order.side,
                px=avg_px,
            )
        ]


@dataclass(frozen=True, slots=True)
class VWAPFill:
    """Volume-weighted average price fill.

    @CLAUDE NOTE: Sketch only. Real implementation would need volume data in
    bars_priced (not currently in BARS_PRICED schema).
    """

    n_slices: int = 10

    def execute(self, order: Order, bars_priced: pl.LazyFrame) -> list[Fill]:
        vwap_px = 0.0  # placeholder
        return [
            Fill(
                t=order.t,
                i=order.i,
                o=order,
                q=order.q,
                side=order.side,
                px=vwap_px,
            )
        ]


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------


class Strategy(Protocol):
    """Hybrid strategy: declare targets (phase 1) and optionally override (phase 2).

    @CLAUDE NOTE: We use Protocol here (like Branch A) because the default
    implementations for optional methods can live as standalone functions that
    strategies call, rather than requiring inheritance. But we need
    runtime_checkable for isinstance checks in the expiry path.
    """

    name: str
    rule: rrule

    @abstractmethod
    def get_target_position(
        self,
        t: datetime,
        bars_priced: pl.LazyFrame,
        current_position: Mapping[Instrument, float],
    ) -> Mapping[Instrument, float]:
        """Phase 1: Declare desired holdings."""
        ...

    def get_override_orders(
        self,
        t: datetime,
        bars_priced: pl.LazyFrame,
        auto_orders: list[Order],
        current_position: Mapping[Instrument, float],
    ) -> list[Order] | None:
        """Phase 2: Optionally override or supplement auto-generated orders.

        Return None to accept auto_orders as-is.
        Return a list to *replace* auto_orders entirely for this strategy.

        @CLAUDE NOTE: Returning a list replaces (not appends to) auto_orders.
        This lets strategies cancel unwanted auto-orders or substitute with
        limit orders. To merely add orders, return auto_orders + extra_orders.
        """
        return None

    def on_expiry(
        self,
        t: datetime,
        instrument: Instrument,
        bars_priced: pl.LazyFrame,
    ) -> list[Order]:
        """Optional: handle contract expiry outside the normal schedule.

        @CLAUDE NOTE: This is a lightweight version of Branch B's OnExpiry event.
        Strategies that don't care about expiries just leave the default (no-op).
        """
        return []


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


@dataclass
class SingleOption:
    """Roll into the option closest to target delta/tenor each period.

    Uses phase 1 (target positions) only — no order overrides needed.
    """

    name: str
    rule: rrule

    option_exchange: str
    option_base: str
    option_quote: str
    option_kind: Literal["c", "p"]

    spot_instrument: SpotInstrument

    target_delta: float
    target_tenor: timedelta
    qty: float

    def get_target_position(
        self,
        t: datetime,
        bars_priced: pl.LazyFrame,
        current_position: Mapping[Instrument, float],
    ) -> Mapping[Instrument, float]:
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

    def get_override_orders(
        self,
        t: datetime,
        bars_priced: pl.LazyFrame,
        auto_orders: list[Order],
        current_position: Mapping[Instrument, float],
    ) -> list[Order] | None:
        return None  # accept auto-orders

    def on_expiry(
        self,
        t: datetime,
        instrument: Instrument,
        bars_priced: pl.LazyFrame,
    ) -> list[Order]:
        return []  # handled by next scheduled target


class Straddle:
    """Maintain a straddle, with limit-order overrides for better fills.

    Phase 1: declare target call + put positions.
    Phase 2: replace market orders with limit orders at mid-price.

    @CLAUDE NOTE: This demonstrates the hybrid approach's value — simple
    strategies use phase 1 only, sophisticated ones use phase 2 to control
    execution quality without reimplementing the entire position-diff logic.
    """

    def __init__(
        self,
        name: str,
        rule: rrule,
        option_exchange: str,
        option_base: str,
        option_quote: str,
        spot_instrument: SpotInstrument,
        target_delta: float,
        target_tenor: timedelta,
        qty: float,
    ) -> None:
        self.name = name
        self.rule = rule
        self.option_exchange = option_exchange
        self.option_base = option_base
        self.option_quote = option_quote
        self.spot_instrument = spot_instrument
        self.target_delta = target_delta
        self.target_tenor = target_tenor
        self.qty = qty

    def get_target_position(
        self,
        t: datetime,
        bars_priced: pl.LazyFrame,
        current_position: Mapping[Instrument, float],
    ) -> Mapping[Instrument, float]:
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

    def get_override_orders(
        self,
        t: datetime,
        bars_priced: pl.LazyFrame,
        auto_orders: list[Order],
        current_position: Mapping[Instrument, float],
    ) -> list[Order] | None:
        """Replace MarketOrders with LimitOrders at mid-price.

        @CLAUDE NOTE: This is a sketch. Real implementation would query
        bars_priced for bid/ask at time t for each instrument and compute mid.
        """
        limit_orders: list[Order] = []
        for order in auto_orders:
            if isinstance(order, MarketOrder):
                # Pseudocode: mid = (bars[px_bid] + bars[px_ask]) / 2
                mid_price = 100.0  # placeholder
                limit_orders.append(
                    LimitOrder(
                        t=order.t,
                        i=order.i,
                        q=order.q,
                        side=order.side,
                        px_limit=mid_price,
                    )
                )
            else:
                limit_orders.append(order)
        return limit_orders

    def on_expiry(
        self,
        t: datetime,
        instrument: Instrument,
        bars_priced: pl.LazyFrame,
    ) -> list[Order]:
        return []


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------


@dataclass
class PositionBook:
    """Tracks current holdings per strategy + aggregated net.

    @CLAUDE NOTE: Keeping per-strategy books allows PnL attribution while
    the aggregated view is used for netting in phase 1. This is the same
    as Branch A's PositionBook but with explicit aggregate helpers.
    """

    current: dict[Instrument, float] = field(default_factory=dict)

    def net_quantity(self, instrument: Instrument) -> float:
        return self.current.get(instrument, 0.0)

    def update(self, instrument: Instrument, new_qty: float) -> None:
        if abs(new_qty) < 1e-12:
            self.current.pop(instrument, None)
        else:
            self.current[instrument] = new_qty

    def expiring_at(self, t: datetime) -> list[Instrument]:
        return [
            inst
            for inst in self.current
            if isinstance(inst, OptionInstrument) and inst.expiry <= t
        ]


def diff_positions(
    target: dict[Instrument, float],
    current: dict[Instrument, float],
    t: datetime,
) -> list[MarketOrder]:
    """Diff target vs current → MarketOrders (same as Branch A)."""
    orders: list[MarketOrder] = []
    for inst in set(target) | set(current):
        delta = target.get(inst, 0.0) - current.get(inst, 0.0)
        if delta == 0.0:
            continue
        orders.append(
            MarketOrder(
                t=t,
                i=inst,
                q=abs(delta),
                side=Side.BUY if delta > 0 else Side.SELL,
            )
        )
    return orders


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------


class Backtest:
    """Hybrid backtest engine.

    Per-timestep flow:
      Phase 1 — Targets:
        1a. Ask each Strategy for target positions.
        1b. Aggregate targets across strategies (sum per instrument).
        1c. Diff aggregated target vs current portfolio → auto MarketOrders.

      Phase 2 — Overrides:
        2a. Ask each Strategy for order overrides (passing auto-orders for review).
        2b. Replace auto-orders with overrides where strategies opted in.

      Execution:
        3. Execute final orders through the FillModel.
        4. Update position books.

      Expiry:
        5. Check for expiring instruments; call on_expiry for affected strategies.
    """

    def __init__(
        self,
        bars_priced: pl.LazyFrame,
        strategies: list[Strategy],
        fill_model: FillModel | None = None,
    ) -> None:
        self.bars_priced = bars_priced
        self.strategies = strategies
        self.fill_model: FillModel = fill_model or InstantFill()

        self.books: dict[str, PositionBook] = {
            s.name: PositionBook() for s in strategies
        }
        self.orders: list[Order] = []
        self.fills: list[Fill] = []

    def run(self, t0: datetime, tf: datetime) -> None:
        ruleset = rruleset()
        for strategy in self.strategies:
            ruleset.rrule(strategy.rule)

        # @CLAUDE NOTE: Like Branch A, we check for expiries after each timestep
        # rather than injecting them into the schedule. Unlike Branch B's full
        # event queue, this is simpler but means expiry handling is coarser
        # (only checked at schedule times, not at exact expiry moments).
        # A middle ground: scan for expiries between t_prev and t_current
        # and handle them before the regular schedule logic.

        t_prev: datetime | None = None

        for t in tqdm(ruleset.between(t0, tf, inc=True)):
            # --- Handle expiries since last timestep ---
            if t_prev is not None:
                self._handle_expiries(t_prev, t)

            # --- Phase 1: Collect targets ---
            strategy_targets: dict[str, Mapping[Instrument, float]] = {}
            for strategy in self.strategies:
                strategy_targets[strategy.name] = strategy.get_target_position(
                    t,
                    self.bars_priced,
                    self.books[strategy.name].current,
                )

            # Aggregate targets.
            agg_target: dict[Instrument, float] = {}
            for targets in strategy_targets.values():
                for inst, qty in targets.items():
                    agg_target[inst] = agg_target.get(inst, 0.0) + qty

            agg_current: dict[Instrument, float] = {}
            for book in self.books.values():
                for inst, qty in book.current.items():
                    agg_current[inst] = agg_current.get(inst, 0.0) + qty

            # Diff → auto-orders.
            auto_orders = diff_positions(agg_target, agg_current, t)

            # --- Phase 2: Allow overrides ---
            # @CLAUDE NOTE: Per-strategy overrides operate on the *strategy's
            # share* of auto-orders, not the aggregated set. We partition
            # auto-orders by which strategy's target generated each one.
            # For this sketch, we simplify and pass all auto-orders to each
            # strategy (the strategy should only override orders for its own
            # instruments).
            final_orders: list[Order] = []
            for strategy in self.strategies:
                overrides = strategy.get_override_orders(
                    t,
                    self.bars_priced,
                    auto_orders,
                    self.books[strategy.name].current,
                )
                if overrides is not None:
                    final_orders.extend(overrides)
                else:
                    final_orders.extend(auto_orders)

            # @CLAUDE NOTE: The above double-counts when multiple strategies
            # don't override — a real implementation would partition orders by
            # strategy before the override step. Omitted for sketch brevity.

            # --- Execution ---
            for order in final_orders:
                self.orders.append(order)
                fills = self.fill_model.execute(order, self.bars_priced)
                self.fills.extend(fills)

            # --- Update per-strategy books ---
            for strategy in self.strategies:
                book = self.books[strategy.name]
                for inst, qty in strategy_targets[strategy.name].items():
                    book.update(inst, qty)
                for inst in list(book.current):
                    if inst not in strategy_targets[strategy.name]:
                        book.update(inst, 0.0)

            t_prev = t

    def _handle_expiries(self, t_prev: datetime, t_now: datetime) -> None:
        """Check for instruments that expired between t_prev and t_now.

        Calls each strategy's on_expiry hook and executes returned orders.
        """
        for strategy in self.strategies:
            book = self.books[strategy.name]
            for inst in book.expiring_at(t_now):
                expiry_orders = strategy.on_expiry(
                    inst.expiry if isinstance(inst, OptionInstrument) else t_now,
                    inst,
                    self.bars_priced,
                )
                for order in expiry_orders:
                    self.orders.append(order)
                    fills = self.fill_model.execute(order, self.bars_priced)
                    self.fills.extend(fills)

                # Remove expired instrument from book.
                book.update(inst, 0.0)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------


def main() -> None:
    """Example: hybrid backtest with TWAP fill model.

    @CLAUDE NOTE on user interface: The fill_model parameter makes execution
    strategy explicit at the Backtest level. Strategies that want per-order
    control (e.g., some orders TWAP, some instant) would need a richer model
    — perhaps a FillModel that dispatches based on order metadata. Open question.
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
                name="atm_straddle_limit",
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
        fill_model=TWAPFill(n_slices=5),
    ).run(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
