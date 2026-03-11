"""Round 3: Target-Position Architecture.

Strategies emit StrategyOutput(target, orders) — declarative intent plus
imperative execution. The engine collects target position histories per
strategy and nets orders across strategies before filling.

Key changes from r2 portfolio_aware:
  - StrategyOutput: strategies return both target positions and orders.
  - No on_roll / _detect_rolls — rolls are trivially derivable from target
    history post-hoc (instruments dropped between consecutive targets).
  - Batched timestep processing: pop_all_at_t() collects all events at t,
    then net orders across strategies before filling.
  - Active orders mapping: scaffolding for future limit/TWAP/VWAP.
  - Signal/bar filtering uses <= not ==.
  - Fluent API: Backtest().with_bars().with_strategy().run().
"""

from __future__ import annotations

import heapq
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from dateutil.rrule import DAILY, rrule
from tqdm import tqdm

import polars as pl

from backtester.dtypes import (
    Fill,
    Instrument,
    MarketOrder,
    OptionInstrument,
    Order,
    Side,
    SpotInstrument,
)
from backtester.io import get_target_option
from utils import schemas


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

TargetPosition = dict[Instrument, float]
"""Declarative intent: what the strategy wants to hold."""


@dataclass(frozen=True, slots=True)
class StrategyOutput:
    """Return type for strategy event handlers.

    target: declarative — what the strategy wants to hold (analysis artifact).
    orders: imperative — how to get there (execution artifact).
    """

    target: TargetPosition
    orders: list[Order]


_EMPTY_OUTPUT = StrategyOutput(target={}, orders=[])


# ---------------------------------------------------------------------------
# Position views — read-only windows into engine state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionView:
    """Read-only view of a single strategy's positions."""

    _holdings: Mapping[Instrument, float]

    @property
    def current(self) -> Mapping[Instrument, float]:
        return self._holdings

    def qty(self, instrument: Instrument) -> float:
        return self._holdings.get(instrument, 0.0)

    @property
    def instruments(self) -> list[Instrument]:
        return list(self._holdings.keys())

    @property
    def is_flat(self) -> bool:
        return all(abs(q) < 1e-12 for q in self._holdings.values())


@dataclass(frozen=True, slots=True)
class PortfolioView:
    """Read-only view of the entire portfolio (all strategies)."""

    _positions: Mapping[str, Mapping[Instrument, float]]

    def strategy_position(self, name: str) -> Mapping[Instrument, float]:
        return self._positions.get(name, {})

    @property
    def all_instruments(self) -> set[Instrument]:
        result: set[Instrument] = set()
        for pos in self._positions.values():
            result.update(pos.keys())
        return result

    def net_qty(self, instrument: Instrument) -> float:
        """Net quantity across all strategies for an instrument."""
        return sum(pos.get(instrument, 0.0) for pos in self._positions.values())


# ---------------------------------------------------------------------------
# Event context — <= filtering
# ---------------------------------------------------------------------------


class EventContext:
    """Rich context object passed to strategy event handlers.

    Signals and bars are filtered with <= (up to and including t).
    """

    def __init__(
        self,
        t: datetime,
        bars_priced: pl.LazyFrame,
        position: PositionView,
        portfolio: PortfolioView,
        signals: dict[str, pl.LazyFrame],
    ) -> None:
        self._t = t
        self._bars_priced = bars_priced
        self._position = position
        self._portfolio = portfolio
        self._signals = signals

    @property
    def t(self) -> datetime:
        return self._t

    @property
    def position(self) -> PositionView:
        return self._position

    @property
    def portfolio(self) -> PortfolioView:
        return self._portfolio

    def bars_up_to_t(self) -> pl.LazyFrame:
        """Return bars_priced filtered to time_end <= t."""
        return self._bars_priced.filter(pl.col("time_end") <= self._t)

    def get_signal(self, name: str) -> pl.LazyFrame:
        """Get a named signal LazyFrame, pre-filtered to time_end <= t."""
        if name not in self._signals:
            raise KeyError(
                f"Signal '{name}' not registered. "
                f"Available: {list(self._signals.keys())}"
            )
        return self._signals[name]

    def get_price(self, instrument: Instrument) -> float:
        """Look up the mark price for an instrument at time t.

        Placeholder — would filter bars_priced for the instrument at time t.
        """
        _ = instrument
        return 0.0


# ---------------------------------------------------------------------------
# Events — no OnRoll
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OnStart:
    ctx: EventContext


@dataclass(frozen=True, slots=True)
class OnEnd:
    ctx: EventContext


@dataclass(frozen=True, slots=True)
class OnSchedule:
    ctx: EventContext


@dataclass(frozen=True, slots=True)
class OnExpiry:
    instrument: Instrument
    ctx: EventContext


@dataclass(frozen=True, slots=True)
class OnFill:
    fill: Fill
    ctx: EventContext


Event = OnStart | OnEnd | OnSchedule | OnExpiry | OnFill


# ---------------------------------------------------------------------------
# Strategy ABC — returns StrategyOutput
# ---------------------------------------------------------------------------


class Strategy(ABC):
    """Strategy base class. All output-producing handlers return StrategyOutput."""

    name: str
    rule: rrule

    @abstractmethod
    def on_schedule(self, ctx: EventContext) -> StrategyOutput: ...

    def on_start(self, ctx: EventContext) -> StrategyOutput:  # noqa: ARG002
        return _EMPTY_OUTPUT

    def on_end(self, ctx: EventContext) -> StrategyOutput:  # noqa: ARG002
        return _EMPTY_OUTPUT

    def on_expiry(  # noqa: ARG002
        self,
        instrument: Instrument,
        ctx: EventContext,
    ) -> StrategyOutput:
        return _EMPTY_OUTPUT

    def on_fill(self, fill: Fill, ctx: EventContext) -> None:  # noqa: ARG002
        """Informational only — no output."""
        pass


# ---------------------------------------------------------------------------
# Concrete strategies — construct their own orders
# ---------------------------------------------------------------------------


class SingleOption(Strategy):
    option_kind: Literal["c", "p"]

    def __init__(
        self,
        name: str,
        rule: rrule,
        option_exchange: str,
        option_base: str,
        option_quote: str,
        option_kind: Literal["c", "p"],
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
        self.option_kind = option_kind
        self.spot_instrument = spot_instrument
        self.target_delta = target_delta
        self.target_tenor = target_tenor
        self.qty = qty

    def _select_instrument(
        self,
        t: datetime,
        bars: pl.LazyFrame,
    ) -> OptionInstrument:
        lf_rate = bars.select("time_start", "time_end", "rate").unique()
        lf_spot = bars.select(
            "time_start", "time_end",
            pl.lit(self.spot_instrument.exchange).alias("exchange"),
            pl.lit(self.spot_instrument.base).alias("base"),
            pl.lit(self.spot_instrument.quote).alias("quote"),
            pl.col("spot").alias("px_bid"),
            pl.col("spot").alias("px_ask"),
            pl.col("spot").alias("px_mark"),
        ).unique(["time_start", "time_end"])  # fmt: off
        lf_option = bars.select(schemas.BARS_OPTION.names()).unique()

        return get_target_option(
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

    def on_schedule(self, ctx: EventContext) -> StrategyOutput:
        target_inst = self._select_instrument(ctx.t, ctx.bars_up_to_t())
        target: TargetPosition = {target_inst: self.qty}
        orders: list[Order] = []

        # Close current if different instrument.
        for inst in ctx.position.instruments:
            if inst != target_inst:
                orders.append(
                    MarketOrder(
                        t=ctx.t,
                        i=inst,
                        q=abs(ctx.position.qty(inst)),
                        side=Side.SELL if ctx.position.qty(inst) > 0 else Side.BUY,
                    )
                )

        # Open new if not held.
        if abs(ctx.position.qty(target_inst)) < 1e-12:
            orders.append(
                MarketOrder(
                    t=ctx.t,
                    i=target_inst,
                    q=abs(self.qty),
                    side=Side.BUY if self.qty > 0 else Side.SELL,
                )
            )

        return StrategyOutput(target=target, orders=orders)


class Straddle(Strategy):
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

    def on_schedule(self, ctx: EventContext) -> StrategyOutput:
        bars = ctx.bars_up_to_t()
        lf_rate = bars.select("time_start", "time_end", "rate").unique()
        lf_spot = bars.select(
            "time_start", "time_end",
            pl.lit(self.spot_instrument.exchange).alias("exchange"),
            pl.lit(self.spot_instrument.base).alias("base"),
            pl.lit(self.spot_instrument.quote).alias("quote"),
            pl.col("spot").alias("px_bid"),
            pl.col("spot").alias("px_ask"),
            pl.col("spot").alias("px_mark"),
        ).unique(["time_start", "time_end"])  # fmt: off
        lf_option = bars.select(schemas.BARS_OPTION.names()).unique()

        call = get_target_option(
            lf_rate=lf_rate,
            lf_spot=lf_spot,
            lf_option=lf_option,
            option_exchange=self.option_exchange,
            option_base=self.option_base,
            option_quote=self.option_quote,
            option_kind="c",
            spot_instrument=self.spot_instrument,
            target_time=ctx.t,
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

        target: TargetPosition = {call: self.qty, put: self.qty}
        orders: list[Order] = []

        # Close legs that differ from target.
        held_call: OptionInstrument | None = None
        held_put: OptionInstrument | None = None
        for inst in ctx.position.instruments:
            if isinstance(inst, OptionInstrument) and inst.kind == "c":
                held_call = inst
            elif isinstance(inst, OptionInstrument) and inst.kind == "p":
                held_put = inst

        for held, new in [(held_call, call), (held_put, put)]:
            if held is not None and held != new:
                orders.append(
                    MarketOrder(
                        t=ctx.t,
                        i=held,
                        q=abs(self.qty),
                        side=Side.SELL if self.qty > 0 else Side.BUY,
                    )
                )

        # Open new legs if not held.
        if held_call != call:
            orders.append(
                MarketOrder(
                    t=ctx.t,
                    i=call,
                    q=abs(self.qty),
                    side=Side.BUY if self.qty > 0 else Side.SELL,
                )
            )
        if held_put != put:
            orders.append(
                MarketOrder(
                    t=ctx.t,
                    i=put,
                    q=abs(self.qty),
                    side=Side.BUY if self.qty > 0 else Side.SELL,
                )
            )

        return StrategyOutput(target=target, orders=orders)


# ---------------------------------------------------------------------------
# Event queue — min-heap with pop_all_at_t()
# ---------------------------------------------------------------------------


# Event priority within a timestep: start < expiry < schedule < end.
_EVENT_PRIORITY: dict[str, int] = {
    "start": 0,
    "expiry": 1,
    "schedule": 2,
    "end": 3,
    "fill": 4,
}


@dataclass(order=True, frozen=True, slots=True)
class QueueEntry:
    t: datetime
    priority: int = field(compare=True)
    seq: int = field(compare=True)
    strategy_name: str = field(compare=False)
    event_kind: str = field(compare=False)
    payload: Any = field(compare=False)


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[QueueEntry] = []
        self._seq = 0
        self._seen_expiries: set[tuple[str, datetime, Instrument]] = set()

    def push(
        self,
        t: datetime,
        strategy_name: str,
        event_kind: str,
        payload: Any = None,
    ) -> None:
        priority = _EVENT_PRIORITY.get(event_kind, 99)
        entry = QueueEntry(t, priority, self._seq, strategy_name, event_kind, payload)
        heapq.heappush(self._heap, entry)
        self._seq += 1

    def push_expiry(
        self,
        t: datetime,
        strategy_name: str,
        instrument: Instrument,
    ) -> None:
        key = (strategy_name, t, instrument)
        if key not in self._seen_expiries:
            self._seen_expiries.add(key)
            self.push(t, strategy_name, "expiry", instrument)

    def pop_all_at_t(self) -> list[QueueEntry]:
        """Pop all entries at the earliest time in the queue."""
        if not self._heap:
            return []
        first = heapq.heappop(self._heap)
        batch = [first]
        while self._heap and self._heap[0].t == first.t:
            batch.append(heapq.heappop(self._heap))
        return batch

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)


# ---------------------------------------------------------------------------
# Active orders — scaffolding for future limit/TWAP/VWAP
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ActiveOrder:
    id: str
    t_submitted: datetime
    strategy: str
    instrument: Instrument
    delta_qty: float  # signed: positive = buy, negative = sell
    order: MarketOrder
    status: Literal["pending", "filled", "cancelled"] = "pending"


# ---------------------------------------------------------------------------
# Position-as-LazyFrame history
# ---------------------------------------------------------------------------


@dataclass
class PositionSlice:
    """A position slice: bars_priced from fill_t to close_t, with qty."""

    strategy: str
    instrument: Instrument
    qty: float
    fill_t: datetime
    close_t: datetime | None = None

    def to_lazyframe(self, bars_priced: pl.LazyFrame) -> pl.LazyFrame:
        lf = bars_priced.filter(pl.col("time_end") >= self.fill_t)
        if self.close_t is not None:
            lf = lf.filter(pl.col("time_end") <= self.close_t)
        return lf.with_columns(pl.lit(self.qty).alias("qty"))


# ---------------------------------------------------------------------------
# Backtest engine — fluent API, batched timestep, netting
# ---------------------------------------------------------------------------


class Backtest:
    """Event-driven backtest engine with target-position architecture.

    Fluent API:
        Backtest()
            .with_bars(bars_priced)
            .with_signal("momentum", lf)
            .with_strategy(short_put)
            .run(t0, tf)
    """

    def __init__(self) -> None:
        self._bars_priced: pl.LazyFrame | None = None
        self._signals: dict[str, pl.LazyFrame] = {}
        self._strategies: list[Strategy] = []

    # --- Fluent configuration ---

    def with_bars(self, bars_priced: pl.LazyFrame) -> Backtest:
        self._bars_priced = bars_priced
        return self

    def with_signal(self, name: str, lf: pl.LazyFrame) -> Backtest:
        self._signals[name] = lf
        return self

    def with_strategy(self, strategy: Strategy) -> Backtest:
        self._strategies.append(strategy)
        return self

    # --- Execution ---

    def run(self, t0: datetime, tf: datetime) -> BacktestResult:
        if self._bars_priced is None:
            raise ValueError("bars_priced is required — call .with_bars()")
        if not self._strategies:
            raise ValueError("At least one strategy is required")

        return _run_backtest(
            bars_priced=self._bars_priced,
            signals=dict(self._signals),
            strategies=list(self._strategies),
            t0=t0,
            tf=tf,
        )


@dataclass
class BacktestResult:
    """Holds all output from a backtest run."""

    target_history: dict[str, list[tuple[datetime, TargetPosition]]]
    """Per-strategy target position history: strategy_name -> [(t, target)]."""

    positions: dict[str, dict[Instrument, float]]
    """Final per-strategy positions at end of run."""

    fills: list[tuple[str, Fill]]
    """All fills: (strategy_name, fill)."""

    closed_slices: list[PositionSlice]
    """Finalized position slices."""

    active_orders: dict[str, ActiveOrder]
    """Active orders at end of run (should be empty for market-order-only)."""


def _run_backtest(
    bars_priced: pl.LazyFrame,
    signals: dict[str, pl.LazyFrame],
    strategies: list[Strategy],
    t0: datetime,
    tf: datetime,
) -> BacktestResult:
    """Core backtest loop: batched timestep processing with netting."""

    strategy_map: dict[str, Strategy] = {s.name: s for s in strategies}
    positions: dict[str, dict[Instrument, float]] = {s.name: {} for s in strategies}

    # Target position histories per strategy.
    target_history: dict[str, list[tuple[datetime, TargetPosition]]] = {
        s.name: [] for s in strategies
    }

    # Position-as-LazyFrame slices.
    open_slices: dict[str, dict[Instrument, PositionSlice]] = {
        s.name: {} for s in strategies
    }
    closed_slices: list[PositionSlice] = []

    # Active orders mapping.
    active_orders: dict[str, ActiveOrder] = {}

    # All fills for post-hoc analysis.
    all_fills: list[tuple[str, Fill]] = []

    # --- Helpers ---

    def _make_context(t: datetime, strategy_name: str) -> EventContext:
        filtered_signals = {
            name: lf.filter(pl.col("time_end") <= t) for name, lf in signals.items()
        }
        position = PositionView(dict(positions[strategy_name]))
        portfolio = PortfolioView({name: dict(pos) for name, pos in positions.items()})
        return EventContext(
            t=t,
            bars_priced=bars_priced,
            position=position,
            portfolio=portfolio,
            signals=filtered_signals,
        )

    def _apply_fill(strategy_name: str, fill: Fill) -> None:
        pos = positions[strategy_name]
        sign = 1.0 if fill.side == Side.BUY else -1.0
        current = pos.get(fill.i, 0.0)
        new_qty = current + sign * fill.q

        if abs(new_qty) < 1e-12:
            pos.pop(fill.i, None)
            slices = open_slices[strategy_name]
            if fill.i in slices:
                s = slices.pop(fill.i)
                s.close_t = fill.t
                closed_slices.append(s)
        else:
            pos[fill.i] = new_qty
            if abs(current) < 1e-12:
                open_slices[strategy_name][fill.i] = PositionSlice(
                    strategy=strategy_name,
                    instrument=fill.i,
                    qty=new_qty,
                    fill_t=fill.t,
                )

    def _fill_market_order(order: MarketOrder) -> Fill:
        px = 0.0  # placeholder — would look up bid/ask from bars
        return Fill(
            t=order.t,
            i=order.i,
            o=order,
            q=order.q,
            side=order.side,
            px=px,
        )

    def _expiring_at(strategy_name: str, t: datetime) -> list[Instrument]:
        return [
            inst
            for inst in positions[strategy_name]
            if isinstance(inst, OptionInstrument) and inst.expiry <= t
        ]

    # --- Seed event queue ---

    queue = EventQueue()

    for strategy in strategies:
        queue.push(t0, strategy.name, "start")

    for strategy in strategies:
        for t in strategy.rule.between(t0, tf, inc=True):
            queue.push(t, strategy.name, "schedule")

    for strategy in strategies:
        queue.push(tf, strategy.name, "end")

    # --- Main loop: batched timestep processing ---

    total_events = len(queue)
    with tqdm(total=total_events, desc="Backtesting") as pbar:
        while queue:
            batch = queue.pop_all_at_t()
            t = batch[0].t
            if t > tf:
                break

            pbar.update(len(batch))

            # 1. Dispatch all events at t, collect StrategyOutput per strategy.
            #    Last target wins per strategy; orders accumulate.
            strategy_targets: dict[str, TargetPosition] = {}
            strategy_orders: dict[str, list[Order]] = defaultdict(list)

            for entry in batch:
                s_name = entry.strategy_name
                strategy = strategy_map[s_name]
                ctx = _make_context(t, s_name)

                if entry.event_kind == "start":
                    output = strategy.on_start(ctx)
                elif entry.event_kind == "end":
                    output = strategy.on_end(ctx)
                elif entry.event_kind == "schedule":
                    output = strategy.on_schedule(ctx)
                elif entry.event_kind == "expiry":
                    output = strategy.on_expiry(entry.payload, ctx)
                elif entry.event_kind == "fill":
                    strategy.on_fill(entry.payload, ctx)
                    output = None
                else:
                    output = None

                if output is not None:
                    # Last target wins for this strategy at this t.
                    strategy_targets[s_name] = output.target
                    strategy_orders[s_name].extend(output.orders)

            # 2. Record final target positions in history.
            for s_name, tgt in strategy_targets.items():
                target_history[s_name].append((t, tgt))

            # 3. Collect all orders and compute signed qty per instrument.
            #    Track which strategy submitted each order for fill attribution.
            order_attribution: list[tuple[str, MarketOrder]] = []
            for s_name, orders in strategy_orders.items():
                for order in orders:
                    if isinstance(order, MarketOrder):
                        order_attribution.append((s_name, order))

            # 4. Net orders across strategies per instrument.
            instrument_net: dict[Instrument, float] = defaultdict(float)
            instrument_orders: dict[Instrument, list[tuple[str, MarketOrder]]] = \
                defaultdict(list)  # fmt: off

            for s_name, order in order_attribution:
                sign = 1.0 if order.side == Side.BUY else -1.0
                instrument_net[order.i] += sign * order.q
                instrument_orders[order.i].append((s_name, order))

            # 5. Submit to active_orders + fill immediately (MarketOrder).
            for inst, net_qty in instrument_net.items():
                if abs(net_qty) < 1e-12:
                    # Fully internally crossed — fill each side at mark (no spread).
                    for s_name, order in instrument_orders[inst]:
                        fill = _fill_market_order(order)
                        ao = ActiveOrder(
                            id=str(uuid.uuid4()),
                            t_submitted=t,
                            strategy=s_name,
                            instrument=inst,
                            delta_qty=(1.0 if order.side == Side.BUY else -1.0)
                            * order.q,
                            order=order,
                            status="filled",
                        )
                        active_orders[ao.id] = ao
                        _apply_fill(s_name, fill)
                        all_fills.append((s_name, fill))
                        queue.push(t, s_name, "fill", fill)
                else:
                    # Residual goes to market — fill all orders (simplified).
                    for s_name, order in instrument_orders[inst]:
                        fill = _fill_market_order(order)
                        ao = ActiveOrder(
                            id=str(uuid.uuid4()),
                            t_submitted=t,
                            strategy=s_name,
                            instrument=inst,
                            delta_qty=(1.0 if order.side == Side.BUY else -1.0)
                            * order.q,
                            order=order,
                            status="filled",
                        )
                        active_orders[ao.id] = ao
                        _apply_fill(s_name, fill)
                        all_fills.append((s_name, fill))
                        queue.push(t, s_name, "fill", fill)

            # 6. Expiry injection for positions held at this t.
            for s_name in positions:
                for inst in _expiring_at(s_name, t):
                    queue.push_expiry(t, s_name, inst)

    # Finalize remaining open slices.
    for s_name, slices in open_slices.items():
        for inst, s in slices.items():
            s.close_t = tf
            closed_slices.append(s)

    return BacktestResult(
        target_history=target_history,
        positions=positions,
        fills=all_fills,
        closed_slices=closed_slices,
        active_orders=active_orders,
    )


# ---------------------------------------------------------------------------
# Example usage — fluent ergonomics + target history inspection
# ---------------------------------------------------------------------------


def main() -> None:
    spot = SpotInstrument(exchange="deribit", base="BTC", quote="USD")

    momentum_lf = pl.LazyFrame(
        schema=pl.Schema(
            {
                "time_end": pl.Datetime(time_zone="UTC"),
                "value": pl.Float64(),
            }
        )
    )

    short_put = SingleOption(
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
    )

    straddle = Straddle(
        name="atm_straddle",
        rule=rrule(DAILY),
        option_exchange="deribit",
        option_base="BTC",
        option_quote="USD",
        spot_instrument=spot,
        target_delta=0.50,
        target_tenor=timedelta(days=30),
        qty=1.0,
    )

    result = Backtest() \
        .with_bars(pl.LazyFrame(schema=schemas.BARS_PRICED)) \
        .with_signal("momentum", momentum_lf) \
        .with_strategy(short_put) \
        .with_strategy(straddle) \
        .run(
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 12, 31, tzinfo=timezone.utc),
        )  # fmt: off

    # Post-hoc: inspect target position histories per strategy.
    for s_name, history in result.target_history.items():
        for t, target in history:
            print(f"{s_name} @ {t}: {target}")

    # Post-hoc: derive rolls from target history (no engine complexity).
    from itertools import pairwise

    for s_name, history in result.target_history.items():
        for (t1, tgt1), (t2, tgt2) in pairwise(history):
            closed = set(tgt1) - set(tgt2)
            opened = set(tgt2) - set(tgt1)
            if closed and opened:
                print(f"{s_name} rolled at {t2}: {closed} -> {opened}")
