"""Variant A: Minimal — simplest working completion of the r4/manual.py sketch.

Frozen dataclass strategies, match-statement dispatch, market-order-only fills,
minimal EventContext, no netting. ~300 lines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import IntEnum

import heapq as hq

import polars as pl
from dateutil.rrule import rrule, rruleset
from tqdm import tqdm

from backtester.dtypes import (
    Fill,
    Instrument,
    MarketOrder,
    OptionInstrument,
    OptionKind,
    Order,
)
from utils import checks


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class Priority(IntEnum):
    START = 0
    EXPIRY = 1
    SCHEDULE = 2
    END = 3


@dataclass(frozen=True, slots=True)
class OnScheduled:
    t: datetime
    strategy: Strategy


@dataclass(frozen=True, slots=True)
class OnExpiry:
    t: datetime
    instrument: Instrument


Event = OnScheduled | OnExpiry


def _priority(event: Event) -> int:
    match event:
        case OnExpiry():
            return Priority.EXPIRY
        case OnScheduled():
            return Priority.SCHEDULE


# ---------------------------------------------------------------------------
# Event queue
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, order=True)
class QueueEntry:
    t: datetime
    priority: int
    seq: int
    event: Event = field(compare=False)


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[QueueEntry] = []
        self._seen: set[Event] = set()
        self._seq = 0

    def push(self, event: Event) -> None:
        if event in self._seen:
            return
        entry = QueueEntry(event.t, _priority(event), self._seq, event)
        hq.heappush(self._heap, entry)
        self._seq += 1
        self._seen.add(event)

    def pop_all_at_t(self) -> list[QueueEntry]:
        if not self._heap:
            return []
        first = hq.heappop(self._heap)
        batch = [first]
        self._seen.discard(first.event)
        while self._heap and self._heap[0].t == first.t:
            entry = hq.heappop(self._heap)
            self._seen.discard(entry.event)
            batch.append(entry)
        return batch

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)


# ---------------------------------------------------------------------------
# Context & strategy output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventContext:
    """Information visible to a strategy at time ``t``."""

    t: datetime
    bars_rate: pl.LazyFrame | None
    bars_spot: pl.LazyFrame | None
    bars_futures_calendar: pl.LazyFrame | None
    bars_futures_perpetual: pl.LazyFrame | None
    bars_option: pl.LazyFrame | None
    signals: dict[str, pl.LazyFrame]
    position: dict[Instrument, float]  # strategy positions
    portfolio_position: dict[Instrument, float]  # portfolio-wide positions


@dataclass(frozen=True, slots=True)
class StrategyOutput:
    targets: dict[Instrument, float] = field(default_factory=dict)
    orders: list[Order] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Strategy(ABC):
    name: str
    rule: rrule | rruleset

    def on_event(self, event: Event, ctx: EventContext) -> StrategyOutput:
        match event:
            case OnScheduled():
                return self.on_scheduled(ctx)
            case OnExpiry(instrument=instrument):
                return self.on_expiry(ctx, instrument)

    @abstractmethod
    def on_scheduled(self, ctx: EventContext) -> StrategyOutput: ...

    def on_expiry(self, _ctx: EventContext, _i: Instrument) -> StrategyOutput:
        return StrategyOutput()


# ---------------------------------------------------------------------------
# Schemas (placeholders)
# ---------------------------------------------------------------------------


class Schemas:
    BARS_RATE = pl.Schema(
        {"time_end": pl.Datetime(time_zone="UTC"), "rate": pl.Float64}
    )
    BARS_SPOT = pl.Schema({"time_end": pl.Datetime(time_zone="UTC"), "px": pl.Float64})
    BARS_FUTURES_CALENDAR = pl.Schema({"time_end": pl.Datetime(time_zone="UTC"), "px": pl.Float64})  # fmt: off
    BARS_FUTURES_PERPETUAL = pl.Schema({"time_end": pl.Datetime(time_zone="UTC"), "px": pl.Float64})  # fmt: off
    BARS_OPTION = pl.Schema(
        {  # fmt: off
            "time_end": pl.Datetime(time_zone="UTC"),
            "px": pl.Float64,
            "delta": pl.Float64,
        }
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BacktestResult:
    fills: list[Fill]
    positions: dict[Instrument, float]


# ---------------------------------------------------------------------------
# Fill helper
# ---------------------------------------------------------------------------

_PLACEHOLDER_PX = 100.0


def _fill_market_order(order: MarketOrder) -> Fill:
    return Fill(t=order.t, i=order.i, o=order, q=order.q, px=_PLACEHOLDER_PX)


# ---------------------------------------------------------------------------
# Filtering helper
# ---------------------------------------------------------------------------


def _filter_lf(lf: pl.LazyFrame | None, t: datetime) -> pl.LazyFrame | None:
    if lf is None:
        return None
    return lf.filter(pl.col("time_end") <= t)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Backtest:
    bars_rate: pl.LazyFrame | None = None
    bars_spot: pl.LazyFrame | None = None
    bars_futures_calendar: pl.LazyFrame | None = None
    bars_futures_perpetual: pl.LazyFrame | None = None
    bars_option: pl.LazyFrame | None = None
    signals: dict[str, pl.LazyFrame] = field(default_factory=dict)
    strategies: dict[str, Strategy] = field(default_factory=dict)

    def with_bars_rate(self, bars: pl.LazyFrame) -> Backtest:
        checks.require(
            checks.is_none("bars_rate", self.bars_rate),
            checks.has_schema(bars, Schemas.BARS_RATE),
        )
        return replace(self, bars_rate=bars)

    def with_bars_spot(self, bars: pl.LazyFrame) -> Backtest:
        checks.require(
            checks.is_none("bars_spot", self.bars_spot),
            checks.has_schema(bars, Schemas.BARS_SPOT),
        )
        return replace(self, bars_spot=bars)

    def with_bars_futures_calendar(self, bars: pl.LazyFrame) -> Backtest:
        checks.require(
            checks.is_none("bars_futures_calendar", self.bars_futures_calendar),
            checks.has_schema(bars, Schemas.BARS_FUTURES_CALENDAR),
        )
        return replace(self, bars_futures_calendar=bars)

    def with_bars_futures_perpetual(self, bars: pl.LazyFrame) -> Backtest:
        checks.require(
            checks.is_none("bars_futures_perpetual", self.bars_futures_perpetual),
            checks.has_schema(bars, Schemas.BARS_FUTURES_PERPETUAL),
        )
        return replace(self, bars_futures_perpetual=bars)

    def with_bars_option(self, bars: pl.LazyFrame) -> Backtest:
        checks.require(
            checks.is_none("bars_option", self.bars_option),
            checks.has_schema(bars, Schemas.BARS_OPTION),
        )
        return replace(self, bars_option=bars)

    def with_signal(self, name: str, signal: pl.LazyFrame) -> Backtest:
        checks.require(checks.not_in(name, name, self.signals.keys()))
        return replace(self, signals={**self.signals, name: signal})

    def with_strategy(self, strategy: Strategy) -> Backtest:
        checks.require(
            checks.not_in(strategy.name, strategy.name, self.strategies.keys())
        )
        return replace(self, strategies={**self.strategies, strategy.name: strategy})

    def run(self, t0: datetime, tf: datetime) -> BacktestResult:
        fills: list[Fill] = []
        # strategy name -> instrument -> signed qty
        strat_positions: dict[str, dict[Instrument, float]] = {
            name: {} for name in self.strategies
        }
        portfolio_positions: dict[Instrument, float] = {}

        queue = EventQueue()
        for strategy in self.strategies.values():
            for t in strategy.rule.between(t0, tf, inc=True):
                t_utc = t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t
                queue.push(OnScheduled(t_utc, strategy))

        with tqdm(total=len(queue)) as pbar:
            while queue:
                batch = queue.pop_all_at_t()
                t = batch[0].t
                if t > tf:
                    break
                pbar.update(len(batch))

                filtered_signals = {
                    k: _filter_lf(v, t) for k, v in self.signals.items()
                }  # fmt: off

                for entry in batch:
                    match entry.event:
                        case OnScheduled(strategy=strategy):
                            ctx = EventContext(
                                t=t,
                                bars_rate=_filter_lf(self.bars_rate, t),
                                bars_spot=_filter_lf(self.bars_spot, t),
                                bars_futures_calendar=_filter_lf(
                                    self.bars_futures_calendar, t
                                ),
                                bars_futures_perpetual=_filter_lf(
                                    self.bars_futures_perpetual, t
                                ),
                                bars_option=_filter_lf(self.bars_option, t),
                                signals=filtered_signals,
                                position=dict(strat_positions[strategy.name]),
                                portfolio_position=dict(portfolio_positions),
                            )
                            output = strategy.on_event(entry.event, ctx)
                            for order in output.orders:
                                if not isinstance(order, MarketOrder):
                                    continue
                                fill = _fill_market_order(order)
                                fills.append(fill)
                                # Update positions
                                sp = strat_positions[strategy.name]
                                sp[fill.i] = sp.get(fill.i, 0.0) + fill.q
                                if sp[fill.i] == 0.0:
                                    del sp[fill.i]
                                portfolio_positions[fill.i] = (
                                    portfolio_positions.get(fill.i, 0.0) + fill.q
                                )
                                if portfolio_positions[fill.i] == 0.0:
                                    del portfolio_positions[fill.i]
                                # Inject expiry event for option instruments
                                if (
                                    isinstance(fill.i, OptionInstrument)
                                    and fill.i in portfolio_positions
                                ):
                                    queue.push(OnExpiry(fill.i.expiry, fill.i))

                        case OnExpiry():
                            for sname, strategy in self.strategies.items():
                                ctx = EventContext(
                                    t=t,
                                    bars_rate=_filter_lf(self.bars_rate, t),
                                    bars_spot=_filter_lf(self.bars_spot, t),
                                    bars_futures_calendar=_filter_lf(
                                        self.bars_futures_calendar, t
                                    ),
                                    bars_futures_perpetual=_filter_lf(
                                        self.bars_futures_perpetual, t
                                    ),
                                    bars_option=_filter_lf(self.bars_option, t),
                                    signals=filtered_signals,
                                    position=dict(strat_positions[sname]),
                                    portfolio_position=dict(portfolio_positions),
                                )
                                output = strategy.on_event(entry.event, ctx)
                                for order in output.orders:
                                    if not isinstance(order, MarketOrder):
                                        continue
                                    fill = _fill_market_order(order)
                                    fills.append(fill)
                                    sp = strat_positions[sname]
                                    sp[fill.i] = sp.get(fill.i, 0.0) + fill.q
                                    if sp[fill.i] == 0.0:
                                        del sp[fill.i]
                                    portfolio_positions[fill.i] = (
                                        portfolio_positions.get(fill.i, 0.0) + fill.q
                                    )
                                    if portfolio_positions[fill.i] == 0.0:
                                        del portfolio_positions[fill.i]

        return BacktestResult(fills=fills, positions=portfolio_positions)


# ---------------------------------------------------------------------------
# Example strategies
# ---------------------------------------------------------------------------


def _select_option(_ctx: EventContext, kind: OptionKind) -> OptionInstrument:
    """Placeholder instrument selector — returns a dummy option."""
    return OptionInstrument(
        exchange="drbt",
        base="btc",
        quote="usd",
        strike=50_000.0,
        listing=datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc),
        expiry=datetime(2025, 2, 1, 8, 0, 0, tzinfo=timezone.utc),
        kind=kind,
    )


@dataclass(frozen=True, slots=True)
class SingleOption(Strategy):
    option_kind: OptionKind = OptionKind.CALL
    target_delta: float = 0.25
    target_tenor: float = 30.0
    qty: float = 1.0

    def on_scheduled(self, ctx: EventContext) -> StrategyOutput:
        target = _select_option(ctx, self.option_kind)
        current_instruments = list(ctx.position.keys())
        orders: list[Order] = []

        # Close old position if it differs from target
        for inst in current_instruments:
            if inst != target:
                q = ctx.position[inst]
                orders.append(MarketOrder(t=ctx.t, i=inst, q=-q))

        # Open target if not already held at desired qty
        held = ctx.position.get(target, 0.0)
        delta = self.qty - held
        if delta != 0.0:
            orders.append(MarketOrder(t=ctx.t, i=target, q=delta))

        return StrategyOutput(targets={target: self.qty}, orders=orders)


@dataclass(frozen=True, slots=True)
class Straddle(Strategy):
    target_delta: float = 0.50
    target_tenor: float = 30.0
    qty: float = 1.0

    def on_scheduled(self, ctx: EventContext) -> StrategyOutput:
        call = _select_option(ctx, OptionKind.CALL)
        put = replace(call, kind=OptionKind.PUT)
        targets = {call: self.qty, put: self.qty}
        orders: list[Order] = []

        # Close positions that aren't part of the target straddle
        for inst, q in ctx.position.items():
            if inst not in targets:
                orders.append(MarketOrder(t=ctx.t, i=inst, q=-q))

        # Open / adjust each leg
        for inst, tgt_q in targets.items():
            held = ctx.position.get(inst, 0.0)
            delta = tgt_q - held
            if delta != 0.0:
                orders.append(MarketOrder(t=ctx.t, i=inst, q=delta))

        return StrategyOutput(targets=targets, orders=orders)
