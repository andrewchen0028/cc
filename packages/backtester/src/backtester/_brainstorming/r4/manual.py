from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from dateutil.rrule import rrule, rruleset, DAILY
from tqdm import tqdm

import heapq as hq
import polars as pl

from backtester.dtypes import Instrument, Order, OrderID, Position, PositionID, Fill
from utils import checks


@dataclass(frozen=True, slots=True)
class OnScheduled:
    t: datetime
    strategy: Strategy


@dataclass(frozen=True, slots=True)
class OnExpiry:
    t: datetime
    instrument: Instrument


Event = OnScheduled | OnExpiry


@dataclass(frozen=True, slots=True, order=True)
class QueueEntry:
    t: datetime
    seq: int = field(compare=True)
    event: Event


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[QueueEntry] = []
        self._seen: set[Event] = set()
        self._seq = 0

    def push(self, t: datetime, event: Event) -> None:
        if event in self._seen:
            return

        hq.heappush(self._heap, QueueEntry(t, self._seq, event))
        self._seq += 1
        self._seen.add(event)

    def pop_all_at_t(self) -> list[QueueEntry]:
        if not self._heap:
            return []

        first = hq.heappop(self._heap)
        batch = [first]
        self._seen.remove(first.event)

        while self._heap and self._heap[0].t == first.t:
            entry = hq.heappop(self._heap)
            self._seen.remove(entry.event)
            batch.append(entry)

        return batch

    def __len__(self):
        return len(self._heap)

    def __bool__(self):
        return bool(self._heap)


@dataclass(frozen=True, slots=True)
class EventContext:
    """Wraps information visible to a Strategy upon event evaluation (<= time `t`)."""

    ...


@dataclass(frozen=True, slots=True)
class StrategyOutput:
    orders: list[Order] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Strategy(ABC):
    rule: rrule | rruleset

    def on_event(self, event: Event, ctx: EventContext) -> StrategyOutput:
        match event:
            case OnScheduled():
                return self._on_scheduled(ctx)
            case OnExpiry(instrument=instrument):
                return self._on_expiry(ctx, instrument)

    def _on_scheduled(self, ctx: EventContext) -> StrategyOutput: ...
    def _on_expiry(self, ctx: EventContext, i: Instrument) -> StrategyOutput:
        """By default, do nothing on expiry events."""
        ...


@dataclass(frozen=True, slots=True)
class Schemas:
    BARS_RATE = pl.Schema(...)
    BARS_SPOT = pl.Schema(...)
    BARS_FUTURES_CALENDAR = pl.Schema(...)
    BARS_FUTURES_PERPETUAL = pl.Schema(...)
    BARS_OPTION = pl.Schema(...)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    # TODO [@CLAUDE]: Decide what goes in here. E.g. list of fills, PnL, etc.
    ...


@dataclass(frozen=True, slots=True)
class Backtest:
    bars_rate: pl.LazyFrame | None = None
    bars_spot: pl.LazyFrame | None = None
    bars_futures_calendar: pl.LazyFrame | None = None
    bars_futures_perpetual: pl.LazyFrame | None = None
    bars_option: pl.LazyFrame | None = None
    signals: dict[str, pl.LazyFrame] = {}
    strategies: dict[str, Strategy] = {}

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

    def with_strategy(self, name: str, strategy: Strategy) -> Backtest:
        checks.require(checks.not_in(name, name, self.strategies.keys()))
        return replace(self, strategies={**self.strategies, name: strategy})

    def run(self, t0: datetime, tf: datetime) -> BacktestResult:
        fills: dict[OrderID, list[Fill]] = {}  # noqa: F841
        orders_closed: dict[OrderID, Order] = {}  # noqa: F841
        orders_active: dict[OrderID, Order] = {}  # noqa: F841
        positions_closed: dict[PositionID, Position] = {}  # noqa: F841
        positions_active: dict[PositionID, Position] = {}  # noqa: F841

        queue = EventQueue()

        for strategy in self.strategies.values():
            for t in strategy.rule.between(t0, tf, inc=True):
                queue.push(t, OnScheduled(t, strategy))

        with tqdm(total=len(queue)):
            while queue:
                if (batch := queue.pop_all_at_t())[0].t > tf:
                    break

                # Construct context visible to strategies at time `t`.
                ctx = EventContext()

                # Process all events at time `t`.
                # - Collect gross orders from all strategies (pre-netting).
                orders_gross: dict[Strategy, list[Order]] = {}

                for entry in batch:
                    entry.t  # noqa: F842
                    entry.seq  # noqa: F842
                    entry.event  # noqa: F842

                    match entry.event:
                        case OnScheduled(t=t, strategy=strategy):
                            output = strategy.on_event(entry.event, ctx)
                            orders_gross[strategy].append(output.orders)
                        case OnExpiry(t=t, instrument=instrument):  # noqa: F841
                            for strategy in self.strategies.values():
                                output = strategy.on_event(entry.event, ctx)
                                orders_gross[strategy].append(output.orders)

                # Net orders across strategies.
                orders_netted = self._net_orders(orders_gross)  # noqa: F841

                # TODO [@CLAUDE]:
                # - Submit netted orders.
                # - Process fills.
                # - Inject events into queue as needed (e.g. for order expiries).
                ...

        return BacktestResult()

    def _net_orders(self, orders_gross: dict[Strategy, list[Order]]) -> list[Order]:
        # TODO [@CLAUDE]: Implement in a way that allows netting different order types
        # (e.g. market, limit, TWAP, VWAP, etc.).
        ...


def main() -> None:
    t0 = datetime(2025, 1, 1, timezone.utc)
    tf = datetime(2025, 1, 31, timezone.utc)

    # NOTE: This way, we can leave data attributes unset if not available, and defer
    # data availability requirements until `run()`, which is the point of no return.
    backtest = Backtest() \
        .with_bars_rate(pl.LazyFrame(schema=Schemas.BARS_RATE)) \
        .with_bars_spot(pl.LazyFrame(schema=Schemas.BARS_SPOT)) \
        .with_bars_futures_calendar(pl.LazyFrame(schema=Schemas.BARS_FUTURES_CALENDAR)) \
        .with_bars_futures_perpetual(pl.LazyFrame(schema=Schemas.BARS_FUTURES_PERPETUAL)) \
        .with_bars_option(pl.LazyFrame(schema=Schemas.BARS_OPTION)) \
        .with_signal("signal", pl.LazyFrame()) \
        .with_strategy("strategy", Strategy(rrule(DAILY)))  # fmt: off

    result = backtest.run(t0, tf)  # noqa: F841
