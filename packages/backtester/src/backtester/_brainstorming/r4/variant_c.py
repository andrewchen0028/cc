"""Variant C: Extensible.

FillModel protocol, active order tracking, per-event method dispatch,
richer BacktestResult with @property accessors. Most feature-complete variant.
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from dateutil.rrule import rrule, rruleset
from tqdm import tqdm

import polars as pl

from backtester.dtypes import (
    Fill,
    Instrument,
    MarketOrder,
    OptionInstrument,
    Order,
    OrderID,
    SpotInstrument,
)
from utils import checks


# -- Position views --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionView:
    _holdings: Mapping[Instrument, float]

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
    _positions: Mapping[str, Mapping[Instrument, float]]

    def strategy_position(self, name: str) -> Mapping[Instrument, float]:
        return self._positions.get(name, {})

    @property
    def all_instruments(self) -> set[Instrument]:
        out: set[Instrument] = set()
        for pos in self._positions.values():
            out.update(pos.keys())
        return out

    def net_qty(self, instrument: Instrument) -> float:
        return sum(p.get(instrument, 0.0) for p in self._positions.values())


# -- Context & output ------------------------------------------------------

TargetPosition = dict[Instrument, float]


@dataclass(frozen=True, slots=True)
class EventContext:
    """Information visible to a strategy at time t. Bars/signals filtered <= t."""

    t: datetime
    bars: dict[str, pl.LazyFrame | None]
    signals: dict[str, pl.LazyFrame]
    position: PositionView
    portfolio: PortfolioView

    def get_price(self, _instrument: Instrument) -> float:
        """Placeholder — look up mark price from bars at time t."""
        return 0.0


@dataclass(frozen=True, slots=True)
class StrategyOutput:
    targets: dict[Instrument, float] = field(default_factory=dict)
    orders: list[Order] = field(default_factory=list)


_EMPTY = StrategyOutput()


# -- Strategy ABC ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Strategy(ABC):
    rule: rrule | rruleset
    name: str

    @abstractmethod
    def on_schedule(self, ctx: EventContext) -> StrategyOutput: ...

    def on_start(self, _ctx: EventContext) -> StrategyOutput:
        return _EMPTY

    def on_end(self, _ctx: EventContext) -> StrategyOutput:
        return _EMPTY

    def on_expiry(self, _instrument: Instrument, _ctx: EventContext) -> StrategyOutput:
        return _EMPTY

    def on_fill(self, _fill: Fill, _ctx: EventContext) -> None:
        pass


# -- Active orders ---------------------------------------------------------


@dataclass(slots=True)
class ActiveOrder:
    order: Order
    strategy_name: str
    status: Literal["pending", "filled", "cancelled"] = "pending"


# -- FillModel protocol ---------------------------------------------------


@runtime_checkable
class FillModel(Protocol):
    def execute(
        self, order: Order, bars: dict[str, pl.LazyFrame | None], t: datetime
    ) -> list[Fill]: ...


class InstantFill:
    """Fills MarketOrders at placeholder price. LimitOrders stay pending."""

    def execute(
        self, order: Order, bars: dict[str, pl.LazyFrame | None], t: datetime
    ) -> list[Fill]:
        _ = bars  # placeholder — would look up price from bars
        if isinstance(order, MarketOrder):
            return [Fill(t=t, i=order.i, o=order, q=order.q, px=1.0)]
        return []  # LimitOrder: placeholder, would check price condition


# -- Event queue -----------------------------------------------------------

_EVENT_PRIORITY = {"start": 0, "expiry": 1, "schedule": 2, "end": 3, "fill": 4}  # fmt: off


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
        self._seen: set[tuple[str, datetime, str, int]] = set()
        self._seq = 0

    def push(
        self, t: datetime, strategy_name: str, event_kind: str, payload: Any = None
    ) -> None:
        key = (strategy_name, t, event_kind, id(payload) if payload is not None else 0)
        if key in self._seen:
            return
        self._seen.add(key)
        priority = _EVENT_PRIORITY.get(event_kind, 99)
        heapq.heappush(
            self._heap,
            QueueEntry(t, priority, self._seq, strategy_name, event_kind, payload),
        )
        self._seq += 1

    def push_expiry(
        self, t: datetime, strategy_name: str, instrument: Instrument
    ) -> None:
        key = (strategy_name, t, "expiry", hash(instrument))
        if key in self._seen:
            return
        self._seen.add(key)
        heapq.heappush(
            self._heap,
            QueueEntry(
                t,
                _EVENT_PRIORITY["expiry"],
                self._seq,
                strategy_name,
                "expiry",
                instrument,
            ),
        )
        self._seq += 1

    def pop_all_at_t(self) -> list[QueueEntry]:
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


# -- Schemas (placeholder) -------------------------------------------------


class Schemas:
    BARS_RATE = pl.Schema({"time_end": pl.Datetime("us", "UTC"), "rate": pl.Float64})
    BARS_SPOT = pl.Schema({"time_end": pl.Datetime("us", "UTC"), "px_mark": pl.Float64})
    BARS_FUTURES_CALENDAR = pl.Schema(
        {"time_end": pl.Datetime("us", "UTC"), "px_mark": pl.Float64}
    )
    BARS_FUTURES_PERPETUAL = pl.Schema(
        {"time_end": pl.Datetime("us", "UTC"), "px_mark": pl.Float64}
    )
    BARS_OPTION = pl.Schema({"time_end": pl.Datetime("us", "UTC"), "px_mark": pl.Float64})  # fmt: off


_BAR_SCHEMAS = {
    "rate": Schemas.BARS_RATE, "spot": Schemas.BARS_SPOT,
    "futures_calendar": Schemas.BARS_FUTURES_CALENDAR,
    "futures_perpetual": Schemas.BARS_FUTURES_PERPETUAL,
    "option": Schemas.BARS_OPTION,
}  # fmt: off


# -- BacktestResult --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BacktestResult:
    fills: list[tuple[str, Fill]]
    final_positions: dict[str, dict[Instrument, float]]
    target_history: dict[str, list[tuple[datetime, TargetPosition]]]
    active_orders: dict[OrderID, ActiveOrder]

    @property
    def n_fills(self) -> int:
        return len(self.fills)

    @property
    def instruments(self) -> set[Instrument]:
        return {fill.i for _, fill in self.fills}

    @property
    def strategies(self) -> list[str]:
        return list(self.final_positions.keys())

    @property
    def remaining_active(self) -> dict[OrderID, ActiveOrder]:
        return {k: v for k, v in self.active_orders.items() if v.status == "pending"}


# -- Backtest --------------------------------------------------------------

_BAR_FIELDS = (
    "bars_rate",
    "bars_spot",
    "bars_futures_calendar",
    "bars_futures_perpetual",
    "bars_option",
)
_BAR_KEYS = ("rate", "spot", "futures_calendar", "futures_perpetual", "option")


@dataclass(frozen=True, slots=True)
class Backtest:
    bars_rate: pl.LazyFrame | None = None
    bars_spot: pl.LazyFrame | None = None
    bars_futures_calendar: pl.LazyFrame | None = None
    bars_futures_perpetual: pl.LazyFrame | None = None
    bars_option: pl.LazyFrame | None = None
    signals: dict[str, pl.LazyFrame] = field(default_factory=dict)
    strategies: dict[str, Strategy] = field(default_factory=dict)
    fill_model: FillModel = field(default_factory=InstantFill)

    def _with_bars(self, field_name: str, key: str, bars: pl.LazyFrame) -> Backtest:
        checks.require(
            checks.is_none(field_name, getattr(self, field_name)),
            checks.has_schema(bars, _BAR_SCHEMAS[key]),
        )
        return replace(self, **{field_name: bars})

    def with_bars_rate(self, bars: pl.LazyFrame) -> Backtest:
        return self._with_bars("bars_rate", "rate", bars)

    def with_bars_spot(self, bars: pl.LazyFrame) -> Backtest:
        return self._with_bars("bars_spot", "spot", bars)

    def with_bars_futures_calendar(self, bars: pl.LazyFrame) -> Backtest:
        return self._with_bars("bars_futures_calendar", "futures_calendar", bars)

    def with_bars_futures_perpetual(self, bars: pl.LazyFrame) -> Backtest:
        return self._with_bars("bars_futures_perpetual", "futures_perpetual", bars)

    def with_bars_option(self, bars: pl.LazyFrame) -> Backtest:
        return self._with_bars("bars_option", "option", bars)

    def with_signal(self, name: str, signal: pl.LazyFrame) -> Backtest:
        checks.require(checks.not_in(name, name, self.signals.keys()))
        return replace(self, signals={**self.signals, name: signal})

    def with_strategy(self, name: str, strategy: Strategy) -> Backtest:
        checks.require(checks.not_in(name, name, self.strategies.keys()))
        return replace(self, strategies={**self.strategies, name: strategy})

    def with_fill_model(self, model: FillModel) -> Backtest:
        return replace(self, fill_model=model)

    # -- Run ---------------------------------------------------------------

    def run(self, t0: datetime, tf: datetime) -> BacktestResult:
        positions: dict[str, dict[Instrument, float]] = {n: {} for n in self.strategies}
        target_history: dict[str, list[tuple[datetime, TargetPosition]]] = {
            n: [] for n in self.strategies
        }
        all_fills: list[tuple[str, Fill]] = []
        active_orders: dict[OrderID, ActiveOrder] = {}

        bars_raw = {k: getattr(self, f) for f, k in zip(_BAR_FIELDS, _BAR_KEYS)}

        def _filter_bars(t: datetime) -> dict[str, pl.LazyFrame | None]:
            return {
                k: (v.filter(pl.col("time_end") <= t) if v is not None else None)
                for k, v in bars_raw.items()
            }

        def _filter_signals(t: datetime) -> dict[str, pl.LazyFrame]:
            return {
                k: v.filter(pl.col("time_end") <= t) for k, v in self.signals.items()
            }

        def _ctx(t: datetime, s: str) -> EventContext:
            return EventContext(
                t=t,
                bars=_filter_bars(t),
                signals=_filter_signals(t),
                position=PositionView(dict(positions[s])),
                portfolio=PortfolioView({n: dict(p) for n, p in positions.items()}),
            )

        def _apply_fill(s: str, fill: Fill) -> None:
            pos = positions[s]
            nq = pos.get(fill.i, 0.0) + fill.q
            if abs(nq) < 1e-12:
                pos.pop(fill.i, None)
            else:
                pos[fill.i] = nq

        def _net_orders(
            strat_orders: dict[str, list[Order]],
        ) -> list[tuple[str, Order]]:
            per_inst: dict[Instrument, list[tuple[str, Order]]] = defaultdict(list)
            for s, orders in strat_orders.items():
                for o in orders:
                    per_inst[o.i].append((s, o))
            return [pair for pairs in per_inst.values() for pair in pairs]

        def _evaluate_pending(t: datetime) -> None:
            for ao in list(active_orders.values()):
                if ao.status != "pending":
                    continue
                fills = self.fill_model.execute(ao.order, _filter_bars(t), t)
                if fills:
                    ao.status = "filled"
                    for fill in fills:
                        _apply_fill(ao.strategy_name, fill)
                        all_fills.append((ao.strategy_name, fill))

        # Seed queue.
        queue = EventQueue()
        for s_name in self.strategies:
            queue.push(t0, s_name, "start")
        for s_name, strat in self.strategies.items():
            for t in strat.rule.between(t0, tf, inc=True):
                queue.push(t, s_name, "schedule")
        for s_name in self.strategies:
            queue.push(tf, s_name, "end")

        # Main loop.
        with tqdm(total=len(queue), desc="Backtesting") as pbar:
            while queue:
                batch = queue.pop_all_at_t()
                t = batch[0].t
                if t > tf:
                    break
                pbar.update(len(batch))

                # a. Evaluate pending active orders.
                _evaluate_pending(t)

                # b. Dispatch events, collect StrategyOutput.
                strat_targets: dict[str, TargetPosition] = {}
                strat_orders: dict[str, list[Order]] = defaultdict(list)

                for entry in batch:
                    s = entry.strategy_name
                    strategy = self.strategies[s]
                    ctx = _ctx(t, s)
                    output: StrategyOutput | None = None

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

                    if output is not None:
                        strat_targets[s] = output.targets
                        strat_orders[s].extend(output.orders)

                # c. Record targets.
                for s, tgt in strat_targets.items():
                    target_history[s].append((t, tgt))

                # d+e. Net orders and submit to fill model.
                for s, order in _net_orders(strat_orders):
                    fills = self.fill_model.execute(order, _filter_bars(t), t)
                    if fills:
                        active_orders[order.id] = ActiveOrder(order, s, "filled")
                        for fill in fills:
                            _apply_fill(s, fill)
                            all_fills.append((s, fill))
                            queue.push(t, s, "fill", fill)
                    else:
                        active_orders[order.id] = ActiveOrder(order, s, "pending")

                # f. Inject expiry events for option positions.
                for s, pos in positions.items():
                    for inst in list(pos.keys()):
                        if isinstance(inst, OptionInstrument) and inst.expiry > t:
                            queue.push_expiry(inst.expiry, s, inst)

        return BacktestResult(
            fills=all_fills,
            final_positions=positions,
            target_history=target_history,
            active_orders=active_orders,
        )


# -- Concrete strategies ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class SingleOption(Strategy):
    """Single option, rolls on schedule. Placeholder instrument selection."""

    qty: float = 1.0

    def on_schedule(self, ctx: EventContext) -> StrategyOutput:
        target_inst = SpotInstrument("deribit", "BTC", "USD")  # placeholder
        orders: list[Order] = []
        for inst in ctx.position.instruments:
            if inst != target_inst:
                orders.append(MarketOrder(t=ctx.t, i=inst, q=-ctx.position.qty(inst)))
        if abs(ctx.position.qty(target_inst)) < 1e-12:
            orders.append(MarketOrder(t=ctx.t, i=target_inst, q=self.qty))
        return StrategyOutput(targets={target_inst: self.qty}, orders=orders)


@dataclass(frozen=True, slots=True)
class Straddle(Strategy):
    """Call + put at same strike/expiry. Placeholder instrument selection."""

    qty: float = 1.0

    def on_schedule(self, ctx: EventContext) -> StrategyOutput:
        call = SpotInstrument("deribit", "BTC-C", "USD")  # placeholder
        put = SpotInstrument("deribit", "BTC-P", "USD")
        orders: list[Order] = []
        for inst in ctx.position.instruments:
            if inst not in (call, put):
                orders.append(MarketOrder(t=ctx.t, i=inst, q=-ctx.position.qty(inst)))
        for leg in (call, put):
            if abs(ctx.position.qty(leg)) < 1e-12:
                orders.append(MarketOrder(t=ctx.t, i=leg, q=self.qty))
        return StrategyOutput(targets={call: self.qty, put: self.qty}, orders=orders)
