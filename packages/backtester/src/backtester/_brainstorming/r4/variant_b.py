"""Variant B: Portfolio-aware.

Adds PositionView/PortfolioView, lifecycle events (OnStart/OnEnd/OnFill),
order netting with internal crossing, and target position history.
Richer than variant_a but still concise.
"""

from __future__ import annotations

import heapq as hq
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from dateutil.rrule import rrule, rruleset
from tqdm import tqdm

import polars as pl

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
# Position views
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionView:
    """Read-only view of a single strategy's positions."""

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
    """Read-only view of all strategies' positions."""

    _positions: Mapping[str, Mapping[Instrument, float]]

    def net_qty(self, instrument: Instrument) -> float:
        return sum(pos.get(instrument, 0.0) for pos in self._positions.values())

    @property
    def all_instruments(self) -> set[Instrument]:
        out: set[Instrument] = set()
        for pos in self._positions.values():
            out.update(pos.keys())
        return out

    def strategy_position(self, name: str) -> Mapping[Instrument, float]:
        return self._positions.get(name, {})


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

_PRIORITY = {"start": 0, "expiry": 1, "schedule": 2, "end": 3, "fill": 4}


@dataclass(frozen=True, slots=True)
class OnStart:
    t: datetime
    strategy: Strategy


@dataclass(frozen=True, slots=True)
class OnEnd:
    t: datetime
    strategy: Strategy


@dataclass(frozen=True, slots=True)
class OnScheduled:
    t: datetime
    strategy: Strategy


@dataclass(frozen=True, slots=True)
class OnExpiry:
    t: datetime
    instrument: Instrument


@dataclass(frozen=True, slots=True)
class OnFill:
    t: datetime
    strategy: Strategy
    fill: Fill


Event = OnStart | OnEnd | OnScheduled | OnExpiry | OnFill


def _event_kind(event: Event) -> str:
    match event:
        case OnStart():
            return "start"
        case OnExpiry():
            return "expiry"
        case OnScheduled():
            return "schedule"
        case OnEnd():
            return "end"
        case OnFill():
            return "fill"  # fmt: off


# ---------------------------------------------------------------------------
# Event queue
# ---------------------------------------------------------------------------


@dataclass(order=True, frozen=True, slots=True)
class QueueEntry:
    t: datetime
    priority: int = field(compare=True)
    seq: int = field(compare=True)
    event: Event = field(compare=False)


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[QueueEntry] = []
        self._seen: set[Event] = set()
        self._seq = 0

    def push(self, event: Event) -> None:
        if event in self._seen:
            return
        t = event.t
        priority = _PRIORITY.get(_event_kind(event), 99)
        entry = QueueEntry(t, priority, self._seq, event)
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
# Event context
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventContext:
    """Information visible to a strategy at time t."""

    t: datetime
    bars: dict[str, pl.LazyFrame]
    signals: dict[str, pl.LazyFrame]
    position: PositionView
    portfolio: PortfolioView


# ---------------------------------------------------------------------------
# Strategy output & ABC
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategyOutput:
    targets: dict[Instrument, float] = field(default_factory=dict)
    orders: list[Order] = field(default_factory=list)


_EMPTY = StrategyOutput()


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

    def on_expiry(self, _ctx: EventContext, _instrument: Instrument) -> StrategyOutput:
        return _EMPTY

    def on_fill(self, _ctx: EventContext, _fill: Fill) -> None:
        pass


# ---------------------------------------------------------------------------
# Schemas (placeholder)
# ---------------------------------------------------------------------------


class Schemas:
    BARS_RATE = pl.Schema({"t": pl.Datetime("us", "UTC"), "rate": pl.Float64})
    BARS_SPOT = pl.Schema({"t": pl.Datetime("us", "UTC"), "px_mark": pl.Float64})
    BARS_FUTURES_CALENDAR = pl.Schema(
        {"t": pl.Datetime("us", "UTC"), "px_mark": pl.Float64}
    )
    BARS_FUTURES_PERPETUAL = pl.Schema(
        {"t": pl.Datetime("us", "UTC"), "px_mark": pl.Float64}
    )
    BARS_OPTION = pl.Schema({"t": pl.Datetime("us", "UTC"), "px_mark": pl.Float64})  # fmt: off


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BacktestResult:
    fills: list[tuple[str, Fill]]
    positions: dict[str, dict[Instrument, float]]
    target_history: dict[str, list[tuple[datetime, dict[Instrument, float]]]]

    @property
    def fill_count(self) -> int:
        return len(self.fills)

    @property
    def instruments_traded(self) -> set[Instrument]:
        return {f.i for _, f in self.fills}

    @property
    def strategy_names(self) -> list[str]:
        return list(self.target_history.keys())


# ---------------------------------------------------------------------------
# Backtest (fluent frozen dataclass)
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

    # --- Builder methods ---

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

    # --- Run ---

    def run(self, t0: datetime, tf: datetime) -> BacktestResult:
        positions: dict[str, dict[Instrument, float]] = \
            {n: {} for n in self.strategies}  # fmt: off
        target_history: dict[str, list[tuple[datetime, dict[Instrument, float]]]] = \
            {n: [] for n in self.strategies}  # fmt: off
        all_fills: list[tuple[str, Fill]] = []

        # --- Helpers ---

        def _bars_dict(t: datetime) -> dict[str, pl.LazyFrame]:
            out: dict[str, pl.LazyFrame] = {}
            for key, lf in [
                ("rate", self.bars_rate),
                ("spot", self.bars_spot),
                ("futures_calendar", self.bars_futures_calendar),
                ("futures_perpetual", self.bars_futures_perpetual),
                ("option", self.bars_option),
            ]:
                if lf is not None:
                    out[key] = lf.filter(pl.col("t") <= t)
            return out

        def _signals_dict(t: datetime) -> dict[str, pl.LazyFrame]:
            return {k: lf.filter(pl.col("t") <= t) for k, lf in self.signals.items()}

        def _make_ctx(t: datetime, strat_name: str) -> EventContext:
            return EventContext(
                t=t,
                bars=_bars_dict(t),
                signals=_signals_dict(t),
                position=PositionView(dict(positions[strat_name])),
                portfolio=PortfolioView({n: dict(p) for n, p in positions.items()}),
            )

        def _apply_fill(strat_name: str, fill: Fill) -> None:
            pos = positions[strat_name]
            current = pos.get(fill.i, 0.0)
            new_qty = current + fill.q  # q is signed
            if abs(new_qty) < 1e-12:
                pos.pop(fill.i, None)
            else:
                pos[fill.i] = new_qty

        def _get_mark_price(_instrument: Instrument) -> float:
            return 1.0  # placeholder

        # --- Seed queue ---

        queue = EventQueue()

        for _, strat in self.strategies.items():
            queue.push(OnStart(t0, strat))
            queue.push(OnEnd(tf, strat))
            for t in strat.rule.between(t0, tf, inc=True):
                queue.push(OnScheduled(t, strat))

        # --- Main loop ---

        with tqdm(total=len(queue), desc="Backtesting") as pbar:
            while queue:
                batch = queue.pop_all_at_t()
                t = batch[0].t
                if t > tf:
                    break
                pbar.update(len(batch))

                # 1. Dispatch events, collect outputs per strategy.
                strat_orders: dict[str, list[Order]] = defaultdict(list)
                strat_targets: dict[str, dict[Instrument, float]] = {}

                for entry in batch:
                    ev = entry.event
                    match ev:
                        case OnStart(strategy=s):
                            ctx = _make_ctx(t, s.name)
                            out = s.on_start(ctx)
                        case OnExpiry(instrument=inst):
                            # Fire for ALL strategies holding this instrument.
                            for sn, pos in positions.items():
                                if inst in pos:
                                    s = self.strategies[sn]
                                    ctx = _make_ctx(t, sn)
                                    out = s.on_expiry(ctx, inst)
                                    strat_targets[sn] = out.targets
                                    strat_orders[sn].extend(out.orders)
                            continue  # already dispatched per-strategy above
                        case OnScheduled(strategy=s):
                            ctx = _make_ctx(t, s.name)
                            out = s.on_schedule(ctx)
                        case OnEnd(strategy=s):
                            ctx = _make_ctx(t, s.name)
                            out = s.on_end(ctx)
                        case OnFill(strategy=s, fill=fill):
                            ctx = _make_ctx(t, s.name)
                            s.on_fill(ctx, fill)
                            continue  # informational, no output
                        case _:
                            continue

                    strat_targets[s.name] = out.targets
                    strat_orders[s.name].extend(out.orders)

                # 2. Record targets.
                for sn, tgt in strat_targets.items():
                    target_history[sn].append((t, tgt))

                # 3. Net orders per instrument across strategies.
                all_orders: list[tuple[str, Order]] = []
                for sn, orders in strat_orders.items():
                    for o in orders:
                        all_orders.append((sn, o))

                net_qty: dict[Instrument, float] = defaultdict(float)
                per_inst: dict[Instrument, list[tuple[str, Order]]] = defaultdict(list)
                for sn, o in all_orders:
                    net_qty[o.i] += o.q
                    per_inst[o.i].append((sn, o))

                # 4. Fill.
                for inst, _nq in net_qty.items():
                    px = _get_mark_price(inst)
                    for sn, o in per_inst[inst]:
                        fill = Fill(t=t, i=inst, o=o, q=o.q, px=px)
                        _apply_fill(sn, fill)
                        all_fills.append((sn, fill))
                        queue.push(OnFill(t, self.strategies[sn], fill))

                # 5. Inject expiry events for newly held options.
                for sn, pos in positions.items():
                    for inst in list(pos.keys()):
                        if isinstance(inst, OptionInstrument) and inst.expiry <= tf:
                            queue.push(OnExpiry(inst.expiry, inst))

        return BacktestResult(
            fills=all_fills,
            positions=positions,
            target_history=target_history,
        )


# ---------------------------------------------------------------------------
# Example strategies (frozen slotted dataclasses)
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
    target_qty: float = 1.0

    def on_schedule(self, ctx: EventContext) -> StrategyOutput:
        target_inst = _select_option(ctx, self.option_kind)
        targets: dict[Instrument, float] = {target_inst: self.target_qty}
        orders: list[Order] = []

        # Close current holdings if different instrument.
        for inst in ctx.position.instruments:
            if inst != target_inst:
                orders.append(MarketOrder(t=ctx.t, i=inst, q=-ctx.position.qty(inst)))

        # Open new if not held at desired qty.
        delta = self.target_qty - ctx.position.qty(target_inst)
        if abs(delta) > 1e-12:
            orders.append(MarketOrder(t=ctx.t, i=target_inst, q=delta))

        return StrategyOutput(targets=targets, orders=orders)


@dataclass(frozen=True, slots=True)
class Straddle(Strategy):
    target_qty: float = 1.0

    def on_schedule(self, ctx: EventContext) -> StrategyOutput:
        call = _select_option(ctx, OptionKind.CALL)
        put = replace(call, kind=OptionKind.PUT)
        targets: dict[Instrument, float] = {call: self.target_qty, put: self.target_qty}
        orders: list[Order] = []

        # Close legs that differ.
        for inst in ctx.position.instruments:
            if inst not in targets:
                orders.append(MarketOrder(t=ctx.t, i=inst, q=-ctx.position.qty(inst)))

        # Open/adjust legs.
        for inst, tgt_q in targets.items():
            delta = tgt_q - ctx.position.qty(inst)
            if abs(delta) > 1e-12:
                orders.append(MarketOrder(t=ctx.t, i=inst, q=delta))

        return StrategyOutput(targets=targets, orders=orders)
