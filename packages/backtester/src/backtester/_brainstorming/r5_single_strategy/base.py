"""NOTE [@CLAUDE]:
- Simplifying to single-strategy for now.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from dateutil.rrule import rrule, rruleset, MONTHLY

import heapq as hq
import polars as pl

from backtester.dtypes import (
    ClosedPosition,
    Fill,
    Instrument,
    LimitOrder,
    MarketOrder,
    OpenPosition,
    OptionInstrument,
    OptionKind,
    Order,
    OrderID,
    PositionID,
    SpotInstrument,
)
from utils import checks, schemas


@dataclass(frozen=True, slots=True)
class OnScheduled:
    strategy: Strategy


@dataclass(frozen=True, slots=True)
class OnExpiry:
    instrument: Instrument


@dataclass(frozen=True, slots=True)
class OnFill:
    fill: Fill


Event = OnScheduled | OnExpiry | OnFill


@dataclass(frozen=True, slots=True, order=True)
class QueueEntry:
    fire_at: datetime
    seq: int
    event: Event = field(compare=False)


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[QueueEntry] = []
        self._seen: set[tuple[datetime, Event]] = set()
        self._seq = 0

    def push(self, fire_at: datetime, event: Event) -> None:
        key = (fire_at, event)
        if key in self._seen:
            return

        hq.heappush(self._heap, QueueEntry(fire_at, self._seq, event))
        self._seq += 1
        self._seen.add(key)

    def peek(self) -> QueueEntry | None:
        return self._heap[0] if self._heap else None

    def pop_batch(self) -> list[QueueEntry]:
        if not self._heap:
            return []

        first = hq.heappop(self._heap)
        batch = [first]
        self._seen.discard((first.fire_at, first.event))

        while self._heap and self._heap[0].fire_at == first.fire_at:
            entry = hq.heappop(self._heap)
            self._seen.discard((entry.fire_at, entry.event))
            batch.append(entry)

        return batch

    def __len__(self):
        return len(self._heap)

    def __bool__(self):
        return bool(self._heap)


@dataclass(frozen=True, slots=True)
class EventContext:
    t: datetime
    bars_rate: pl.LazyFrame | None = None
    bars_spot: pl.LazyFrame | None = None
    bars_priced: pl.LazyFrame | None = None
    signals: dict[str, pl.LazyFrame] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StrategyOutput:
    orders: list[Order] = field(default_factory=list)
    cancels: list[OrderID] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Strategy(ABC):
    rule: rrule | rruleset

    def on_event(self, event: Event, ctx: EventContext) -> StrategyOutput:
        match event:
            case OnScheduled():
                return self._on_scheduled(ctx)
            case OnExpiry(instrument=instrument):
                return self._on_expiry(ctx, instrument)
            case OnFill(fill=fill):
                return self._on_fill(ctx, fill)

    def _on_scheduled(self, ctx: EventContext) -> StrategyOutput: ...
    def _on_expiry(self, ctx: EventContext, i: Instrument) -> StrategyOutput: ...
    def _on_fill(self, ctx: EventContext, fill: Fill) -> StrategyOutput: ...


@dataclass(frozen=True, slots=True)
class Schemas:
    PATH_RATE = schemas.PATH_RATE
    BARS_SPOT = schemas.BARS_SPOT
    BARS_PRICED = schemas.BARS_PRICED


@dataclass(frozen=True, slots=True)
class BacktestResult:
    fills: list[Fill] = field(default_factory=list)
    positions_closed: list[ClosedPosition] = field(default_factory=list)
    positions_open: list[OpenPosition] = field(default_factory=list)
    orders_unfilled: list[Order] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TryFillResult:
    orders_completed: list[Order] = field(default_factory=list)
    orders_remaining: list[Order] = field(default_factory=list)
    fills_complete: list[Fill] = field(default_factory=list)
    fills_partial: list[Fill] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Backtest:
    bars_rate: pl.LazyFrame | None = None
    bars_spot: pl.LazyFrame | None = None
    bars_priced: pl.LazyFrame | None = None
    signals: dict[str, pl.LazyFrame] = field(default_factory=dict)

    def with_bars_rate(self, bars: pl.LazyFrame) -> Backtest:
        checks.require(
            checks.is_none("bars_rate", self.bars_rate),
            checks.has_schema(bars, Schemas.PATH_RATE),
        )
        return replace(self, bars_rate=bars)

    def with_bars_spot(self, bars: pl.LazyFrame) -> Backtest:
        checks.require(
            checks.is_none("bars_spot", self.bars_spot),
            checks.has_schema(bars, Schemas.BARS_SPOT),
        )
        return replace(self, bars_spot=bars)

    def with_bars_priced(self, bars: pl.LazyFrame) -> Backtest:
        checks.require(
            checks.is_none("bars_priced", self.bars_priced),
            checks.has_schema(bars, Schemas.BARS_PRICED),
        )
        return replace(self, bars_priced=bars)

    def with_signal(self, name: str, signal: pl.LazyFrame) -> Backtest:
        checks.require(checks.not_in(name, name, self.signals.keys()))
        return replace(self, signals={**self.signals, name: signal})

    def run(self, strategy: Strategy, t0: datetime, tf: datetime) -> BacktestResult:
        fills: list[Fill] = []
        orders_active: dict[OrderID, Order] = {}
        positions_closed: dict[PositionID, ClosedPosition] = {}
        positions_active: dict[PositionID, OpenPosition] = {}

        queue = EventQueue()

        for t in strategy.rule.between(t0, tf, inc=True):
            queue.push(t, OnScheduled(strategy))

        while queue:
            if (top := queue.peek()) is None or top.fire_at > tf:
                break
            t = top.fire_at

            ctx = EventContext(
                t,
                self._filter(t, self.bars_rate),
                self._filter(t, self.bars_spot),
                self._filter(t, self.bars_priced),
                {k: self._filter(t, v) for k, v in self.signals.items()},
            )

            # --- Dispatch events to strategy ---
            # NOTE: All cases do the same thing (strategy dispatches internally),
            # so no need to match here.
            for entry in queue.pop_batch():
                output = strategy.on_event(entry.event, ctx)
                for oid in output.cancels:
                    orders_active.pop(oid, None)
                orders_active.update({o.id: o for o in output.orders})

            # --- Try to fill active orders ---
            # NOTE: No netting because we've simplified to single-strategy.
            # We assume the strategy will not emit crossing orders.
            orders_remaining: dict[OrderID, Order] = {}
            for order in orders_active.values():
                result = self._try_fill(t, order)
                orders_remaining.update({o.id: o for o in result.orders_remaining})
                for fill in result.fills_complete + result.fills_partial:
                    fills.append(fill)
                    queue.push(t, OnFill(fill))
            orders_active = orders_remaining

            # --- Update positions from new fills ---
            for fill in fills:
                if fill.t != t:
                    continue

                # Find existing open position for this instrument.
                existing_pid: PositionID | None = None
                for pid, pos in positions_active.items():
                    if pos.i == fill.i:
                        existing_pid = pid
                        break

                if existing_pid is not None:
                    pos = positions_active.pop(existing_pid)
                    new_q = pos.q + fill.q
                    if new_q == 0.0:
                        closed = ClosedPosition(t0=pos.t0, tf=t, i=pos.i, q=pos.q)
                        positions_closed[closed.id] = closed
                    else:
                        updated = OpenPosition(t0=pos.t0, i=pos.i, q=new_q)
                        positions_active[updated.id] = updated
                else:
                    opened = OpenPosition(t0=t, i=fill.i, q=fill.q)
                    positions_active[opened.id] = opened

            # --- Schedule expiry events for option positions ---
            for position in positions_active.values():
                match position.i:
                    case SpotInstrument():
                        pass
                    case OptionInstrument():
                        queue.push(position.i.expiry, OnExpiry(position.i))

        return BacktestResult(
            fills=fills,
            positions_closed=list(positions_closed.values()),
            positions_open=list(positions_active.values()),
            orders_unfilled=list(orders_active.values()),
        )

    def _filter(self, t: datetime, lf: pl.LazyFrame | None) -> pl.LazyFrame | None:
        if lf is None:
            return None
        return lf.filter(pl.col("time_end") <= t)

    def _get_fill_price(self, t: datetime, i: Instrument, q: float) -> float | None:
        px_col = "px_ask" if q > 0 else "px_bid"  # buy at ask, sell at bid
        match i:
            case SpotInstrument(exchange=exch, base=base, quote=quote):
                lf = self.bars_spot
                if lf is None:
                    return None
                row = lf \
                    .filter(
                        (pl.col("time_end") <= t)
                        & (pl.col("exchange") == exch)
                        & (pl.col("base") == base)
                        & (pl.col("quote") == quote)
                    ) \
                    .sort("time_end") \
                    .last() \
                    .collect()  # fmt: off
            case OptionInstrument(
                exchange=exch,
                base=base,
                quote=quote,
                strike=strike,
                listing=listing,
                expiry=expiry,
                kind=kind,
            ):
                lf = self.bars_priced
                if lf is None:
                    return None
                row = lf \
                    .filter(
                        (pl.col("time_end") <= t)
                        & (pl.col("exchange") == exch)
                        & (pl.col("base") == base)
                        & (pl.col("quote") == quote)
                        & (pl.col("strike") == strike)
                        & (pl.col("listing") == listing)
                        & (pl.col("expiry") == expiry)
                        & (pl.col("kind") == kind.value)
                    ) \
                    .sort("time_end") \
                    .last() \
                    .collect()  # fmt: off
        if row.is_empty():
            return None
        return row[px_col][0]

    def _try_fill(self, t: datetime, order: Order) -> TryFillResult:
        match order:
            case MarketOrder(i=i, q=q):
                px = self._get_fill_price(t, i, q)
                if px is None:
                    return TryFillResult(orders_remaining=[order])
                fill = Fill(t=t, i=i, o=order, q=q, px=px)
                return TryFillResult(
                    orders_completed=[order],
                    fills_complete=[fill],
                )
            case LimitOrder():
                raise NotImplementedError("Limit order fills not yet implemented")


@dataclass(frozen=True, slots=True)
class PrintStrategy(Strategy):
    def _on_scheduled(self, ctx: EventContext) -> StrategyOutput:
        print(ctx.t)
        if ctx.bars_priced is not None:
            # NOTE: this should likely be replaced by `backtester.io.get_target_option`?
            target = ctx.bars_priced \
                .filter(
                    (pl.col("exchange") == "drbt")
                    & (pl.col("base") == "btc")
                    & (pl.col("quote") == "usd")
                    & (pl.col("kind") == "c")
                ) \
                .with_columns(
                    (pl.col("expiry") - pl.col("time_end")).alias("tenor"),
                ) \
                .with_columns(
                    (pl.col("time_end") - ctx.t).abs().alias("abs_err_time"),
                    (pl.col("delta") - 0.5).abs().alias("abs_err_delta"),
                    (pl.col("tenor") - timedelta(days=30)).abs().alias("abs_err_tenor"),
                ) \
                .sort(["abs_err_time", "abs_err_tenor", "abs_err_delta"]) \
                .head(1) \
                .collect()  # fmt: off
            if not target.is_empty():
                kind_raw = target["kind"].item()
                kind = OptionKind.CALL if kind_raw == "c" else OptionKind.PUT
                option = OptionInstrument(
                    exchange=target["exchange"].item(),
                    base=target["base"].item(),
                    quote=target["quote"].item(),
                    strike=target["strike"].item(),
                    listing=target["listing"].item(),
                    expiry=target["expiry"].item(),
                    kind=kind,
                )
                print(f"  target: {option}")
        return StrategyOutput()

    def _on_expiry(self, ctx: EventContext, i: Instrument) -> StrategyOutput:
        return StrategyOutput()

    def _on_fill(self, ctx: EventContext, fill: Fill) -> StrategyOutput:
        return StrategyOutput()


def main() -> None:
    from backtester.io import _build_lf_priced
    from utils.samplers import (
        get_path_rate,
        get_paths_mark,
        to_bars_spot,
        to_bars_option,
    )

    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    tf = datetime(2025, 12, 31, tzinfo=timezone.utc)

    lf_rate = get_path_rate(t0, tf)
    lf_mark = get_paths_mark(t0, tf)
    lf_spot = to_bars_spot(lf_mark, exchanges="drbt", quotes="usd")
    lf_option = to_bars_option(lf_mark, exchange="drbt", base="btc", quote="usd")

    lf_priced = _build_lf_priced(
        lf_rate,
        lf_spot,
        lf_option,
        option_exchange="drbt",
        option_base="btc",
        option_quote="usd",
        spot_exchange="drbt",
        spot_base="btc",
        spot_quote="usd",
    )

    _result = Backtest() \
        .with_bars_rate(lf_rate) \
        .with_bars_spot(lf_spot) \
        .with_bars_priced(lf_priced) \
        .run(PrintStrategy(rrule(MONTHLY, dtstart=t0)), t0, tf)  # fmt: off

    print(_result)


if __name__ == "__main__":
    main()
