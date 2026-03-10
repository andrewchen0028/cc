"""Sketch 1: Strategy-Autonomous — strategies own their state.

Strategies are self-contained: they manage their own positions, orders, and fills.
The engine is a thin event dispatcher that only tracks positions for expiry detection.

Tradeoffs:
  + Simplest engine implementation — strategies have full autonomy.
  + Easy to reason about: each strategy is a self-contained unit.
  + Post-hoc analysis via strategy-owned order/fill lists.
  - Strategies must reimplement position tracking (duplication across strategies).
  - No cross-strategy awareness: each strategy operates in isolation.
  - Signals are untyped (flat dict) — easy to misuse, no IDE autocompletion.

@CLAUDE: This is the "batteries NOT included" variant. The engine is minimal,
pushing complexity to strategies. Good for simple, independent strategies.
Bad for portfolios where strategies need to cooperate or share state.
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
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
# Events — flat `signals` dict on every event
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OnSchedule:
    """Fired at each rrule-scheduled time."""

    t: datetime
    bars_priced: pl.LazyFrame
    signals: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class OnExpiry:
    """Fired when a held instrument expires."""

    t: datetime
    instrument: Instrument
    bars_priced: pl.LazyFrame
    signals: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class OnFill:
    """Fired when an order is filled."""

    t: datetime
    fill: Fill
    bars_priced: pl.LazyFrame
    signals: Mapping[str, Any]


Event = OnSchedule | OnExpiry | OnFill


# ---------------------------------------------------------------------------
# Strategy ABC — autonomous, owns its own state
# ---------------------------------------------------------------------------


class Strategy(ABC):
    """Strategy base class. Strategies own their positions and history.

    @CLAUDE: No on_start/on_end here — lifecycle is minimal. Strategies that
    need initialization do it in __init__; teardown is post-hoc via self.fills.
    """

    name: str
    rule: rrule

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    @abstractmethod
    def on_schedule(self, event: OnSchedule) -> list[Order]:
        """Handle a scheduled evaluation. Return orders to submit."""
        ...

    def on_expiry(self, event: OnExpiry) -> list[Order]:  # noqa: ARG002
        """Handle an instrument expiry. Default: no action."""
        return []

    def on_fill(self, event: OnFill) -> list[Order]:  # noqa: ARG002
        """Handle a fill confirmation. Default: no action."""
        return []


# ---------------------------------------------------------------------------
# Concrete strategies — self-contained with own position tracking
# ---------------------------------------------------------------------------


class SingleOption(Strategy):
    """Roll into the option closest to target delta/tenor each period.

    Owns its state: `self.held`, `self.orders`, `self.fills`.
    """

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

        # Strategy-owned state.
        self.held: OptionInstrument | None = None
        self.orders: list[Order] = []
        self.fills: list[Fill] = []

    def _select_instrument(self, t: datetime, bars: pl.LazyFrame) -> OptionInstrument:
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

    def on_schedule(self, event: OnSchedule) -> list[Order]:
        # Example signal access: event.signals["momentum"]
        orders: list[Order] = []
        target = self._select_instrument(event.t, event.bars_priced)

        if self.held is not None and self.held != target:
            orders.append(MarketOrder(
                t=event.t, i=self.held, q=abs(self.qty),
                side=Side.SELL if self.qty > 0 else Side.BUY,
            ))
            self.held = None

        if self.held is None:
            orders.append(MarketOrder(
                t=event.t, i=target, q=abs(self.qty),
                side=Side.BUY if self.qty > 0 else Side.SELL,
            ))
            self.held = target

        self.orders.extend(orders)
        return orders

    def on_expiry(self, event: OnExpiry) -> list[Order]:
        if self.held == event.instrument:
            self.held = None
        return []

    def on_fill(self, event: OnFill) -> list[Order]:
        self.fills.append(event.fill)
        return []


class Straddle(Strategy):
    """Maintain a straddle (call + put at same strike/expiry)."""

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

        self.held_call: OptionInstrument | None = None
        self.held_put: OptionInstrument | None = None
        self.orders: list[Order] = []
        self.fills: list[Fill] = []

    def on_schedule(self, event: OnSchedule) -> list[Order]:
        orders: list[Order] = []

        lf_rate = event.bars_priced.select("time_start", "time_end", "rate").unique()
        lf_spot = event.bars_priced.select(
            "time_start", "time_end",
            pl.lit(self.spot_instrument.exchange).alias("exchange"),
            pl.lit(self.spot_instrument.base).alias("base"),
            pl.lit(self.spot_instrument.quote).alias("quote"),
            pl.col("spot").alias("px_bid"),
            pl.col("spot").alias("px_ask"),
            pl.col("spot").alias("px_mark"),
        ).unique(["time_start", "time_end"])  # fmt: off
        lf_option = event.bars_priced.select(schemas.BARS_OPTION.names()).unique()

        call = get_target_option(
            lf_rate=lf_rate, lf_spot=lf_spot, lf_option=lf_option,
            option_exchange=self.option_exchange, option_base=self.option_base,
            option_quote=self.option_quote, option_kind="c",
            spot_instrument=self.spot_instrument, target_time=event.t,
            target_delta=self.target_delta, target_tenor=self.target_tenor,
        )
        put = OptionInstrument(
            exchange=call.exchange, base=call.base, quote=call.quote,
            strike=call.strike, listing=call.listing, expiry=call.expiry,
            kind="p",
        )

        for held, new in [(self.held_call, call), (self.held_put, put)]:
            if held is not None and held != new:
                orders.append(MarketOrder(
                    t=event.t, i=held, q=abs(self.qty),
                    side=Side.SELL if self.qty > 0 else Side.BUY,
                ))

        if self.held_call != call:
            orders.append(MarketOrder(
                t=event.t, i=call, q=abs(self.qty),
                side=Side.BUY if self.qty > 0 else Side.SELL,
            ))
        if self.held_put != put:
            orders.append(MarketOrder(
                t=event.t, i=put, q=abs(self.qty),
                side=Side.BUY if self.qty > 0 else Side.SELL,
            ))

        self.held_call = call
        self.held_put = put
        self.orders.extend(orders)
        return orders

    def on_expiry(self, event: OnExpiry) -> list[Order]:
        if event.instrument in (self.held_call, self.held_put):
            self.held_call = None
            self.held_put = None
        return []

    def on_fill(self, event: OnFill) -> list[Order]:
        self.fills.append(event.fill)
        return []


# ---------------------------------------------------------------------------
# Event queue — min-heap with dedup set for expiries
# ---------------------------------------------------------------------------


@dataclass(order=True, frozen=True, slots=True)
class QueueEntry:
    t: datetime
    seq: int = field(compare=True)
    event_factory: object = field(compare=False)


class EventQueue:
    """Min-heap of (time, event) pairs with expiry dedup."""

    def __init__(self) -> None:
        self._heap: list[QueueEntry] = []
        self._seq = 0
        self._seen_expiries: set[tuple[str, datetime, Instrument]] = set()

    def push(self, t: datetime, event_factory: object) -> None:
        heapq.heappush(self._heap, QueueEntry(t, self._seq, event_factory))
        self._seq += 1

    def push_expiry(
        self,
        t: datetime,
        strategy_name: str,
        instrument: Instrument,
        event_factory: Callable[[], tuple[str, OnExpiry]],
    ) -> None:
        """Push an expiry event, deduplicating by (strategy, time, instrument)."""
        key = (strategy_name, t, instrument)
        if key not in self._seen_expiries:
            self._seen_expiries.add(key)
            self.push(t, event_factory)

    def pop(self) -> QueueEntry:
        return heapq.heappop(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)


# ---------------------------------------------------------------------------
# Position book — engine-side, for expiry detection only
# ---------------------------------------------------------------------------


@dataclass
class PositionBook:
    """Engine tracks positions only for expiry injection."""

    holdings: dict[Instrument, float] = field(default_factory=dict)

    def apply_fill(self, fill: Fill) -> None:
        sign = 1.0 if fill.side == Side.BUY else -1.0
        current = self.holdings.get(fill.i, 0.0)
        new_qty = current + sign * fill.q
        if abs(new_qty) < 1e-12:
            self.holdings.pop(fill.i, None)
        else:
            self.holdings[fill.i] = new_qty

    def expiring_at(self, t: datetime) -> list[Instrument]:
        return [
            inst for inst in self.holdings
            if isinstance(inst, OptionInstrument) and inst.expiry <= t
        ]


# ---------------------------------------------------------------------------
# Signal resolver — converts LazyFrame to flat dict at time t
# ---------------------------------------------------------------------------


def _resolve_signals(
    signals_lf: pl.LazyFrame | None,
    t: datetime,
) -> dict[str, Any]:
    """Filter signals LazyFrame to time_end == t, return {signal_name: value}.

    @CLAUDE: Signals LazyFrame has schema (time_end, signal_name, value).
    At each timestep we materialize just the row(s) matching t.
    """
    if signals_lf is None:
        return {}
    df = signals_lf.filter(pl.col("time_end") == t).collect()
    return dict(zip(df["signal_name"].to_list(), df["value"].to_list()))


# ---------------------------------------------------------------------------
# Backtester — flat constructor, thin dispatcher
# ---------------------------------------------------------------------------


class Backtest:
    """Event-driven backtest engine — strategy-autonomous variant.

    @CLAUDE: The engine is deliberately thin. It dispatches events and fills
    orders, but strategies own their positions and history. The engine only
    tracks positions for expiry detection.
    """

    def __init__(
        self,
        bars_priced: pl.LazyFrame,
        strategies: list[Strategy],
        signals: pl.LazyFrame | None = None,
    ) -> None:
        self.bars_priced = bars_priced
        self.strategies = strategies
        self.signals = signals
        self.strategy_map: dict[str, Strategy] = {s.name: s for s in strategies}
        self.books: dict[str, PositionBook] = {
            s.name: PositionBook() for s in strategies
        }

    def _fill_order(self, order: MarketOrder) -> Fill:
        """Instant fill at best bid/ask. MarketOrder only."""
        # Placeholder: look up price from bars_priced at order.t for order.i
        px = 0.0
        return Fill(
            t=order.t, i=order.i, o=order,
            q=order.q, side=order.side, px=px,
        )

    def run(self, t0: datetime, tf: datetime) -> None:
        queue = EventQueue()
        signals_at: dict[datetime, dict[str, Any]] = {}

        # --- Seed schedule events ---
        for strategy in self.strategies:
            for t in strategy.rule.between(t0, tf, inc=True):
                _name = strategy.name

                def _make_schedule(
                    t: datetime = t, name: str = _name,
                ) -> tuple[str, OnSchedule]:
                    sigs = signals_at.get(t)
                    if sigs is None:
                        sigs = _resolve_signals(self.signals, t)
                        signals_at[t] = sigs
                    return name, OnSchedule(
                        t=t, bars_priced=self.bars_priced, signals=sigs,
                    )

                queue.push(t, _make_schedule)

        # --- Process events ---
        with tqdm(total=len(queue), desc="Backtesting") as pbar:
            while queue:
                entry = queue.pop()
                pbar.update(1)

                if entry.t > tf:
                    break

                strategy_name, event = entry.event_factory()  # type: ignore[operator]
                strategy = self.strategy_map[strategy_name]

                # --- Dispatch ---
                if isinstance(event, OnSchedule):
                    new_orders = strategy.on_schedule(event)
                elif isinstance(event, OnExpiry):
                    new_orders = strategy.on_expiry(event)
                elif isinstance(event, OnFill):
                    new_orders = strategy.on_fill(event)
                else:
                    new_orders = []

                # --- Process orders -> fills ---
                for order in new_orders:
                    if not isinstance(order, MarketOrder):
                        continue  # MarketOrder only in this sketch
                    fill = self._fill_order(order)

                    self.books[strategy_name].apply_fill(fill)

                    sigs = signals_at.get(fill.t)
                    if sigs is None:
                        sigs = _resolve_signals(self.signals, fill.t)
                        signals_at[fill.t] = sigs

                    def _make_fill(
                        t: datetime = fill.t, f: Fill = fill,
                        name: str = strategy_name, s: dict[str, Any] = sigs,
                    ) -> tuple[str, OnFill]:
                        return name, OnFill(
                            t=t, fill=f, bars_priced=self.bars_priced, signals=s,
                        )

                    queue.push(fill.t, _make_fill)

                # --- Expiry injection with dedup ---
                for s_name, book in self.books.items():
                    for inst in book.expiring_at(entry.t):

                        def _make_expiry(
                            t: datetime = entry.t, i: Instrument = inst,
                            name: str = s_name,
                        ) -> tuple[str, OnExpiry]:
                            sigs = signals_at.get(t, {})
                            return name, OnExpiry(
                                t=t, instrument=i,
                                bars_priced=self.bars_priced, signals=sigs,
                            )

                        queue.push_expiry(entry.t, s_name, inst, _make_expiry)


# ---------------------------------------------------------------------------
# Example usage — flat constructor with external signal
# ---------------------------------------------------------------------------


def main() -> None:
    spot = SpotInstrument(exchange="deribit", base="BTC", quote="USD")

    # External signal: a LazyFrame with (time_end, signal_name, value).
    momentum_signal = pl.LazyFrame(
        schema=pl.Schema({
            "time_end": pl.Datetime(time_zone="UTC"),
            "signal_name": pl.String(),
            "value": pl.Float64(),
        }),
    )

    Backtest(
        bars_priced=pl.LazyFrame(schema=schemas.BARS_PRICED),
        signals=momentum_signal,
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

    # Post-hoc analysis: strategies own their order/fill history.
    # short_put = backtest.strategy_map["short_put_25d"]
    # print(f"Orders: {len(short_put.orders)}, Fills: {len(short_put.fills)}")
