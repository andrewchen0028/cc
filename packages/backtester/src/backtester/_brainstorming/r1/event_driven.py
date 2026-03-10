"""Branch B: Event-Callback Architecture.

Strategies receive typed events (`OnSchedule`, `OnExpiry`, `OnFill`) and return
orders in response. The Backtester manages an event queue sorted by time,
merging rrule schedule events with contract expiry events.

Pros:
  - Flexible: handles irregular scheduling (expiries, fills) naturally.
  - Clean separation: events are the only interface between engine and strategy.
  - Strategies have full control over order type/timing.
Cons:
  - Strategies must manage their own position state.
  - More complex to implement simple strategies (boilerplate for events).
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

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
# Events
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OnSchedule:
    """Fired at each rrule-scheduled time."""

    t: datetime
    bars_priced: pl.LazyFrame


@dataclass(frozen=True, slots=True)
class OnExpiry:
    """Fired when a held instrument expires.

    @CLAUDE NOTE: This directly addresses base.py's TODO about triggering
    evaluation at times outside the rrule schedule (e.g., contract expiries).
    The engine detects upcoming expiries from the position book and injects
    OnExpiry events into the queue.
    """

    t: datetime
    instrument: Instrument
    bars_priced: pl.LazyFrame


@dataclass(frozen=True, slots=True)
class OnFill:
    """Fired when an order is filled (for strategies that want confirmation)."""

    t: datetime
    fill: Fill
    bars_priced: pl.LazyFrame


Event = OnSchedule | OnExpiry | OnFill


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------


class Strategy(ABC):
    """Event-driven strategy base class.

    @CLAUDE NOTE: Using ABC instead of Protocol here because strategies need
    mutable state (position tracking) and we want to provide default no-op
    implementations for event handlers the strategy doesn't care about.
    A Protocol can't provide defaults without a mixin, which is more awkward.
    """

    name: str
    rule: rrule

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
# Concrete strategies
# ---------------------------------------------------------------------------


class SingleOption(Strategy):
    """Roll into the option closest to target delta/tenor each period.

    Manages its own position: tracks the currently held instrument and quantity.
    On expiry, rolls into a new contract.
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

        # Mutable state: currently held instrument.
        self.held: OptionInstrument | None = None

    def _select_instrument(self, t: datetime, bars: pl.LazyFrame) -> OptionInstrument:
        """Find the option closest to target delta/tenor at time t."""
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
        orders: list[Order] = []
        target = self._select_instrument(event.t, event.bars_priced)

        # Close existing position if instrument changed.
        if self.held is not None and self.held != target:
            orders.append(
                MarketOrder(
                    t=event.t,
                    i=self.held,
                    q=abs(self.qty),
                    side=Side.SELL if self.qty > 0 else Side.BUY,
                )
            )
            self.held = None

        # Open new position.
        if self.held is None:
            orders.append(
                MarketOrder(
                    t=event.t,
                    i=target,
                    q=abs(self.qty),
                    side=Side.BUY if self.qty > 0 else Side.SELL,
                )
            )
            self.held = target

        return orders

    def on_expiry(self, event: OnExpiry) -> list[Order]:
        """On expiry, clear the held reference — the contract is settled.

        The next on_schedule call will open a new position.
        """
        if self.held == event.instrument:
            self.held = None
        return []


class Straddle(Strategy):
    """Maintain a straddle (call + put at same strike/expiry).

    On each schedule event, checks whether current legs match the target and
    rolls if needed. On expiry of either leg, clears both legs.
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

        self.held_call: OptionInstrument | None = None
        self.held_put: OptionInstrument | None = None

    def on_schedule(self, event: OnSchedule) -> list[Order]:
        orders: list[Order] = []

        # Find the call leg; derive the put from it.
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
            lf_rate=lf_rate,
            lf_spot=lf_spot,
            lf_option=lf_option,
            option_exchange=self.option_exchange,
            option_base=self.option_base,
            option_quote=self.option_quote,
            option_kind="c",
            spot_instrument=self.spot_instrument,
            target_time=event.t,
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

        # Close old legs if rolling.
        for held, new in [(self.held_call, call), (self.held_put, put)]:
            if held is not None and held != new:
                orders.append(
                    MarketOrder(
                        t=event.t,
                        i=held,
                        q=abs(self.qty),
                        side=Side.SELL if self.qty > 0 else Side.BUY,
                    )
                )

        # Open new legs.
        if self.held_call != call:
            orders.append(
                MarketOrder(
                    t=event.t,
                    i=call,
                    q=abs(self.qty),
                    side=Side.BUY if self.qty > 0 else Side.SELL,
                )
            )
        if self.held_put != put:
            orders.append(
                MarketOrder(
                    t=event.t,
                    i=put,
                    q=abs(self.qty),
                    side=Side.BUY if self.qty > 0 else Side.SELL,
                )
            )

        self.held_call = call
        self.held_put = put
        return orders

    def on_expiry(self, event: OnExpiry) -> list[Order]:
        """Clear both legs on expiry of either — they share the same expiry."""
        if event.instrument in (self.held_call, self.held_put):
            self.held_call = None
            self.held_put = None
        return []


# ---------------------------------------------------------------------------
# Event queue
# ---------------------------------------------------------------------------


@dataclass(order=True, frozen=True, slots=True)
class QueueEntry:
    """Priority-queue entry, ordered by time then sequence number."""

    t: datetime
    seq: int = field(compare=True)  # tie-break for deterministic ordering
    event_factory: object = field(compare=False)  # () -> tuple[str, Event]


class EventQueue:
    """Min-heap of (time, event) pairs.

    @CLAUDE NOTE: Using a heap rather than a sorted list because we need to
    inject expiry events dynamically as positions are opened. A sorted list
    would require re-sorting; the heap gives O(log n) insertion.
    """

    def __init__(self) -> None:
        self._heap: list[QueueEntry] = []
        self._seq = 0

    def push(self, t: datetime, event_factory: object) -> None:
        heapq.heappush(self._heap, QueueEntry(t, self._seq, event_factory))
        self._seq += 1

    def pop(self) -> QueueEntry:
        return heapq.heappop(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)


# ---------------------------------------------------------------------------
# Position book (strategy-managed, engine-tracked for expiry detection)
# ---------------------------------------------------------------------------


@dataclass
class PositionBook:
    """Engine-side position tracker for expiry detection.

    Unlike Branch A where the engine owns position state, here the engine only
    tracks positions to know *when* instruments expire and inject OnExpiry events.
    Strategies manage their own state.
    """

    holdings: dict[Instrument, float] = field(default_factory=dict)

    def apply_fill(self, fill: Fill) -> None:
        inst = fill.i
        sign = 1.0 if fill.side == Side.BUY else -1.0
        current = self.holdings.get(inst, 0.0)
        new_qty = current + sign * fill.q
        if abs(new_qty) < 1e-12:
            self.holdings.pop(inst, None)
        else:
            self.holdings[inst] = new_qty

    def expiring_at(self, t: datetime) -> list[Instrument]:
        """Return instruments whose expiry is at time t."""
        return [
            inst
            for inst in self.holdings
            if isinstance(inst, OptionInstrument) and inst.expiry <= t
        ]


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------


class Backtest:
    """Event-driven backtest engine.

    Flow:
      1. Seed event queue with rrule schedule events for each strategy.
      2. Pop events in time order; dispatch to the relevant strategy.
      3. Strategy returns orders → engine fills them → emits OnFill events.
      4. Engine checks for upcoming expiries → injects OnExpiry events.
      5. Repeat until queue is exhausted or t > tf.
    """

    def __init__(
        self,
        bars_priced: pl.LazyFrame,
        strategies: list[Strategy],
    ) -> None:
        self.bars_priced = bars_priced
        self.strategies = strategies
        self.strategy_map: dict[str, Strategy] = {s.name: s for s in strategies}

        self.books: dict[str, PositionBook] = {
            s.name: PositionBook() for s in strategies
        }
        self.orders: list[Order] = []
        self.fills: list[Fill] = []

    def _fill_order(self, order: Order) -> Fill:
        """Instant fill at best bid/ask.

        @CLAUDE NOTE: In the event-driven model, more sophisticated fill models
        could emit orders into a separate "exchange simulator" that produces
        fills asynchronously (as events). For now, fills are synchronous.
        """
        # Pseudocode: look up price from bars_priced at order.t for order.i
        px = 0.0  # placeholder
        return Fill(
            t=order.t,
            i=order.i,
            o=order,
            q=order.q,
            side=order.side,
            px=px,
        )

    def run(self, t0: datetime, tf: datetime) -> None:
        queue = EventQueue()

        # --- Seed schedule events ---
        for strategy in self.strategies:
            for t in strategy.rule.between(t0, tf, inc=True):
                # Capture strategy name for the factory closure.
                _name = strategy.name

                def _make_schedule(
                    t: datetime = t, name: str = _name
                ) -> tuple[str, OnSchedule]:
                    return name, OnSchedule(t=t, bars_priced=self.bars_priced)

                queue.push(t, _make_schedule)

        # --- Process events ---
        # @CLAUDE NOTE: We use tqdm on the queue length, but the queue grows
        # dynamically (expiry/fill events), so the progress bar is approximate.
        with tqdm(total=len(queue), desc="Backtesting") as pbar:
            while queue:
                entry = queue.pop()
                pbar.update(1)

                if entry.t > tf:
                    break

                strategy_name, event = entry.event_factory()  # type: ignore[operator]
                strategy = self.strategy_map[strategy_name]

                # --- Dispatch event ---
                if isinstance(event, OnSchedule):
                    new_orders = strategy.on_schedule(event)
                elif isinstance(event, OnExpiry):
                    new_orders = strategy.on_expiry(event)
                elif isinstance(event, OnFill):
                    new_orders = strategy.on_fill(event)
                else:
                    new_orders = []

                # --- Process orders → fills ---
                for order in new_orders:
                    self.orders.append(order)
                    fill = self._fill_order(order)
                    self.fills.append(fill)

                    # Update engine-side position book.
                    self.books[strategy_name].apply_fill(fill)

                    # Emit OnFill event.
                    def _make_fill(
                        t: datetime = fill.t,
                        f: Fill = fill,
                        name: str = strategy_name,
                    ) -> tuple[str, OnFill]:
                        return name, OnFill(t=t, fill=f, bars_priced=self.bars_priced)

                    queue.push(fill.t, _make_fill)

                # --- Check for expiries and inject OnExpiry events ---
                # @CLAUDE NOTE: We check all strategies' books for expiring
                # instruments after processing each event. Expiry events are
                # only injected once (we could track "already injected" to
                # avoid duplicates; omitted here for sketch simplicity).
                for s_name, book in self.books.items():
                    for inst in book.expiring_at(entry.t):

                        def _make_expiry(
                            t: datetime = entry.t,
                            i: Instrument = inst,
                            name: str = s_name,
                        ) -> tuple[str, OnExpiry]:
                            return name, OnExpiry(
                                t=t,
                                instrument=i,
                                bars_priced=self.bars_priced,
                            )

                        queue.push(entry.t, _make_expiry)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------


def main() -> None:
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
