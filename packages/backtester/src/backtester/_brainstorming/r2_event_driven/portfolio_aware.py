"""Sketch 3: Portfolio-Aware Cooperation — strategies see the full portfolio.

Strategies hold a read-only PositionView (backed by the engine's book) and
receive the full portfolio view (all strategies' positions) via EventContext.
Strategies can condition their logic on what other strategies hold.

Tradeoffs:
  + Strategies can cooperate: e.g., hedge strategy sees what others hold.
  + Rich EventContext: `.bars_at_t()`, `.get_signal(name)`, `.get_price(inst)`.
  + on_roll lifecycle hook: engine detects close+open at same timestep.
  + Position-as-LazyFrame history: each position is a slice of bars_priced.
  - Most complex engine — tracks portfolio view, detects rolls, manages slices.
  - Strategies can create implicit coupling via portfolio view.
  - PositionView is a read-only snapshot — stale if strategy caches it.

@CLAUDE: This is the "portfolio-first" variant. Best for multi-strategy
portfolios where strategies need awareness of each other. The roll detection
is unique to this sketch and useful for accounting (e.g., roll P&L attribution).
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
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
# Position views — read-only windows into engine state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionView:
    """Read-only view of a single strategy's positions.

    @CLAUDE: This is a snapshot — strategies get a fresh one each event.
    The `current` property returns the position dict at call time.
    """

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
    """Read-only view of the entire portfolio (all strategies' positions).

    @CLAUDE: Strategies can inspect what other strategies hold. This enables
    hedging strategies, portfolio-level risk limits, etc.
    """

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
        return sum(
            pos.get(instrument, 0.0) for pos in self._positions.values()
        )


# ---------------------------------------------------------------------------
# Event context — rich accessor object
# ---------------------------------------------------------------------------


class EventContext:
    """Rich context object passed to strategy event handlers.

    @CLAUDE: Unlike the flat dict (autonomous) or typed bundle (engine_managed),
    this provides method-based access. Signals are pre-filtered to time_end == t.
    The portfolio view gives cross-strategy awareness.
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

    def bars_at_t(self) -> pl.LazyFrame:
        """Return bars_priced filtered to time_end == t."""
        return self._bars_priced.filter(pl.col("time_end") == self._t)

    def get_signal(self, name: str) -> pl.LazyFrame:
        """Get a named signal LazyFrame, pre-filtered to time_end == t.

        @CLAUDE: Signals are registered via .with_signal(name, lf) on the
        builder. They're filtered once per timestep and cached in the context.
        """
        if name not in self._signals:
            raise KeyError(f"Signal '{name}' not registered. "
                           f"Available: {list(self._signals.keys())}")
        return self._signals[name]

    def get_price(self, instrument: Instrument) -> float:
        """Look up the mark price for an instrument at time t.

        Placeholder — would filter bars_priced for the instrument at time t.
        """
        # Pseudocode: filter bars_priced for instrument at time_end == t
        _ = instrument
        return 0.0


# ---------------------------------------------------------------------------
# Events — carry EventContext
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


@dataclass(frozen=True, slots=True)
class OnRoll:
    """Fired when the engine detects a roll (close old + open new at same t).

    @CLAUDE: The engine detects rolls by looking at orders from a single
    strategy at the same timestep: if there's a close on instrument A and
    an open on instrument B, it fires on_roll(old=A, new=B, ctx).
    This is a post-processing hook — the orders have already been filled.
    """

    old: Instrument
    new: Instrument
    ctx: EventContext


Event = OnStart | OnEnd | OnSchedule | OnExpiry | OnFill | OnRoll


# ---------------------------------------------------------------------------
# Strategy ABC — portfolio-aware
# ---------------------------------------------------------------------------


class Strategy(ABC):
    """Strategy base class with full lifecycle and portfolio awareness."""

    name: str
    rule: rrule

    def on_start(self, event: OnStart) -> list[Order]:  # noqa: ARG002
        return []

    def on_end(self, event: OnEnd) -> list[Order]:  # noqa: ARG002
        return []

    @abstractmethod
    def on_schedule(self, event: OnSchedule) -> list[Order]:
        ...

    def on_expiry(self, event: OnExpiry) -> list[Order]:  # noqa: ARG002
        return []

    def on_fill(self, event: OnFill) -> list[Order]:  # noqa: ARG002
        return []

    def on_roll(self, event: OnRoll) -> None:  # noqa: ARG002
        """Post-processing hook for roll detection. No orders returned.

        @CLAUDE: on_roll is informational — strategies can use it for
        accounting, logging, or adjusting internal state. It's called after
        fills are processed, so returning orders would be out of sequence.
        """
        pass


# ---------------------------------------------------------------------------
# Concrete strategies — portfolio-aware
# ---------------------------------------------------------------------------


class SingleOption(Strategy):
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
        ctx = event.ctx
        orders: list[Order] = []

        # Access signal via context.
        # momentum = ctx.get_signal("momentum")

        # Access portfolio: check net exposure across all strategies.
        # net_btc = ctx.portfolio.net_qty(some_instrument)

        target = self._select_instrument(ctx.t, ctx.bars_at_t())

        # Read held position from the position view (engine-backed).
        held: OptionInstrument | None = None
        for inst in ctx.position.instruments:
            if isinstance(inst, OptionInstrument):
                held = inst
                break

        if held is not None and held != target:
            orders.append(MarketOrder(
                t=ctx.t, i=held, q=abs(self.qty),
                side=Side.SELL if self.qty > 0 else Side.BUY,
            ))

        if held is None or held != target:
            orders.append(MarketOrder(
                t=ctx.t, i=target, q=abs(self.qty),
                side=Side.BUY if self.qty > 0 else Side.SELL,
            ))

        return orders

    def on_roll(self, event: OnRoll) -> None:
        """Track roll for accounting."""
        # e.g., log roll from event.old -> event.new
        pass


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

    def on_schedule(self, event: OnSchedule) -> list[Order]:
        ctx = event.ctx
        orders: list[Order] = []

        bars = ctx.bars_at_t()
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
            lf_rate=lf_rate, lf_spot=lf_spot, lf_option=lf_option,
            option_exchange=self.option_exchange, option_base=self.option_base,
            option_quote=self.option_quote, option_kind="c",
            spot_instrument=self.spot_instrument, target_time=ctx.t,
            target_delta=self.target_delta, target_tenor=self.target_tenor,
        )
        put = OptionInstrument(
            exchange=call.exchange, base=call.base, quote=call.quote,
            strike=call.strike, listing=call.listing, expiry=call.expiry,
            kind="p",
        )

        held_call: OptionInstrument | None = None
        held_put: OptionInstrument | None = None
        for inst in ctx.position.instruments:
            if isinstance(inst, OptionInstrument) and inst.kind == "c":
                held_call = inst
            elif isinstance(inst, OptionInstrument) and inst.kind == "p":
                held_put = inst

        for held, new in [(held_call, call), (held_put, put)]:
            if held is not None and held != new:
                orders.append(MarketOrder(
                    t=ctx.t, i=held, q=abs(self.qty),
                    side=Side.SELL if self.qty > 0 else Side.BUY,
                ))

        if held_call != call:
            orders.append(MarketOrder(
                t=ctx.t, i=call, q=abs(self.qty),
                side=Side.BUY if self.qty > 0 else Side.SELL,
            ))
        if held_put != put:
            orders.append(MarketOrder(
                t=ctx.t, i=put, q=abs(self.qty),
                side=Side.BUY if self.qty > 0 else Side.SELL,
            ))

        return orders


# ---------------------------------------------------------------------------
# Event queue — min-heap with dedup set for expiries
# ---------------------------------------------------------------------------


@dataclass(order=True, frozen=True, slots=True)
class QueueEntry:
    t: datetime
    seq: int = field(compare=True)
    event_factory: object = field(compare=False)


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[QueueEntry] = []
        self._seq = 0
        self._seen_expiries: set[tuple[str, datetime, Instrument]] = set()

    def push(self, t: datetime, event_factory: object) -> None:
        heapq.heappush(self._heap, QueueEntry(t, self._seq, event_factory))
        self._seq += 1

    def push_expiry(
        self, t: datetime, strategy_name: str, instrument: Instrument,
        event_factory: object,
    ) -> None:
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
# Position-as-LazyFrame history
# ---------------------------------------------------------------------------


@dataclass
class PositionSlice:
    """A position slice: bars_priced from fill_t to close_t, with qty.

    @CLAUDE: Mirrors the base.py pattern. Each open position is a "live" slice
    of bars_priced. On close, the slice is finalized and appended to closed list.
    """

    strategy: str
    instrument: Instrument
    qty: float
    fill_t: datetime
    close_t: datetime | None = None

    def to_lazyframe(self, bars_priced: pl.LazyFrame) -> pl.LazyFrame:
        """Materialize this slice as a LazyFrame."""
        lf = bars_priced.filter(pl.col("time_end") >= self.fill_t)
        if self.close_t is not None:
            lf = lf.filter(pl.col("time_end") <= self.close_t)
        return lf.with_columns(pl.lit(self.qty).alias("qty"))


# ---------------------------------------------------------------------------
# Roll detection
# ---------------------------------------------------------------------------


def _detect_rolls(
    orders: list[Order],
    position_before: Mapping[Instrument, float],
) -> list[tuple[Instrument, Instrument]]:
    """Detect rolls: close of old instrument + open of new instrument.

    @CLAUDE: A roll is detected when, in a single batch of orders:
    - An instrument goes from non-zero to zero (close)
    - A different instrument goes from zero to non-zero (open)
    Both must happen in the same order batch from the same strategy.
    """
    closes: list[Instrument] = []
    opens: list[Instrument] = []

    for order in orders:
        if not isinstance(order, MarketOrder):
            continue
        current_qty = position_before.get(order.i, 0.0)
        sign = 1.0 if order.side == Side.BUY else -1.0
        new_qty = current_qty + sign * order.q

        if abs(current_qty) > 1e-12 and abs(new_qty) < 1e-12:
            closes.append(order.i)
        elif abs(current_qty) < 1e-12 and abs(new_qty) > 1e-12:
            opens.append(order.i)

    # Pair closes with opens — simple 1:1 pairing.
    rolls: list[tuple[Instrument, Instrument]] = []
    for old, new in zip(closes, opens):
        rolls.append((old, new))
    return rolls


# ---------------------------------------------------------------------------
# Backtester — builder pattern, portfolio-aware
# ---------------------------------------------------------------------------


class BacktestBuilder:
    """Builder for portfolio-aware backtest."""

    def __init__(self) -> None:
        self._bars_priced: pl.LazyFrame | None = None
        self._signals: dict[str, pl.LazyFrame] = {}
        self._strategies: list[Strategy] = []

    def with_bars(self, bars_priced: pl.LazyFrame) -> BacktestBuilder:
        self._bars_priced = bars_priced
        return self

    def with_signal(self, name: str, lf: pl.LazyFrame) -> BacktestBuilder:
        """Register a named signal LazyFrame.

        @CLAUDE: Signals are stored by name. At each timestep, the engine
        filters each to time_end == t and makes them available via
        ctx.get_signal(name).
        """
        self._signals[name] = lf
        return self

    def with_strategy(self, strategy: Strategy) -> BacktestBuilder:
        self._strategies.append(strategy)
        return self

    def build(self) -> Backtest:
        if self._bars_priced is None:
            raise ValueError("bars_priced is required")
        if not self._strategies:
            raise ValueError("At least one strategy is required")
        return Backtest(
            bars_priced=self._bars_priced,
            signals=dict(self._signals),
            strategies=list(self._strategies),
        )


class Backtest:
    """Event-driven backtest engine — portfolio-aware variant.

    @CLAUDE: Key differences from other sketches:
    - Strategies get a PositionView (own positions) and PortfolioView (all).
    - Roll detection fires on_roll() after fills are processed.
    - Position history stored as LazyFrame slices (position-as-LazyFrame).
    """

    def __init__(
        self,
        bars_priced: pl.LazyFrame,
        signals: dict[str, pl.LazyFrame],
        strategies: list[Strategy],
    ) -> None:
        self.bars_priced = bars_priced
        self.signals = signals
        self.strategies = strategies
        self.strategy_map: dict[str, Strategy] = {s.name: s for s in strategies}
        self.positions: dict[str, dict[Instrument, float]] = {
            s.name: {} for s in strategies
        }

        # Position-as-LazyFrame history.
        self.open_slices: dict[str, dict[Instrument, PositionSlice]] = {
            s.name: {} for s in strategies
        }
        self.closed_slices: list[PositionSlice] = []

    @classmethod
    def builder(cls) -> BacktestBuilder:
        return BacktestBuilder()

    def _make_context(self, t: datetime, strategy_name: str) -> EventContext:
        """Build an EventContext for strategy_name at time t."""
        # Filter signals to time_end == t.
        filtered_signals = {
            name: lf.filter(pl.col("time_end") == t)
            for name, lf in self.signals.items()
        }

        position = PositionView(dict(self.positions[strategy_name]))
        portfolio = PortfolioView({
            name: dict(pos) for name, pos in self.positions.items()
        })

        return EventContext(
            t=t,
            bars_priced=self.bars_priced,
            position=position,
            portfolio=portfolio,
            signals=filtered_signals,
        )

    def _apply_fill(self, strategy_name: str, fill: Fill) -> None:
        pos = self.positions[strategy_name]
        sign = 1.0 if fill.side == Side.BUY else -1.0
        current = pos.get(fill.i, 0.0)
        new_qty = current + sign * fill.q

        if abs(new_qty) < 1e-12:
            pos.pop(fill.i, None)
            # Close the position slice.
            slices = self.open_slices[strategy_name]
            if fill.i in slices:
                s = slices.pop(fill.i)
                s.close_t = fill.t
                self.closed_slices.append(s)
        else:
            pos[fill.i] = new_qty
            # Open a position slice if new.
            if abs(current) < 1e-12:
                self.open_slices[strategy_name][fill.i] = PositionSlice(
                    strategy=strategy_name,
                    instrument=fill.i,
                    qty=new_qty,
                    fill_t=fill.t,
                )

    def _fill_order(self, order: MarketOrder) -> Fill:
        px = 0.0  # placeholder
        return Fill(
            t=order.t, i=order.i, o=order,
            q=order.q, side=order.side, px=px,
        )

    def _expiring_at(self, strategy_name: str, t: datetime) -> list[Instrument]:
        return [
            inst for inst in self.positions[strategy_name]
            if isinstance(inst, OptionInstrument) and inst.expiry <= t
        ]

    def run(self, t0: datetime, tf: datetime) -> None:
        queue = EventQueue()

        # --- Seed on_start ---
        for strategy in self.strategies:
            _name = strategy.name

            def _make_start(name: str = _name) -> tuple[str, OnStart]:
                return name, OnStart(ctx=self._make_context(t0, name))

            queue.push(t0, _make_start)

        # --- Seed schedule events ---
        for strategy in self.strategies:
            for t in strategy.rule.between(t0, tf, inc=True):
                _name = strategy.name

                def _make_schedule(
                    t: datetime = t, name: str = _name,
                ) -> tuple[str, OnSchedule]:
                    return name, OnSchedule(ctx=self._make_context(t, name))

                queue.push(t, _make_schedule)

        # --- Seed on_end ---
        for strategy in self.strategies:
            _name = strategy.name

            def _make_end(name: str = _name) -> tuple[str, OnEnd]:
                return name, OnEnd(ctx=self._make_context(tf, name))

            queue.push(tf, _make_end)

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
                if isinstance(event, OnStart):
                    new_orders = strategy.on_start(event)
                elif isinstance(event, OnEnd):
                    new_orders = strategy.on_end(event)
                elif isinstance(event, OnSchedule):
                    new_orders = strategy.on_schedule(event)
                elif isinstance(event, OnExpiry):
                    new_orders = strategy.on_expiry(event)
                elif isinstance(event, OnFill):
                    new_orders = strategy.on_fill(event)
                elif isinstance(event, OnRoll):
                    strategy.on_roll(event)
                    new_orders = []
                else:
                    new_orders = []

                # --- Snapshot position before fills (for roll detection) ---
                position_before = dict(self.positions[strategy_name])

                # --- Process orders -> fills ---
                for order in new_orders:
                    if not isinstance(order, MarketOrder):
                        continue
                    fill = self._fill_order(order)
                    self._apply_fill(strategy_name, fill)

                    def _make_fill(
                        f: Fill = fill, name: str = strategy_name,
                    ) -> tuple[str, OnFill]:
                        return name, OnFill(
                            fill=f, ctx=self._make_context(f.t, name),
                        )

                    queue.push(fill.t, _make_fill)

                # --- Roll detection ---
                if new_orders:
                    rolls = _detect_rolls(new_orders, position_before)
                    for old_inst, new_inst in rolls:

                        def _make_roll(
                            old: Instrument = old_inst,
                            new: Instrument = new_inst,
                            name: str = strategy_name,
                            t: datetime = entry.t,
                        ) -> tuple[str, OnRoll]:
                            return name, OnRoll(
                                old=old, new=new,
                                ctx=self._make_context(t, name),
                            )

                        queue.push(entry.t, _make_roll)

                # --- Expiry injection ---
                for s_name in self.positions:
                    for inst in self._expiring_at(s_name, entry.t):

                        def _make_expiry(
                            t: datetime = entry.t, i: Instrument = inst,
                            name: str = s_name,
                        ) -> tuple[str, OnExpiry]:
                            return name, OnExpiry(
                                instrument=i,
                                ctx=self._make_context(t, name),
                            )

                        queue.push_expiry(entry.t, s_name, inst, _make_expiry)

        # Finalize any remaining open slices.
        for s_name, slices in self.open_slices.items():
            for inst, s in slices.items():
                s.close_t = tf
                self.closed_slices.append(s)


# ---------------------------------------------------------------------------
# Example usage — builder pattern with named signals
# ---------------------------------------------------------------------------


def main() -> None:
    spot = SpotInstrument(exchange="deribit", base="BTC", quote="USD")

    # Named signal: a LazyFrame registered by name.
    momentum_lf = pl.LazyFrame(schema=pl.Schema({
        "time_end": pl.Datetime(time_zone="UTC"),
        "value": pl.Float64(),
    }))

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

    bt = Backtest.builder() \
        .with_bars(pl.LazyFrame(schema=schemas.BARS_PRICED)) \
        .with_signal("momentum", momentum_lf) \
        .with_strategy(short_put) \
        .with_strategy(straddle) \
        .build() \
        .run(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 12, 31, tzinfo=timezone.utc),
        )  # fmt: off

    # Post-hoc: closed position slices as LazyFrames.
    # for s in bt.closed_slices:
    #     lf = s.to_lazyframe(bars_priced)
