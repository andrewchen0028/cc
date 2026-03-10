"""Sketch 2: Engine-Managed State — engine owns positions, strategies are ~stateless.

The engine manages all position state, fill history, and signal resolution.
Strategies receive everything they need via event payloads and return orders.
Strategies are nearly pure functions: `(event, position) -> list[Order]`.

Tradeoffs:
  + Strategies are simple and testable — no mutable state to manage.
  + Engine handles netting: opposing orders from different strategies are netted.
  + Typed SignalBundle: user defines a frozen dataclass, gets IDE autocompletion.
  + Centralized fill history as LazyFrame — easy to analyze post-hoc.
  - Strategies can't cache intermediate computations across events.
  - SignalBundle requires user to define a subclass — more setup friction.
  - Engine is more complex: it must manage per-strategy positions, netting, lifecycle.

@CLAUDE: This is the "batteries included" variant. The engine does the heavy
lifting so strategies stay simple. The typed SignalBundle is the cleanest signal
API but requires more upfront work from the user.
"""

from __future__ import annotations

import heapq
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
# Typed signal bundle — user subclasses this
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SignalBundle:
    """Base class for typed signal payloads.

    @CLAUDE: User defines their own subclass with typed fields. The engine
    filters each LazyFrame field to time_end == t before passing to strategies.
    This gives type safety and IDE autocompletion, unlike a flat dict.
    """
    pass


@dataclass(frozen=True, slots=True)
class EmptySignals(SignalBundle):
    """Default when no signals are provided."""
    pass


# ---------------------------------------------------------------------------
# Events — position passed in, typed SignalBundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OnStart:
    """Fired once at t0 for initialization."""

    t: datetime
    bars_priced: pl.LazyFrame
    signals: SignalBundle


@dataclass(frozen=True, slots=True)
class OnEnd:
    """Fired once at tf for teardown/flatten."""

    t: datetime
    bars_priced: pl.LazyFrame
    position: Mapping[Instrument, float]
    signals: SignalBundle


@dataclass(frozen=True, slots=True)
class OnSchedule:
    """Fired at each rrule-scheduled time."""

    t: datetime
    bars_priced: pl.LazyFrame
    position: Mapping[Instrument, float]
    signals: SignalBundle


@dataclass(frozen=True, slots=True)
class OnExpiry:
    """Fired when a held instrument expires."""

    t: datetime
    instrument: Instrument
    bars_priced: pl.LazyFrame
    position: Mapping[Instrument, float]
    signals: SignalBundle


@dataclass(frozen=True, slots=True)
class OnFill:
    """Fired when an order is filled."""

    t: datetime
    fill: Fill
    bars_priced: pl.LazyFrame
    position: Mapping[Instrument, float]
    signals: SignalBundle


Event = OnStart | OnEnd | OnSchedule | OnExpiry | OnFill


# ---------------------------------------------------------------------------
# Strategy ABC — ~stateless, receives position via events
# ---------------------------------------------------------------------------


class Strategy(ABC):
    """Strategy base class. Engine owns position state; strategies are ~stateless.

    @CLAUDE: Strategies receive position as a read-only mapping in every event.
    The return type is always list[Order]. No mutable state needed.
    """

    name: str
    rule: rrule

    def on_start(self, event: OnStart) -> list[Order]:  # noqa: ARG002
        """Called once at t0. Default: no action."""
        return []

    def on_end(self, event: OnEnd) -> list[Order]:  # noqa: ARG002
        """Called once at tf. Default: no action (override to flatten)."""
        return []

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
# Concrete strategies — ~stateless
# ---------------------------------------------------------------------------


class SingleOption(Strategy):
    """Roll into the option closest to target delta/tenor each period.

    @CLAUDE: Note this strategy is ~stateless: it reads its held position from
    event.position rather than maintaining self.held. The only state it needs
    is the instrument selection parameters (immutable config).
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

    def _held_instrument(
        self, position: Mapping[Instrument, float],
    ) -> OptionInstrument | None:
        """Find our currently held option from the position map."""
        for inst, qty in position.items():
            if isinstance(inst, OptionInstrument) and abs(qty) > 1e-12:
                return inst
        return None

    def on_schedule(self, event: OnSchedule) -> list[Order]:
        orders: list[Order] = []
        target = self._select_instrument(event.t, event.bars_priced)
        held = self._held_instrument(event.position)

        # Example typed signal access: event.signals (user's SignalBundle subclass)

        if held is not None and held != target:
            orders.append(MarketOrder(
                t=event.t, i=held, q=abs(self.qty),
                side=Side.SELL if self.qty > 0 else Side.BUY,
            ))

        if held is None or held != target:
            orders.append(MarketOrder(
                t=event.t, i=target, q=abs(self.qty),
                side=Side.BUY if self.qty > 0 else Side.SELL,
            ))

        return orders

    def on_end(self, event: OnEnd) -> list[Order]:
        """Flatten position at end of backtest."""
        orders: list[Order] = []
        for inst, qty in event.position.items():
            if abs(qty) > 1e-12:
                orders.append(MarketOrder(
                    t=event.t, i=inst, q=abs(qty),
                    side=Side.SELL if qty > 0 else Side.BUY,
                ))
        return orders


class Straddle(Strategy):
    """Maintain a straddle (call + put at same strike/expiry). ~Stateless."""

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

        # Derive held state from position map.
        held_call: OptionInstrument | None = None
        held_put: OptionInstrument | None = None
        for inst in event.position:
            if isinstance(inst, OptionInstrument) and inst.kind == "c":
                held_call = inst
            elif isinstance(inst, OptionInstrument) and inst.kind == "p":
                held_put = inst

        # Close old legs if rolling.
        for held, new in [(held_call, call), (held_put, put)]:
            if held is not None and held != new:
                orders.append(MarketOrder(
                    t=event.t, i=held, q=abs(self.qty),
                    side=Side.SELL if self.qty > 0 else Side.BUY,
                ))

        # Open new legs.
        if held_call != call:
            orders.append(MarketOrder(
                t=event.t, i=call, q=abs(self.qty),
                side=Side.BUY if self.qty > 0 else Side.SELL,
            ))
        if held_put != put:
            orders.append(MarketOrder(
                t=event.t, i=put, q=abs(self.qty),
                side=Side.BUY if self.qty > 0 else Side.SELL,
            ))

        return orders

    def on_end(self, event: OnEnd) -> list[Order]:
        orders: list[Order] = []
        for inst, qty in event.position.items():
            if abs(qty) > 1e-12:
                orders.append(MarketOrder(
                    t=event.t, i=inst, q=abs(qty),
                    side=Side.SELL if qty > 0 else Side.BUY,
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
# Signal bundle resolver
# ---------------------------------------------------------------------------


def _resolve_signal_bundle(
    bundle: SignalBundle,
    t: datetime,
) -> SignalBundle:
    """Filter any LazyFrame fields in the bundle to time_end == t.

    @CLAUDE: This uses dataclass introspection to find LazyFrame fields and
    filter them. The result is a new frozen dataclass with filtered frames.
    Non-LazyFrame fields pass through unchanged.
    """
    if isinstance(bundle, EmptySignals):
        return bundle

    filtered: dict[str, Any] = {}
    for f in bundle.__dataclass_fields__:
        val = getattr(bundle, f)
        if isinstance(val, pl.LazyFrame):
            filtered[f] = val.filter(pl.col("time_end") == t)
        else:
            filtered[f] = val
    return type(bundle)(**filtered)


# ---------------------------------------------------------------------------
# Order netting
# ---------------------------------------------------------------------------


def _net_orders(
    all_orders: list[tuple[str, MarketOrder]],
) -> list[tuple[str, MarketOrder]]:
    """Net opposing orders on the same instrument across strategies.

    @CLAUDE: Netting collapses opposing BUY/SELL orders on the same instrument
    at the same time into a single residual order. Fills are attributed back
    to originating strategies proportionally.

    Example: Strategy A buys 2 BTC-C, Strategy B sells 1 BTC-C →
    net execution: buy 1 BTC-C. Fill of 1 attributed to A's buy, fill of 1
    attributed to B's sell, remaining 1 from A executed on market.
    """
    if not all_orders:
        return []

    # Group by instrument.
    by_inst: dict[Instrument, list[tuple[str, MarketOrder]]] = defaultdict(list)
    for name, order in all_orders:
        by_inst[order.i].append((name, order))

    result: list[tuple[str, MarketOrder]] = []
    for inst, orders in by_inst.items():
        # Sum signed quantities.
        net_qty = 0.0
        for _, order in orders:
            sign = 1.0 if order.side == Side.BUY else -1.0
            net_qty += sign * order.q

        if abs(net_qty) < 1e-12:
            continue  # Fully netted — no market execution needed.

        # Attribute residual to the strategies on the "winning" side.
        residual_side = Side.BUY if net_qty > 0 else Side.SELL
        remaining = abs(net_qty)

        for name, order in orders:
            if order.side == residual_side and remaining > 1e-12:
                exec_qty = min(order.q, remaining)
                result.append((name, MarketOrder(
                    t=order.t, i=inst, q=exec_qty, side=residual_side,
                )))
                remaining -= exec_qty

    return result


# ---------------------------------------------------------------------------
# Backtester — engine-managed state, dataclass config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Configuration for an engine-managed backtest."""

    bars_priced: pl.LazyFrame
    strategies: tuple[Strategy, ...]
    t0: datetime
    tf: datetime
    signals: SignalBundle = field(default_factory=EmptySignals)


class Backtest:
    """Event-driven backtest engine — engine-managed state variant.

    Engine owns:
      - Per-strategy position: dict[Instrument, float]
      - Fill history: list of (t, strategy, instrument, qty, side, px) rows
      - Signal resolution: filters SignalBundle fields to time_end == t

    @CLAUDE: The netting step collects all orders at the same timestep and
    nets opposing orders before execution. This is more realistic (a real
    exchange would net internally) and prevents artificial trading costs.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.strategy_map: dict[str, Strategy] = {
            s.name: s for s in config.strategies
        }
        self.positions: dict[str, dict[Instrument, float]] = {
            s.name: {} for s in config.strategies
        }
        self.fill_rows: list[dict[str, Any]] = []

    def _apply_fill(self, strategy_name: str, fill: Fill) -> None:
        pos = self.positions[strategy_name]
        sign = 1.0 if fill.side == Side.BUY else -1.0
        current = pos.get(fill.i, 0.0)
        new_qty = current + sign * fill.q
        if abs(new_qty) < 1e-12:
            pos.pop(fill.i, None)
        else:
            pos[fill.i] = new_qty

        self.fill_rows.append({
            "t": fill.t,
            "strategy": strategy_name,
            "instrument": str(fill.i),
            "qty": fill.q,
            "side": fill.side.value,
            "px": fill.px,
        })

    def _fill_order(self, order: MarketOrder) -> Fill:
        px = 0.0  # placeholder
        return Fill(
            t=order.t, i=order.i, o=order,
            q=order.q, side=order.side, px=px,
        )

    def _expiring_at(
        self, strategy_name: str, t: datetime,
    ) -> list[Instrument]:
        return [
            inst for inst in self.positions[strategy_name]
            if isinstance(inst, OptionInstrument) and inst.expiry <= t
        ]

    @property
    def fill_history(self) -> pl.LazyFrame:
        """Accumulated fill history as a LazyFrame."""
        if not self.fill_rows:
            return pl.LazyFrame(schema={
                "t": pl.Datetime(time_zone="UTC"),
                "strategy": pl.String(),
                "instrument": pl.String(),
                "qty": pl.Float64(),
                "side": pl.String(),
                "px": pl.Float64(),
            })
        return pl.DataFrame(self.fill_rows).lazy()

    def run(self) -> None:
        cfg = self.config
        queue = EventQueue()

        # --- Seed on_start ---
        signals_t0 = _resolve_signal_bundle(cfg.signals, cfg.t0)
        for strategy in cfg.strategies:
            _name = strategy.name

            def _make_start(
                name: str = _name, sigs: SignalBundle = signals_t0,
            ) -> tuple[str, OnStart]:
                return name, OnStart(
                    t=cfg.t0, bars_priced=cfg.bars_priced, signals=sigs,
                )

            queue.push(cfg.t0, _make_start)

        # --- Seed schedule events ---
        for strategy in cfg.strategies:
            for t in strategy.rule.between(cfg.t0, cfg.tf, inc=True):
                _name = strategy.name

                def _make_schedule(
                    t: datetime = t, name: str = _name,
                ) -> tuple[str, OnSchedule]:
                    sigs = _resolve_signal_bundle(cfg.signals, t)
                    return name, OnSchedule(
                        t=t, bars_priced=cfg.bars_priced,
                        position=dict(self.positions[name]),
                        signals=sigs,
                    )

                queue.push(t, _make_schedule)

        # --- Seed on_end ---
        for strategy in cfg.strategies:
            _name = strategy.name

            def _make_end(
                name: str = _name,
            ) -> tuple[str, OnEnd]:
                sigs = _resolve_signal_bundle(cfg.signals, cfg.tf)
                return name, OnEnd(
                    t=cfg.tf, bars_priced=cfg.bars_priced,
                    position=dict(self.positions[name]),
                    signals=sigs,
                )

            queue.push(cfg.tf, _make_end)

        # --- Process events ---
        with tqdm(total=len(queue), desc="Backtesting") as pbar:
            while queue:
                entry = queue.pop()
                pbar.update(1)

                if entry.t > cfg.tf:
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
                else:
                    new_orders = []

                # --- Collect orders for netting ---
                # @CLAUDE: In a full implementation, we'd batch all orders at the
                # same timestep before netting. Here we net per-event for simplicity.
                tagged_orders = [
                    (strategy_name, o) for o in new_orders
                    if isinstance(o, MarketOrder)
                ]
                netted = _net_orders(tagged_orders)

                for name, order in netted:
                    fill = self._fill_order(order)
                    self._apply_fill(name, fill)

                    def _make_fill(
                        t: datetime = fill.t, f: Fill = fill,
                        n: str = name,
                    ) -> tuple[str, OnFill]:
                        sigs = _resolve_signal_bundle(cfg.signals, t)
                        return n, OnFill(
                            t=t, fill=f, bars_priced=cfg.bars_priced,
                            position=dict(self.positions[n]),
                            signals=sigs,
                        )

                    queue.push(fill.t, _make_fill)

                # --- Expiry injection ---
                for s_name in self.positions:
                    for inst in self._expiring_at(s_name, entry.t):

                        def _make_expiry(
                            t: datetime = entry.t, i: Instrument = inst,
                            name: str = s_name,
                        ) -> tuple[str, OnExpiry]:
                            sigs = _resolve_signal_bundle(cfg.signals, t)
                            return name, OnExpiry(
                                t=t, instrument=i, bars_priced=cfg.bars_priced,
                                position=dict(self.positions[name]),
                                signals=sigs,
                            )

                        queue.push_expiry(entry.t, s_name, inst, _make_expiry)


# ---------------------------------------------------------------------------
# Example usage — dataclass config with typed SignalBundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MySignals(SignalBundle):
    """User-defined typed signal bundle."""

    momentum: pl.LazyFrame
    volatility_regime: pl.LazyFrame


def main() -> None:
    spot = SpotInstrument(exchange="deribit", base="BTC", quote="USD")

    # Typed signal bundle — each field is a LazyFrame filtered per-timestep.
    signals = MySignals(
        momentum=pl.LazyFrame(schema=pl.Schema({
            "time_end": pl.Datetime(time_zone="UTC"),
            "value": pl.Float64(),
        })),
        volatility_regime=pl.LazyFrame(schema=pl.Schema({
            "time_end": pl.Datetime(time_zone="UTC"),
            "regime": pl.String(),
        })),
    )

    config = BacktestConfig(
        bars_priced=pl.LazyFrame(schema=schemas.BARS_PRICED),
        strategies=(
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
        ),
        t0=datetime(2024, 1, 1, tzinfo=timezone.utc),
        tf=datetime(2024, 12, 31, tzinfo=timezone.utc),
        signals=signals,
    )

    bt = Backtest(config)
    bt.run()

    # Post-hoc: fill_history is a LazyFrame.
    # df_fills = bt.fill_history.collect()
