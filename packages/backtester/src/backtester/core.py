# packages/backtester/src/backtester/backtester.py
"""Option backtester module."""

# NOTE [@CLAUDE]: This file is just an initial sketch, don't consider it strongly.

from __future__ import annotations
from datetime import datetime, timedelta
from dateutil import rrule
from tqdm import tqdm
from typing import Literal, Mapping, Protocol

import polars as pl

from backtester.dtypes import Fill, Instrument, Order
from utils import checks
from utils import schemas


class Strategy(Protocol):
    """Logic-object defining when and how to trade."""

    rule: rrule.rrule
    """Defines when to consider trading."""

    def get_target_position(self) -> Mapping[Instrument, float]:
        """Returns the target position as a mapping from Instrument to quantity.

        If "no trade", the current position holdings will be returned."""
        ...


class SingleOption:
    option_exchange: str
    option_base: str
    option_quote: str
    option_kind: Literal["c", "p"]

    target_delta: float
    target_tenor: timedelta

    rule: rrule.rrule
    hedge: Literal["notional", "delta"] | None = None

    def get_target_position(self) -> Mapping[Instrument, float]: ...


class Straddle:
    option_exchange: str
    option_base: str
    option_quote: str

    target_delta: float
    target_tenor: timedelta

    rule: rrule.rrule

    def get_target_position(self) -> Mapping[Instrument, float]: ...


class Backtester:
    """Logic-object for backtesting a trading strategy against historical data."""

    def __init__(self, lf: pl.LazyFrame) -> None:
        """Initialize Backtester."""
        checks.require(checks.has_schema(lf, schemas.BARS_PRICED))
        self.lf = lf

    def run(
        self, strategy: Strategy, t0: datetime, tf: datetime, dt: timedelta
    ) -> None:
        checks.require(
            checks.is_utc("t0", t0),
            checks.is_utc("tf", tf),
            checks.is_gt("dt", dt, "0", timedelta()),
        )

        # Track active positions (time-series, not just current quantity)
        self.positions_active: pl.LazyFrame = pl.LazyFrame(schemas.POSITION)
        # Collect closed positions
        self.positions_closed: pl.LazyFrame = pl.LazyFrame(schemas.POSITION)
        # Collect orders and fills
        self.orders: list[Order] = []
        self.fills: list[Fill] = []

        for t in tqdm(strategy.rule.between(t0, tf, inc=True)):
            print(t.isoformat())

    def emit_orders(self, position_target: Mapping[Instrument, float]) -> list[Order]:
        """Compare target position to active positions to determine orders."""
        ...

    def emit_fills(self, orders: list[Order]) -> list[Fill]:
        """Compare orders to market data to determine fills.

        For now, assume all orders fill at best bid/ask."""
        ...
