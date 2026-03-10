# packages/backtester/src/backtester/notes/architecture.py
"""Loose, WIP sketch on backtester architecture."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from dateutil.rrule import rrule, rruleset, DAILY
from tqdm import tqdm

import polars as pl

from backtester.dtypes import Order, Fill
from backtester.io import (
    get_bars_spot,  # noqa: F401
    get_bars_option,  # noqa: F401
    get_target_option,  # noqa: F401
    _build_lf_priced,  # noqa: F401
)
from utils import schemas


"""
GENERAL ITEMS [@CLAUDE]:

Requirements:
- each Backtest has zero or more Strategies
- each Strategy manages zero or more related positions, e.g.:
    - straddle:  call and put with same strike and expiry
    - butterfly: call, put (x2), and call with different strikes and same expiry
- each position represents (time, instrument) => quantity

Notes:
- position/order-generation logic should be independent between Strategies
- the Backtester combines Strategies and nets their positions/orders

Questions:
- should we do accounting in terms of positions or orders?
"""


@dataclass(frozen=True, slots=True)
class Strategy:
    """Logic-object defining when and how to trade."""

    name: str
    """Strategy name for reporting purposes."""
    rule: rrule
    """Recurrence rule defining *when* to consider trading."""

    # TODO [@CLAUDE]:
    # - Decide how to define trading logic, i.e. *how* to trade.
    # - Decide how to define multiple Strategies conforming to a common interface.
    # E.g., maybe a typing.Protocol with a method like
    #   `get_target_position(t: datetime, ...) -> Mapping[Instrument, float]`?
    ...


SCHEMA_POSITIONS_SPOT = schemas.BARS_SPOT | {"qty": pl.Float64()}
SCHEMA_POSITIONS_OPTION = schemas.BARS_OPTION | {"qty": pl.Float64()}


class Backtest:
    bars_rate: pl.LazyFrame
    bars_spot: pl.LazyFrame
    bars_option: pl.LazyFrame
    rules: rruleset = rruleset()
    strategies: list[Strategy] = []

    # NOTE [@CLAUDE]: We may want some way to map positions to the originating Strategy.
    closed_positions_spot = pl.LazyFrame(schema=SCHEMA_POSITIONS_SPOT)
    closed_positions_option = pl.LazyFrame(schema=SCHEMA_POSITIONS_OPTION)
    orders: dict[Strategy, list[Order]] = {}
    fills: dict[Strategy, list[Fill]] = {}

    def __init__(
        self,
        bars_rate: pl.LazyFrame,
        bars_spot: pl.LazyFrame,
        bars_option: pl.LazyFrame,
        strategies: list[Strategy],
    ) -> None:
        self.bars_rate = bars_rate
        self.bars_spot = bars_spot
        self.bars_option = bars_option
        self.strategies = strategies

    def run(self, t0: datetime, tf: datetime) -> None:
        """Evaluate backtest at each time in the rruleset."""

        # TODO [@CLAUDE]: Come up with a way to ensure we can trigger evaluation at
        # times not defined in the rruleset, e.g. upon any contract expiries that occur
        # for instruments chosen by the Strategy(ies).

        for strategy in self.strategies:
            self.rules.rrule(strategy.rule)

        # NOTE [@CLAUDE]: Should these be represented as LazyFrames?
        open_positions_spot = pl.LazyFrame(schema=SCHEMA_POSITIONS_SPOT)  # noqa: F841
        open_positions_option = pl.LazyFrame(schema=SCHEMA_POSITIONS_OPTION)  # noqa: F841

        _closed_positions_spot: list[pl.LazyFrame] = []
        _closed_positions_option: list[pl.LazyFrame] = []

        for t in tqdm(self.rules.xafter(t0, inc=True)):
            # ROUGHLY:
            # 1. Ask each Strategy for its target position / desired Orders at time `t`.
            # 2. Net across Strategies to get total target position / desired Orders.
            # 3. Compare to current position and emit Orders & Fills.
            # 4. For each open position to be closed, filter the position LazyFrame up
            #    to the time `t` and append the subset to the list of closed positions,
            #    then remove the position LazyFrame.
            # 5. For each new position to be opened, get bars for the corresponding
            #    Instrument from time `t` onward, and append/insert to open positions.

            # NOTE [@CLAUDE]:
            #   For now, our only required "Orders => Fills" logic is "instant fill at
            # best bid/ask". But let's also sketch ways to express "Order generators",
            # e.g. TWAP/VWAP. While not a requirement for now, it would be nice for our
            # code to be extensible to support this in the future.
            ...

        # NOTE [@CLAUDE]: Is this an efficient way to accumulate closed positions?
        self.closed_positions_spot = pl.concat(_closed_positions_spot)
        self.closed_positions_option = pl.concat(_closed_positions_option)


def main() -> None:
    # TODO [@CLAUDE]: Brainstorm cleaner Backtest user interfaces.
    Backtest(
        bars_rate=pl.LazyFrame(schema=schemas.PATH_RATE),
        bars_spot=pl.LazyFrame(schema=schemas.BARS_SPOT),
        bars_option=pl.LazyFrame(schema=schemas.BARS_OPTION),
        strategies=[
            Strategy("Strategy 1", rrule(DAILY)),
            Strategy("Strategy 2", rrule(DAILY)),
        ],
    ).run(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
