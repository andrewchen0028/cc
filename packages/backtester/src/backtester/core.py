# packages/backtester/src/backtester/backtester.py
"""Option backtester module."""

from __future__ import annotations
from datetime import timedelta
from typing import Literal, Mapping, Protocol

from backtester.dtypes import Instrument


class Strategy(Protocol):
    def skip_trade(self) -> bool: ...
    def get_target_position(self) -> Mapping[Instrument, float]: ...


class SingleOptionStrategy:
    option_exchange: str
    option_base: str
    option_quote: str
    option_kind: Literal["c", "p"]

    target_delta: float
    target_tenor: timedelta

    hedge: Literal["notional", "delta"] | None = None

    def skip_trade(self) -> bool: ...
    def get_target_position(self) -> Mapping[Instrument, float]: ...


class Backtester:
    def __init__(self) -> None: ...
    def run(self, strategy: Strategy) -> None: ...
