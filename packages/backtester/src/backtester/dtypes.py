from dataclasses import dataclass
from datetime import datetime, time
from enum import StrEnum
from typing import Literal


from utils import checks


@dataclass(frozen=True, slots=True)
class OptionInstrument:
    exchange: str
    base: str
    quote: str
    strike: float
    listing: datetime
    expiry: datetime
    kind: Literal["c", "p"]

    def __post_init__(self) -> None:
        checks.require(
            checks.is_gt("strike", self.strike, "0", 0),
            checks.is_utc("listing", self.listing),
            checks.is_utc("expiry", self.expiry),
            checks.is_lt("listing", self.listing, "expiry", self.expiry),
            checks.is_in("kind", self.kind, ("c", "p")),
        )
        checks.recommend(
            checks.has_time("listing", self.listing, time(8, 0, 0)),
            checks.has_time("expiry", self.expiry, time(8, 0, 0)),
        )


@dataclass(frozen=True, slots=True)
class SpotInstrument:
    exchange: str
    base: str
    quote: str


Instrument = OptionInstrument | SpotInstrument


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class MarketOrder:
    t: datetime
    i: Instrument
    q: float
    side: Side

    def __post_init__(self) -> None:
        checks.require(
            checks.is_utc("t", self.t),
            checks.is_gt("q", self.q, "0", 0),
        )


@dataclass(frozen=True, slots=True)
class LimitOrder:
    t: datetime
    i: Instrument
    q: float
    side: Side
    px_limit: float

    def __post_init__(self) -> None:
        checks.require(
            checks.is_utc("t", self.t),
            checks.is_gt("q", self.q, "0", 0),
            checks.is_gt("px_limit", self.px_limit, "0", 0),
        )


Order = MarketOrder | LimitOrder


@dataclass(frozen=True, slots=True)
class Fill:
    t: datetime
    i: Instrument
    o: Order
    q: float
    side: Side
    px: float

    def __post_init__(self) -> None:
        checks.require(
            checks.is_utc("t", self.t),
            checks.is_gt("q", self.q, "0", 0),
            checks.is_gt("px", self.px, "0", 0),
        )
