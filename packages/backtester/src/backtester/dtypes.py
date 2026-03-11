from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, time
from enum import StrEnum


from utils import checks
from utils.dtypes import ID


class OptionKind(StrEnum):
    CALL = "CALL"
    PUT = "PUT"


@dataclass(frozen=True, slots=True)
class OptionInstrument:
    exchange: str
    base: str
    quote: str
    strike: float
    listing: datetime
    expiry: datetime
    kind: OptionKind

    def __post_init__(self) -> None:
        checks.require(
            checks.is_gt("strike", self.strike, 0),
            checks.is_utc("listing", self.listing),
            checks.is_utc("expiry", self.expiry),
            checks.is_lt("listing", self.listing, self.expiry),
            checks.is_in("kind", self.kind, OptionKind.__members__.values()),
        )
        checks.recommend(
            checks.has_time("listing", self.listing, time(8, 0, 0)),
            checks.has_time("expiry", self.expiry, time(8, 0, 0)),
        )

    @property
    def id(self) -> ID[OptionInstrument]:
        return ID[OptionInstrument](self.__hash__())


@dataclass(frozen=True, slots=True)
class SpotInstrument:
    exchange: str
    base: str
    quote: str

    @property
    def id(self) -> ID[SpotInstrument]:
        return ID[SpotInstrument](self.__hash__())


Instrument = OptionInstrument | SpotInstrument


class TakerSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class MakerSide(StrEnum):
    BID = "BID"
    ASK = "ASK"


OrderSide = TakerSide | MakerSide


class PositionSide(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True, slots=True)
class MarketOrder:
    t: datetime
    i: Instrument
    q: float

    def __post_init__(self) -> None:
        checks.require(
            checks.is_utc("t", self.t),
            checks.is_ne("q", self.q, 0),
        )

    @property
    def id(self) -> ID[MarketOrder]:
        return ID[MarketOrder](self.__hash__())

    @property
    def side(self) -> TakerSide:
        return TakerSide.BUY if self.q > 0 else TakerSide.SELL


@dataclass(frozen=True, slots=True)
class LimitOrder:
    t: datetime
    i: Instrument
    q: float
    px_limit: float

    def __post_init__(self) -> None:
        checks.require(
            checks.is_utc("t", self.t),
            checks.is_ne("q", self.q, 0),
            checks.is_gt("px_limit", self.px_limit, 0),
        )

    @property
    def id(self) -> ID[LimitOrder]:
        return ID[LimitOrder](self.__hash__())

    @property
    def side(self) -> MakerSide:
        return MakerSide.BID if self.q > 0 else MakerSide.ASK


Order = MarketOrder | LimitOrder
OrderID = ID[MarketOrder] | ID[LimitOrder]


@dataclass(frozen=True, slots=True)
class Fill:
    t: datetime
    i: Instrument
    o: Order
    q: float
    px: float

    def __post_init__(self) -> None:
        checks.require(
            checks.is_utc("t", self.t),
            checks.is_ne("q", self.q, 0),
            checks.is_gt("px", self.px, 0),
        )

    @property
    def id(self) -> ID[Fill]:
        return ID[Fill](self.__hash__())

    @property
    def side(self) -> OrderSide:
        return self.o.side


@dataclass(frozen=True, slots=True)
class OpenPosition:
    t0: datetime
    i: Instrument
    q: float

    @property
    def id(self) -> ID[OpenPosition]:
        return ID[OpenPosition](self.__hash__())

    @property
    def side(self) -> PositionSide:
        return PositionSide.LONG if self.q > 0 else PositionSide.SHORT


@dataclass(frozen=True, slots=True)
class ClosedPosition:
    t0: datetime
    tf: datetime
    i: Instrument
    q: float

    @property
    def id(self) -> ID[ClosedPosition]:
        return ID[ClosedPosition](self.__hash__())

    @property
    def side(self) -> PositionSide:
        return PositionSide.LONG if self.q > 0 else PositionSide.SHORT


Position = OpenPosition | ClosedPosition
PositionID = ID[OpenPosition] | ID[ClosedPosition]
