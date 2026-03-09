from dataclasses import dataclass
from datetime import datetime, time
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
