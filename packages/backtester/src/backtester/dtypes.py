from dataclasses import dataclass
from datetime import datetime, time
from typing import Literal
import warnings

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
        if errors := [
            *checks.check_positive("strike", self.strike),
            *checks.check_is_utc("listing", self.listing),
            *checks.check_is_utc("expiry", self.expiry),
            *checks.check_datetime_order(self.listing, self.expiry),
            *checks.check_one_of("kind", self.kind, ("c", "p")),
        ]:
            raise ValueError("\n".join(errors))
        if warnings_ := [
            *checks.check_datetime_time("listing", self.listing, time(8, 0, 0)),
            *checks.check_datetime_time("expiry", self.expiry, time(8, 0, 0)),
        ]:
            for w in warnings_:
                warnings.warn(w)


@dataclass(frozen=True, slots=True)
class SpotInstrument:
    exchange: str
    base: str
    quote: str


Instrument = OptionInstrument | SpotInstrument
