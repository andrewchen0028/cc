from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Literal


@dataclass(frozen=True, slots=True)
class OptionInstrument:
    exchange: str
    base: str
    quote: str
    expiry: datetime
    strike: float
    kind: Literal["c", "p"]

    def __post_init__(self) -> None:
        if self.expiry.tzinfo != UTC:
            raise ValueError(f"expiry must be UTC, got {self.expiry.tzinfo}")


@dataclass(frozen=True, slots=True)
class SpotInstrument:
    exchange: str
    base: str
    quote: str


Instrument = OptionInstrument | SpotInstrument
