from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal
import warnings


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
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.listing.utcoffset() != timedelta(0):
            raise ValueError(f"listing must be timezone.utc, got {self.listing.tzinfo}")
        if self.expiry.utcoffset() != timedelta(0):
            raise ValueError(f"expiry must be timezone.utc, got {self.expiry.tzinfo}")
        if (
            self.listing.hour != 8
            or self.listing.minute != 0
            or self.listing.second != 0
        ):
            warnings.warn(f"listing time is not 08:00:00, got {self.listing.time()}")
        if self.expiry.hour != 8 or self.expiry.minute != 0 or self.expiry.second != 0:
            warnings.warn(f"expiry time is not 08:00:00, got {self.expiry.time()}")
        if self.expiry <= self.listing:
            raise ValueError(
                f"expiry precedes listing: listing={self.listing}, expiry={self.expiry}"
            )
        if self.kind not in ("c", "p"):
            raise ValueError(f"kind must be 'c' or 'p', got {self.kind}")


@dataclass(frozen=True, slots=True)
class SpotInstrument:
    exchange: str
    base: str
    quote: str


Instrument = OptionInstrument | SpotInstrument
