from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from backtester.dtypes import OptionInstrument, SpotInstrument


LISTING = datetime(2023, 1, 13, 8, 0, 0, tzinfo=timezone.utc)
EXPIRY = datetime(2023, 1, 20, 8, 0, 0, tzinfo=timezone.utc)


def _make_option(**overrides) -> OptionInstrument:
    defaults = dict(
        exchange="drbt",
        base="btc",
        quote="usd",
        strike=100_000.0,
        listing=LISTING,
        expiry=EXPIRY,
        kind="c",
    )
    return OptionInstrument(**{**defaults, **overrides})


# --- OptionInstrument ---


def test_option_instrument_valid():
    opt = _make_option()
    assert opt.strike == 100_000.0
    assert opt.kind == "c"


def test_option_instrument_frozen():
    opt = _make_option()
    with pytest.raises(AttributeError):
        opt.strike = 200_000.0  # type: ignore[misc]


def test_option_instrument_rejects_negative_strike():
    with pytest.raises(ValueError):
        _make_option(strike=-1.0)


def test_option_instrument_rejects_zero_strike():
    with pytest.raises(ValueError):
        _make_option(strike=0.0)


def test_option_instrument_rejects_naive_listing():
    with pytest.raises(ValueError):
        _make_option(listing=datetime(2023, 1, 13, 8, 0, 0))


def test_option_instrument_rejects_non_utc_expiry():
    with pytest.raises(ValueError):
        _make_option(expiry=datetime(2023, 1, 20, 8, 0, 0, tzinfo=ZoneInfo("US/Eastern")))


def test_option_instrument_rejects_expiry_before_listing():
    with pytest.raises(ValueError):
        _make_option(listing=EXPIRY, expiry=LISTING)


def test_option_instrument_accepts_zoneinfo_utc():
    opt = _make_option(
        listing=datetime(2023, 1, 13, 8, 0, 0, tzinfo=ZoneInfo("UTC")),
        expiry=datetime(2023, 1, 20, 8, 0, 0, tzinfo=ZoneInfo("UTC")),
    )
    assert opt.listing.tzinfo == ZoneInfo("UTC")


def test_option_instrument_warns_non_8am(recwarn):
    _make_option(listing=datetime(2023, 1, 13, 9, 0, 0, tzinfo=timezone.utc))
    assert len(recwarn) >= 1


def test_option_instrument_no_warning_at_8am(recwarn):
    _make_option()
    assert len(recwarn) == 0


# --- SpotInstrument ---


def test_spot_instrument_valid():
    spot = SpotInstrument(exchange="cbse", base="btc", quote="usd")
    assert spot.exchange == "cbse"


def test_spot_instrument_frozen():
    spot = SpotInstrument(exchange="cbse", base="btc", quote="usd")
    with pytest.raises(AttributeError):
        spot.exchange = "binc"  # type: ignore[misc]
