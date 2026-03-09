# packages/backtester/src/backtester/schemas.py
"""Reference schemas for backtester module."""

from datetime import timezone
import narwhals as nw


PATHS_MARK = nw.Schema({
    "time_start": nw.Datetime(time_zone=timezone.utc),
    "time_end": nw.Datetime(time_zone=timezone.utc),
    "name": nw.String(),
    "price": nw.Float64(),
})  # fmt: off


PATH_RATE = nw.Schema({
    # Timestamps
    "time_start": nw.Datetime(time_zone=timezone.utc),
    "time_end": nw.Datetime(time_zone=timezone.utc),
    # Value
    "rate": nw.Float64(),
})  # fmt: off

BARS_SPOT = nw.Schema({
    # Timestamps
    "time_start": nw.Datetime(time_zone=timezone.utc),
    "time_end": nw.Datetime(time_zone=timezone.utc),
    # Identifiers
    "exchange": nw.String(),
    "base": nw.String(),
    "quote": nw.String(),
    # Values
    "px_bid": nw.Float64(),
    "px_ask": nw.Float64(),
    "px_mark": nw.Float64(),
})  # fmt: off

BARS_OPTION = nw.Schema({
    # Timestamps
    "time_start": nw.Datetime(time_zone=timezone.utc),
    "time_end": nw.Datetime(time_zone=timezone.utc),
    # Identifiers
    "exchange": nw.String(),
    "base": nw.String(),
    "quote": nw.String(),
    "strike": nw.Float64(),
    "listing": nw.Datetime(time_zone=timezone.utc),
    "expiry": nw.Datetime(time_zone=timezone.utc),
    "kind": nw.String(),
    # Values
    "iv_bid": nw.Float64(),
    "iv_ask": nw.Float64(),
    "iv_mark": nw.Float64(),
})  # fmt: off

BARS_PRICED = nw.Schema({
    # Timestamps
    "time_start": nw.Datetime(time_zone=timezone.utc),
    "time_end": nw.Datetime(time_zone=timezone.utc),
    # Identifiers
    "exchange": nw.String(),
    "base": nw.String(),
    "quote": nw.String(),
    "strike": nw.Float64(),
    "listing": nw.Datetime(time_zone=timezone.utc),
    "expiry": nw.Datetime(time_zone=timezone.utc),
    "is_call": nw.Boolean(),
    # Values (spot & rate)
    "spot": nw.Float64(),
    "rate": nw.Float64(),
    # Values (option)
    "iv_bid": nw.Float64(),
    "iv_ask": nw.Float64(),
    "iv_mark": nw.Float64(),
    "px_bid": nw.Float64(),
    "px_ask": nw.Float64(),
    "px_mark": nw.Float64(),
    "delta": nw.Float64(),
    "gamma": nw.Float64(),
    "vega": nw.Float64(),
    "theta": nw.Float64(),
    "rho": nw.Float64(),
})  # fmt: off
