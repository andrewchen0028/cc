# packages/backtester/src/backtester/schemas.py
"""Reference schemas for backtester module."""

from datetime import timezone
import polars as pl


PATHS_MARK = pl.Schema({
    "time_start": pl.Datetime(time_zone=timezone.utc),
    "time_end": pl.Datetime(time_zone=timezone.utc),
    "name": pl.String(),
    "price": pl.Float64(),
})  # fmt: off


PATH_RATE = pl.Schema({
    # Timestamps
    "time_start": pl.Datetime(time_zone=timezone.utc),
    "time_end": pl.Datetime(time_zone=timezone.utc),
    # Value
    "rate": pl.Float64(),
})  # fmt: off

BARS_SPOT = pl.Schema({
    # Timestamps
    "time_start": pl.Datetime(time_zone=timezone.utc),
    "time_end": pl.Datetime(time_zone=timezone.utc),
    # Identifiers
    "exchange": pl.String(),
    "base": pl.String(),
    "quote": pl.String(),
    # Values
    "px_bid": pl.Float64(),
    "px_ask": pl.Float64(),
    "px_mark": pl.Float64(),
})  # fmt: off

BARS_OPTION = pl.Schema({
    # Timestamps
    "time_start": pl.Datetime(time_zone=timezone.utc),
    "time_end": pl.Datetime(time_zone=timezone.utc),
    # Identifiers
    "exchange": pl.String(),
    "base": pl.String(),
    "quote": pl.String(),
    "strike": pl.Float64(),
    "listing": pl.Datetime(time_zone=timezone.utc),
    "expiry": pl.Datetime(time_zone=timezone.utc),
    "kind": pl.String(),
    # Values
    "iv_bid": pl.Float64(),
    "iv_ask": pl.Float64(),
    "iv_mark": pl.Float64(),
})  # fmt: off

BARS_PRICED = pl.Schema({
    # Timestamps
    "time_start": pl.Datetime(time_zone=timezone.utc),
    "time_end": pl.Datetime(time_zone=timezone.utc),
    # Identifiers
    "exchange": pl.String(),
    "base": pl.String(),
    "quote": pl.String(),
    "strike": pl.Float64(),
    "listing": pl.Datetime(time_zone=timezone.utc),
    "expiry": pl.Datetime(time_zone=timezone.utc),
    "kind": pl.String(),
    # Values (spot & rate)
    "spot": pl.Float64(),
    "rate": pl.Float64(),
    # Values (option)
    "iv_bid": pl.Float64(),
    "iv_ask": pl.Float64(),
    "iv_mark": pl.Float64(),
    "px_bid": pl.Float64(),
    "px_ask": pl.Float64(),
    "px_mark": pl.Float64(),
    "delta": pl.Float64(),
    "gamma": pl.Float64(),
    "vega": pl.Float64(),
    "theta": pl.Float64(),
    "rho": pl.Float64(),
})  # fmt: off
