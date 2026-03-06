# packages/backtester/src/backtester/backtester.py
"""Option backtester module."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Collection, Literal, Mapping, Protocol

import narwhals as nw
import polars as pl


RATE_SCHEMA = nw.Schema({
    "time_start": nw.Datetime("us", "utc"),
    "time_end": nw.Datetime("us", "utc"),
    "rate": nw.Float64(),
})  # fmt: off

SPOT_SCHEMA = nw.Schema({
    "time_start": nw.Datetime("us", "utc"),
    "time_end": nw.Datetime("us", "utc"),
    "exchange": nw.String(),
    "base": nw.String(),
    "quote": nw.String(),
    "px_bid": nw.Float64(),
    "px_ask": nw.Float64(),
    "px_mark": nw.Float64(),
})  # fmt: off

OPTION_SCHEMA = nw.Schema({
    "time_start": nw.Datetime("us", "utc"),
    "time_end": nw.Datetime("us", "utc"),
    "exchange": nw.String(),
    "base": nw.String(),
    "quote": nw.String(),
    "strike": nw.Float64(),
    "expiry": nw.Datetime("us", "utc"),
    "kind": nw.String(),
    "iv_bid": nw.Float64(),
    "iv_ask": nw.Float64(),
    "iv_mark": nw.Float64(),
})  # fmt: off

PRICED_SCHEMA = nw.Schema({
    # Timestamps and risk-free rate
    "time_start": nw.Datetime("us", "utc"),
    "time_end": nw.Datetime("us", "utc"),
    "rate": nw.Float64(),
    # Spot identifiers
    "exchange_spot": nw.String(),
    "base_spot": nw.String(),
    "quote_spot": nw.String(),
    # Spot prices
    "px_bid_spot": nw.Float64(),
    "px_ask_spot": nw.Float64(),
    "px_mark_spot": nw.Float64(),
    # Option identifiers
    "exchange_option": nw.String(),
    "base_option": nw.String(),
    "quote_option": nw.String(),
    "strike": nw.Float64(),
    "expiry": nw.Datetime("us", "utc"),
    "kind": nw.String(),
    # Option prices and IVs
    "px_bid_option": nw.Float64(),
    "px_ask_option": nw.Float64(),
    "px_mark_option": nw.Float64(),
    "iv_bid": nw.Float64(),
    "iv_ask": nw.Float64(),
    "iv_mark": nw.Float64(),
    # Option greeks
    "delta": nw.Float64(),
    "gamma": nw.Float64(),
    "vega": nw.Float64(),
    "theta": nw.Float64(),
    "rho": nw.Float64(),
})  # fmt: off


@dataclass(frozen=True, slots=True)
class SpotInstrument:
    exchange: str
    base: str
    quote: str


@dataclass(frozen=True, slots=True)
class OptionInstrument:
    exchange: str
    base: str
    quote: str
    strike: float
    expiry: datetime
    kind: Literal["c", "p"]


Instrument = SpotInstrument | OptionInstrument


class MarketDataProvider:
    def __init__(
        self,
        lf_rate: nw.LazyFrame,
        lf_spot: nw.LazyFrame,
        lf_option: nw.LazyFrame,
    ) -> None:
        self.lf_rate = self._check_schema(lf_rate, RATE_SCHEMA)
        self.lf_spot = self._check_schema(lf_spot, SPOT_SCHEMA)
        self.lf_option = self._check_schema(lf_option, OPTION_SCHEMA)

        # Get priced table by joining all tables and computing Black-numerical greeks.
        self.lf_priced = self._get_lf_priced()

    def _check_schema(self, lf: nw.LazyFrame, schema: nw.Schema) -> nw.LazyFrame:
        if False:  # TODO: implement schema checking
            raise ValueError(f"Schema mismatch. Expected {schema}, got{lf.schema}")
        return lf

    def _get_lf_priced(self) -> nw.LazyFrame:
        # TODO: implement actual pricing logic
        return nw.from_native(self.lf_option).join(
            nw.from_native(self.lf_spot), on=["time_start", "time_end"], how="inner"
        ).rename({
            "exchange":       "exchange_option",
            "base":           "base_option",
            "quote":          "quote_option",
            "exchange_right": "exchange_spot",
            "base_right":     "base_spot",
            "quote_right":    "quote_spot",
            "px_bid":         "px_bid_spot",
            "px_ask":         "px_ask_spot",
            "px_mark":        "px_mark_spot",
        }).join(nw.from_native(self.lf_rate), on=["time_start", "time_end"], how="inner"
        ).with_columns([
            nw.col("px_mark_spot").alias(col) for col in [
                "px_bid_option",
                "px_ask_option",
                "px_mark_option",
                "delta",
                "gamma",
                "vega",
                "theta",
                "rho",
            ]
        ])  # fmt: off

    def get_bars_spot(
        self,
        spot: SpotInstrument,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> nw.LazyFrame:
        return nw.from_native(self.lf_spot).filter(
            nw.col("exchange") == spot.exchange,
            nw.col("base") == spot.base,
            nw.col("quote") == spot.quote,
            nw.col("time_start") > start_time if start_time is not None else True,
            nw.col("time_end") < end_time if end_time is not None else True,
        )

    def get_bars_option(
        self,
        option: OptionInstrument,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> nw.LazyFrame:
        return nw.from_native(self.lf_option).filter(
            nw.col("exchange") == option.exchange,
            nw.col("base") == option.base,
            nw.col("quote") == option.quote,
            nw.col("strike") == option.strike,
            nw.col("expiry") == option.expiry,
            nw.col("kind") == option.kind,
            nw.col("time_start") > start_time if start_time is not None else True,
            nw.col("time_end") < end_time if end_time is not None else True,
        )

    def get_target_option(
        self,
        exchange: str,
        base: str,
        quote: str,
        kind: Literal["c", "p"],
        *,
        target_time: datetime,
        target_delta: float,
        target_tenor: timedelta,
    ) -> OptionInstrument:
        df = nw.from_native(self.lf_priced).filter(
            nw.col("exchange") == exchange,
            nw.col("base") == base,
            nw.col("quote") == quote,
            nw.col("kind") == kind
        ).with_columns(
            (nw.col("expiry") - nw.col("time_end")).alias("tenor"),
        ).with_columns(
            (nw.col("time_end") - target_time).abs().alias("abs_err_time"),
            (nw.col("delta") - target_delta).abs().alias("abs_err_delta"),
            (nw.col("tenor") - target_tenor).abs().alias("abs_err_tenor"),
        ).sort(["abs_err_time", "abs_err_tenor", "abs_err_delta"]).head(1).collect()  # fmt: off

        return OptionInstrument(
            exchange=df["exchange"].item(),
            base=df["base"].item(),
            quote=df["quote"].item(),
            strike=df["strike"].item(),
            expiry=df["expiry"].item(),
            kind=df["kind"].item(),
        )


def sample_polars_rate(
    start_time: datetime = datetime(2020, 1, 1),
    end_time: datetime = datetime(2020, 1, 10),
    freq: timedelta = timedelta(minutes=1),
) -> pl.LazyFrame:
    # TODO: implement actual sampling logic
    return pl.LazyFrame(schema={
        "time_start": pl.Datetime("us", "utc"),
        "time_end": pl.Datetime("us", "utc"),
        "rate": pl.Float64(),
    })  # fmt: off


def sample_polars_spot(
    exchanges: Collection[str] | None = None,
    bases: Collection[str] | None = None,
    quotes: Collection[str] | None = None,
) -> pl.LazyFrame:
    # TODO: implement actual sampling logic
    return pl.LazyFrame(schema={
        "time_start": pl.Datetime("us", "utc"),
        "time_end": pl.Datetime("us", "utc"),
        "exchange": pl.String(),
        "base": pl.String(),
        "quote": pl.String(),
        "px_bid": pl.Float64(),
        "px_ask": pl.Float64(),
        "px_mark": pl.Float64(),
    })  # fmt: off


def sample_polars_option(
    exchanges: Collection[str] | None = None,
    bases: Collection[str] | None = None,
    quotes: Collection[str] | None = None,
    strikes: Collection[float] | None = None,
    expiries: Collection[datetime] | None = None,
    kinds: Collection[Literal["c", "p"]] | None = None,
) -> pl.LazyFrame:
    # TODO: implement actual sampling logic
    return pl.LazyFrame(schema={
        "time_start": pl.Datetime("us", "utc"),
        "time_end": pl.Datetime("us", "utc"),
        "exchange": pl.String(),
        "base": pl.String(),
        "quote": pl.String(),
        "strike": pl.Float64(),
        "expiry": pl.Datetime("us", "utc"),
        "kind": pl.String(),
        "iv_bid": pl.Float64(),
        "iv_ask": pl.Float64(),
        "iv_mark": pl.Float64(),
    })  # fmt: off


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
    def __init__(self, mdp: MarketDataProvider) -> None: ...
    def run(self, strategy: Strategy) -> None: ...
