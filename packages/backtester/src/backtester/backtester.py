# packages/backtester/src/backtester/backtester.py
"""Option backtester module."""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Literal, Mapping, Protocol

import narwhals as nw

from backtester.instruments import Instrument, OptionInstrument, SpotInstrument
from backtester import schemas
from utils import checks


class MarketDataProvider:
    def __init__(
        self,
        lf_rate: nw.LazyFrame,
        lf_spot: nw.LazyFrame,
        lf_option: nw.LazyFrame,
    ) -> None:
        self.lf_rate = checks.check_schema(lf_rate, schemas.BARS_RATE)
        self.lf_spot = checks.check_schema(lf_spot, schemas.BARS_SPOT)
        self.lf_option = checks.check_schema(lf_option, schemas.BARS_OPTION)
        self.lf_priced = self._get_lf_priced()

    def _get_lf_priced(self) -> nw.LazyFrame:
        lf_option = nw.from_native(self.lf_option)
        lf_spot = nw.from_native(self.lf_spot)
        lf_rate = nw.from_native(self.lf_rate)
        try:
            # TODO: implement actual pricing logic
            out = lf_option.join(
                lf_spot, on=["time_start", "time_end"], how="inner"
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
            }).join(lf_rate, on=["time_start", "time_end"], how="inner"
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
            return checks.check_schema(out, schemas.BARS_PRICED)
        except Exception as e:
            print("WARNING: ensure rate/spot/option LazyFrames have same backend")
            raise e

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
            nw.col("listing") == option.listing,
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
            listing=df["listing"].item(),
            expiry=df["expiry"].item(),
            kind=df["kind"].item(),
        )


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
