# packages/backtester/src/backtester/io.py
"""Input/output utilities for backtester module."""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Literal

import polars as pl

from backtester.dtypes import SpotInstrument, OptionInstrument
from utils import checks
from utils import schemas


def _build_lf_priced(
    lf_rate: pl.LazyFrame,
    lf_spot: pl.LazyFrame,
    lf_option: pl.LazyFrame,
    option_exchange: str,
    option_base: str,
    option_quote: str,
    spot_exchange: str,
    spot_base: str,
    spot_quote: str,
    *,
    ds: float = 0.01,
    dr: float = 0.0001,
    dv: float = 0.01,
    dt: float = 1 / (365.25 * 24 * 60 * 60),
) -> pl.LazyFrame:
    """Black-Scholes pricing with central-difference Greeks.

    Args:
        ds: spot shock (default: 0.01 [one cent])
        dr: rate shock (default: 0.0001 [one basis point])
        dv: ivol shock (default: 0.01 [one ivol point])
        dt: time shock (default: 1 / (365.25 * 24 * 60 * 60) [one second])
    """

    def _px_bs(
        s: pl.Expr,
        k: pl.Expr,
        r: pl.Expr,
        sigma: pl.Expr,
        tau: pl.Expr,
        is_call: pl.Expr,
    ) -> pl.Expr:
        from utils.stats import norm_cdf

        dp = ((s / k).log() + (r + sigma * sigma / 2) * tau) / (sigma * tau.sqrt())
        dm = ((s / k).log() + (r - sigma * sigma / 2) * tau) / (sigma * tau.sqrt())
        c = s * norm_cdf(dp) - k * (0 - r * tau).exp() * norm_cdf(dm)
        p = k * (0 - r * tau).exp() * norm_cdf(0 - dm) - s * norm_cdf(0 - dp)
        return pl.when(is_call).then(c).otherwise(p).clip(0)

    checks.require(
        checks.has_schema(lf_rate, schemas.PATH_RATE),
        checks.has_schema(lf_spot, schemas.BARS_SPOT),
        checks.has_schema(lf_option, schemas.BARS_OPTION),
    )

    lff_rate = lf_rate

    lff_spot = lf_spot.filter([
        pl.col("exchange") == spot_exchange,
        pl.col("base") == spot_base,
        pl.col("quote") == spot_quote,
    ]).select([
        "time_start",
        "time_end",
        "px_mark",
    ]).rename({"px_mark": "spot"})  # fmt: off

    lff_option = lf_option.filter([
        pl.col("exchange") == option_exchange,
        pl.col("base") == option_base,
        pl.col("quote") == option_quote,
    ]).with_columns([
        pl.when(pl.col("kind") == "c").then(True).otherwise(False).alias("is_call")
    ])  # fmt: off

    s = pl.col("spot")
    k = pl.col("strike")
    r = pl.col("rate")
    v = pl.col("iv_mark")
    tau = (pl.col("expiry") - pl.col("time_end")).dt.total_seconds() / (365 * 24 * 3600)  # fmt: off
    is_call = pl.col("is_call")

    keys = ["time_start", "time_end"] \
        + ["exchange", "base", "quote", "strike", "listing", "expiry", "kind"] \
        + ["spot", "rate", "iv_bid", "iv_ask", "iv_mark"]  # fmt: off

    lff_priced = lff_option \
        .join(lff_spot, ["time_start", "time_end"]) \
        .join(lff_rate, ["time_start", "time_end"]) \
    .with_columns([
        _px_bs(s, k, r, pl.col("iv_bid"), tau, is_call).alias("px_bid"),
        _px_bs(s, k, r, pl.col("iv_ask"), tau, is_call).alias("px_ask"),
        _px_bs(s, k, r, pl.col("iv_mark"), tau, is_call).alias("px_mark"),
    ]).with_columns([
        _px_bs(s, k, r, v, tau, is_call).alias("_"),
        _px_bs((s + ds).clip(0), k, r, v, tau, is_call).alias("_s_up"),
        _px_bs((s - ds).clip(0), k, r, v, tau, is_call).alias("_s_dn"),
        _px_bs(s, k, r, (v + dv).clip(0), tau, is_call).alias("_v_up"),
        _px_bs(s, k, r, (v - dv).clip(0), tau, is_call).alias("_v_dn"),
        _px_bs(s, k, r, v, (tau + dt).clip(0), is_call).alias("_tau_up"),
        _px_bs(s, k, r, v, (tau - dt).clip(0), is_call).alias("_tau_dn"),
        _px_bs(s, k, (r + dr).clip(0), v, tau, is_call).alias("_r_up"),
        _px_bs(s, k, (r - dr).clip(0), v, tau, is_call).alias("_r_dn"),
    ]).with_columns([
        ((pl.col("_s_up") - pl.col("_s_dn")) / (2 * ds)).alias("delta"),
        ((pl.col("_s_up") - 2 * pl.col("_") + pl.col("_s_dn")) / (ds * ds)).alias("gamma"),
        ((pl.col("_v_up") - pl.col("_v_dn")) / (2 * dv)).alias("vega"),
        ((pl.col("_tau_dn") - pl.col("_tau_up")) / (2 * dt)).alias("theta"),
        ((pl.col("_r_up") - pl.col("_r_dn")) / (2 * dr)).alias("rho"),
    ]).select([
        *keys,
        *["px_bid", "px_ask", "px_mark"],
        *["delta", "gamma", "vega", "theta", "rho"],
    ])  # fmt: off

    checks.require(checks.has_schema(lff_priced, schemas.BARS_PRICED))
    return lff_priced


def get_bars_spot(
    lf_spot: pl.LazyFrame,
    spot: SpotInstrument,
    *,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> pl.LazyFrame:
    predicates: list[pl.Expr] = [
        pl.col("exchange") == spot.exchange,
        pl.col("base") == spot.base,
        pl.col("quote") == spot.quote,
    ]
    if start_time is not None:
        predicates.append(pl.col("time_start") >= start_time)
    if end_time is not None:
        predicates.append(pl.col("time_end") <= end_time)

    checks.require(checks.has_schema(lf_spot, schemas.BARS_SPOT))
    return lf_spot.filter(predicates)


def get_bars_option(
    lf_option: pl.LazyFrame,
    option: OptionInstrument,
    *,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> pl.LazyFrame:
    predicates: list[pl.Expr] = [
        pl.col("exchange") == option.exchange,
        pl.col("base") == option.base,
        pl.col("quote") == option.quote,
        pl.col("strike") == option.strike,
        pl.col("listing") == option.listing,
        pl.col("expiry") == option.expiry,
        pl.col("kind") == option.kind,
    ]
    if start_time is not None:
        predicates.append(pl.col("time_start") >= start_time)
    if end_time is not None:
        predicates.append(pl.col("time_end") <= end_time)

    checks.require(checks.has_schema(lf_option, schemas.BARS_OPTION))
    return lf_option.filter(predicates)


def get_target_option(
    lf_rate: pl.LazyFrame,
    lf_spot: pl.LazyFrame,
    lf_option: pl.LazyFrame,
    option_exchange: str,
    option_base: str,
    option_quote: str,
    option_kind: Literal["c", "p"],
    spot_instrument: SpotInstrument,
    *,
    target_time: datetime,
    target_delta: float,
    target_tenor: timedelta,
) -> OptionInstrument:
    if option_base != spot_instrument.base:
        raise ValueError("Option and spot base assets must match.")

    df = _build_lf_priced(
        lf_rate=lf_rate,
        lf_spot=lf_spot,
        lf_option=lf_option,
        option_exchange=option_exchange,
        option_base=option_base,
        option_quote=option_quote,
        spot_exchange=spot_instrument.exchange,
        spot_base=spot_instrument.base,
        spot_quote=spot_instrument.quote,
    ).filter(
        pl.col("exchange") == option_exchange,
        pl.col("base") == option_base,
        pl.col("quote") == option_quote,
        pl.col("kind") == option_kind
    ).with_columns(
        (pl.col("expiry") - pl.col("time_end")).alias("tenor"),
    ).with_columns(
        (pl.col("time_end") - target_time).abs().alias("abs_err_time"),
        (pl.col("delta") - target_delta).abs().alias("abs_err_delta"),
        (pl.col("tenor") - target_tenor).abs().alias("abs_err_tenor"),
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
