# packages/backtester/src/backtester/io.py
"""Input/output utilities for backtester module."""
# TODO: test Claude modifications

from __future__ import annotations
from datetime import datetime, timedelta
from narwhals.typing import IntoLazyFrameT
from typing import Literal

import narwhals as nw
import narwhals.sql as ns

from backtester.types import SpotInstrument, OptionInstrument
from backtester import schemas
from utils import checks


# NOTE: required for schemas with UTC datetimes
ns.CONN.query("SET TimeZone = 'UTC'")


def _get_lf_priced(
    lf_rate: IntoLazyFrameT,
    lf_spot: IntoLazyFrameT,
    lf_option: IntoLazyFrameT,
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
) -> nw.LazyFrame:
    """Black-Scholes pricing with central-difference Greeks.

    Args:
        ds: spot shock (default: 0.01 [one cent])
        dr: rate shock (default: 0.0001 [one basis point])
        dv: ivol shock (default: 0.01 [one ivol point])
        dt: time shock (default: 1 / (365.25 * 24 * 60 * 60) [one second])
    """

    def _px_bs(
        s: nw.Expr,
        k: nw.Expr,
        r: nw.Expr,
        sigma: nw.Expr,
        tau: nw.Expr,
        is_call: nw.Expr,
    ) -> nw.Expr:
        from utils.stats import norm_cdf

        dp = ((s / k).log() + (r + sigma * sigma / 2) * tau) / (sigma * tau.sqrt())
        dm = ((s / k).log() + (r - sigma * sigma / 2) * tau) / (sigma * tau.sqrt())
        c = s * norm_cdf(dp) - k * (0 - r * tau).exp() * norm_cdf(dm)
        p = k * (0 - r * tau).exp() * norm_cdf(0 - dm) - s * norm_cdf(0 - dp)
        return nw.when(is_call).then(c).otherwise(p).clip(0)

    lf_rate_ = checks.check_schema(nw.from_native(lf_rate), schemas.PATH_RATE)
    lf_spot_ = checks.check_schema(nw.from_native(lf_spot), schemas.BARS_SPOT)
    lf_option_ = checks.check_schema(nw.from_native(lf_option), schemas.BARS_OPTION)

    lff_option = lf_option_.filter([
        nw.col("exchange") == option_exchange,
        nw.col("base") == option_base,
        nw.col("quote") == option_quote,
    ]).with_columns([
        nw.when(nw.col("kind") == "c").then(True).otherwise(False).alias("is_call")
    ])  # fmt: off

    lff_spot = lf_spot_.filter([
        nw.col("exchange") == spot_exchange,
        nw.col("base") == spot_base,
        nw.col("quote") == spot_quote,
    ]).select([
        "time_start",
        "time_end",
        "px_mark",
    ]).rename({"px_mark": "spot"})  # fmt: off

    s = nw.col("spot")
    k = nw.col("strike")
    r = nw.col("rate")
    v = nw.col("iv_mark")
    tau = (nw.col("expiry") - nw.col("time_end")).dt.total_seconds() / (365 * 24 * 3600)  # fmt: off
    is_call = nw.col("is_call")

    keys = ["time_start", "time_end"] \
        + ["exchange", "base", "quote", "strike", "listing", "expiry", "kind"] \
        + ["spot", "rate", "iv_bid", "iv_ask", "iv_mark"]  # fmt: off

    lff_priced = lff_option \
        .join(lff_spot, ["time_start", "time_end"]) \
        .join(lf_rate_, ["time_start", "time_end"]) \
    .with_columns([
        _px_bs(s, k, r, nw.col("iv_bid"), tau, is_call).alias("px_bid"),
        _px_bs(s, k, r, nw.col("iv_ask"), tau, is_call).alias("px_ask"),
        _px_bs(s, k, r, nw.col("iv_mark"), tau, is_call).alias("px_mark"),
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
        ((nw.col("_s_up") - nw.col("_s_dn")) / (2 * ds)).alias("delta"),
        ((nw.col("_s_up") - 2 * nw.col("_") + nw.col("_s_dn")) / (ds * ds)).alias("gamma"),
        ((nw.col("_v_up") - nw.col("_v_dn")) / (2 * dv)).alias("vega"),
        ((nw.col("_tau_dn") - nw.col("_tau_up")) / (2 * dt)).alias("theta"),
        ((nw.col("_r_up") - nw.col("_r_dn")) / (2 * dr)).alias("rho"),
    ]).select([
        *keys,
        *["px_bid", "px_ask", "px_mark"],
        *["delta", "gamma", "vega", "theta", "rho"],
    ])  # fmt: off

    return checks.check_schema(lff_priced, schemas.BARS_PRICED)


def get_bars_spot(
    lf_spot: IntoLazyFrameT,
    spot: SpotInstrument,
    *,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> IntoLazyFrameT:
    return (
        checks.check_schema(nw.from_native(lf_spot), schemas.BARS_SPOT)
        .filter(
            nw.col("exchange") == spot.exchange,
            nw.col("base") == spot.base,
            nw.col("quote") == spot.quote,
            nw.col("time_start") > start_time if start_time is not None else True,
            nw.col("time_end") < end_time if end_time is not None else True,
        )
        .to_native()
    )


def get_bars_option(
    lf_option: IntoLazyFrameT,
    option: OptionInstrument,
    *,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> IntoLazyFrameT:
    return (
        checks.check_schema(nw.from_native(lf_option), schemas.BARS_OPTION)
        .filter(
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
        .to_native()
    )


def get_target_option(
    lf_rate: IntoLazyFrameT,
    lf_spot: IntoLazyFrameT,
    lf_option: IntoLazyFrameT,
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

    df = _get_lf_priced(
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
        nw.col("exchange") == option_exchange,
        nw.col("base") == option_base,
        nw.col("quote") == option_quote,
        nw.col("kind") == option_kind
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
