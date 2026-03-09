# packages/backtester/src/backtester/backtester.py
"""Option backtester module."""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Literal, Mapping, Protocol

import narwhals as nw

from backtester.instruments import Instrument, OptionInstrument, SpotInstrument
from backtester import schemas
from utils import checks
from utils.stats import norm_cdf


_US_PER_YEAR = 365.25 * 24 * 3600 * 1e6

_KEYS = [
    "time_start", "time_end",
    "exchange_spot", "base_spot", "quote_spot",
    "exchange_option", "base_option", "quote_option",
    "strike", "listing", "expiry", "kind",
]  # fmt: off


def _bs_price(
    s: nw.Expr,
    k: nw.Expr,
    r: nw.Expr,
    sigma: nw.Expr,
    tau: nw.Expr,
) -> nw.Expr:
    """Black-Scholes price, dispatched on 'kind' column."""
    d1 = ((s / k).log() + (r + 0.5 * sigma * sigma) * tau) / (sigma * tau.sqrt())
    d2 = d1 - sigma * tau.sqrt()
    df = (0 - r * tau).exp()  # discount factor
    call = s * norm_cdf(d1) - k * df * norm_cdf(d2)
    put = k * df * norm_cdf(0 - d2) - s * norm_cdf(0 - d1)
    return nw.when(nw.col("kind") == "c").then(call).otherwise(put)


class MarketDataProvider:
    def __init__(
        self,
        lf_rate: nw.LazyFrame,
        lf_spot: nw.LazyFrame,
        lf_option: nw.LazyFrame,
    ) -> None:
        self.lf_rate = checks.check_schema(nw.from_native(lf_rate), schemas.PATH_RATE)  # fmt: off
        self.lf_spot = checks.check_schema(nw.from_native(lf_spot), schemas.BARS_SPOT)  # fmt: off
        self.lf_option = checks.check_schema(nw.from_native(lf_option), schemas.BARS_OPTION)  # fmt: off
        # self.lf_priced = self._get_lf_priced()

    def _get_lf_priced(
        self,
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
            return nw.when(is_call).then(c).otherwise(p)

        # Note that lf_option and lf_spot may contain multiple instruments.
        # We'll accept a mapping from option (exchange, base, quote) to spot (exchange, base quote),
        # to get a spot reference instrument for each option instrument.
        # For now just handle one item from the mapping.
        exchange_option, base_option, quote_option = "drbt", "btc", "usd"
        exchange_spot, base_spot, quote_spot = "cbse", "btc", "usd"

        lff_option = self.lf_option.filter([
            nw.col("exchange") == exchange_option,
            nw.col("base") == base_option,
            nw.col("quote") == quote_option,
        ]).with_columns([
            nw.when(nw.col("kind") == "c").then(True).otherwise(False).alias("is_call")
        ])  # fmt: off
        lff_spot = self.lf_spot.filter([
            nw.col("exchange") == exchange_spot,
            nw.col("base") == base_spot,
            nw.col("quote") == quote_spot,
        ]).select(["time_start", "time_end", "px_mark"]).rename({"px_mark": "spot"})  # fmt: off

        s = nw.col("spot")
        k = nw.col("strike")
        r = nw.col("rate")
        v = nw.col("iv_mark")
        tau = (nw.col("expiry") - nw.col("time_end")).dt.total_seconds() / (365 * 24 * 3600)  # fmt: off
        is_call = nw.col("is_call")

        keys = ["time_start", "time_end"] \
            + ["exchange", "base", "quote", "strike", "listing", "expiry", "is_call"] \
            + ["spot", "rate", "iv_bid", "iv_ask", "iv_mark"]  # fmt: off

        lff_priced = lff_option \
            .join(lff_spot, ["time_start", "time_end"]) \
            .join(self.lf_rate, ["time_start", "time_end"]) \
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

    # def get_target_option(
    #     self,
    #     exchange: str,
    #     base: str,
    #     quote: str,
    #     kind: Literal["c", "p"],
    #     *,
    #     target_time: datetime,
    #     target_delta: float,
    #     target_tenor: timedelta,
    # ) -> OptionInstrument:
    #     df = nw.from_native(self.lf_priced).filter(
    #         nw.col("exchange") == exchange,
    #         nw.col("base") == base,
    #         nw.col("quote") == quote,
    #         nw.col("kind") == kind
    #     ).with_columns(
    #         (nw.col("expiry") - nw.col("time_end")).alias("tenor"),
    #     ).with_columns(
    #         (nw.col("time_end") - target_time).abs().alias("abs_err_time"),
    #         (nw.col("delta") - target_delta).abs().alias("abs_err_delta"),
    #         (nw.col("tenor") - target_tenor).abs().alias("abs_err_tenor"),
    #     ).sort(["abs_err_time", "abs_err_tenor", "abs_err_delta"]).head(1).collect()  # fmt: off

    #     return OptionInstrument(
    #         exchange=df["exchange"].item(),
    #         base=df["base"].item(),
    #         quote=df["quote"].item(),
    #         strike=df["strike"].item(),
    #         listing=df["listing"].item(),
    #         expiry=df["expiry"].item(),
    #         kind=df["kind"].item(),
    #     )


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
