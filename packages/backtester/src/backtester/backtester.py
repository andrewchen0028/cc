from datetime import datetime, timedelta
from backtester.instruments import Instrument, OptionInstrument
from ibis import Schema, Table, _
from ibis.expr.datatypes import Float64, String, Timestamp
from typing import Literal, Mapping, Protocol, runtime_checkable

import utils


@runtime_checkable
class Strategy(Protocol):
    def skip_trade(self) -> bool: ...
    def get_target_position(self) -> Mapping[Instrument, float]: ...


class Backtester:
    SCHEMA_BARS_SPOT = Schema({
        "t": Timestamp(timezone="UTC", scale=6),
        "exchange": String(),
        "base": String(),
        "quote": String(),
        "spot": Float64(),
    })  # fmt: off

    SCHEMA_BARS_OPTION = Schema({
        "t": Timestamp(timezone="UTC", scale=6),
        "exchange": String(),
        "base": String(),
        "quote": String(),
        "kind": String(),
        "strike": Float64(),
        "expiry": Timestamp(timezone="UTC", scale=6),
        "ivol": Float64(),
        # TODO: These depend on spot reference instrument.
        # We should have a separate spot table, and join
        # on it given a selected reference spot.
        "spot": Float64(),
        # TODO: Similarly for rate.
        "rate": Float64(),
        # TODO: These should be computed using the given
        # spot reference instrument via Black-numerical.
        "delta": Float64(),
        "gamma": Float64(),
        "vega": Float64(),
        "theta": Float64(),
        "rho": Float64(),
    })  # fmt: off

    def __init__(self, spot_bars: Table, option_bars: Table) -> None:
        self.spot_bars = utils.check_schema(spot_bars, self.SCHEMA_BARS_SPOT)
        self.option_bars = utils.check_schema(option_bars, self.SCHEMA_BARS_OPTION)

    def get_bars_spot(
        self,
        exchange: str,
        base: str,
        quote: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Table:
        filtered = self.spot_bars.filter([
            _["exchange"] == exchange,
            _["base"] == base,
            _["quote"] == quote,
        ])  # fmt: off

        if start is not None:
            filtered = filtered.filter(_["t"] >= start)
        if end is not None:
            filtered = filtered.filter(_["t"] < end)
        return filtered.order_by("t")

    def get_bars_option(
        self,
        exchange: str,
        base: str,
        quote: str,
        kind: Literal["c", "p"],
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Table:
        filtered = self.option_bars.filter([
            _["exchange"] == exchange,
            _["base"] == base,
            _["quote"] == quote,
            _["kind"] == kind,
        ])  # fmt: off

        if start is not None:
            filtered = filtered.filter(_["t"] >= start)
        if end is not None:
            filtered = filtered.filter(_["t"] < end)
        return filtered.order_by("t")

    def get_target_option(
        self,
        exchange: str,
        base: str,
        quote: str,
        kind: Literal["c", "p"],
        *,
        t: datetime,
        delta: float,
        tenor: timedelta,
    ) -> OptionInstrument:
        row = self.option_bars.filter([
            _["exchange"] == exchange,
            _["base"] == base,
            _["quote"] == quote,
            _["kind"] == kind,
        ]).mutate({
            "tenor": _["expiry"] - _["t"],
        }).mutate({
            "abs_err_t": (_["t"] - t).abs(),
            "abs_err_delta": (_["delta"] - delta).abs(),
            "abs_err_tenor": (_["tenor"] - tenor).abs(),
        }).order_by([
            _["abs_err_t"],
            _["abs_err_delta"],
            _["abs_err_tenor"],
        ]).limit(1).to_polars()  # fmt: off

        return OptionInstrument(
            exchange=row["exchange"].item(),
            base=row["base"].item(),
            quote=row["quote"].item(),
            kind=row["kind"].item(),
            strike=row["strike"].item(),
            expiry=row["expiry"].item(),
        )
