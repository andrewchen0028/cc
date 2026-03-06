from ibis import Schema, Table
from ibis.expr.datatypes import Float64, String, Timestamp

import utils


class Backtester:
    SCHEMA_BARS_SPOT = Schema(
        {
            "t": Timestamp(timezone="UTC", scale=6),
            "exchange": String(),
            "base": String(),
            "quote": String(),
            "spot": Float64(),
        }
    )

    SCHEMA_BARS_OPTION = Schema(
        {
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
            # "spot": Float64(),
            # TODO: Similarly for rate.
            # "rate": Float64(),
            # TODO: These should be computed using the given
            # spot reference instrument via Black-numerical.
            # "delta": Float64(),
            # "gamma": Float64(),
            # "vega": Float64(),
            # "theta": Float64(),
            # "rho": Float64(),
        }
    )

    def __init__(self, option_bars: Table) -> None:
        self.option_bars = utils.check_schema(option_bars, self.SCHEMA_BARS_OPTION)
