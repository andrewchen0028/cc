from typing import Final, final

import polars as pl


@final
class Schemas:
    """Schemas for Stake Optimizer"""

    EQY_SH_OUT: Final[pl.Schema] = pl.Schema({
        "date": pl.Date(),
        "security": pl.Utf8(),
        "eqy_sh_out": pl.Float64(),
    })  # fmt: off
    """From Bloomberg Excel Plugin (in shares)."""

    QUEUE_DELAYS: Final[pl.Schema] = pl.Schema({
        "date": pl.Date(),
        "entry": pl.Duration(),
        "exit": pl.Duration(),
    })  # fmt: off
    """From validatorqueue.com (https://github.com/etheralpha/validatorqueue-com)."""

    REWARD_RATE: Final[pl.Schema] = pl.Schema({
        "date": pl.Date(),
        "rate": pl.Float64(),
    })  # fmt: off
    """From somewhere."""
