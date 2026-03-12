import polars as pl

from stake_optimizer.schemas import Schemas
from utils import checks


def adapt_eqy_sh_out(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Adapt raw shares outstanding data to standard schema."""
    checks.require(checks.has_schema(lf, Schemas.EQY_SH_OUT))
    return lf


def adapt_queue_delays(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Adapt raw queue delay data to standard schema."""
    checks.require(checks.has_schema(lf, Schemas.QUEUE_DELAYS))
    return lf


def adapt_reward_rate(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Adapt raw reward rate data to standard schema."""
    checks.require(checks.has_schema(lf, Schemas.REWARD_RATE))
    return lf
