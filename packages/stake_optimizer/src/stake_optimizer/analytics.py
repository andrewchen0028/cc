import numpy as np
import polars as pl


def max_drawdown_n(lf: pl.LazyFrame, n: int) -> pl.LazyFrame:
    """Compute N-day drawdown in EQY_SH_OUT (percent, <= 0).

    Decline from the rolling N-day peak of *prior* values, clamped at zero.
    For N=1 this is min(daily_pct_change, 0).
    Returns LazyFrame with columns: date, security, max_drawdown.
    """
    return lf \
        .sort("security", "date") \
        .with_columns(
            max_drawdown=(
                pl.col("eqy_sh_out")
                / pl.col("eqy_sh_out").shift(1).rolling_max(window_size=n)
                - 1
            ).over("security").clip(upper_bound=0).mul(100),
        ) \
        .select("date", "security", "max_drawdown")  # fmt: off


def drawdown_var_cvar(
    lf: pl.LazyFrame, n: int, q: float,
) -> pl.DataFrame:
    """VaR and CVaR of N-day drawdown at quantile q, per security.

    q is the tail fraction (e.g. 0.05 = worst 5% of days).
    VaR = q-th percentile of drawdowns (negative).
    CVaR = mean of drawdowns <= VaR (more negative).
    Returns DataFrame: security, var, cvar.
    """
    dd = max_drawdown_n(lf, n).collect()
    rows = []
    for sec in dd["security"].unique().sort().to_list():
        vals = dd.filter(pl.col("security") == sec)["max_drawdown"].drop_nulls().to_numpy()
        var = float(np.percentile(vals, q * 100))
        tail = vals[vals <= var]
        cvar = float(tail.mean()) if len(tail) > 0 else var
        rows.append({"security": sec, "var": var, "cvar": cvar})
    return pl.DataFrame(rows)
