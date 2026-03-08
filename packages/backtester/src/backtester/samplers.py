# packages/backtester/src/backtester/samplers3.py
"""Third-generation sample data generators for backtester module."""

from datetime import datetime, timedelta, timezone
from dateutil.rrule import rrule, DAILY, WEEKLY, MONTHLY, FR
from typing import Collection, Sequence

import narwhals as nw
import numpy as np
import polars as pl


from backtester.instruments import SpotInstrument
from backtester import schemas
from utils import checks


def get_path_rate(
    t0: datetime,
    tf: datetime,
    dt: timedelta,
    *,
    kappa: float = 0.5,
    theta: float = 0.05,
    sigma: float = 0.01,
) -> pl.LazyFrame:
    """Sample risk-free rate path via Vasicek model.

    Args:
        t0: start time
        tf: end time
        dt: time step
        kappa: mean reversion speed (default: 0.5)
        theta: long-term mean (default: 0.05)
        sigma: volatility (default: 0.01)

    Returns:
        LazyFrame with columns:
            - time_start
            - time_end
            - rate
    """
    if errors := [
        *checks.check_datetime_timezone(t0, timezone.utc),
        *checks.check_datetime_timezone(tf, timezone.utc),
        *checks.check_datetime_order(t0, tf),
    ]:
        raise ValueError(f"Invalid input\n{'\n'.join(f'- {e}' for e in errors)}")

    # Build time grid using polars datetime_range
    times_start = pl.datetime_range(t0, tf - dt, dt, eager=True).alias("time_start")
    times_end = pl.datetime_range(t0 + dt, tf, dt, eager=True).alias("time_end")
    n_steps = len(times_start)

    # Vasicek: dr = kappa*(theta - r)*dt + sigma*dW
    dt_years = dt / timedelta(days=365.25)
    dW = np.random.standard_normal(n_steps)
    r = theta
    rates = []
    for i in range(n_steps):
        rates.append(r)
        r = r + kappa * (theta - r) * dt_years + sigma * np.sqrt(dt_years) * dW[i]

    return checks.check_schema(nw.from_native(pl.LazyFrame({
        "time_start": times_start,
        "time_end": times_end,
        "rate": rates,
    })), schemas.PATH_RATE).to_native()  # fmt: off


def get_paths_mark(
    t0: datetime,
    tf: datetime,
    dt: timedelta,
    *,
    names: str | Sequence[str] | None = None,
    s0: float | Sequence[float] | np.typing.ArrayLike | None = None,
    mu: float | Sequence[float] | np.typing.ArrayLike | None = None,
    sigma: float | Sequence[float] | np.typing.ArrayLike | None = None,
) -> pl.LazyFrame:
    """Sample fair-price paths via multivariate geometric Brownian motion.

    Args:
        t0: start time (UTC)
        tf: end time (UTC)
        dt: time step
        names: single asset name or list of asset names (default: auto-generated)
        s0: initial price(s) (scalar or [n_assets] array, default: 1.0 per asset)
        mu: drift(s) (scalar or [n_assets] array, default: 0.0 per asset)
        sigma: covariance matrix [n,n], diagonal variances [n], or scalar variance
               (default: identity matrix)

    Returns:
        LazyFrame with columns: time_start, time_end, name, price
    """

    # If *no* optional arguments are provided, then set defaults.
    if names is None and s0 is None and mu is None and sigma is None:
        names = ["btc", "eth", "sol", "hype"]
        s0 = [100_000, 5_000, 150, 30]
        mu = [0.10, 0.15, 0.20, 0.25]
        # volatilities [50, 70, 100, 150] and pairwise correlations which decrease with price
        sigma = np.array([
            [0.50, 0.40, 0.30, 0.20],
            [0.40, 0.70, 0.25, 0.15],
            [0.30, 0.25, 1.00, 0.10],
            [0.20, 0.15, 0.10, 1.50],
        ])  # fmt: off

    # Infer n_assets from the first sized argument provided
    def _infer_n(x: object) -> int | None:
        if x is None or isinstance(x, (int, float)):
            return None
        if isinstance(x, str):
            return 1
        if isinstance(x, Sequence) and not isinstance(x, str):
            return len(x)
        a = np.asarray(x)
        if a.ndim == 0:
            return None
        return a.shape[0]

    dims = [d for x in (names, s0, mu, sigma) if (d := _infer_n(x)) is not None]
    n_assets = dims[0] if dims else 1

    # Normalize names
    if names is None:
        names = [f"asset_{i}" for i in range(n_assets)]
    elif isinstance(names, str):
        names = [names]

    # Normalize s0
    if s0 is None:
        s0 = np.ones(n_assets)
    elif isinstance(s0, (int, float)):
        s0 = np.array([float(s0)])
    else:
        s0 = np.asarray(s0, dtype=float)

    # Normalize mu
    if mu is None:
        mu = np.zeros(n_assets)
    elif isinstance(mu, (int, float)):
        mu = np.array([float(mu)])
    else:
        mu = np.asarray(mu, dtype=float)

    # Normalize sigma
    if sigma is None:
        sigma_mat = np.eye(n_assets)
    else:
        sigma_arr = np.asarray(sigma, dtype=float)
        if sigma_arr.ndim == 0:
            sigma_mat = np.eye(n_assets) * sigma_arr.item()
        elif sigma_arr.ndim == 1:
            sigma_mat = np.diag(sigma_arr)
        else:
            sigma_mat = sigma_arr

    # Validation
    if errors := [
        *checks.check_datetime_timezone(t0, timezone.utc),
        *checks.check_datetime_timezone(tf, timezone.utc),
        *checks.check_datetime_order(t0, tf),
        *checks.check_vector_length("s0", s0, n_assets),
        *checks.check_vector_length("mu", mu, n_assets),
        *checks.check_matrix_shape("sigma", sigma_mat, (n_assets, n_assets)),
        *checks.check_matrix_positive_semidefinite("sigma", sigma_mat),
    ]:
        raise ValueError(f"Invalid input\n{'\n'.join(f'- {e}' for e in errors)}")

    # Build time grid
    times_start = pl.datetime_range(t0, tf - dt, dt, eager=True).alias("time_start")
    times_end = pl.datetime_range(t0 + dt, tf, dt, eager=True).alias("time_end")
    n_steps = len(times_start)

    # GBM: S(t+dt) = S(t) * exp((mu - 0.5*diag(sigma))*dt + L @ dW)
    dt_years = dt / timedelta(days=365.25)
    L = np.linalg.cholesky(sigma_mat)
    dW = np.random.standard_normal((n_steps, n_assets))
    dX = dW @ L.T  # (n_steps, n_assets)

    drift = (mu - 0.5 * np.diag(sigma_mat)) * dt_years
    log_returns = drift + np.sqrt(dt_years) * dX @ L.T  # (n_steps, n_assets)

    prices = s0 * np.exp(np.cumsum(log_returns, axis=0))  # (n_steps, n_assets)

    # Build long-format DataFrame: repeat times for each asset, stack prices
    dfs: list[pl.DataFrame] = []
    for i, name in enumerate(names):
        dfs.append(
            pl.DataFrame(
                {
                    "time_start": times_start,
                    "time_end": times_end,
                    "name": pl.Series([name] * n_steps),
                    "price": prices[:, i],
                }
            )
        )

    out = pl.concat(dfs).lazy()
    return checks.check_schema(nw.from_native(out), schemas.PATHS_MARK).to_native()


def get_bars_spot(
    t0: datetime,
    tf: datetime,
    dt: timedelta,
    paths_mark: pl.LazyFrame,
    instruments: Collection[SpotInstrument],
) -> pl.LazyFrame:
    """Sample spot bars by adding noise to mark price paths.

    Args:
        t0: start time (UTC)
        tf: end time (UTC)
        dt: time step
        paths_mark: mark price paths (output of get_paths_mark)
        instruments: collection of spot instruments to sample bars for

    Returns:
        LazyFrame with columns:
            - time_start
            - time_end
            - exchange
            - base
            - quote
            - px_bid
            - px_ask
            - px_mark
    
    TODO: implementation
    """
    paths_mark = checks.check_schema(
        nw.from_native(paths_mark), schemas.PATHS_MARK
    ).to_native()
    out = pl.LazyFrame(schema=schemas.BARS_SPOT)
    return checks.check_schema(nw.from_native(out), schemas.BARS_SPOT).to_native()


# Seven daily contracts (08:00 UTC)
_DAILY = rrule(DAILY, byhour=8)
# Four weekly contracts (Fridays at 08:00 UTC)
_WEEKLY = rrule(WEEKLY, byweekday=FR, byhour=8)
# Three monthly contracts (last Friday of each month at 08:00 UTC)
_MONTHLY = rrule(MONTHLY, byweekday=FR(-1), byhour=8)
# Four quarterly contracts (last Friday of Mar, Jun, Sep, Dec at 08:00 UTC)
_QUARTERLY = rrule(MONTHLY, bymonth=(3, 6, 9, 12), byweekday=FR(-1), byhour=8)


def get_bars_option(
    marks: pl.LazyFrame,
    exchange: str,
    base: str,
    quote: str,
    rules: rrule | Collection[rrule] = (_WEEKLY, _MONTHLY, _QUARTERLY),
    n_moneynesses: int = 5,
    d_moneynesses: float = 0.1,
) -> pl.LazyFrame:
    """Sample option bars.

    Args:
        marks: mark price path LazyFrame (output of get_paths_mark)
        exchange: exchange identifier
        base: base asset identifier
        quote: quote asset identifier
        rules: one or more dateutil.rrule objects defining listing and expiry rules
        n_moneynesses: number of moneynesses to sample (default: 5, must be odd)
        d_moneynesses: step size between moneynesses (default: 0.1)

    Returns:
        LazyFrame with columns corresponding to BARS_OPTION schema
    """

    if base not in (names := marks.select(pl.col("name").unique()).collect().to_numpy().flatten().tolist()):  # fmt: off
        raise ValueError(f"Base asset {base} not found in marks (found: {names})")

    marks = marks.filter(pl.col("name").eq(base))
    t0: datetime = marks.select(pl.col("time_end").min()).collect().item()
    tf: datetime = marks.select(pl.col("time_end").max()).collect().item()

    def _get_listings_and_expiries(
        t0: datetime,
        tf: datetime,
        rules: rrule | Collection[rrule],
    ) -> pl.LazyFrame:
        if isinstance(rules, rrule):
            rules = (rules,)

        lfs: list[pl.LazyFrame] = []

        for rule in rules:
            r = rule.replace(dtstart=datetime(1, 1, 1, tzinfo=timezone.utc))
            listings = [r.before(t0, inc=True)] + r.between(t0, tf, inc=True)
            expiries = r.between(t0, tf, inc=True) + [r.after(tf, inc=True)]
            listings = pl.Series("listing", listings)
            expiries = pl.Series("expiry", expiries)
            lfs.append(pl.select(listings, expiries, eager=False))

        return pl.concat(lfs).sort("listing").group_by("expiry").first()

    def _get_moneynesses(n: int, dm: float) -> list[float]:
        if n % 2 == 0:
            raise ValueError("n must be odd")
        return [1 + dm * i for i in range(-n // 2 + 1, n // 2 + 1)]

    listings_and_expiries = _get_listings_and_expiries(t0, tf, rules)
    moneynesses = pl.Series(_get_moneynesses(5, 0.2)).to_frame("moneyness").lazy()
    kinds = pl.Series(["c", "p"]).to_frame("kind").lazy()

    out = listings_and_expiries \
        .join(marks, left_on="listing", right_on="time_end") \
        .join(kinds, how="cross") \
        .join(moneynesses, how="cross") \
        .with_columns((pl.col("price") * pl.col("moneyness")).round_sig_figs(2).alias("strike")) \
        .select(["listing", "expiry", "strike", "kind"]) \
        .join(marks, how="cross") \
        .filter([
            pl.col("time_start") <= pl.col("listing"),
            pl.col("time_end") <= pl.col("expiry"),
        ]) \
        .with_columns([
            # TODO: replace with SVI
            pl.lit(1.00).alias("iv_mark"),
            pl.lit(0.99).alias("iv_bid"),
            pl.lit(1.01).alias("iv_ask"),
        ]) \
        .with_columns([
            # TODO: replace with actual identifiers
            pl.lit("drbt").alias("exchange"),
            pl.lit("btc").alias("base"),
            pl.lit("usd").alias("quote"),
        ]) \
        .select([
            # Timestamps
            "time_start",
            "time_end",
            # Identifiers
            "exchange",
            "base",
            "quote",
            "strike",
            "listing",
            "expiry",
            "kind",
            # Values
            "iv_bid",
            "iv_ask",
            "iv_mark",
        ])  # fmt: off

    return checks.check_schema(nw.from_native(out), schemas.BARS_OPTION).to_native()
