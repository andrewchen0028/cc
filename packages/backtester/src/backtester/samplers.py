# packages/backtester/src/backtester/samplers3.py
"""Third-generation sample data generators for backtester module."""

from datetime import datetime, timedelta, timezone

import narwhals as nw
import numpy as np
import polars as pl

from backtester.backtester import checks


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
    SCHEMA = nw.Schema({
        "time_start": nw.Datetime(time_zone=timezone.utc),
        "time_end": nw.Datetime(time_zone=timezone.utc),
        "rate": nw.Float64(),
    })  # fmt: off

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
    })), SCHEMA).to_native()  # fmt: off
