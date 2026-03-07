# packages/rng/src/rng/spot.py
"""GBM spot price path sampler."""
# NOTE: consider deprecating
from __future__ import annotations
from datetime import UTC, datetime, timedelta
from collections.abc import Collection, Sequence
from itertools import product

import numpy as np
from numpy.typing import ArrayLike
import polars as pl

from rng import gbm

_DEFAULT_START = datetime(2025, 1, 1, tzinfo=UTC)
_DEFAULT_END = datetime(2025, 12, 31, tzinfo=UTC)
_DEFAULT_INTERVAL = timedelta(minutes=1)


def sample_spot_bars(
    exchanges: Collection[str] = ("cbse",),
    base_assets: Collection[str] = ("btc",),
    quote_assets: Collection[str] = ("usd",),
    *,
    start: datetime = _DEFAULT_START,
    end: datetime = _DEFAULT_END,
    interval: timedelta = _DEFAULT_INTERVAL,
    s0: Sequence[float] | None = None,
    mu: float | ArrayLike | None = None,
    sigma: float | ArrayLike | None = None,
    spread: float = 0.001,
    seed: int | None = None,
) -> pl.LazyFrame:
    """Sample GBM spot price paths into a Polars LazyFrame.

    Args:
        exchanges:    Exchanges to generate bars for.
        base_assets:  Base assets to generate bars for.
        quote_assets: Quote assets to generate bars for.
        start:        Start time (UTC) of the generated bars.
        end:          End time (UTC, exclusive) of the generated bars.
        interval:     Duration of each bar (time_end - time_start).
        s0:           Initial mid-price per instrument. Defaults to ones.
        mu:           Annualised drift(s). Defaults to zeros.
        sigma:        Annualised vol (scalar, 1-D) or covariance matrix (d, d).
                      Defaults to identity.
        spread:       Fractional bid/ask spread around mid. Default 0.001.
        seed:         Random seed for reproducibility.

    Returns:
        LazyFrame with columns:
        exchange, base, quote, time_start, time_end, px_mark, px_bid, px_ask.
    """
    for name, dt in [("start", start), ("end", end)]:
        if dt.tzinfo is None:
            raise ValueError(f"{name} must be timezone-aware (UTC), got naive datetime")

    instruments = list(product(exchanges, base_assets, quote_assets))
    d = len(instruments)
    if s0 is None:
        s0 = [1.0] * d
    if mu is None:
        mu = np.zeros(d) if d > 1 else 0.0
    if sigma is None:
        sigma = np.eye(d) if d > 1 else 1.0
    if len(s0) != d:
        raise ValueError(f"len(s0)={len(s0)} != len(instruments)={d}")

    interval_seconds = interval.total_seconds()
    total_seconds = (end - start).total_seconds()
    n = int(total_seconds / interval_seconds)
    if n < 1:
        raise ValueError("end - start must be >= interval")

    dt = interval_seconds / (365.25 * 86400)

    if d == 1:
        paths = gbm(
            s0=s0[0],
            mu=float(np.asarray(mu).flat[0]),
            n=n,
            sigma=sigma,
            dt=dt,
            seed=seed,
        )
        paths = paths[1:].reshape(-1, 1)
    else:
        paths = gbm(s0=np.array(s0), mu=mu, n=n, sigma=sigma, dt=dt, seed=seed)
        paths = paths[1:]  # (n, d)

    spread_rng = np.random.default_rng(None if seed is None else seed + 1)
    half_spreads = spread / 2 * np.exp(spread_rng.normal(0, 0.4, size=(n, d)))
    half_spreads = np.clip(half_spreads, spread / 10, spread * 5)

    times_start = [start + interval * i for i in range(n)]
    times_end = [t + interval for t in times_start]

    frames: list[pl.LazyFrame] = []
    for j, (exchange, base, quote) in enumerate(instruments):
        mid = paths[:, j]
        hs = half_spreads[:, j]
        frames.append(
            pl.LazyFrame(
                {
                    "exchange": pl.Series([exchange] * n, dtype=pl.Utf8),
                    "base": pl.Series([base] * n, dtype=pl.Utf8),
                    "quote": pl.Series([quote] * n, dtype=pl.Utf8),
                    "time_start": times_start,
                    "time_end": times_end,
                    "px_mark": mid,
                    "px_bid": mid * (1 - hs),
                    "px_ask": mid * (1 + hs),
                }
            )
        )

    return pl.concat(frames)
