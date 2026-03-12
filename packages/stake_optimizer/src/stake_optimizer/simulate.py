from datetime import date

import numpy as np
import polars as pl

SECURITIES: list[str] = ["ETHA", "ETHB", "ETHC", "ETHD"]
_START = date(2024, 1, 1)
_END = date(2024, 12, 31)
_US_PER_HOUR = 3_600_000_000


def _ou_process(
    rng: np.random.Generator, n: int, mu: float, theta: float, sigma: float, x0: float,
) -> np.ndarray:
    """Discretized Ornstein-Uhlenbeck mean-reverting process."""
    x = np.empty(n)
    x[0] = x0
    noise = rng.normal(0, sigma, n)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) + noise[i]
    return x


def simulate_eqy_sh_out(
    start: date = _START, end: date = _END,
    securities: list[str] | None = None, seed: int = 42,
) -> pl.LazyFrame:
    """Simulate daily shares outstanding for ETH ETFs.

    Flat most days, with intermittent autocorrelated create/redeem episodes.
    Each security starts at a different scale (30k–100M) to justify log-y.
    """
    rng = np.random.default_rng(seed)
    securities = securities or SECURITIES
    dates = pl.date_range(start, end, "1d", eager=True)
    n = len(dates)

    frames = []
    for sec in securities:
        s0 = 10 ** rng.uniform(4.5, 8)  # wide scale spread

        # Latent AR(1) "activity" process — drives clustered create/redeem episodes
        activity = np.zeros(n)
        for i in range(1, n):
            activity[i] = 0.92 * activity[i - 1] + rng.normal(0, 0.3)

        # Only move shares when activity is strong (sparse, autocorrelated bursts)
        active = np.abs(activity) > 1.3
        log_ret = np.where(active, activity * 0.04, 0.0)
        log_ret[0] = 0.0

        values = s0 * np.exp(np.cumsum(log_ret))
        frames.append(pl.DataFrame({
            "date": dates,
            "security": [sec] * n,
            "eqy_sh_out": values,
        }))

    return pl.concat(frames).lazy()


def simulate_queue_delays(
    start: date = _START, end: date = _END,
    mean_h: float = 24, seed: int = 43,
) -> pl.LazyFrame:
    """Simulate daily entry/exit queue delays.

    Mostly near *mean_h* hours, with autocorrelated spikes via
    exponentiated OU: hours = mean_h * exp(latent), latent ~ OU(0).
    """
    rng = np.random.default_rng(seed)
    dates = pl.date_range(start, end, "1d", eager=True)
    n = len(dates)

    # Latent OU centred at 0 — exp(0)=1 so baseline = mean_h.
    # theta=0.12 gives multi-day autocorrelation; sigma=0.35 gives occasional 2-5x spikes.
    entry_latent = _ou_process(rng, n, mu=0, theta=0.12, sigma=0.35, x0=0)
    exit_latent = _ou_process(rng, n, mu=0, theta=0.12, sigma=0.35, x0=0)
    entry_h = np.maximum(mean_h * np.exp(entry_latent), 0.5)
    exit_h = np.maximum(mean_h * np.exp(exit_latent), 0.5)

    return pl.DataFrame({
        "date": dates,
        "entry": pl.Series("entry", (entry_h * _US_PER_HOUR).astype(np.int64), dtype=pl.Duration("us")),
        "exit": pl.Series("exit", (exit_h * _US_PER_HOUR).astype(np.int64), dtype=pl.Duration("us")),
    }).lazy()


def simulate_reward_rate(
    start: date = _START, end: date = _END, seed: int = 44,
) -> pl.LazyFrame:
    """Simulate daily staking reward rate (OU process around ~4%)."""
    rng = np.random.default_rng(seed)
    dates = pl.date_range(start, end, "1d", eager=True)
    n = len(dates)

    rate = np.clip(
        _ou_process(rng, n, mu=0.04, theta=0.03, sigma=0.003, x0=0.04), 0.01, 0.10,
    )  # fmt: off

    return pl.DataFrame({"date": dates, "rate": rate}).lazy()
