# packages/sandbox/sandbox.py
"""Sandbox for testing out code snippets."""

from backtester.backtester import (
    sample_polars_spot,
    sample_polars_rate,
    sample_polars_option,
    MarketDataProvider,
)

lf_rate = sample_polars_rate()
lf_spot = sample_polars_spot()
lf_option = sample_polars_option()
mdp = MarketDataProvider(lf_rate, lf_spot, lf_option)
mdp.lf_priced.to_native()
