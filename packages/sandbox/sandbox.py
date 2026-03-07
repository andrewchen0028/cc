# packages/sandbox/sandbox.py
"""Sandbox for testing out code snippets."""

from backtester.backtester import (
    # sample_polars_spot,
    # sample_polars_rate,
    # sample_polars_option,
    sample_duckdb_rate,
    sample_duckdb_spot,
    sample_duckdb_option,
    MarketDataProvider,
)
import duckdb


# lf_rate = sample_polars_rate()
# lf_spot = sample_polars_spot()
# lf_option = sample_polars_option()
with duckdb.connect() as con:
    lf_rate = sample_duckdb_rate(con)
    lf_spot = sample_duckdb_spot(con)
    lf_option = sample_duckdb_option(con)
    mdp = MarketDataProvider(lf_rate, lf_spot, lf_option)
    mdp.lf_priced.to_native().show()
