# packages/sandbox/sandbox.py
"""Sandbox for testing out code snippets."""

from backtester.backtester import MarketDataProvider
from backtester import samplers
import duckdb


# Test polars
lf_rate = samplers.sample_polars_rate()
lf_spot = samplers.sample_polars_spot()
lf_option = samplers.sample_polars_option()
mdp = MarketDataProvider(lf_rate, lf_spot, lf_option)
print(mdp.lf_priced.collect_schema())


# Test duckdb
with duckdb.connect() as con:
    lf_rate = samplers.sample_duckdb_rate(con)
    lf_spot = samplers.sample_duckdb_spot(con)
    lf_option = samplers.sample_duckdb_option(con)
    mdp = MarketDataProvider(lf_rate, lf_spot, lf_option)
    print(mdp.lf_priced.collect_schema())
