# packages/backtester/src/backtester/samplers.py
"""Sample data generators for backtester module."""

from datetime import datetime, timedelta
from typing import Collection, Literal
import duckdb
import polars as pl
import narwhals as nw


def sample_polars_rate(
    start_time: datetime = datetime(2020, 1, 1),
    end_time: datetime = datetime(2020, 1, 10),
    freq: timedelta = timedelta(minutes=1),
) -> pl.LazyFrame:
    # TODO: implement actual sampling logic
    return pl.LazyFrame(schema={
        "time_start": pl.Datetime("us", "UTC"),
        "time_end": pl.Datetime("us", "UTC"),
        "rate": pl.Float64(),
    })  # fmt: off


def sample_duckdb_rate(
    con: duckdb.DuckDBPyConnection,
    start_time: datetime = datetime(2020, 1, 1),
    end_time: datetime = datetime(2020, 1, 10),
    freq: timedelta = timedelta(minutes=1),
) -> nw.LazyFrame:
    con.execute("SET TimeZone='UTC'")
    con.register("rate", sample_polars_rate(start_time, end_time, freq))
    return nw.from_native(con.table("rate"))


def sample_polars_spot(
    exchanges: Collection[str] | None = None,
    bases: Collection[str] | None = None,
    quotes: Collection[str] | None = None,
) -> pl.LazyFrame:
    # TODO: implement actual sampling logic
    return pl.LazyFrame(schema={
        "time_start": pl.Datetime("us", "UTC"),
        "time_end": pl.Datetime("us", "UTC"),
        "exchange": pl.String(),
        "base": pl.String(),
        "quote": pl.String(),
        "px_bid": pl.Float64(),
        "px_ask": pl.Float64(),
        "px_mark": pl.Float64(),
    })  # fmt: off


def sample_duckdb_spot(
    con: duckdb.DuckDBPyConnection,
    exchanges: Collection[str] | None = None,
    bases: Collection[str] | None = None,
    quotes: Collection[str] | None = None,
) -> nw.LazyFrame:
    con.execute("SET TimeZone='UTC'")
    con.register("spot", sample_polars_spot(exchanges, bases, quotes))
    return nw.from_native(con.table("spot"))


def sample_polars_option(
    exchanges: Collection[str] | None = None,
    bases: Collection[str] | None = None,
    quotes: Collection[str] | None = None,
    strikes: Collection[float] | None = None,
    expiries: Collection[datetime] | None = None,
    kinds: Collection[Literal["c", "p"]] | None = None,
) -> pl.LazyFrame:
    # TODO: implement actual sampling logic
    return pl.LazyFrame(schema={
        "time_start": pl.Datetime("us", "UTC"),
        "time_end": pl.Datetime("us", "UTC"),
        "exchange": pl.String(),
        "base": pl.String(),
        "quote": pl.String(),
        "strike": pl.Float64(),
        "expiry": pl.Datetime("us", "UTC"),
        "kind": pl.String(),
        "iv_bid": pl.Float64(),
        "iv_ask": pl.Float64(),
        "iv_mark": pl.Float64(),
    })  # fmt: off


def sample_duckdb_option(
    con: duckdb.DuckDBPyConnection,
    exchanges: Collection[str] | None = None,
    bases: Collection[str] | None = None,
    quotes: Collection[str] | None = None,
    strikes: Collection[float] | None = None,
    expiries: Collection[datetime] | None = None,
    kinds: Collection[Literal["c", "p"]] | None = None,
) -> nw.LazyFrame:
    con.execute("SET TimeZone='UTC'")
    con.register(
        "option",
        sample_polars_option(exchanges, bases, quotes, strikes, expiries, kinds),
    )
    return nw.from_native(con.table("option"))
