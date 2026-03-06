from __future__ import annotations
from datetime import datetime
from dataclasses import dataclass
from typing import ClassVar
from backtester.instruments import OptionInstrument, SpotInstrument

import duckdb
import polars as pl


@dataclass(slots=True)
class CombinedTable:
    database: str
    """E.g. "~/combined.db"."""
    table: str
    """E.g. "combined"."""
    select: str = "*"
    """Used to override default select statement."""

    ORDER_KEYS: ClassVar[tuple[str, ...]] = ("time_start", "time_end")
    GROUP_KEYS: ClassVar[tuple[str, ...]] = (
        "exchange",
        "base",
        "quote",
        "strike",
        "expiry",
        "kind",
    )
    VALUE_KEYS: ClassVar[tuple[str, ...]] = ("iv_mark", "iv_bid", "iv_ask", "px_mark", "px_bid", "px_ask", "")


@dataclass(slots=True)
class OptionBarsTable:
    database: str
    table: str
    select: str = "*"

    GROUP_KEYS: ClassVar[tuple[str, ...]] = ("exchange", "base", "quote", "strike", "expiry", "kind")  # fmt: off
    ORDER_KEYS: ClassVar[tuple[str, ...]] = ("time_start", "time_end")
    VALUE_KEYS: ClassVar[tuple[str, ...]] = ("iv_mark", "iv_bid", "iv_ask")
    where: ClassVar[str] = f"""
        {" AND ".join([f"{group_key} IS NOT NULL" for group_key in GROUP_KEYS])} AND
        {" AND ".join([f"{order_key} IS NOT NULL" for order_key in ORDER_KEYS])} AND
        {" AND ".join([f"{value_key} > 0" for value_key in VALUE_KEYS])}
    """

    _con: duckdb.DuckDBPyConnection | None = None

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect(self.database, read_only=True)
        return self._con

    def __enter__(self) -> OptionBarsTable:
        self._con = duckdb.connect(self.database, read_only=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __post_init__(self) -> None:
        self.con.execute(f"SELECT {self.select} FROM {self.table} WHERE {self.where}")

    def get_bars(
        self,
        option: OptionInstrument,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pl.LazyFrame:
        query = f"""
        SELECT {self.select}
        FROM {self.table}
        WHERE {self.where} AND
            exchange = '{option.exchange}' AND
            base = '{option.base}' AND
            quote = '{option.quote}' AND
            strike = {option.strike} AND
            expiry = '{option.expiry}' AND
            kind = '{option.kind}'
        """
        if start_time is not None:
            query += f" AND time_start >= '{start_time}'"
        if end_time is not None:
            query += f" AND time_end < '{end_time}'"
        return self.con.query(query).pl(lazy=True)


@dataclass(slots=True)
class SpotBarsTable:
    database: str
    table: str
    select: str = "*"

    GROUP_KEYS: ClassVar[tuple[str, ...]] = ("exchange", "base", "quote")
    ORDER_KEYS: ClassVar[tuple[str, ...]] = ("time_start", "time_end")
    VALUE_KEYS: ClassVar[tuple[str, ...]] = ("px_bid", "px_ask")
    where: ClassVar[str] = f"""
        {" AND ".join([f"{group_key} IS NOT NULL" for group_key in GROUP_KEYS])} AND
        {" AND ".join([f"{order_key} IS NOT NULL" for order_key in ORDER_KEYS])} AND
        {" AND ".join([f"{value_key} > 0" for value_key in VALUE_KEYS])}
    """

    _con: duckdb.DuckDBPyConnection | None = None

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect(self.database, read_only=True)
        return self._con

    def __enter__(self) -> SpotBarsTable:
        self._con = duckdb.connect(self.database, read_only=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __post_init__(self) -> None:
        self.con.execute(f"SELECT {self.select} FROM {self.table} WHERE {self.where}")

    def get_bars(
        self,
        spot: SpotInstrument,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pl.LazyFrame:
        query = f"""
        SELECT {self.select}
        FROM {self.table}
        WHERE {self.where} AND
            exchange = '{spot.exchange}' AND
            base = '{spot.base}' AND
            quote = '{spot.quote}'
        """
        if start_time is not None:
            query += f" AND time_start >= '{start_time}'"
        if end_time is not None:
            query += f" AND time_end < '{end_time}'"
        return self.con.query(query).pl(lazy=True)
