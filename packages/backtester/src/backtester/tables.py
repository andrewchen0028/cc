from __future__ import annotations
from datetime import datetime
from dataclasses import dataclass
from typing import ClassVar
from backtester.instruments import OptionInstrument, SpotInstrument

import duckdb
import polars as pl


@dataclass(slots=True)
class OptionBarsTable:
    database: str
    table: str
    select: str = "*"

    group_keys: ClassVar[tuple[str, ...]] = ("exchange", "base", "quote", "kind", "strike", "expiry")  # fmt: off
    order_keys: ClassVar[tuple[str, ...]] = ("time_start", "time_end")
    value_keys: ClassVar[tuple[str, ...]] = ("iv_mark", "iv_bid", "iv_ask")
    where: ClassVar[str] = f"""
        {" AND ".join([f"{group_key} IS NOT NULL" for group_key in group_keys])} AND
        {" AND ".join([f"{order_key} IS NOT NULL" for order_key in order_keys])} AND
        {" AND ".join([f"{value_key} > 0" for value_key in value_keys])}
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
            kind = '{option.kind}' AND
            strike = {option.strike} AND
            expiry = '{option.expiry}'
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

    group_keys: ClassVar[tuple[str, ...]] = ("exchange", "base", "quote")
    order_keys: ClassVar[tuple[str, ...]] = ("time_start", "time_end")
    value_keys: ClassVar[tuple[str, ...]] = ("px_bid", "px_ask")
    where: ClassVar[str] = f"""
        {" AND ".join([f"{group_key} IS NOT NULL" for group_key in group_keys])} AND
        {" AND ".join([f"{order_key} IS NOT NULL" for order_key in order_keys])} AND
        {" AND ".join([f"{value_key} > 0" for value_key in value_keys])}
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
