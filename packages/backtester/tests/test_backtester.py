import pytest
import ibis
from ibis.backends.duckdb import Backend

from backtester.backtester import Backtester


def test_backtester_empty_table():
    con: Backend = ibis.duckdb.connect()
    t = con.create_table("option_bars", schema=Backtester.SCHEMA_BARS_OPTION)
    Backtester(t)

def test_backtester_wrong_schema():
    con: Backend = ibis.duckdb.connect()
    t = con.create_table("option_bars", schema=Backtester.SCHEMA_BARS_SPOT)
    with pytest.raises(TypeError):
        Backtester(t)

