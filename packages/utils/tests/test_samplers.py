from datetime import datetime, timedelta, timezone

import pytest

from utils import checks, samplers, schemas


T0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
TF = datetime(2023, 2, 1, tzinfo=timezone.utc)
DT = timedelta(hours=1)


# --- get_path_rate ---


def test_get_path_rate_schema():
    lf = samplers.get_path_rate(T0, TF, DT)
    assert checks.has_schema(lf, schemas.PATH_RATE) is None


def test_get_path_rate_nonempty():
    df = samplers.get_path_rate(T0, TF, DT).collect()
    assert df.height > 0


def test_get_path_rate_time_order():
    df = samplers.get_path_rate(T0, TF, DT).collect()
    assert (df["time_start"] < df["time_end"]).all()


def test_get_path_rate_rejects_naive_datetime():
    with pytest.raises((ValueError, ExceptionGroup)):
        samplers.get_path_rate(datetime(2023, 1, 1), TF, DT)


def test_get_path_rate_rejects_reversed_times():
    with pytest.raises(ValueError):
        samplers.get_path_rate(TF, T0, DT)


# --- get_paths_mark ---


def test_get_paths_mark_schema():
    lf = samplers.get_paths_mark(T0, TF, DT)
    assert checks.has_schema(lf, schemas.PATHS_MARK) is None


def test_get_paths_mark_defaults():
    df = samplers.get_paths_mark(T0, TF, DT).collect()
    names = df["name"].unique().sort().to_list()
    assert names == ["btc", "eth", "hype", "sol"]


def test_get_paths_mark_single_asset():
    lf = samplers.get_paths_mark(T0, TF, DT, names="foo", s0=100.0, mu=0.1, sigma=0.5)
    df = lf.collect()
    assert df["name"].unique().to_list() == ["foo"]
    assert df.height > 0


def test_get_paths_mark_prices_positive():
    df = samplers.get_paths_mark(T0, TF, DT).collect()
    assert (df["price"] > 0).all()


def test_get_paths_mark_rejects_bad_shape():
    with pytest.raises(ValueError):
        samplers.get_paths_mark(T0, TF, DT, names=["a", "b"], s0=[1.0], mu=[0.0, 0.0])


# --- to_bars_spot ---


def test_to_bars_spot_schema():
    paths = samplers.get_paths_mark(T0, TF, DT)
    lf = paths.pipe(samplers.to_bars_spot, ["binc"], ["usd"])
    assert checks.has_schema(lf, schemas.BARS_SPOT) is None


def test_to_bars_spot_exchanges_and_quotes():
    paths = samplers.get_paths_mark(T0, TF, DT)
    df = paths.pipe(samplers.to_bars_spot, ["binc", "cbse"], ["usd", "usdt"]).collect()
    exchanges = df["exchange"].unique().sort().to_list()
    quotes = df["quote"].unique().sort().to_list()
    assert exchanges == ["binc", "cbse"]
    assert quotes == ["usd", "usdt"]


def test_to_bars_spot_bid_lt_ask():
    paths = samplers.get_paths_mark(T0, TF, DT)
    df = paths.pipe(samplers.to_bars_spot, ["binc"], ["usd"]).collect()
    assert (df["px_bid"] < df["px_ask"]).all()


# --- to_bars_option ---


def test_to_bars_option_schema():
    paths = samplers.get_paths_mark(T0, TF, DT)
    lf = paths.pipe(samplers.to_bars_option, "drbt", "btc", "usd")
    assert checks.has_schema(lf, schemas.BARS_OPTION) is None


def test_to_bars_option_has_calls_and_puts():
    paths = samplers.get_paths_mark(T0, TF, DT)
    df = paths.pipe(samplers.to_bars_option, "drbt", "btc", "usd").collect()
    kinds = df["kind"].unique().sort().to_list()
    assert kinds == ["c", "p"]


def test_to_bars_option_rejects_missing_base():
    paths = samplers.get_paths_mark(T0, TF, DT)
    with pytest.raises(ValueError, match="not found"):
        paths.pipe(samplers.to_bars_option, "drbt", "nonexistent", "usd").collect()
