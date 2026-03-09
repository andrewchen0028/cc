from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from backtester.dtypes import OptionInstrument, SpotInstrument
from backtester import io
from utils import checks, samplers, schemas


T0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
TT = datetime(2023, 1, 15, tzinfo=timezone.utc)
TF = datetime(2023, 2, 28, tzinfo=timezone.utc)
DT = timedelta(hours=1)


@pytest.fixture(scope="module")
def sample_data():
    path_rate = samplers.get_path_rate(T0, TF, DT)
    paths_mark = samplers.get_paths_mark(T0, TF, DT)
    bars_spot = paths_mark.pipe(samplers.to_bars_spot, ["binc", "cbse"], ["usd", "usdt"])
    bars_option = paths_mark.pipe(samplers.to_bars_option, "drbt", "btc", "usd")
    return path_rate, paths_mark, bars_spot, bars_option


# --- _build_lf_priced ---


def test_build_lf_priced_schema(sample_data):
    path_rate, _, bars_spot, bars_option = sample_data
    lf = io._build_lf_priced(
        path_rate, bars_spot, bars_option,
        "drbt", "btc", "usd", "cbse", "btc", "usd",
    )
    assert checks.has_schema(lf, schemas.BARS_PRICED) is None


def test_build_lf_priced_nonempty(sample_data):
    path_rate, _, bars_spot, bars_option = sample_data
    df = io._build_lf_priced(
        path_rate, bars_spot, bars_option,
        "drbt", "btc", "usd", "cbse", "btc", "usd",
    ).collect()
    assert df.height > 0


def test_build_lf_priced_greeks_finite(sample_data):
    path_rate, _, bars_spot, bars_option = sample_data
    df = io._build_lf_priced(
        path_rate, bars_spot, bars_option,
        "drbt", "btc", "usd", "cbse", "btc", "usd",
    ).collect()
    for col in ["delta", "gamma", "vega", "theta", "rho"]:
        assert df[col].is_nan().sum() == 0, f"{col} has NaNs"
        assert df[col].is_infinite().sum() == 0, f"{col} has Infs"


# --- get_bars_spot ---


def test_get_bars_spot_filters(sample_data):
    _, _, bars_spot, _ = sample_data
    spot = SpotInstrument("cbse", "btc", "usd")
    df = io.get_bars_spot(bars_spot, spot).collect()
    assert (df["exchange"] == "cbse").all()
    assert (df["base"] == "btc").all()
    assert (df["quote"] == "usd").all()


def test_get_bars_spot_time_bounds(sample_data):
    _, _, bars_spot, _ = sample_data
    spot = SpotInstrument("cbse", "btc", "usd")
    df = io.get_bars_spot(bars_spot, spot, start_time=TT, end_time=TF).collect()
    assert (df["time_start"] >= TT).all()
    assert (df["time_end"] <= TF).all()


# --- get_bars_option ---


def test_get_bars_option_filters(sample_data):
    _, _, _, bars_option = sample_data
    # Pick the first option from the data
    first = bars_option.head(1).collect()
    opt = OptionInstrument(
        exchange=first["exchange"].item(),
        base=first["base"].item(),
        quote=first["quote"].item(),
        strike=first["strike"].item(),
        listing=first["listing"].item(),
        expiry=first["expiry"].item(),
        kind=first["kind"].item(),
    )
    df = io.get_bars_option(bars_option, opt).collect()
    assert df.height > 0
    assert (df["strike"] == opt.strike).all()


# --- get_target_option ---


def test_get_target_option_returns_option_instrument(sample_data):
    path_rate, _, bars_spot, bars_option = sample_data
    result = io.get_target_option(
        path_rate, bars_spot, bars_option,
        "drbt", "btc", "usd", "c",
        SpotInstrument("cbse", "btc", "usd"),
        target_time=TT,
        target_delta=0.50,
        target_tenor=timedelta(days=30),
    )
    assert isinstance(result, OptionInstrument)
    assert result.kind == "c"


def test_get_target_option_rejects_mismatched_base(sample_data):
    path_rate, _, bars_spot, bars_option = sample_data
    with pytest.raises(ValueError, match="base assets must match"):
        io.get_target_option(
            path_rate, bars_spot, bars_option,
            "drbt", "btc", "usd", "c",
            SpotInstrument("cbse", "eth", "usd"),
            target_time=TT,
            target_delta=0.50,
            target_tenor=timedelta(days=30),
        )
