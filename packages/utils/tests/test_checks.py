from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from utils import checks


# --- require / recommend ---


def test_require_passes_on_all_none():
    checks.require(None, None, None)


def test_require_raises_on_error():
    with pytest.raises(ValueError, match="bad"):
        checks.require(None, "bad", None)


def test_require_joins_multiple_errors():
    with pytest.raises(ValueError, match="a\nb"):
        checks.require("a", "b")


def test_recommend_warns(recwarn):
    checks.recommend("heads up")
    assert len(recwarn) == 1
    assert "heads up" in str(recwarn[0].message)


def test_recommend_skips_none(recwarn):
    checks.recommend(None, None)
    assert len(recwarn) == 0


# --- comparisons ---


def test_is_gt_pass():
    assert checks.is_gt("a", 2, "b", 1) is None


def test_is_gt_fail():
    assert checks.is_gt("a", 1, "b", 2) is not None


def test_is_ge_pass():
    assert checks.is_ge("a", 2, "b", 2) is None


def test_is_ge_fail():
    assert checks.is_ge("a", 1, "b", 2) is not None


def test_is_lt_pass():
    assert checks.is_lt("a", 1, "b", 2) is None


def test_is_lt_fail():
    assert checks.is_lt("a", 2, "b", 1) is not None


def test_is_le_pass():
    assert checks.is_le("a", 2, "b", 2) is None


def test_is_le_fail():
    assert checks.is_le("a", 3, "b", 2) is not None


def test_is_eq_pass():
    assert checks.is_eq("a", 42, "b", 42) is None


def test_is_eq_fail():
    assert checks.is_eq("a", 1, "b", 2) is not None


def test_comparisons_work_with_datetimes():
    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    tf = datetime(2023, 2, 1, tzinfo=timezone.utc)
    assert checks.is_lt("t0", t0, "tf", tf) is None
    assert checks.is_lt("tf", tf, "t0", t0) is not None


# --- is_in ---


def test_is_in_pass():
    assert checks.is_in("kind", "c", ("c", "p")) is None


def test_is_in_fail():
    assert checks.is_in("kind", "x", ("c", "p")) is not None


# --- is_utc ---


def test_is_utc_pass_timezone_utc():
    dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
    assert checks.is_utc("dt", dt) is None


def test_is_utc_pass_zoneinfo_utc():
    dt = datetime(2023, 1, 1, tzinfo=ZoneInfo("UTC"))
    assert checks.is_utc("dt", dt) is None


def test_is_utc_fail_naive():
    dt = datetime(2023, 1, 1)
    assert checks.is_utc("dt", dt) is not None


def test_is_utc_fail_non_utc():
    dt = datetime(2023, 1, 1, tzinfo=ZoneInfo("US/Eastern"))
    assert checks.is_utc("dt", dt) is not None


# --- has_time ---


def test_has_time_pass():
    dt = datetime(2023, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    assert checks.has_time("dt", dt, time(8, 0, 0)) is None


def test_has_time_fail():
    dt = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
    assert checks.has_time("dt", dt, time(8, 0, 0)) is not None


# --- has_shape ---


def test_has_shape_vector_pass():
    assert checks.has_shape("v", [1, 2, 3], 3) is None


def test_has_shape_vector_fail():
    assert checks.has_shape("v", [1, 2], 3) is not None


def test_has_shape_matrix_pass():
    assert checks.has_shape("m", np.eye(3), (3, 3)) is None


def test_has_shape_matrix_fail():
    assert checks.has_shape("m", np.eye(2), (3, 3)) is not None


# --- is_positive_semidefinite ---


def test_is_positive_semidefinite_pass():
    assert checks.is_positive_semidefinite("m", np.eye(3)) is None


def test_is_positive_semidefinite_fail():
    m = np.array([[1, 2], [2, 1]])  # eigenvalues: 3, -1
    assert checks.is_positive_semidefinite("m", m) is not None


# --- has_schema ---


def test_has_schema_pass():
    lf = pl.LazyFrame({"a": [1], "b": ["x"]})
    schema = pl.Schema({"a": pl.Int64(), "b": pl.String()})
    assert checks.has_schema(lf, schema) is None


def test_has_schema_missing_column():
    lf = pl.LazyFrame({"a": [1]})
    schema = pl.Schema({"a": pl.Int64(), "b": pl.String()})
    err = checks.has_schema(lf, schema)
    assert err is not None
    assert "missing column" in err


def test_has_schema_wrong_dtype():
    lf = pl.LazyFrame({"a": [1], "b": [2]})
    schema = pl.Schema({"a": pl.Int64(), "b": pl.String()})
    err = checks.has_schema(lf, schema)
    assert err is not None
    assert "dtype mismatch" in err
