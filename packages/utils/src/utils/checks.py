# packages/utils/src/utils/checks.py
"""Utility functions for validation checks."""

from datetime import datetime, timezone

import numpy as np
import polars as pl


def check_datetime_timezone(dt: datetime, tzinfo: timezone) -> list[str]:
    if dt.tzinfo != tzinfo:
        return [f"{dt} must have {tzinfo=}, got {dt.tzinfo=}"]
    return []


def check_datetime_order(t0: datetime, tf: datetime, strict: bool = True) -> list[str]:
    if strict and tf <= t0:
        return [f"tf must be after t0, got t0={t0} and tf={tf}"]
    if not strict and tf < t0:
        return [f"tf must be at or after t0, got t0={t0} and tf={tf}"]
    return []


def check_array_shape(
    name: str, a: np.typing.ArrayLike, expected: int | tuple[int, ...]
) -> list[str]:
    arr = np.asarray(a, dtype=float)
    if isinstance(expected, int):
        expected = (expected,)
    if arr.shape != expected:
        return [f"{name} must have shape {expected}, got {arr.shape}"]
    return []


def check_matrix_positive_semidefinite(name: str, m: np.typing.ArrayLike) -> list[str]:
    a = np.asarray(m, dtype=float)
    eigenvalues = np.linalg.eigvalsh(a)
    if np.any(eigenvalues < -1e-10):
        return [
            f"{name} must be positive semi-definite, got min eigenvalue {eigenvalues.min():.6e}"
        ]
    return []


def check_schema(lf: pl.LazyFrame, expected: pl.Schema) -> pl.LazyFrame:
    actual = lf.collect_schema()
    errors = []

    for col, dtype in expected.items():
        if col not in actual:
            errors.append(f"missing column: {col}")
        elif str(actual[col]) != str(dtype):
            errors.append(
                f"dtype mismatch for '{col}':"
                f"\n\texpected: {dtype}"
                f"\n\tgot: {actual[col]}"
            )
    if errors:
        raise ValueError("\n".join(errors))
    return lf
