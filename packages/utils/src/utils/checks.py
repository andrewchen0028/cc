# packages/utils/src/utils/checks.py
"""Utility functions for validation checks."""

import warnings
from datetime import datetime, time, timedelta
from typing import Collection

import numpy as np
import polars as pl


def require(*results: str | None) -> None:
    if errors := [e for e in results if e is not None]:
        raise ValueError("\n".join(errors))


def recommend(*results: str | None) -> None:
    for w in results:
        if w is not None:
            warnings.warn(w)


def is_gt(left: str, a: object, right: str, b: object) -> str | None:
    try:
        if not a > b:  # type: ignore[operator]
            return f"expected {left} > {right}, got {a!r} > {b!r}"
    except TypeError as e:
        return f"expected {left} > {right}, got {e}"
    return None


def is_ge(left: str, a: object, right: str, b: object) -> str | None:
    try:
        if not a >= b:  # type: ignore[operator]
            return f"expected {left} >= {right}, got {a!r} >= {b!r}"
    except TypeError as e:
        return f"expected {left} >= {right}, got {e}"
    return None


def is_lt(left: str, a: object, right: str, b: object) -> str | None:
    try:
        if not a < b:  # type: ignore[operator]
            return f"expected {left} < {right}, got {a!r} < {b!r}"
    except TypeError as e:
        return f"expected {left} < {right}, got {e}"
    return None


def is_le(left: str, a: object, right: str, b: object) -> str | None:
    try:
        if not a <= b:  # type: ignore[operator]
            return f"expected {left} <= {right}, got {a!r} <= {b!r}"
    except TypeError as e:
        return f"expected {left} <= {right}, got {e}"
    return None


def is_eq(left: str, a: object, right: str, b: object) -> str | None:
    if not a == b:
        return f"expected {left} == {right}, got {a!r} == {b!r}"
    return None


def is_in(name: str, obj: object, objs: Collection) -> str | None:
    if obj not in objs:
        return f"{name} must be one of {set(objs)}, got {obj!r}"
    return None


def is_utc(name: str, dt: datetime) -> str | None:
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        return f"{name} must be UTC, got tzinfo={dt.tzinfo}"
    return None


def has_time(name: str, dt: datetime, expected: time) -> str | None:
    if dt.time() != expected:
        return f"{name} time is not {expected}, got {dt.time()}"
    return None


def has_shape(
    name: str, a: np.typing.ArrayLike, expected: int | tuple[int, ...]
) -> str | None:
    arr = np.asarray(a, dtype=float)
    if isinstance(expected, int):
        expected = (expected,)
    if arr.shape != expected:
        return f"{name} must have shape {expected}, got {arr.shape}"
    return None


def is_positive_semidefinite(name: str, m: np.typing.ArrayLike) -> str | None:
    a = np.asarray(m, dtype=float)
    eigenvalues = np.linalg.eigvalsh(a)
    if np.any(eigenvalues < -1e-10):
        return f"{name} must be positive semi-definite, got min eigenvalue {eigenvalues.min():.6e}"
    return None


def has_schema(lf: pl.LazyFrame, expected: pl.Schema) -> str | None:
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
        return "\n".join(errors)
    return None
