# packages/utils/src/utils/checks.py
"""Utility functions for validation checks."""

from datetime import datetime, time, timedelta
from typing import Collection

import warnings

import numpy as np
import polars as pl


class WarningGroup(ExceptionGroup, UserWarning):
    """ExceptionGroup that can be issued via warnings.warn."""

    def __new__(cls, message: str, exceptions: list[Warning]) -> "WarningGroup":
        return super().__new__(cls, message, exceptions)


def require(*results: Exception | None) -> None:
    if errors := [e for e in results if e is not None]:
        if len(errors) == 1:
            raise errors[0]
        raise ExceptionGroup("validation errors", errors)


def recommend(*results: Exception | None) -> None:
    warns = [UserWarning(str(e)) for e in results if e is not None]
    if not warns:
        return
    if len(warns) == 1:
        warnings.warn(warns[0])
    else:
        warnings.warn(WarningGroup("validation warnings", warns))


def is_gt(left: str, a: object, right: str, b: object) -> ValueError | TypeError | None:
    try:
        _ = a > b  # type: ignore[operator]
    except Exception as e:
        return TypeError(f"can't compare {left} and {right}: {e}")

    if not a > b:  # type: ignore[operator]
        return ValueError(f"expected {left} > {right}, got {a!r} > {b!r}")
    return None


def is_ge(left: str, a: object, right: str, b: object) -> ValueError | TypeError | None:
    try:
        _ = a >= b  # type: ignore[operator]
    except Exception as e:
        return TypeError(f"can't compare {left} and {right}: {e}")

    if not a >= b:  # type: ignore[operator]
        return ValueError(f"expected {left} >= {right}, got {a!r} >= {b!r}")
    return None


def is_lt(left: str, a: object, right: str, b: object) -> ValueError | TypeError | None:
    try:
        _ = a < b  # type: ignore[operator]
    except Exception as e:
        return TypeError(f"can't compare {left} and {right}: {e}")

    if not a < b:  # type: ignore[operator]
        return ValueError(f"expected {left} < {right}, got {a!r} < {b!r}")
    return None


def is_le(left: str, a: object, right: str, b: object) -> ValueError | TypeError | None:
    try:
        _ = a <= b  # type: ignore[operator]
    except Exception as e:
        return TypeError(f"can't compare {left} and {right}: {e}")

    if not a <= b:  # type: ignore[operator]
        return ValueError(f"expected {left} <= {right}, got {a!r} <= {b!r}")
    return None


def is_eq(left: str, a: object, right: str, b: object) -> ValueError | TypeError | None:
    try:
        _ = a == b  # type: ignore[operator]
    except Exception as e:
        return TypeError(f"can't compare {left} and {right}: {e}")

    if not a == b:
        return ValueError(f"expected {left} == {right}, got {a!r} == {b!r}")
    return None


def is_in(name: str, obj: object, objs: Collection) -> ValueError | TypeError | None:
    try:
        if obj not in objs:
            return ValueError(f"{name} must be one of {set(objs)}, got {obj!r}")
    except Exception as e:
        return TypeError(f"can't check if {name} is in {objs}: {e}")
    return None


def is_utc(name: str, dt: datetime) -> ValueError | TypeError | None:
    if not isinstance(dt, datetime):
        return TypeError(f"{name} must be a datetime, got {type(dt).__name__}")
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        return ValueError(f"{name} must be UTC, got tzinfo={dt.tzinfo}")
    return None


def has_time(name: str, dt: datetime, expected: time) -> ValueError | TypeError | None:
    if not isinstance(dt, datetime):
        return TypeError(f"{name} must be a datetime, got {type(dt).__name__}")
    if dt.time() != expected:
        return ValueError(f"{name} time is not {expected}, got {dt.time()}")
    return None


def has_shape(
    name: str, a: np.typing.ArrayLike, expected: int | tuple[int, ...]
) -> ValueError | TypeError | None:
    arr = np.asarray(a, dtype=float)
    if isinstance(expected, int):
        expected = (expected,)
    if arr.shape != expected:
        return ValueError(f"{name} must have shape {expected}, got {arr.shape}")
    return None


def is_positive_semidefinite(
    name: str, m: np.typing.ArrayLike
) -> ValueError | TypeError | None:
    try:
        a = np.asarray(m, dtype=float)
        eigenvalues = np.linalg.eigvalsh(a)
        if np.any(eigenvalues < -1e-10):
            return ValueError(
                f"{name} must be positive semi-definite, got min eigenvalue {eigenvalues.min():.6e}"
            )
        return None
    except Exception as e:
        return TypeError(f"can't check if {name} is positive semi-definite: {e}")


def has_schema(lf: pl.LazyFrame, expected: pl.Schema) -> ValueError | TypeError | None:
    try:
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
            return ValueError("\n".join(errors))
        return None
    except Exception as e:
        return TypeError(f"can't check schema: {e}")
