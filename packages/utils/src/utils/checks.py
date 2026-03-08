# packages/utils/src/utils/checks.py
"""Utility functions for validation checks."""

from datetime import datetime, timezone

import narwhals as nw


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


def check_schema(lf: nw.LazyFrame, expected: nw.Schema) -> nw.LazyFrame:
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
