# packages/utils/src/utils/stats.py
"""Statistical functions."""

import numpy as np
import polars as pl


def norm_cdf(x: pl.Expr) -> pl.Expr:
    """Abramowitz-Stegun approximation for standard normal CDF.

    Use in place of scipy.stats.norm.cdf/scipy.special.ndtr, which is not supported by Polars.

    Args:
        x: Polars expression representing input value(s)

    Returns:
        Polars expression representing standard normal CDF of x
    """
    p = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    z = x.abs()
    t = 1.0 / (1.0 + p * z)

    poly = (
        b1 * t
        + b2 * t * t
        + b3 * t * t * t
        + b4 * t * t * t * t
        + b5 * t * t * t * t * t
    )

    phi = (-0.5 * z * z).exp() / np.sqrt(2.0 * np.pi)
    cdf_pos = 1.0 - phi * poly

    return pl.when(x >= 0).then(cdf_pos).otherwise(1.0 - cdf_pos)
