# packages/utils/src/utils/stats.py
"""Statistical functions."""

import narwhals as nw
import numpy as np


def norm_cdf(x: nw.Expr) -> nw.Expr:
    """Abramowitz-Stegun approximation for standard normal CDF.
    
    Use in place of scipy.stats.norm.cdf/scipy.special.ndtr, which are not supported by Polars/Narwhals.

    Args:
        x: Narwhals expression representing input value(s)

    Returns:
        Narwhals expression representing standard normal CDF of x    
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

    return nw.when(x >= 0).then(cdf_pos).otherwise(1.0 - cdf_pos)
