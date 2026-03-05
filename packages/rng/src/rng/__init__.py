import numpy as np
from numpy.typing import ArrayLike


def gbm(
    s0: float | ArrayLike = 1.0,
    mu: float | ArrayLike = 1.0,
    n: int = 365,
    *,
    sigma: float | ArrayLike | None = None,
    dt: float = 1 / 365,
    seed: int | None = None,
) -> np.ndarray:
    """Generate one or more correlated Geometric Brownian Motion paths.

    1-D mode (scalar s0 and mu):
        Returns array of shape (n+1,).

    2-D mode (array s0 and mu):
        Returns array of shape (n+1, d) where d = len(s0).

    Args:
        s0:    Initial value(s). Scalar for 1-D, shape (d,) for 2-D.
               Default: 1.0.
        mu:    Drift(s) per unit time. Scalar for 1-D, shape (d,) for 2-D.
               Default: 1.0.
        n:     Number of time steps. Default: 365.
        sigma: Volatility in 1-D mode (scalar); covariance matrix of shape
               (d, d) in 2-D mode. Default: 1.0 in 1-D, identity in 2-D.
        dt:    Size of each time step in annualised units. Default: 1/365.
        seed:  Optional random seed for reproducibility.

    Returns:
        Array of shape (n+1,) in 1-D mode or (n+1, d) in 2-D mode,
        with the initial value(s) at index 0.
    """
    s0 = np.asarray(s0, dtype=float)
    mu = np.asarray(mu, dtype=float)

    if s0.shape != mu.shape:
        raise ValueError(
            f"s0 and mu must have the same shape; got {s0.shape} and {mu.shape}"
        )
    if s0.ndim > 1:
        raise ValueError("s0 and mu must be scalar (1-D mode) or 1-D arrays (2-D mode)")
    if not np.all(s0 > 0):
        raise ValueError("s0 must be positive")
    if not np.all(np.isfinite(mu)):
        raise ValueError("mu must be finite")
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")

    rng = np.random.default_rng(seed)

    if s0.ndim == 0:  # 1-D mode
        sigma = 1.0 if sigma is None else float(sigma)
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be positive and finite")

        z = rng.standard_normal(n)
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        path = np.empty(n + 1)
        path[0] = float(s0)
        np.exp(log_returns, out=path[1:])
        np.cumprod(path[1:], out=path[1:])
        path[1:] *= float(s0)
        return path

    else:  # 2-D mode
        d = len(s0)
        sigma = np.eye(d) if sigma is None else np.asarray(sigma, dtype=float)
        if sigma.shape != (d, d):
            raise ValueError(f"sigma must have shape ({d}, {d}); got {sigma.shape}")
        if not np.allclose(sigma, sigma.T):
            raise ValueError("sigma must be symmetric")
        try:
            L = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            raise ValueError("sigma must be positive definite")

        var = np.diag(sigma)
        z = rng.standard_normal((n, d))
        dw = (z @ L.T) * np.sqrt(dt)
        log_returns = (mu - 0.5 * var) * dt + dw
        paths = np.empty((n + 1, d))
        paths[0] = s0
        np.exp(log_returns, out=paths[1:])
        np.cumprod(paths[1:], out=paths[1:], axis=0)
        paths[1:] *= s0
        return paths
