## Setting

We have a single-asset ETF which wraps Ethereum.

We want a mental model for staking optimization.

We measure system state at daily increments.

We define *total assets* $X_t$ as the amount of ETH held by the fund during interval $(t-1, t)$.

*Staked* ETH pays a *reward rate* $r_t$, which varies over each interval $(t-1, t)$, but is assumed known at time $t$.

Liquid ETH can be staked by a decision at time $t$, but it will be locked and non-reward-bearing until a certain time delay has passed (the *entry queue* delay). This delay is random but known at time $t$, i.e. at time $t$ we know the time $t + \tau_t^{\text{entry}}$ at which an amount staked at $t$ will start earning rewards.

Similarly, there is an *exit queue* which impacts unstaking of staked ETH back into liquid ETH, during which the ETH does not earn rewards *and* is unavailable to service redemptions (see below).

The fund is subject to random creations and redemptions. Redemptions are limited to the size of the fund, while creations are unbounded in size.

Creations and redemptions may carry associated fees, which may be functions of creation/redemption size: e.g., we may charge (or incur) a fixed spread in basis points (bps) out of every create/redeem.

If the fund faces a redemption at time $t$ which is larger than the amount of liquid ETH available at time $t$, the fund will be forced to borrow ETH to service the redemption until a sufficient amount of liquid ETH becomes available. This comes with a substantial borrow cost which varies with time and is known at the time of borrowing. We expect this borrow cost to be correlated with staking queue delays.

In absence of the staking queues, we would want to keep the entire fund staked at all times. But due to the staking queue delays and the financing cost to cover redemptions, we require some optimal policy by which to set our staking ratio.

We have historical time-series data of the following schema which can be used to inform our policy:

| Dataset                | Fields                          | Description                                                                                                                            |
| ---------------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Fund Total Assets      | time, fund, amount              | Historical ETH amounts held by similar funds, which can be used to develop daily redemption time series via drawdowns in total assets. |
| Entry/Exit Queue Times | time, queue (entry/exit), delay | Historical entry/exit queue delay durations at each point in the past.                                                                 |

---

## Model — Queue-Adjusted VaR

**Philosophy**: Choose a single constant staking ratio $s \in [0,1]$ whose liquid buffer covers the cumulative redemptions that arrive while staked ETH is locked in the exit queue.

**Key observations**:

1. *Redemptions are serially correlated*. Stress episodes — large redemption days — cluster together. A single-day VaR ignores this and systematically understates the liquid buffer required.

2. *The exit queue is the binding constraint*. If liquid ETH runs short at time $t$ and we decide to unstake, those funds are unavailable for $\tau_t^{\text{exit}}$ days. During that entire window the fund must service redemptions from existing liquid ETH (or borrowing). The relevant risk quantity is therefore the *cumulative* fractional redemption over a $\tau^{\text{exit}}$-day horizon, not a single-day drawdown.

**Method**:

Define the fractional daily redemption $\rho_t = (X_{t-1} - X_t)^+ / X_{t-1}$ — the NAV drawdown from net outflows, floored at zero (creations are ignored since they add liquidity).

Let $\tau^*$ be a representative exit queue delay (e.g. the 90th-percentile of historical delays). The cumulative fractional redemption over a $\tau^*$-day window is:
$$P_t(\tau^*) = \sum_{k=0}^{\tau^*-1} \rho_{t-k}$$

Because redemptions are autocorrelated, $P_t(\tau^*)$ can substantially exceed $\tau^* \cdot \mathbb{E}[\rho_t]$. Set the required liquid buffer:
$$b^* = \widehat{\text{VaR}}_{95\%}\!\bigl(P_t(\tau^*)\bigr)$$

Optimal staking ratio: $s^* = 1 - b^*$, so that liquid ETH $L_t = (1 - s^*) X_t$ covers the 95th-percentile cumulative redemption demand over the full exit queue window.

**Limitation**: $s^*$ is time-invariant and uses a fixed horizon $\tau^*$; it does not adapt when the queue is currently short (where a higher $s$ is safe) or long (where a lower $s$ is warranted).

---

## Computing VaR Statistics

### Step 1 — Daily fractional redemptions

```python
import polars as pl

fund = pl.scan_csv("fund_total_assets.csv")  # columns: time, fund, amount

rho = (
    fund
    .sort("time")
    .with_columns(
        pl.col("amount").shift(1).over("fund").alias("prev_amount")
    )
    .with_columns(
        ((pl.col("prev_amount") - pl.col("amount")) / pl.col("prev_amount"))
        .clip(lower_bound=0)
        .alias("rho")
    )
    .drop_nulls("rho")
)
```

### Step 2 — Check serial dependence

Confirm autocorrelation before relying on it. Significant positive ACF at lags 1–5 motivates the multi-day horizon.

```python
rho_collected = rho.sort("time").select("rho").collect()

acf = (
    rho_collected
    .with_columns([
        pl.col("rho").shift(k).alias(f"lag_{k}") for k in range(1, 6)
    ])
    .select([
        pl.corr("rho", f"lag_{k}").alias(f"acf_lag_{k}") for k in range(1, 6)
    ])
)
print(acf)
```

### Step 3 — Exit queue delay distribution

```python
queues = pl.scan_csv("queue_times.csv")  # columns: time, queue, delay

tau_stats = (
    queues
    .filter(pl.col("queue") == "exit")
    .select(
        pl.col("delay").quantile(0.50).alias("p50"),
        pl.col("delay").quantile(0.90).alias("p90"),
        pl.col("delay").quantile(0.95).alias("p95"),
        pl.col("delay").max().alias("max"),
    )
    .collect()
)
print(tau_stats)

# Use p90 as the conservative-but-tractable risk horizon
tau_star = int(tau_stats["p90"].item())
```

### Step 4 — Queue-adjusted cumulative redemption VaR

```python
var_stats = (
    rho
    .sort("time")
    .with_columns(
        pl.col("rho")
        .rolling_sum(window_size=tau_star, min_periods=tau_star)
        .alias("rho_cumulative")
    )
    .select(
        pl.col("rho_cumulative").quantile(0.90).alias("var_90"),
        pl.col("rho_cumulative").quantile(0.95).alias("var_95"),
        pl.col("rho_cumulative").quantile(0.99).alias("var_99"),
    )
    .collect()
)
print(var_stats)

b_star = var_stats["var_95"].item()
s_star = 1 - b_star
print(f"Liquid buffer: {b_star:.2%}  →  s* = {s_star:.2%}")
```

### Step 5 — Sensitivity across queue horizons

Sweep over plausible $\tau$ values to understand how sensitive $s^*$ is to the assumed horizon.

```python
quantiles = [0.90, 0.95, 0.99]

rows = []
for tau in range(1, int(tau_stats["max"].item()) + 1):
    row = {"tau": tau}
    cumulative = (
        rho
        .sort("time")
        .select(
            pl.col("rho")
            .rolling_sum(window_size=tau, min_periods=tau)
            .alias("rho_cumulative")
        )
        .collect()
        .get_column("rho_cumulative")
        .drop_nulls()
    )
    for q in quantiles:
        row[f"var_{int(q*100)}"] = cumulative.quantile(q)
    rows.append(row)

sensitivity = pl.DataFrame(rows).with_columns(
    (1 - pl.col("var_95")).alias("s_star_95")
)
print(sensitivity)
```
