# Backtester Architecture Notes

## Decided

### 1. Time Model: Unified Grid

The backtester owns a single time grid stepping at fixed `dt`. At each tick `t`,
each strategy checks its own `rrule` to decide whether to act. The backtester
does **not** union event times across strategies.

```python
for t in grid(t0, tf, dt):
    for strategy, weight in portfolio:
        if strategy.rule.after(t, inc=True) == t:  # rrule hit
            target = strategy.get_target_position(ctx)
```

### 2. Attribution: Per-Strategy

- Unnetted per-strategy target positions are always recorded.
- Combined/netted views are derived via aggregation, never stored directly.
- P&L attribution traces back to strategy-level fills.

### 3. Execution: Instant Fill at Bid/Ask

- Market orders cross the spread immediately.
- Limit orders fill when price is favorable.
- Designed for future order generators (e.g., TWAP that spreads a delta
  across the next N ticks).

### 4. Strategy Input: Frozen Snapshot Context

The strategy is purely functional: frozen context in, target positions out.

```python
class Strategy(Protocol):
    rule: rrule.rrule
    def get_target_position(self, ctx: Context) -> Mapping[Instrument, float]: ...
```

Any persistent state lives inside the strategy object, not in the context.

### 5. Position Lifecycle: Continuous Ledger (Append-Only)

- No position IDs. The ledger is append-only: `(time, instrument, qty_delta, strategy_id)`.
- Positions are derived views, not first-class objects.
- Target goes from +3 to +1 → append qty_delta = -2.

### 6. Composition: Portfolio / Book with `+` / `*` Operators

```python
straddle1 + 2 * straddle2
# builds:
Portfolio([(straddle1, 1.0), (straddle2, 2.0)])
```

The backtester iterates sub-strategies individually for unnetted visibility:

```python
for strategy, weight in portfolio:
    target_i = strategy.get_target_position(ctx)
    scaled_i = {k: v * weight for k, v in target_i.items()}
    record(strategy.id, scaled_i)           # unnetted history

net_target = merge(all_scaled_targets)      # netted target
orders = diff(net_target, positions_current) # netted orders
```

Preferred over `CompositeStrategy` because the backtester must see
individual strategies for the unnetted view requirement.

---

## Open Questions

### A. Backtester API

| Option | Description | Trade-off |
|--------|-------------|-----------|
| **Portfolio-always** | `run()` always takes Portfolio. Single strategy = `Portfolio([(s, 1.0)])`. | One code path; slightly verbose for single-strategy use. |
| **Overloaded run()** | Accepts `Strategy \| Portfolio`. | Convenience; two code paths. |
| **Strategy.iter_components()** | Protocol gains `iter_components()` defaulting to `yield self`. Backtester always uses it. | One code path; operators return Strategy; elegant but slightly magical. |

### B. Ledger Shape

| Option | Description | Trade-off |
|--------|-------------|-----------|
| **Single flat frame** | One LazyFrame, all strategies interleaved. Filter by `strategy_id`. | Simple; large frame. |
| **Per-strategy frames** | `dict[str, LazyFrame]`. Concat for combined view. | Natural grouping; more bookkeeping. |
| **Eager collect per tick** | Build rows as dicts/tuples during loop, convert to LazyFrame at end. | Pragmatic for iterative backtest; most memory-friendly during loop. |

### C. Composition Operator Return Type

Details TBD — what exactly does `__add__` / `__rmul__` return and how
does the type system see it?
