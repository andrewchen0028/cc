# r4 Requirements

Exhaustive requirements inferred from the full brainstorming history (r1 → r2 → r3 → r4/manual.py), WIP.md, and current source code (dtypes.py, checks.py, utils/dtypes.py).

---

## 1. Must-Have

Core requirements without which the backtester does not function.

### Architecture

- Event-driven architecture with a min-heap event queue.
- Batched timestep processing: `pop_all_at_t()` collects all events at the earliest time before any are processed.
- Event priority within a timestep (e.g., start < expiry < schedule < end) so that ordering is deterministic and semantically correct.
- Sequence number tie-breaking for events with the same time and priority.
- Event deduplication (e.g. no duplicate expiry events for same instrument held across multiple strategies).
- Dynamic event injection during the run loop (e.g., expiry events for newly acquired positions).

### Strategy Interface

- Strategy base class (ABC) with a scheduling rule (`rrule` / `rruleset`).
- Each strategy has a unique name.
- `on_schedule` as the primary abstract method.
- Strategies return `StrategyOutput` containing both **target positions** (declarative, for post-hoc analysis, "where the strategy wants to be") and **orders** (imperative, for execution, "how the strategy tries to get there").
- Strategies own "how to execute": they construct their own orders using position context to compute deltas. The engine does NOT convert targets to orders.
- Default no-op implementations for optional lifecycle event handlers.

### Events

- `OnScheduled`: fired per strategy according to its `rrule`.
- `OnExpiry`: fired when a held instrument reaches its expiry time.
- Events must be injectable into the queue mid-run (e.g. expiries discovered as positions change).

### Data Input

- Accept bar data as `pl.LazyFrame`, with schema validation at configuration time.
- Accept named signals as `pl.LazyFrame`, registered by string key.
- Time filtering for bars and signals uses `<=` (up to and including `t`), not `==`.

### Event Context

- `EventContext` (or equivalent) wraps all information visible to a strategy at time `t`.
- Provides access to (e.g.) current time, filtered bars, filtered signals, strategy's own position state, and portfolio-wide position state.

### Orders & Fills

- `MarketOrder` for immediate execution.
- `LimitOrder` for price-conditional execution.
- `Order = MarketOrder | LimitOrder` union type (from `dtypes.py`).
- `Fill` type recording instrument, originating order, quantity, and price.

### Position Management

- Engine-managed position state (strategies are ~stateless regarding positions).
- Collect per-strategy "target"/"desired" position histories (for analysis & diagnostics, not for execution)
- Represent using `OpenPosition` and `ClosedPosition` types (from `dtypes.py`).

### Instruments

- `OptionInstrument` with exchange, base, quote, strike, listing, expiry, kind.
- `SpotInstrument` with exchange, base, quote.
- `Instrument = OptionInstrument | SpotInstrument` union type.
- `OptionKind` enum (CALL / PUT).
- `TakerSide` (BUY/SELL), `MakerSide` (BID/ASK), `PositionSide` (LONG/SHORT) enums.
- Validation on instrument construction (`checks.require` for constraints, `checks.recommend` for conventions like 08:00 UTC expiry time [for Deribit options]).

### Configuration & Ergonomics

- Fluent API on `Backtest` itself (no separate `Builder` class): `.with_bars_*().with_signal().with_strategy().run(t0, tf)`.
- `build` and `run` merged into a single `.run()` call.
- Multiple strategies in one backtest.
- Validation at configuration time: prevent double-setting bars (`checks.is_none`), prevent duplicate signal/strategy names (`checks.not_in`).
- Immutable configuration via frozen dataclass with `replace()` for builder methods.

### Result

- `BacktestResult` returned from `.run()`, capturing the minimal data set necessary to derive everything else. Maybe that set is something like fill history and final positions. `BacktestResult` should have `@property` accessors which compute common aggregations and derived views from the raw result data.

---

## 2. Should-Have

Strongly implied by the design trajectory; expected in a complete implementation.

### Order Netting

- Net orders across strategies per instrument before execution.
- Internal crossing: when the net order quantity is zero (strategies cancel each other out), fill each side at mark price with no spread cost.
- Residual (non-zero net) goes to market.

### Target Position History

- Collect target positions per strategy per timestep for post-hoc analysis.
- `BacktestResult` exposes per-strategy target history.

### Position Materialization

- `Position` objects don't contain any price information etc., so we'll want some way to "materialize" it against an underling `bars` LazyFrame.

### Multiple Bar Types

- Separate bar inputs per asset class: `bars_rate`, `bars_spot`, `bars_futures_calendar`, `bars_futures_perpetual`, `bars_option`.
- Dedicated `with_bars_*()` method for each bar type.
- `Schemas` class defining the expected `pl.Schema` for each bar type.
- Each bar type is optional (leave `None` if data not available); strategies validate data presence lazily at `run()` time.

### Lifecycle Events

- `OnStart`: fired at `t0` before any schedule events.
- `OnEnd`: fired at `tf` after all schedule events.
- `OnFill`: informational callback after a fill occurs.

### Context Utilities

- `get_price(instrument)`: look up mark price from bars at time `t`.
- Progress bar (`tqdm`) during backtest execution.

### Validation

- Use `utils.checks` module consistently: `require()` for hard errors, `recommend()` for warnings.
- Schema validation on bar LazyFrames via `checks.has_schema()`.
- Phantom-typed IDs (`ID[T]` from `utils.dtypes`) for type-safe identification of orders, fills, and positions.

### Multi-Step Order Support

- Facilitation at least, if not implementation, of multi-timestep order types (e.g. limit, TWAP, VWAP) (see below)

---

## 3. Nice-to-Have

Features that appeared in sketches or are natural extensions of the architecture.

### Multi-Step Order Support

- Active orders mapping: track orders that persist across timesteps.
- `ActiveOrder` with status tracking: `pending` / `filled` / `cancelled`.
- Each timestep evaluates active orders against current market state (e.g., limit order checks price condition, TWAP splits across timesteps).
- Order lifecycle: strategies can submit, modify, and cancel active orders.
- `BacktestResult` exposes active orders remaining at end of run (should be empty for market-order-only backtests).

### Extensible Fill Model

- `FillModel` protocol for pluggable execution simulation, facilitating partial execution for multi-step orders (inspired by r1/hybrid.py).
- Required implementation: `InstantFill` (market orders)
- Example/indicative implementations `LimitFill` (price threshold), `TWAPFill` (time-weighted splitting), `VWAPFill` (volume-weighted splitting).
- Fill model receives bars/market data to determine fill price and quantity.

### Strategy Design

- `match`-statement event dispatch in `Strategy.on_event()` (from r4/manual.py) as alternative to per-event methods.
- Frozen slotted dataclass for `Strategy` itself (from r4) vs. mutable class (from r2/r3).
- Strategy-level parameters as dataclass fields rather than `__init__` assignments.

### Event Queue Deduplication

- General-purpose deduplication via `_seen: set[Event]` (from r4), not just expiry-specific dedup (from r2/r3).

### Capital & Margin Tracking

- Explicitly track cash and margin requirements.
- Reject orders that would exceed available capital/margin.

---

## 4. Ideas

Speculative possibilities hinted at across the development history, or natural extrapolations.

### Position Views

- `PositionView`: read-only view of a strategy's own positions (`.qty(instrument)`, `.instruments`, `.is_flat`).
- `PortfolioView`: read-only view of all strategies' positions (`.net_qty(instrument)`, `.all_instruments`, `.strategy_position(name)`).
- Fresh snapshots created per event dispatch (not stale references).

### Protocol vs. ABC

- WIP.md flags this as an open question: whether `Strategy` should be a `Protocol` (structural subtyping, more Pythonic for simple interfaces) or `ABC` (nominal subtyping, clearer contract). r1/position_centric used Protocol; all subsequent rounds used ABC.

### Additional Instrument Types

- `FuturesCalendarInstrument` and `FuturesPerpetualInstrument` types in `dtypes.py` — implied by the separate bar types (`bars_futures_calendar`, `bars_futures_perpetual`) in r4 but not yet defined.
- Extend `Instrument` union accordingly.

### Additional Order Types

- Beyond `MarketOrder` and `LimitOrder`: stop orders, stop-limit orders, or custom order types.
- Extend `Order` union in `dtypes.py`.

### Transaction Cost & Slippage Modeling

- Configurable fee model applied during fill generation. (spreads should be explicit from the provided bars data.)
- Slippage model based on order size relative to available liquidity.
- Extensibility to "quotes" data instead of just "bars", e.g. `with_quotes_spot(...)` where the schema contains level-2 quotes (side, level, price, quantity) [Definitely don't implement yet, but nice if the framework could be loosely extensible to this]

### Portfolio-Level Constraints

- Position limits per instrument or per strategy.
- Portfolio-level risk limits (e.g., max notional, max delta exposure).
- Engine-enforced constraints that reject orders violating limits.

### PnL & Performance Metrics [via `@propert` accessors]

- `BacktestResult` includes computed PnL (realized, unrealized, total).
- Equity curve as a time series.
- Standard performance metrics: Sharpe, drawdown, etc.

### Custom User-Defined Events

- Allow strategies or the engine to inject arbitrary event types beyond the built-in set.
- Have a `strategy_events` list which collects arbitrary (e.g. diagnostic) events from strategies throughout the backtest run.

### Multi-Exchange Support

- Some differences between exchanges, e.g. different fees.

