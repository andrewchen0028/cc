# CLAUDE.md

## Project Overview

Python 3.13+ monorepo using [uv](https://docs.astral.sh/uv/) workspaces. Option backtester with sampled market data.

## Project Structure

```
packages/
  backtester/   - Core: samplers, instruments, schemas, backtester
  utils/        - Shared validation (checks.py) and math (stats.py)
  sandbox/      - Jupyter notebooks for exploration
```

## Commands

- `uv sync` to install/update dependencies
- `uv add <pkg>` to add a dependency
- `task test` to run tests (Taskfile.yml)

## Code Style

### Imports

Stdlib, then third-party, then internal. Blank line between groups:

```python
from datetime import datetime, timedelta, timezone
from typing import Collection, Sequence

import narwhals as nw
import numpy as np
import polars as pl

from backtester import schemas
from utils import checks
```

### Narwhals / Polars

- Public functions accept and return `pl.LazyFrame`
- Wrap with `nw.from_native()` for narwhals ops, unwrap with `.to_native()`
- Schema validation: `checks.check_schema(nw.from_native(lf), schemas.X).to_native()`
- Use `utils.stats.norm_cdf` (Abramowitz-Stegun approximation) as a narwhals expression in place of scipy

### Formatting

- Backslash `\` line continuation for chained `.join().with_columns()` pipelines
- End multi-line expressions with `# fmt: off` (schema dicts, chains, matrices)
- Inline shape comments: `# (n_steps, n_assets)`

### Validation

- Validators return `list[str]` of errors (empty = valid)
- Aggregate with walrus: `if errors := [*check_a(), *check_b()]:`
- Normalize flexible inputs early: `[x] if isinstance(x, str) else x`

### Types & Data

- Union syntax: `str | None`, not `Optional[str]`
- Frozen slotted dataclasses for value objects: `@dataclass(frozen=True, slots=True)`
- Schemas defined as `nw.Schema({...})` in `backtester/schemas.py`

### Learning from Mistakes

When you introduce a bug and then fix it, add a note to the relevant section of this file so you don't repeat it.

### Polars Expression Pitfalls

- **Operator precedence with scalar * expr**: `scalar * expr_a * 60 + expr_b` binds as `((scalar * expr_a) * 60) + expr_b`, NOT `scalar * (expr_a * 60 + expr_b)`. Build the additive expression first, then scale: `(expr_a * 60 + expr_b).mul(scalar)`.
- **Int8 overflow in datetime accessors**: `.dt.hour()`, `.dt.minute()`, `.dt.weekday()` return `Int8`. Arithmetic like `hour * 60` overflows silently. Always cast to `Int64` or `Float64` before arithmetic: `.dt.hour().cast(pl.Int64) * 60`.

### Marimo Pitfalls

- **Variable uniqueness across cells**: Marimo requires all non-private variable names to be unique across the entire notebook. Loop variables, imports, and any other names that appear in multiple cells must be prefixed with `_` to make them cell-private. Always use `_` prefix for: loop iteration variables, temporary variables, and imports that appear in more than one cell (or move shared imports to a single cell).
