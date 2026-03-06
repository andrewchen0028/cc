## Constructs

```python
class SpotInstrument:
    exchange: str
    base: str
    quote: str

class OptionInstrument:
    exchange: str
    base: str
    quote: str
    strike: float
    expiry: datetime
    kind: Literal["c", "p"]

Instrument = SpotInstrument | OptionInstrument
```

## Data

We'll assume the following data tables:

### Rate

| Field      | Type                     |
| ---------- | ------------------------ |
| time_start | timestamp with time zone |
| time_end   | timestamp with time zone |
| rate       | float                    |

### Spot

| Field               | Type                     |
| ------------------- | ------------------------ |
| time_start          | timestamp with time zone |
| time_end            | timestamp with time zone |
| exchange            | string                   |
| base                | string                   |
| quote               | string                   |
| px_{bid, ask, mark} | float                    |

### Options

| Field               | Type                     |
| ------------------- | ------------------------ |
| time_start          | timestamp with time zone |
| time_end            | timestamp with time zone |
| exchange            | string                   |
| base                | string                   |
| quote               | string                   |
| strike              | float                    |
| expiry              | timestamp with time zone |
| kind                | string                   |
| iv_{bid, ask, mark} | float                    |

### Priced

This table is precomputed by joining the three raw tables above for a given spot/option exchange/base/quote, with greeks computed via Black-numerical.

| Field                              | Type                     | Source   |
| ---------------------------------- | ------------------------ | -------- |
| time_start                         | timestamp with time zone | (both)   |
| time_end                           | timestamp with time zone | (both)   |
| exchange_{spot, option}            | string                   | (each)   |
| base_{spot, option}                | string                   | (each)   |
| quote_{spot, option}               | string                   | (each)   |
| strike                             | float                    | option   |
| expiry                             | timestamp with time zone | option   |
| kind                               | string                   | option   |
| px_{bid, ask, mark}_{spot, option} | float                    | (each)   |
| iv_{bid, ask, mark}                | float                    | option   |
| rate                               | float                    | rate     |
| delta                              | float                    | computed |
| gamma                              | float                    | computed |
| vega                               | float                    | computed |
| theta                              | float                    | computed |
| rho                                | float                    | computed |

## Inputs

- `Backtester` will be initialized with a specific spot/option exchange/base/quote.
- `Backtester` will refer to the corresponding precomputed Priced table throughout.
- `Backtester` will take a `Strategy` object with methods:
  - `skip_trade(...) -> bool`
  - `get_target_position(...) -> Mapping[Instrument, float]`

Note, `get_target_position` may call Backtester method `get_target_option`:

```python
def get_target_option(
    self,
    exchange: str,
    base: str,
    quote: str,
    kind: str,
    target_time: datetime,
    target_delta: float,
    target_tenor: timedelta
) -> OptionInstrument:
    # From the Priced table, return an OptionInstrument constructed using the row with
    # `time_end`, `delta`, and `tenor` closest to the target parameters.
    ...
```
