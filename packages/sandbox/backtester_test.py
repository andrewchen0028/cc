from datetime import datetime, timedelta, timezone

from backtester.backtester import MarketDataProvider
from backtester import samplers

t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
tf = datetime(2023, 2, 1, tzinfo=timezone.utc)
dt = timedelta(hours=1)

path_rate = samplers.get_path_rate()
paths_mark = samplers.get_paths_mark()
bars_spot = paths_mark.pipe(samplers.to_bars_spot, ["binc", "cbse"], ["usd", "usdt"])
bars_option = paths_mark.pipe(
    samplers.to_bars_option,
    "drbt",
    "btc",
    "usd",
    rules=samplers._MONTHLY,
    n_log_moneynesses=3,
)

mdp = MarketDataProvider(path_rate, bars_spot, bars_option)
mdp._get_lf_priced().collect()
