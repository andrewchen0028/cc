from datetime import datetime, timedelta, timezone


from backtester import io
from backtester.dtypes import SpotInstrument
from backtester import samplers

t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
tt = datetime(2023, 6, 30, tzinfo=timezone.utc)
tf = datetime(2023, 12, 31, tzinfo=timezone.utc)
dt = timedelta(hours=1)

path_rate = samplers.get_path_rate()
paths_mark = samplers.get_paths_mark()
bars_spot = paths_mark.pipe(samplers.to_bars_spot, ["binc", "cbse"], ["usd", "usdt"])
bars_option = paths_mark.pipe(samplers.to_bars_option, "drbt", "btc", "usd")

print(
    io.get_target_option(
        path_rate,
        bars_spot,
        bars_option,
        "drbt",
        "btc",
        "usd",
        "c",
        SpotInstrument("cbse", "btc", "usd"),
        target_time=tt,
        target_delta=0.50,
        target_tenor=timedelta(days=30),
    )
)
