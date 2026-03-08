import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    from datetime import datetime, timedelta, timezone

    import matplotlib.pyplot as plt

    from backtester import samplers

    return datetime, plt, samplers, timedelta, timezone


@app.cell
def _(plt):
    plt.rcParams["figure.facecolor"] = "darkgray"
    plt.rcParams["axes.facecolor"] = "lightgray"
    return


@app.cell
def _(datetime, timedelta, timezone):
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    tf = datetime(2025, 12, 31, tzinfo=timezone.utc)
    dt = timedelta(hours=1)
    return dt, t0, tf


@app.cell
def _(dt, samplers, t0, tf):
    path_rate = samplers.get_path_rate(t0, tf, dt)
    path_rate.show()
    path_rate.collect().to_pandas().plot.line("time_end", "rate")  # ty:ignore[unresolved-attribute]
    return


@app.cell
def _(dt, samplers, t0, tf):
    paths_mark = samplers.get_paths_mark(
        t0,
        tf,
        dt,
        names=["btc", "eth", "sol"],
        s0=[1500.0, 500.0, 100.0],
        mu=[0.07, 0.12, 0.20],
        sigma=[[0.50, 0.05, 0.20],
               [0.05, 0.70, 0.50],
               [0.20, 0.50, 1.00]],
    )
    paths_mark.show()
    paths_mark.collect().to_pandas().pivot_table("price", "time_end", "name").plot.line()  # ty:ignore[unresolved-attribute]
    return


@app.cell
def _(datetime, timedelta):
    from dateutil.rrule import rrule, DAILY, WEEKLY, MONTHLY, FR
    import polars as pl

    #     For now, hardcode the following contract introduction rules:
    #         - seven dailies (08:00 UTC)
    #         - four weeklies (Fridays at 08:00 UTC)
    #         - three monthlies (last Friday of each month at 08:00 UTC)
    #         - four quarterlies (last Friday of Mar, Jun, Sep, Dec at 08:00 UTC)
    #         - strikes from 10% to 190% moneyness in 10% increments
    #         - on duplicates, keep longest tenor
    daily = rrule(DAILY, byhour=8)
    weekly = rrule(WEEKLY, byweekday=FR, byhour=8)
    monthly = rrule(MONTHLY, byweekday=FR(-1), byhour=8)
    monthly = rrule(MONTHLY, bymonth=(3, 6, 9, 12), byweekday=FR(-1), byhour=8)


    def _get_option_expiries(t0: datetime, tf: datetime, dt: timedelta, rule: rrule) -> list[datetime]:
        rule: rrule = rule.replace(dtstart=t0, until=tf)
        return list(rule.xafter(t0))

    return FR, MONTHLY, daily, monthly, pl, rrule


@app.cell
def _(daily, dt, t0, tf):
    _get_option_expiries(t0, tf, dt, daily)
    return


@app.cell
def _(dt, pl, t0, tf):
    pl.datetime_range(t0, tf, dt, eager=True)[0]
    return


@app.cell
def _(monthly, rrule, t0, tf):
    rule: rrule = monthly.replace(dtstart=t0, until=tf)
    list(rule.xafter(t0))
    return


@app.cell
def _(FR, MONTHLY, rrule, t0, tf):
    list(rrule(freq=MONTHLY, byweekday=FR(-1), dtstart=t0, until=tf, byhour=8).xafter(t0))
    return


if __name__ == "__main__":
    app.run()
