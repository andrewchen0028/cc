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
        mu=[0.07, 0.12, 0.20],
        sigma=[[0.50, 0.05, 0.20],
               [0.05, 0.70, 0.50],
               [0.20, 0.50, 1.00]],
    )
    paths_mark.show()
    paths_mark.collect().to_pandas().pivot_table("price", "time_end", "name").plot.line()  # ty:ignore[unresolved-attribute]
    return


if __name__ == "__main__":
    app.run()
