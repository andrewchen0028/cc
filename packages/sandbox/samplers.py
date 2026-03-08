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
def _(datetime, samplers, timedelta, timezone):
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    tf = datetime(2025, 12, 31, tzinfo=timezone.utc)
    dt = timedelta(hours=1)

    path_rate = samplers.get_path_rate(t0, tf, dt)
    path_rate.show()
    path_rate.collect().to_pandas().plot.line("time_end", "rate")  # ty:ignore[unresolved-attribute]
    return


if __name__ == "__main__":
    app.run()
