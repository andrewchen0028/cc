import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    # Can XGBoost learn seasonalities from phase features alone?
    # We generate minutely data with time-of-day and day-of-week patterns,
    # then compare models with progressively richer feature sets.
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    import marimo as mo
    import numpy as np
    import polars as pl
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error

    return mean_squared_error, mo, np, pl, plt, xgb


@app.cell
def _(plt):
    plt.rcParams["figure.facecolor"] = "darkgray"
    plt.rcParams["axes.facecolor"] = "lightgray"
    return


@app.cell
def _(np, pl, plt):
    # --- Generate minutely data with two seasonalities ---
    rng = np.random.default_rng(42)
    n_days = 30 * 6
    minutes_per_day = 1440
    n = n_days * minutes_per_day

    ts = pl.datetime_range(
        pl.datetime(2025, 1, 1),
        pl.datetime(2025, 1, 1) + pl.duration(minutes=n - 1),
        interval="1m",
        eager=True,
    ).alias("timestamp")

    mod = np.arange(n) % minutes_per_day
    dow = np.repeat(np.arange(n_days), minutes_per_day) % 7

    # Time-of-day seasonality: sine wave with period = 1 day
    y_tod = 5 * np.sin(2 * np.pi * mod / minutes_per_day)
    # Day-of-week seasonality: higher on weekdays (0-4), lower on weekends (5-6)
    y_dow = 2.5 * np.where(dow < 5, 1, -1).astype(np.float64)
    y = y_tod + y_dow + rng.normal(0, 1.0, n)

    df = pl.DataFrame({
        "timestamp": ts,
        "y": y,
    }).with_columns(
        # Phase features — encode cyclic position
        (pl.col("timestamp").dt.hour().cast(pl.Int64) * 60 + pl.col("timestamp").dt.minute().cast(pl.Int64))
        .mul(2 * np.pi)
        .truediv(minutes_per_day)
        .alias("tod_phase"),  # [0, 2π) radians
        pl.col("timestamp").dt.weekday().cast(pl.Float64)
        .mul(2 * np.pi)
        .truediv(7)
        .alias("dow_phase"),  # [0, 2π) radians
    ).with_columns(
        pl.col("tod_phase").cos().alias("tod_cos"),
        pl.col("tod_phase").sin().alias("tod_sin"),
        pl.col("dow_phase").cos().alias("dow_cos"),
        pl.col("dow_phase").sin().alias("dow_sin"),
    ).drop_nulls()  # drop first row (lag1 is null)

    # Train/test split: last 7 days = test
    split_ts = df["timestamp"].max() - pl.duration(days=30)
    train = df.filter(pl.col("timestamp") <= split_ts)
    test = df.filter(pl.col("timestamp") > split_ts)

    print(f"Train: {train.height:,} rows, Test: {test.height:,} rows")
    plt.plot(df["timestamp"], df["y"])
    plt.axvline(train["timestamp"].max(), color="tab:orange")
    return df, test, train


@app.cell
def _(df, plt):
    plt.plot(df["tod_sin"][:1440 * 7])
    plt.plot(df["tod_cos"][:1440 * 7])
    plt.plot(df["dow_sin"][:1440 * 7])
    plt.plot(df["dow_cos"][:1440 * 7])
    return


@app.cell
def _(mean_squared_error, np, test, train, xgb):
    # --- Define feature sets ---
    feature_sets = {
        # "1) lag": ["lag1"],
        "2) + time-of-day": ["tod_cos", "tod_sin"],  # , "lag1"
        "3) + day-of-week": ["tod_cos", "tod_sin", "dow_cos", "dow_sin"],  # , "lag1"
    }

    xgb_params = {
        "n_estimators": 512,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 0,
        "n_jobs": -1,
    }

    results = {}
    models = {}
    predictions = {}

    for _label, _feats in feature_sets.items():
        X_train = train.select(_feats).to_numpy()
        y_train = train["y"].to_numpy()
        X_test = test.select(_feats).to_numpy()
        y_test = test["y"].to_numpy()

        _model = xgb.XGBRegressor(**xgb_params)
        _model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        _preds = _model.predict(X_test)
        _rmse = np.sqrt(mean_squared_error(y_test, _preds))
        results[_label] = _rmse
        models[_label] = _model
        predictions[_label] = _preds

    for _label, _rmse in results.items():
        print(f"{_label:30s}  RMSE = {_rmse:.4f}")
    return feature_sets, models, predictions, results, xgb_params


@app.cell
def _(mo, results):
    # --- Summary table ---
    mo.md(
        "| Model | RMSE |\n|---|---|\n"
        + "\n".join(f"| {k} | {v:.4f} |" for k, v in results.items())
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Plots

    Top: actual vs predicted on the test set (first 2 days).
    Bottom: residuals over the same window.
    """)
    return


@app.cell
def _(plt, predictions, test):
    _fig, _axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Show first 7 days of test set for readability
    _window = 7 * 1440
    _ts = test["timestamp"][:_window].to_numpy()
    _y_actual = test["y"][:_window].to_numpy()

    _ax_pred, _ax_resid = _axes

    _ax_pred.plot(_ts, _y_actual, label="actual", color="black", linewidth=0.6, alpha=0.2, linestyle="none", marker=".")

    _colors = ["tab:red", "tab:blue", "tab:green"]
    for (_label, _preds), _color in zip(predictions.items(), _colors):
        _p = _preds[:_window]
        _ax_pred.plot(_ts, _p, label=_label, color=_color, linewidth=0.8, alpha=0.2, linestyle="none", marker=".")
        _ax_resid.plot(_ts, _y_actual - _p, label=_label, color=_color, linewidth=0.5, alpha=0.2, linestyle="none", marker=".")

    _ax_pred.set_ylabel("y")
    _ax_pred.legend(fontsize=8)
    _ax_pred.set_title("Actual vs Predicted (first 7 days of test)")

    _ax_resid.axhline(0, color="black", linewidth=0.5)
    _ax_resid.set_ylabel("residual")
    _ax_resid.set_xlabel("time")
    _ax_resid.legend(fontsize=8)

    plt.tight_layout()
    _fig
    return


@app.cell
def _(feature_sets, models, plt):
    # --- Feature importance for the full model ---
    _full_label = "3) + day-of-week"
    _full_feats = feature_sets[_full_label]
    _importances = models[_full_label].feature_importances_

    _fig2, _ax2 = plt.subplots(figsize=(8, 4))
    _ax2.barh(_full_feats, _importances)
    _ax2.set_xlabel("Feature importance (gain)")
    _ax2.set_title(f"Feature importance — {_full_label}")
    plt.tight_layout()
    _fig2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Per-round RMSEs
    """)
    return


@app.cell
def _(feature_sets, mean_squared_error, np, test, train, xgb, xgb_params):
    results_by_round = {}
    models_by_round = {}
    predictions_by_round = {}
    rmse_by_round = {}

    for _label, feats in feature_sets.items():
        _X_train = train.select(feats).to_numpy()
        _y_train = train["y"].to_numpy()
        _X_test = test.select(feats).to_numpy()
        _y_test = test["y"].to_numpy()

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            **xgb_params,
        )

        model.fit(
            _X_train,
            _y_train,
            eval_set=[(_X_train, _y_train), (_X_test, _y_test)],
            verbose=False,
        )

        preds = model.predict(_X_test)
        final_rmse = np.sqrt(mean_squared_error(_y_test, preds))

        evals = model.evals_result()
        train_rmse_curve = np.array(evals["validation_0"]["rmse"], dtype=float)
        test_rmse_curve = np.array(evals["validation_1"]["rmse"], dtype=float)

        results_by_round[_label] = final_rmse
        models_by_round[_label] = model
        predictions_by_round[_label] = preds
        rmse_by_round[_label] = {
            "train": train_rmse_curve,
            "test": test_rmse_curve,
        }

    for _label, rmse in results_by_round.items():
        print(f"{_label:30s} final test RMSE = {rmse:.4f}")
    return models_by_round, rmse_by_round


@app.cell
def _(np, plt, rmse_by_round):
    fig, ax = plt.subplots(figsize=(10, 5))

    for _label, _curves in rmse_by_round.items():
        rounds = np.arange(1, len(_curves["train"]) + 1)

        ax.plot(rounds, _curves["train"], label=f"{_label} train", linewidth=1.2)
        ax.plot(rounds, _curves["test"], linestyle="--", label=f"{_label} test", linewidth=1.2)

    ax.set_xlabel("Boosting round")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE by boosting round")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(feature_sets, models_by_round, plt):
    # --- Feature importance for the full model ---
    _full_label = "3) + day-of-week"
    _full_feats = feature_sets[_full_label]
    _importances = models_by_round[_full_label].feature_importances_

    _fig2, _ax2 = plt.subplots(figsize=(8, 4))
    _ax2.barh(_full_feats, _importances)
    _ax2.set_xlabel("Feature importance (gain)")
    _ax2.set_title(f"Feature importance — {_full_label}")
    plt.tight_layout()
    _fig2
    return


if __name__ == "__main__":
    app.run()
