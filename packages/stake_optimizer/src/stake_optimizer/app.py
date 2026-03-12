import dash
from dash import Input, Output, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from stake_optimizer.adapters import (
    adapt_eqy_sh_out,
    adapt_queue_delays,
    adapt_reward_rate,
)
from stake_optimizer.analytics import drawdown_var_cvar, max_drawdown_n
from stake_optimizer.simulate import (
    SECURITIES,
    simulate_eqy_sh_out,
    simulate_queue_delays,
    simulate_reward_rate,
)

_COLORS = dict(zip(SECURITIES, px.colors.qualitative.Plotly[: len(SECURITIES)]))
_N_VALUES = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90]
_Q_VALUES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
_MKR = 4  # marker size
_PLOT_H = 800  # shared height for both columns

# 3 panels sharing one xaxis.
_DOMAINS = {
    "y":  [0.68, 1.0],   # Row 1: Shares Outstanding
    "y2": [0.36, 0.64],  # Row 2: Max Drawdown
    "y3": [0.0,  0.32],  # Row 3: Queue Delays
}


def create_app() -> dash.Dash:
    eqy_df = adapt_eqy_sh_out(simulate_eqy_sh_out()).collect()
    delays_df = adapt_queue_delays(simulate_queue_delays()).collect()
    adapt_reward_rate(simulate_reward_rate())

    # Pre-extract raw and scaled series per security
    eqy_by_sec: dict[str, dict] = {}
    for sec in SECURITIES:
        s = eqy_df.filter(pl.col("security") == sec)
        raw = s["eqy_sh_out"].to_list()
        peak = max(raw)
        eqy_by_sec[sec] = {
            "dates": s["date"].to_list(),
            "raw": raw,
            "scaled": [v / peak for v in raw],
        }

    delays_hours = delays_df.with_columns(
        (pl.col("entry").dt.total_seconds() / 3600).alias("entry_h"),
        (pl.col("exit").dt.total_seconds() / 3600).alias("exit_h"),
    )  # fmt: off
    delay_dates = delays_hours["date"].to_list()
    entry_h = delays_hours["entry_h"].to_list()
    exit_h = delays_hours["exit_h"].to_list()

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

    app.layout = dbc.Container([
        html.H2("Stake Optimizer", className="my-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("N-day Max Drawdown Window"),
                dcc.Slider(
                    id="n-slider", min=1, max=90, step=None, value=30,
                    marks={n: str(n) for n in _N_VALUES},
                    updatemode="drag",
                ),
            ], width=4, className="my-3"),
            dbc.Col([
                dbc.Label("VaR/CVaR Tail Quantile"),
                dcc.Slider(
                    id="q-slider", min=0.01, max=0.99, step=None, value=0.05,
                    marks={q: f"{q:.0%}" for q in _Q_VALUES},
                    updatemode="drag",
                ),
            ], width=4, className="my-3"),
            dbc.Col([
                dbc.Switch(id="scale-toggle", label="Scale by Max", value=False),
            ], width=3, className="d-flex align-items-center my-3"),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="main-plot"), width=8),
            dbc.Col(dcc.Graph(id="var-plot"), width=4),
        ]),
    ], fluid=True)  # fmt: off

    @app.callback(
        Output("main-plot", "figure"),
        [Input("n-slider", "value"), Input("scale-toggle", "value")],
    )
    def update_figure(n: int, scaled: bool) -> go.Figure:
        dd_df = max_drawdown_n(eqy_df.lazy(), n).collect()
        val_key = "scaled" if scaled else "raw"

        fig = go.Figure()

        # Row 1: shares outstanding & Row 2: shaded drawdown
        for sec in SECURITIES:
            c = _COLORS[sec]
            eqy = eqy_by_sec[sec]
            fig.add_trace(
                go.Scatter(
                    x=eqy["dates"], y=eqy[val_key],
                    xaxis="x", yaxis="y",
                    name=sec, legendgroup=sec, mode="lines+markers",
                    line=dict(color=c, width=1.5), marker=dict(size=_MKR, color=c),
                )
            )

            sec_dd = dd_df.filter(pl.col("security") == sec)
            fig.add_trace(
                go.Scatter(
                    x=sec_dd["date"].to_list(), y=sec_dd["max_drawdown"].to_list(),
                    xaxis="x", yaxis="y2",
                    name=sec, legendgroup=sec, showlegend=False,
                    mode="lines+markers", fill="tozeroy",
                    line=dict(color=c, width=1.5), marker=dict(size=_MKR, color=c),
                    fillcolor=f"rgba({_hex_to_rgb(c)}, 0.15)",
                )
            )

        # Row 3: queue delays
        fig.add_trace(
            go.Scatter(
                x=delay_dates, y=entry_h, xaxis="x", yaxis="y3",
                name="Entry Queue", mode="lines+markers",
                line=dict(color="#e74c3c", width=1.5), marker=dict(size=_MKR, color="#e74c3c"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=delay_dates, y=exit_h, xaxis="x", yaxis="y3",
                name="Exit Queue", mode="lines+markers",
                line=dict(color="#3498db", width=1.5), marker=dict(size=_MKR, color="#3498db"),
            )
        )

        y1_title = "Scaled (max=1)" if scaled else "Shares"
        fig.update_layout(
            xaxis=dict(
                title="Date", domain=[0.0, 1.0], anchor="y3", showspikes=False,
            ),
            yaxis=dict(
                title=y1_title, domain=_DOMAINS["y"],
                anchor="free", position=0,
            ),
            yaxis2=dict(
                title="Max DD (%)", domain=_DOMAINS["y2"],
                anchor="free", position=0, overlaying=None,
            ),
            yaxis3=dict(
                title="Hours", domain=_DOMAINS["y3"],
                anchor="free", position=0, overlaying=None,
            ),
            hovermode="x",
            hoversubplots="axis",
            height=_PLOT_H,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5),
            uirevision="constant",
            margin=dict(t=40, l=60),
        )

        return fig

    @app.callback(
        Output("var-plot", "figure"),
        [Input("n-slider", "value"), Input("q-slider", "value")],
    )
    def update_var(n: int, q: float) -> go.Figure:
        vc = drawdown_var_cvar(eqy_df.lazy(), n, q)
        secs = vc["security"].to_list()
        var_vals = vc["var"].to_list()
        cvar_vals = vc["cvar"].to_list()
        colors = [_COLORS[s] for s in secs]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=secs, x=var_vals, name="VaR", orientation="h",
            marker_color=colors, marker_line_width=0, opacity=0.6,
        ))
        fig.add_trace(go.Bar(
            y=secs, x=cvar_vals, name="CVaR", orientation="h",
            marker_color=colors, marker_line_width=2,
            marker_line_color=colors, opacity=1.0,
            marker_pattern_shape="/",
        ))

        fig.update_layout(
            barmode="group",
            xaxis_title="Drawdown (%)",
            yaxis_title="",
            title=f"VaR / CVaR \u2014 {n}d, {q:.0%} tail",
            height=_PLOT_H,
            margin=dict(t=50, b=40, l=50, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5),
            uirevision="constant",
        )

        return fig

    return app


def _hex_to_rgb(h: str) -> str:
    """'#e74c3c' -> '231, 76, 60'"""
    h = h.lstrip("#")
    return f"{int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}"


if __name__ == "__main__":
    create_app().run(debug=True)
