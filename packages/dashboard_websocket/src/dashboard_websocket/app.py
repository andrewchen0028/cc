"""
Dash app that connects to the generator WebSocket and plots live data.

Run standalone (generator must already be running):
    python -m dashboard_websocket.app

Or launch both together:
    python -m dashboard_websocket
"""

import json

import plotly.graph_objs as go
from dash import Dash, Input, Output, State, dcc, html
from dash_extensions import WebSocket

MAX_POINTS = 200

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H3("Live Random Walk"),
        WebSocket(id="ws", url="ws://localhost:8765"),
        dcc.Store(id="store", data={"t": [], "v": []}),
        dcc.Graph(id="graph", animate=False),
    ],
    style={"fontFamily": "sans-serif", "padding": "1rem"},
)


@app.callback(
    Output("store", "data"),
    Input("ws", "message"),
    State("store", "data"),
    prevent_initial_call=True,
)
def on_message(message: dict | None, data: dict) -> dict:
    if message is None:
        return data
    point = json.loads(message["data"])
    t = data["t"] + [point["t"]]
    v = data["v"] + [point["v"]]
    return {"t": t[-MAX_POINTS:], "v": v[-MAX_POINTS:]}


@app.callback(
    Output("graph", "figure"),
    Input("store", "data"),
)
def update_graph(data: dict) -> go.Figure:
    fig = go.Figure(
        go.Scatter(x=data["t"], y=data["v"], mode="lines", line={"width": 1.5})
    )
    fig.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title="Value",
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
        uirevision="constant",  # keep zoom/pan across updates
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)
