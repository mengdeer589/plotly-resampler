# -*- coding: utf-8 -*-
"""Demo: 多条 trace . 多人支持 . GIL 释放 + 并行化

每个浏览器 session 持有独立的 FigureResampler，互不干扰。
动态采样通过 construct_update_data_patch（公开方法）实现。
"""
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback_context, dcc, html, no_update
from plotly_resampler import FigureResampler
import flask
from uuid import uuid4

# -- 数据 --
n_traces = 30
n_points = 2_000_000
x = np.arange(n_points)
y_list = []
for i in range(n_traces):
    y = (i + 1) * np.sin(np.linspace(0, 50 * (i + 1), n_points)) + np.random.randn(n_points) * 0.3
    y_list.append(y)

# -- 服务端 session 存储 --
_session_figs: dict = {}        # sid -> FigureResampler

# -- Dash app --
app = Dash(__name__)
app.server.secret_key = "demo-secret-key-change-in-production"

app.layout = html.Div([
    html.H1(f"{n_traces} traces x {n_points:,} points  (per-session Fig)",
            style={"textAlign": "center"}),
    html.Button("plot chart", id="plot-button", n_clicks=0),
    html.Hr(),
    dcc.Graph(id="graph-id"),
])

# -- 1) plot : 为当前 session 创建专属 FigureResampler --
@app.callback(
    Output("graph-id", "figure"),
    Input("plot-button", "n_clicks"),
    prevent_initial_call=True,
)
def plot_graph(n_clicks):
    ctx = callback_context
    if len(ctx.triggered) and "plot-button" in ctx.triggered[0]["prop_id"]:
        sid = str(uuid4())
        flask.session["sid"] = sid
        fig = FigureResampler(go.Figure())
        for i in range(n_traces):
            fig.add_trace(go.Scattergl(name=f"trace-{i}"), hf_x=x, hf_y=y_list[i])
        fig.update_xaxes(range=[500_000, 900_000])
        _session_figs[sid] = fig
        return fig
    return no_update

# -- 2) relayout : 取出当前 session 的 fig，调用公开方法动态采样 --
@app.callback(
    Output("graph-id", "figure", allow_duplicate=True),
    Input("graph-id", "relayoutData"),
    prevent_initial_call=True,
)
def update_fig(relayoutdata):
    sid = flask.session.get("sid")
    if sid is None or sid not in _session_figs:
        return no_update
    return _session_figs[sid].construct_update_data_patch(relayoutdata)

if __name__ == "__main__":
    app.run(debug=True, port=9025)
