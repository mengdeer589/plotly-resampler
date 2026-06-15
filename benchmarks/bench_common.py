"""Shared data & FigureResampler setup for server benchmarks."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import plotly.graph_objects as go
from plotly_resampler import FigureResampler

N_TRACES = 3
N_POINTS_LIST = [100_000, 300_000, 500_000]

def build_fig(seed=42):
    np.random.seed(seed)
    fig = FigureResampler(go.Figure(), default_n_shown_samples=2000)
    for i, n in enumerate(N_POINTS_LIST):
        x = np.arange(n, dtype=np.float64)
        y = np.random.randn(n).cumsum().astype(np.float64)
        fig.add_trace(
            go.Scattergl(x=x, y=y, name=f"trace-{i}"),
            hf_x=x,
            hf_y=y,
        )
    return fig

def generate_events(count=50, seed=123):
    np.random.seed(seed)
    total_max = max(N_POINTS_LIST)
    events = []
    for _ in range(count):
        center = np.random.uniform(0, total_max)
        width = np.random.uniform(total_max * 0.01, total_max * 0.5)
        events.append({
            "xaxis.range[0]": float(max(0, center - width / 2)),
            "xaxis.range[1]": float(min(total_max, center + width / 2)),
        })
    # include autorange+reset event
    events.append({"xaxis.autorange": True, "xaxis.showspikes": True})
    return events
