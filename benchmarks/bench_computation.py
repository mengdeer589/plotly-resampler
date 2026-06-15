"""Benchmark 1: pure construct_update_data_patch latency.

Measures the raw computation time of FigureResamcher.construct_update_data_patch()
with 3 traces of 100K-500K points, without any server overhead.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import statistics
import numpy as np
import plotly.graph_objects as go
from plotly_resampler import FigureResampler

N_TRACES = 3
N_POINTS_LIST = [100_000, 300_000, 500_000]  # one per trace
N_EVENTS = 100  # simulated user interactions
AGG_SEED = 42

np.random.seed(AGG_SEED)

# --- Generate data ---
traces_x = []
traces_y = []
for n in N_POINTS_LIST:
    x = np.arange(n, dtype=np.float64)
    y = np.random.randn(n).cumsum().astype(np.float64)
    traces_x.append(x)
    traces_y.append(y)

# --- Build FigureResampler ---
fig = FigureResampler(go.Figure(), default_n_shown_samples=2000)
for i in range(N_TRACES):
    fig.add_trace(
        go.Scattergl(x=traces_x[i], y=traces_y[i], name=f"trace-{i}"),
        hf_x=traces_x[i],
        hf_y=traces_y[i],
    )

# --- Generate simulated relayout events ---
total_max = max(n for n in N_POINTS_LIST)
events = []
for _ in range(N_EVENTS):
    center = np.random.uniform(0, total_max)
    width = np.random.uniform(total_max * 0.01, total_max * 0.5)
    events.append({
        "xaxis.range[0]": max(0, center - width / 2),
        "xaxis.range[1]": min(total_max, center + width / 2),
    })

# --- Also include some autorange/reset events ---
events.append({"xaxis.autorange": True, "xaxis.showspikes": True})

# --- Warm-up ---
for _ in range(5):
    fig.construct_update_data_patch(events[0])
    fig._construct_update_data(events[0])

# --- Benchmark ---
latencies = []
no_update_count = 0

for ev in events:
    t0 = time.perf_counter()
    result = fig.construct_update_data_patch(ev)
    elapsed = time.perf_counter() - t0
    latencies.append(elapsed * 1000)  # convert to ms
    if result is None or str(result) == "no_update":
        no_update_count += 1

# --- Report ---
latencies_ms = np.array(latencies)
print(f"{'='*60}")
print(f"  Pure Computation Benchmark")
print(f"  Traces: {N_TRACES}, points: {N_POINTS_LIST}")
print(f"  Events: {N_EVENTS}")
print(f"{'='*60}")
print(f"  min:      {latencies_ms.min():8.3f} ms")
print(f"  max:      {latencies_ms.max():8.3f} ms")
print(f"  mean:     {latencies_ms.mean():8.3f} ms")
print(f"  median:   {np.median(latencies_ms):8.3f} ms")
sorted_lat = np.sort(latencies_ms)
print(f"  P50:      {sorted_lat[int(len(sorted_lat)*0.50)]:8.3f} ms")
print(f"  P90:      {sorted_lat[int(len(sorted_lat)*0.90)]:8.3f} ms")
print(f"  P95:      {sorted_lat[int(len(sorted_lat)*0.95)]:8.3f} ms")
print(f"  P99:      {sorted_lat[int(len(sorted_lat)*0.99)]:8.3f} ms")
print(f"{'='*60}")
print(f"  no_update count: {no_update_count}/{len(events)}")
print(f"{'='*60}")
