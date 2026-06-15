"""Server benchmark: compares Flask (sync) vs FastAPI+async (HTTP) vs FastAPI+WS.

Usage:
    python benchmarks/bench_server.py

Tests three configurations:
  1. Flask sync HTTP  (baseline - current impl)
  2. FastAPI async HTTP
  3. FastAPI async WebSocket
"""
import sys, os, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import aiohttp
import asyncio
import numpy as np
from bench_common import generate_events

FLASK_PORT = 9001
FASTAPI_PORT = 9002
CONCURRENCY_LEVELS = [1, 5, 10, 20]
N_REQUESTS_PER_LEVEL = 50

SCRIPT_DIR = os.path.dirname(__file__)


def start_server(script_name: str, port: int):
    script_path = os.path.join(SCRIPT_DIR, script_name)
    return subprocess.Popen(
        [sys.executable, script_path, str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for_server(port: int, timeout: float = 15.0, interval: float = 0.2):
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(interval)
    return False


async def bench_http(session, url, events, concurrency):
    sem = asyncio.Semaphore(concurrency)
    latencies = []

    async def worker(ev):
        async with sem:
            t0 = time.perf_counter()
            async with session.post(url, json=ev) as resp:
                await resp.json()
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)

    tasks = [worker(events[i % len(events)]) for i in range(N_REQUESTS_PER_LEVEL)]
    await asyncio.gather(*tasks)
    return latencies


async def bench_ws(url: str, events: list, concurrency: int) -> list:
    """WebSocket benchmark: each "user" gets a dedicated WS connection."""
    latencies = []
    sem = asyncio.Semaphore(concurrency)

    async def worker(ev):
        async with sem:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url, timeout=5) as ws:
                        t0 = time.perf_counter()
                        await ws.send_json(ev)
                        resp = await ws.receive_json(timeout=5)
                        lat = (time.perf_counter() - t0) * 1000
                        latencies.append(lat)
            except Exception as e:
                latencies.append(-1)  # mark error

    tasks = [worker(events[i % len(events)]) for i in range(N_REQUESTS_PER_LEVEL)]
    await asyncio.gather(*tasks)
    return latencies


def report(label: str, latencies_ms):
    arr = np.array([l for l in latencies_ms if l >= 0])
    if len(arr) == 0:
        print(f"  {label:>24s}:  (no valid data)")
        return
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    p50 = sorted_arr[int(n * 0.50)]
    p90 = sorted_arr[int(n * 0.90)]
    p95 = sorted_arr[int(n * 0.95)]
    p99 = sorted_arr[int(n * 0.99)] if n > 100 else sorted_arr[-1]
    print(f"  {label:>24s}: "
          f"min={arr.min():7.3f}  "
          f"mean={arr.mean():7.3f}  "
          f"P50={p50:7.3f}  "
          f"P90={p90:7.3f}  "
          f"P95={p95:7.3f}  "
          f"P99={p99:7.3f}  "
          f"max={arr.max():7.3f}  ms   (n={n})")


def compare_section(label: str, *results):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    for name, lat in results:
        report(name, lat)
    # Compare all vs Flask
    if results:
        flask_mean = np.mean([l for l in results[0][1] if l >= 0])
        for name, lat in results[1:]:
            arr = np.array([l for l in lat if l >= 0])
            if len(arr) and flask_mean > 0:
                ratio = flask_mean / np.mean(arr)
                print(f"  {'':>24s}  speedup vs Flask: {ratio:.2f}x")
    print(f"{'='*80}")


async def main():
    events = generate_events(count=20, seed=123)

    print("=" * 80)
    print("  Starting servers...")
    print("=" * 80)

    proc_flask = start_server("server_flask.py", FLASK_PORT)
    proc_fastapi = start_server("server_fastapi.py", FASTAPI_PORT)

    try:
        if not wait_for_server(FLASK_PORT):
            print("  ERROR: Flask server did not start")
            return
        print(f"  Flask server     -> http://127.0.0.1:{FLASK_PORT}/aggregate")
        if not wait_for_server(FASTAPI_PORT):
            print("  ERROR: FastAPI server did not start")
            return
        print(f"  FastAPI server   -> http://127.0.0.1:{FASTAPI_PORT}/aggregate")
        print(f"  FastAPI WS       -> ws://127.0.0.1:{FASTAPI_PORT}/ws/aggregate")

        http_flask = f"http://127.0.0.1:{FLASK_PORT}/aggregate"
        http_fastapi = f"http://127.0.0.1:{FASTAPI_PORT}/aggregate"
        ws_fastapi = f"ws://127.0.0.1:{FASTAPI_PORT}/ws/aggregate"

        for concurrency in CONCURRENCY_LEVELS:
            print(f"\n  --- Concurrency = {concurrency} ---")
            async with aiohttp.ClientSession() as session:
                flask_lat = await bench_http(session, http_flask, events, concurrency)
                fastapi_lat = await bench_http(session, http_fastapi, events, concurrency)
            # WebSocket: each concurrent "user" opens their own connection
            ws_lat = await bench_ws(ws_fastapi, events, concurrency)
            compare_section(
                f"Concurrency = {concurrency}",
                ("Flask HTTP (sync)", flask_lat),
                ("FastAPI HTTP (async)", fastapi_lat),
                ("FastAPI WS (async+ws)", ws_lat),
            )

    finally:
        print("\n  Shutting down servers...")
        proc_flask.terminate()
        proc_fastapi.terminate()
        proc_flask.wait(timeout=3)
        proc_fastapi.wait(timeout=3)
        print("  Done.")


if __name__ == "__main__":
    asyncio.run(main())
