"""Granian ASGI server with FastAPI async HTTP endpoint."""
import sys, os, time, asyncio
# Ensure both the benchmarks dir and the project root are on sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, ".."))
for _p in (_script_dir, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from bench_common import build_fig

fig = build_fig()
app = FastAPI()

@app.post("/aggregate")
async def aggregate(request: Request):
    data = await request.json()
    loop = asyncio.get_event_loop()
    t0 = time.perf_counter()
    result = await loop.run_in_executor(
        None, fig.construct_update_data_patch, data
    )
    elapsed = time.perf_counter() - t0
    return JSONResponse({"status": "ok", "time_ms": round(elapsed * 1000, 3)})

@app.websocket("/ws/aggregate")
async def ws_aggregate(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    try:
        while True:
            data = await websocket.receive_json()
            t0 = time.perf_counter()
            result = await loop.run_in_executor(
                None, fig.construct_update_data_patch, data
            )
            elapsed = time.perf_counter() - t0
            await websocket.send_json({
                "status": "ok",
                "time_ms": round(elapsed * 1000, 3),
            })
    except Exception:
        pass

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9003
    from granian import Granian
    Granian(
        "benchmarks.server_granian:app",
        address="127.0.0.1",
        port=port,
        interface="asgi",
        log_level="warning",
    ).serve()
