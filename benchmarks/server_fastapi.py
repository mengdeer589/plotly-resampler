"""FastAPI server with both HTTP async endpoint and WebSocket endpoint."""
import sys, os, time, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
import uvicorn
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
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9002
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
