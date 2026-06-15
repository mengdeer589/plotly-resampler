"""Flask server with sync aggregation endpoint."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify
from bench_common import build_fig

fig = build_fig()
app = Flask(__name__)

@app.route("/aggregate", methods=["POST"])
def aggregate():
    data = request.get_json(force=True)
    t0 = time.perf_counter()
    result = fig.construct_update_data_patch(data)
    elapsed = time.perf_counter() - t0
    return jsonify({"status": "ok", "time_ms": round(elapsed * 1000, 3)})

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9001
    app.run(host="127.0.0.1", port=port, debug=False)
