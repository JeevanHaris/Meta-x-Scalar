"""
FastAPI application for the Ticket Triage Environment.

Uses openenv.core create_app to expose standard OpenEnv endpoints:
    POST /reset    — start new episode
    POST /step     — execute an action
    GET  /state    — current environment state
    GET  /schema   — action / observation JSON schemas
    WS   /ws       — WebSocket for persistent sessions
    GET  /health   — health check
"""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from e

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from dotenv import load_dotenv

# Load environment variables from .env
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

try:
    from ..models import TicketAction, TicketObservation
    from .ticket_triage_environment import TicketTriageEnvironment
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import TicketAction, TicketObservation
    from server.ticket_triage_environment import TicketTriageEnvironment


app = create_app(
    TicketTriageEnvironment,
    TicketAction,
    TicketObservation,
    env_name="ticket-triage",
    max_concurrent_envs=4,
)

# Dashboard static files path
dashboard_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(dashboard_path, "index.html"))

@app.get("/styles.css")
async def read_css():
    return FileResponse(os.path.join(dashboard_path, "styles.css"), media_type="text/css")

@app.get("/app.js")
async def read_js():
    return FileResponse(os.path.join(dashboard_path, "app.js"), media_type="application/javascript")

@app.get("/baseline_results.json")
async def read_results():
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baseline_results.json")
    if os.path.exists(results_path):
        return FileResponse(results_path)
    else:
        return {"error": "results not found"}


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
