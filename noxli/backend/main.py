"""Noxli ingress backend — stream discovery & config persistence."""

import json
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import db, ha

app = FastAPI()

CONFIG_PATH = Path("/data/config.json")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

CONFIG_DEFAULTS: dict = {
    "rtsp_url": "",
    "detection_sensitivity": 0.5,
    "mqtt_topic": "noxli/detection",
    "log_level": "info",
}


def _read_config() -> dict:
    persisted: dict = {}
    if CONFIG_PATH.exists():
        try:
            persisted = json.loads(CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {**CONFIG_DEFAULTS, **persisted}


def _write_config(data: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(data, indent=2))


# --- Routes ---


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = FRONTEND_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")

    html = html_path.read_text()
    ingress_entry = os.environ.get("INGRESS_ENTRY", "")
    inject = f'<script>window.INGRESS_ENTRY="{ingress_entry}";</script>'
    html = html.replace("</head>", f"{inject}</head>")
    return HTMLResponse(html)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/cameras")
async def list_cameras():
    cameras = await ha.get_stream_entities()
    return {"cameras": cameras}


@app.post("/api/cameras/{entity_id:path}/stream_source")
async def resolve_stream_source(entity_id: str):
    rtsp_url = await ha.get_stream_source(entity_id)
    if not rtsp_url:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Could not resolve stream source for {entity_id}. "
                "The RTSP URL is not exposed in entity attributes and "
                "go2rtc did not return a source. "
                "Please enter the RTSP URL manually below."
            ),
        )

    # Persist the selection
    config = _read_config()
    config["entity_id"] = entity_id
    config["rtsp_url"] = rtsp_url
    _write_config(config)

    return {"entity_id": entity_id, "rtsp_url": rtsp_url}


@app.get("/api/config")
async def get_config():
    return JSONResponse(_read_config())


class ConfigUpdate(BaseModel):
    rtsp_url: Optional[str] = None
    detection_sensitivity: Optional[float] = None
    mqtt_topic: Optional[str] = None
    log_level: Optional[str] = None


@app.post("/api/config")
async def set_config(body: ConfigUpdate):
    config = _read_config()
    updates = body.model_dump(exclude_none=True)
    if "rtsp_url" in updates:
        config.pop("entity_id", None)
    config.update(updates)
    _write_config(config)
    return JSONResponse(config)


# --- Events ---


class EventCreate(BaseModel):
    timestamp: Optional[float] = None
    duration: float = 0
    confidence: float = 0
    source: str = "test"


@app.get("/api/events")
async def list_events(hours: float = Query(default=24, ge=0.1, le=168)):
    since = time.time() - hours * 3600
    with db.get_db() as conn:
        rows = conn.execute(
            "SELECT id, timestamp, duration, confidence, source, created_at "
            "FROM events WHERE timestamp >= ? ORDER BY timestamp ASC",
            (since,),
        ).fetchall()
    return {
        "events": [dict(r) for r in rows],
        "query": {"hours": hours, "since": since},
    }


@app.post("/api/events")
async def create_event(body: EventCreate):
    ts = body.timestamp if body.timestamp is not None else time.time()
    with db.get_db() as conn:
        cur = conn.execute(
            "INSERT INTO events (timestamp, duration, confidence, source) "
            "VALUES (?, ?, ?, ?)",
            (ts, body.duration, body.confidence, body.source),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, timestamp, duration, confidence, source, created_at "
            "FROM events WHERE id = ?",
            (cur.lastrowid,),
        ).fetchone()
    return dict(row)
