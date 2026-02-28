"""Noxli ingress backend — stream discovery & config persistence."""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import audio_stream, db, ha

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Auto-start detection if rtsp_url is configured
    config = _read_config()
    rtsp_url = config.get("rtsp_url", "")
    if rtsp_url:
        sensitivity = config.get("detection_sensitivity", 0.5)
        try:
            audio_stream.loop.start(rtsp_url, sensitivity)
        except Exception as e:
            print(f"[noxli] Auto-start detection failed: {e}")
    yield
    # Shutdown: stop detection loop
    audio_stream.loop.stop()


app = FastAPI(lifespan=_lifespan)

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


# --- Detection control ---


@app.post("/api/detection/start")
async def start_detection():
    config = _read_config()
    rtsp_url = config.get("rtsp_url", "")
    if not rtsp_url:
        raise HTTPException(status_code=400, detail="No rtsp_url configured")
    sensitivity = config.get("detection_sensitivity", 0.5)
    try:
        audio_stream.loop.start(rtsp_url, sensitivity)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"status": "started", "rtsp_url": rtsp_url, "sensitivity": sensitivity}


@app.post("/api/detection/stop")
async def stop_detection():
    audio_stream.loop.stop()
    return {"status": "stopped"}


@app.get("/api/detection/status")
async def detection_status():
    s = audio_stream.loop.stats
    return {
        "running": s.running,
        "rtsp_url": s.rtsp_url,
        "events_detected": s.events_detected,
        "last_event_time": s.last_event_time,
        "chunks_processed": s.chunks_processed,
        "started_at": s.started_at,
        "error": s.error,
        "ffmpeg_error": s.ffmpeg_error,
    }


@app.get("/api/detection/debug")
async def detection_debug():
    """Live debug view — last 30 chunks with audio levels and classifications."""
    s = audio_stream.loop.stats
    chunks = audio_stream.loop.debug_chunks
    return {
        "running": s.running,
        "rtsp_url": s.rtsp_url,
        "chunks_processed": s.chunks_processed,
        "error": s.error,
        "ffmpeg_error": s.ffmpeg_error,
        "chunks": chunks,
    }


@app.get("/api/detection/audio")
async def detection_audio():
    """Stream live audio from the configured source as MP3 for browser playback."""
    config = _read_config()
    rtsp_url = config.get("rtsp_url", "")
    if not rtsp_url:
        raise HTTPException(status_code=400, detail="No rtsp_url configured")

    cmd = audio_stream.loop._build_ffmpeg_cmd(rtsp_url)
    # Replace raw s16le output with MP3 for browser compatibility
    # Find the output format args and replace them
    cmd = cmd[: cmd.index("-acodec")] + [
        "-acodec", "libmp3lame", "-ab", "64k",
        "-ar", "16000", "-ac", "1",
        "-f", "mp3", "-",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def generate():
        try:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                yield chunk
        finally:
            proc.terminate()
            proc.wait()

    return StreamingResponse(generate(), media_type="audio/mpeg")


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


# --- Sleep Sessions ---

SLEEP_METHODS = ["nursing", "rocking", "laying_down", "contact_nap", "stroller", "car_ride"]
SLEEP_TYPES = ["nap", "night"]
TIME_TO_SLEEP_PRESETS = [5, 10, 15, 20, 30, 45, 60]

_SLEEP_COLS = "id, start_time, end_time, session_type, method, time_to_sleep_minutes, notes, created_at"


class SleepSessionCreate(BaseModel):
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    session_type: str = "nap"
    method: Optional[str] = None
    time_to_sleep_minutes: Optional[int] = None
    notes: str = ""


class SleepSessionUpdate(BaseModel):
    end_time: Optional[float] = None
    session_type: Optional[str] = None
    method: Optional[str] = None
    time_to_sleep_minutes: Optional[int] = None
    notes: Optional[str] = None


def _session_dict(row) -> dict:
    d = dict(row)
    if d["end_time"] is not None:
        d["duration_minutes"] = round((d["end_time"] - d["start_time"]) / 60, 1)
    else:
        d["duration_minutes"] = round((time.time() - d["start_time"]) / 60, 1)
    return d


def _attach_wake_ups(conn, sessions: List[dict]) -> None:
    for s in sessions:
        start = s["start_time"]
        end = s["end_time"] if s["end_time"] is not None else time.time()
        rows = conn.execute(
            "SELECT id, timestamp, duration, confidence "
            "FROM events WHERE timestamp BETWEEN ? AND ? "
            "ORDER BY timestamp ASC",
            (start, end),
        ).fetchall()
        s["wake_ups"] = [dict(r) for r in rows]


@app.get("/api/sleep/active")
async def get_active_sleep_session():
    with db.get_db() as conn:
        row = conn.execute(
            f"SELECT {_SLEEP_COLS} FROM sleep_sessions WHERE end_time IS NULL "
            "ORDER BY start_time DESC LIMIT 1",
        ).fetchone()
        if not row:
            return {"active_session": None}
        result = _session_dict(row)
        _attach_wake_ups(conn, [result])
    return {"active_session": result}


@app.get("/api/sleep/methods")
async def get_sleep_methods():
    return {
        "methods": SLEEP_METHODS,
        "types": SLEEP_TYPES,
        "time_to_sleep_presets": TIME_TO_SLEEP_PRESETS,
    }


@app.get("/api/sleep")
async def list_sleep_sessions(hours: float = Query(default=24, ge=0.1, le=168)):
    since = time.time() - hours * 3600
    with db.get_db() as conn:
        rows = conn.execute(
            f"SELECT {_SLEEP_COLS} FROM sleep_sessions "
            "WHERE start_time >= ? OR end_time IS NULL "
            "ORDER BY start_time DESC",
            (since,),
        ).fetchall()
        sessions = [_session_dict(r) for r in rows]
        _attach_wake_ups(conn, sessions)

        active = conn.execute(
            f"SELECT {_SLEEP_COLS} FROM sleep_sessions WHERE end_time IS NULL "
            "ORDER BY start_time DESC LIMIT 1",
        ).fetchone()

    active_dict = None
    if active:
        active_dict = _session_dict(active)
        for s in sessions:
            if s["id"] == active_dict["id"]:
                active_dict = s
                break

    return {
        "sessions": sessions,
        "active_session": active_dict,
        "query": {"hours": hours, "since": since},
    }


@app.post("/api/sleep")
async def create_sleep_session(body: SleepSessionCreate):
    if body.session_type not in SLEEP_TYPES:
        raise HTTPException(status_code=422, detail=f"session_type must be one of {SLEEP_TYPES}")
    if body.method is not None and body.method not in SLEEP_METHODS:
        raise HTTPException(status_code=422, detail=f"method must be one of {SLEEP_METHODS}")

    ts = body.start_time if body.start_time is not None else time.time()

    with db.get_db() as conn:
        # Guard: only one active session at a time
        if body.end_time is None:
            existing = conn.execute(
                "SELECT id FROM sleep_sessions WHERE end_time IS NULL",
            ).fetchone()
            if existing:
                raise HTTPException(status_code=409, detail="An active sleep session already exists")

        cur = conn.execute(
            "INSERT INTO sleep_sessions (start_time, end_time, session_type, method, time_to_sleep_minutes, notes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ts, body.end_time, body.session_type, body.method, body.time_to_sleep_minutes, body.notes),
        )
        conn.commit()
        row = conn.execute(
            f"SELECT {_SLEEP_COLS} FROM sleep_sessions WHERE id = ?",
            (cur.lastrowid,),
        ).fetchone()

    return _session_dict(row)


@app.patch("/api/sleep/{session_id}")
async def update_sleep_session(session_id: int, body: SleepSessionUpdate):
    if body.session_type is not None and body.session_type not in SLEEP_TYPES:
        raise HTTPException(status_code=422, detail=f"session_type must be one of {SLEEP_TYPES}")
    if body.method is not None and body.method not in SLEEP_METHODS:
        raise HTTPException(status_code=422, detail=f"method must be one of {SLEEP_METHODS}")

    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values())

    with db.get_db() as conn:
        row = conn.execute(
            f"SELECT {_SLEEP_COLS} FROM sleep_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")

        conn.execute(
            f"UPDATE sleep_sessions SET {set_clause} WHERE id = ?",
            (*values, session_id),
        )
        conn.commit()
        row = conn.execute(
            f"SELECT {_SLEEP_COLS} FROM sleep_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        result = _session_dict(row)
        _attach_wake_ups(conn, [result])

    return result


@app.delete("/api/sleep/{session_id}")
async def delete_sleep_session(session_id: int):
    with db.get_db() as conn:
        row = conn.execute(
            "SELECT id FROM sleep_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        conn.execute("DELETE FROM sleep_sessions WHERE id = ?", (session_id,))
        conn.commit()
    return {"deleted": session_id}
