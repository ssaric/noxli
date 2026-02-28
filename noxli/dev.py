#!/usr/bin/env python3
"""Local dev server — run from the noxli/ addon directory.

Usage:
    python dev.py

Then open http://localhost:8099
Insert test events:
    curl -X POST http://localhost:8099/api/events -H 'Content-Type: application/json' -d '{}'
    curl -X POST http://localhost:8099/api/events -H 'Content-Type: application/json' -d '{"confidence":0.87,"duration":45}'
"""

import os
import sys
from pathlib import Path

# Patch DB path to local directory before importing anything
os.environ.setdefault("INGRESS_ENTRY", "")
data_dir = Path(__file__).parent / ".devdata"
data_dir.mkdir(exist_ok=True)

import backend.db as db_mod
db_mod.DB_PATH = data_dir / "events.db"

import backend.detector as det_mod
models_dir = data_dir / "models"
models_dir.mkdir(exist_ok=True)
det_mod.MODEL_DIR = models_dir

import backend.main as main_mod
main_mod.CONFIG_PATH = data_dir / "config.json"

import uvicorn
from backend.main import app

if __name__ == "__main__":
    print(f"Dev server: http://localhost:8099")
    print(f"DB: {db_mod.DB_PATH}")
    print(f"Models: {det_mod.MODEL_DIR}")
    print(f"Insert test event: curl -X POST http://localhost:8099/api/events -H 'Content-Type: application/json' -d '{{}}'")
    uvicorn.run(app, host="0.0.0.0", port=8099)
