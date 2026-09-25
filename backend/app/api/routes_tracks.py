"""Unified track storage endpoints used by both models' frontends.

Generation itself still goes straight to the active model's own API via
routes_proxy.py; once a track is finished, the frontend uploads it here so
it lands in one shared place (DATA_DIR/files/<model>/ + one SQLite DB) instead
of each model's own, separate storage.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from .. import db
from ..config import MODELS

router = APIRouter(prefix="/api/tracks", tags=["tracks"])

ALLOWED_AUDIO_EXT = {"wav", "mp3", "flac"}
ALLOWED_TRACK_MODELS = set(MODELS.keys()) | {"editor", "upload", "youtube"}


def _sanitize(text: str) -> str:
    text = re.sub(r"\W+", "_", (text or "")[:40], flags=re.UNICODE)
    return text.strip("_") or "track"


def _row_to_dict(row) -> dict:
    audio_path = Path(row["audio_path"])
    return {
        "id": row["id"],
        "model": row["model"],
        "created_at": row["created_at"],
        "title": row["title"],
        "lyrics": row["lyrics"],
        "seed": row["seed"],
        "duration_ms": row["duration_ms"],
        "wall_ms": row["wall_ms"],
        "params": json.loads(row["params_json"] or "{}"),
        "filename": audio_path.name,
        "audio_url": f"/api/tracks/{row['id']}/audio",
        "abc_url": f"/api/tracks/{row['id']}/abc" if row["abc_path"] else None,
        "stems": (
            {n: f"/api/tracks/{row['id']}/stems/{n}" for n in json.loads(row["stems_json"]).keys()}
            if row["stems_json"]
            else None
        ),
        "midi": (
            {s: f"/api/tracks/{row['id']}/midi/{s}" for s in json.loads(row["midi_json"]).keys()}
            if row["midi_json"]
            else None
        ),
    }


@router.post("")
async def save_track(
    model: str = Form(...),
    title: str = Form(""),
    lyrics: str = Form(""),
    seed: Optional[int] = Form(None),
    duration_ms: Optional[float] = Form(None),
    wall_ms: Optional[float] = Form(None),
    params: str = Form("{}"),
    abc: Optional[str] = Form(None),
    audio: UploadFile = File(...),
):
    if model not in ALLOWED_TRACK_MODELS:
        raise HTTPException(status_code=400, detail=f"unknown track model/origin '{model}'")
    try:
        params_dict = json.loads(params) if params else {}
    except json.JSONDecodeError:
        params_dict = {}

    ext = (audio.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_AUDIO_EXT:
        ext = "wav"
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = f"{ts}_{_sanitize(title)}"
    target_dir = db.model_dir(model)
    audio_path = target_dir / f"{base}.{ext}"
    abc_path = target_dir / f"{base}.abc" if abc and abc.strip() else None

    try:
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        if abc_path is not None:
            abc_path.write_text(abc, encoding="utf-8")

        track_id = db.insert_track(
            model=model,
            title=title,
            lyrics=lyrics,
            seed=seed,
            duration_ms=duration_ms,
            wall_ms=wall_ms,
            params=params_dict,
            audio_path=audio_path,
            abc_path=abc_path,
        )
    except Exception:
        audio_path.unlink(missing_ok=True)
        if abc_path is not None:
            abc_path.unlink(missing_ok=True)
        raise

    row = db.get_track(track_id)
    return _row_to_dict(row)


@router.post("/upload")
async def upload_track(
    audio: UploadFile = File(...),
    title: Optional[str] = Form(None),
):
    ext = (audio.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_AUDIO_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(ALLOWED_AUDIO_EXT)}",
        )

    track_title = title.strip() if title and title.strip() else (audio.filename or "uploaded_track").rsplit(".", 1)[0]
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = f"{ts}_{_sanitize(track_title)}"
    target_dir = db.model_dir("upload")
    audio_path = target_dir / f"{base}.{ext}"

    try:
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        track_id = db.insert_track(
            model="upload",
            title=track_title,
            lyrics="",
            seed=None,
            duration_ms=None,
            wall_ms=None,
            params={"source": "user_upload"},
            audio_path=audio_path,
            abc_path=None,
        )
    except Exception:
        audio_path.unlink(missing_ok=True)
        raise

    row = db.get_track(track_id)
    return _row_to_dict(row)


@router.get("")
async def list_tracks(model: Optional[str] = None):
    if model is not None and model not in ALLOWED_TRACK_MODELS:
        raise HTTPException(status_code=400, detail=f"unknown track model/origin '{model}'")
    return {"data": [_row_to_dict(r) for r in db.list_tracks(model)]}


@router.get("/{track_id}/audio")
async def track_audio(track_id: int):
    row = db.get_track(track_id)
    if not row or not Path(row["audio_path"]).exists():
        return JSONResponse({"error": "audio not found"}, status_code=404)
    return FileResponse(row["audio_path"])


@router.get("/{track_id}/abc")
async def track_abc(track_id: int):
    row = db.get_track(track_id)
    if not row or not row["abc_path"] or not Path(row["abc_path"]).exists():
        return JSONResponse({"error": "track has no ABC plan"}, status_code=404)
    return PlainTextResponse(Path(row["abc_path"]).read_text(encoding="utf-8"))


@router.put("/{track_id}")
async def rename_track(track_id: int, title: str = Body(..., embed=True)):
    if not db.update_track_title(track_id, title):
        raise HTTPException(status_code=404, detail="track not found")
    return _row_to_dict(db.get_track(track_id))


@router.get("/{track_id}/mix")
async def get_mix_settings(track_id: int):
    row = db.get_track(track_id)
    if not row:
        raise HTTPException(status_code=404, detail="track not found")
    return {"settings": db.get_mix_settings(track_id)}


@router.put("/{track_id}/mix")
async def put_mix_settings(track_id: int, settings: dict = Body(...)):
    row = db.get_track(track_id)
    if not row:
        raise HTTPException(status_code=404, detail="track not found")
    db.update_track_mix_settings(track_id, settings)
    return {"settings": settings}


@router.delete("/{track_id}")
async def delete_track(track_id: int):
    if not db.delete_track(track_id):
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"deleted": True}
