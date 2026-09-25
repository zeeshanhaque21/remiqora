"""Remix-from-URL endpoints: import a video, edit its lyrics/score, run stages.

An imported source is a normal track row (``model="youtube"``) with import
state in ``params_json``; this router is the only place that starts the
download/lyrics/melody pipeline in app/remix/pipeline.py.

The ABC tool endpoint is how the UI applies transpose/range/cleanup to an
already-extracted score without re-running SheetSage2: the frontend sends the
(current) ABC text plus the tool's own kwargs and gets the transformed text
back. The pure-text implementations live in app/remix/abc_tools.py.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from .. import db
from .. import stems
from ..config import REMIX_MAX_DURATION_S
from ..orchestrator.manager import manager
from ..orchestrator.state import ModelStatus
from ..remix import abc_tools, pipeline
from .routes_tracks import _row_to_dict, _sanitize

router = APIRouter(prefix="/api/remix", tags=["remix"])

MODEL = "youtube"

# Tool name -> the pure-text function in abc_tools.py it maps to.
_TOOLS = {
    "analyze": abc_tools.analyze,
    "modify": abc_tools.modify,
    "retarget": abc_tools.retarget,
    "clean": abc_tools.clean,
    "fit": abc_tools.lyric_melody_fit,
    "style": abc_tools.build_style,
}


class ImportRequest(BaseModel):
    url: str


class LyricsRequest(BaseModel):
    lyrics: str


class AbcRequest(BaseModel):
    abc: str


class AbcToolRequest(BaseModel):
    abc: str = ""
    kwargs: dict[str, Any] = {}


def _track_or_404(track_id: int) -> "sqlite3.Row":
    row = db.get_track(track_id)
    if not row or row["model"] != MODEL:
        raise HTTPException(status_code=404, detail="remix source not found")
    return row


def _payload(track_id: int) -> dict:
    row = db.get_track(track_id)
    data = _row_to_dict(row)
    params = json.loads(row["params_json"] or "{}")
    data["remix"] = {
        "source_url": params.get("source_url"),
        "webpage_url": params.get("webpage_url"),
        "video_id": params.get("video_id"),
        "extractor": params.get("extractor"),
        "uploader": params.get("uploader"),
        "duration_s": params.get("duration_s"),
        "thumbnail": params.get("thumbnail"),
        "lyrics_source": params.get("lyrics_source"),
        "stages": pipeline.status(track_id)["stages"],
        "active": pipeline.is_active(track_id),
    }
    return data


@router.post("/import")
async def import_remix(body: ImportRequest):
    url = (body.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    try:
        info = await asyncio.to_thread(pipeline._ytdlp_probe, url)
    except Exception as exc:  # noqa: BLE001 - yt-dlp raises many error types
        raise HTTPException(status_code=400, detail=f"could not read URL: {exc}") from exc

    if info.get("_type") == "playlist" or info.get("entries"):
        raise HTTPException(status_code=400, detail="playlists are not supported; paste a single video URL")

    duration = info.get("duration")
    if not duration:
        raise HTTPException(status_code=400, detail="could not determine video duration")
    if float(duration) > REMIX_MAX_DURATION_S:
        raise HTTPException(
            status_code=400,
            detail=f"video is {float(duration):.0f}s; the limit is {REMIX_MAX_DURATION_S}s",
        )

    video_id = info.get("id")
    extractor = info.get("extractor_key") or info.get("extractor")
    existing = _find_existing(extractor, video_id)
    if existing is not None:
        return _payload(existing["id"])

    out_dir = db.model_dir(MODEL)
    ts = info.get("upload_date") or ""
    title = info.get("title") or "remix source"
    # Reserve a unique audio filename up front; the download stage overwrites
    # the row's audio_path with the real file once yt-dlp reports it.
    stem_dir = out_dir / f"{ts}_{_sanitize(title)}" if ts else out_dir / _sanitize(title)
    stem_dir.mkdir(parents=True, exist_ok=True)
    audio_path = stem_dir / "source.mp3"

    params = {
        "source_url": url,
        "webpage_url": info.get("webpage_url") or url,
        "video_id": video_id,
        "extractor": extractor,
        "uploader": info.get("uploader"),
        "duration_s": float(duration),
        "thumbnail": info.get("thumbnail"),
        "lyrics_source": None,
        "stages": pipeline._default_stages(),
    }
    track_id = db.insert_track(
        model=MODEL,
        title=title,
        lyrics="",
        seed=None,
        duration_ms=float(duration) * 1000,
        wall_ms=None,
        params=params,
        audio_path=audio_path,
        abc_path=None,
    )

    # The download stage needs its own output dir; keep it next to the reserved
    # audio path so the finished file resolves without a rename.
    await pipeline.start(track_id, info.get("webpage_url") or url, {**info, "_out_dir": stem_dir})
    return _payload(track_id)


def _find_existing(extractor: Optional[str], video_id: Optional[str]):
    if not video_id:
        return None
    for row in db.list_tracks(MODEL):
        params = json.loads(row["params_json"] or "{}")
        if params.get("video_id") == video_id and (not extractor or params.get("extractor") == extractor):
            return row
    return None


@router.get("/{track_id}")
async def get_remix(track_id: int):
    _track_or_404(track_id)
    return _payload(track_id)


@router.post("/{track_id}/lyrics/retry")
async def retry_lyrics(track_id: int):
    _track_or_404(track_id)
    await pipeline.start_lyrics_retry(track_id)
    return _payload(track_id)


@router.post("/{track_id}/melody")
async def start_melody(track_id: int, force: bool = False):
    _track_or_404(track_id)
    if manager.state.models["yue2"].status != ModelStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail="melody extraction runs on the YuE2 server - make YuE2 the active model first",
        )
    await pipeline.start_melody(track_id, force=force)
    return _payload(track_id)


@router.post("/{track_id}/cancel")
async def cancel_remix(track_id: int):
    _track_or_404(track_id)
    await pipeline.cancel(track_id)
    return _payload(track_id)


@router.put("/{track_id}/lyrics")
async def put_lyrics(track_id: int, body: LyricsRequest):
    row = _track_or_404(track_id)
    db.update_track_lyrics(track_id, body.lyrics)
    params = json.loads(row["params_json"] or "{}")
    params["lyrics_source"] = "edited"
    db.update_track_params(track_id, params)
    return _payload(track_id)


@router.put("/{track_id}/abc")
async def put_abc(track_id: int, body: AbcRequest):
    row = _track_or_404(track_id)
    out_path = Path(row["audio_path"]).with_suffix(".abc")
    out_path.write_text(body.abc, encoding="utf-8")
    db.update_track_abc(track_id, out_path)
    return _payload(track_id)


@router.post("/abc/{tool}")
async def run_abc_tool(tool: str, body: AbcToolRequest):
    func = _TOOLS.get(tool)
    if func is None:
        raise HTTPException(status_code=404, detail=f"unknown ABC tool '{tool}'")
    kwargs = dict(body.kwargs or {})
    if tool == "fit":
        kwargs.setdefault("lyrics", "")
    try:
        result = func(body.abc, **kwargs)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid arguments for '{tool}': {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(result, tuple):
        return {"tool": tool, "result": list(result)}
    return {"tool": tool, "result": result}


@router.delete("/{track_id}")
async def delete_remix(track_id: int):
    row = _track_or_404(track_id)
    if pipeline.is_active(track_id):
        raise HTTPException(status_code=409, detail="import is in progress")
    if stems.is_active(track_id):
        raise HTTPException(status_code=409, detail="stem separation is in progress")
    db.delete_track(track_id)
    pipeline.forbid(track_id)
    stems.forget(track_id)
    return {"deleted": True, "id": track_id}
