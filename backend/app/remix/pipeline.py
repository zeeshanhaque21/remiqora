"""Import a video URL as a remix source: download, transcribe lyrics, score melody.

A remix source is an ordinary track row with ``model="youtube"``; the extra
import state (source metadata and per-stage progress) lives in that row's
``params_json`` so it survives a backend restart. The stage machine is
``download -> lyrics -> melody``: download and lyrics run automatically on
``start()``, melody only runs when YuE2 is already RUNNING (otherwise its
stage is parked as ``waiting_for_yue2`` and can be kicked off later from the
melody endpoint).

Like stems.py this module keeps its own in-memory job registry rather than
reusing the orchestrator's model state: an import is a sequence of one-shot
jobs (yt-dlp, Whisper, SheetSage2), not a persistent server. Unlike stems.py
every stage transition is also mirrored into ``params_json``, so a restart
shows the last known state instead of pretending a job is still in flight;
``_reconcile_interrupted()`` reports any stage left ``queued``/``running`` as
failed on startup.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import httpx

from .. import db, stems
from ..config import (
    FFMPEG_BIN_DIR,
    LOG_DIR,
    MODELS,
    REMIX_MAX_DURATION_S,
    WHISPER_BIN,
    WHISPER_MODEL_PATH,
    yue2_specs,
)
from ..orchestrator.manager import manager
from ..orchestrator.process import tail_log
from ..orchestrator.state import ModelStatus
from . import lyrics_tools

IS_WINDOWS = sys.platform == "win32"

MODEL = "youtube"
SOURCE = "remix"
STAGES = ("download", "lyrics", "melody")

StageStatus = Literal[
    "idle", "queued", "running", "done", "failed", "cancelled", "waiting_for_yue2"
]

# Manual (uploaded) subtitles only. Auto-generated captions are Whisper's own
# output re-served by the platform, and we run Whisper locally anyway.
_SUBTITLE_EXTS = ("srt", "vtt")

# The transcription is a single long request; a read timeout would abandon a
# job that is still progressing fine.
_client = httpx.AsyncClient(timeout=None)

# Serializes transcriptions against each other, matching midi.py's own lock
# rationale: Whisper and SheetSage2 are small next to YuE2-3B but there is no
# reason to put two decoders on the GPU at once.
_gpu_lock = asyncio.Lock()


@dataclass
class RemixJob:
    status: StageStatus
    error: Optional[str] = None
    stage: Optional[str] = None
    proc: Optional[asyncio.subprocess.Process] = None
    cancel_requested: bool = False
    # Set by cancel(); watched by the yt-dlp progress hook, which cannot see
    # cancel_requested directly because it runs on a worker thread.
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)


_jobs: dict[int, RemixJob] = {}


def _server_path(path: Path) -> str:
    # The native server's JSON parser mishandles backslash escapes in paths it
    # is handed back; forward slashes round-trip fine (same fix as midi.py).
    return str(path).replace("\\", "/")


def ffmpeg_bin() -> Optional[str]:
    """Same locator as routes_yue2_upload.get_ffmpeg_bin, kept here so the
    pipeline does not import a router just for a path helper."""
    candidate = FFMPEG_BIN_DIR / ("ffmpeg.exe" if IS_WINDOWS else "ffmpeg")
    if candidate.is_file():
        return str(candidate)
    env_bin = os.environ.get("FFMPEG_BIN")
    if env_bin:
        found = shutil.which(env_bin)
        if found:
            return found
        if Path(env_bin).is_file():
            return env_bin
    return shutil.which("ffmpeg")


# --- params_json access ------------------------------------------------------


def _default_stages() -> dict[str, dict]:
    return {s: {"status": "idle", "error": None} for s in STAGES}


def load_params(track_id: int) -> dict:
    row = db.get_track(track_id)
    if not row:
        return {}
    return json.loads(row["params_json"] or "{}")


def _save_stages(track_id: int, stages: dict[str, dict]) -> None:
    params = load_params(track_id)
    params["stages"] = stages
    db.update_track_params(track_id, params)


def get_stages(track_id: int) -> dict[str, dict]:
    """Current stage map, merged so a row written before a stage existed still
    reports every stage."""
    params = load_params(track_id)
    stages = params.get("stages") or {}
    return {s: {**_default_stages()[s], **stages.get(s, {})} for s in STAGES}


def _set_stage(track_id: int, stage: str, status: StageStatus, error: Optional[str] = None) -> None:
    stages = get_stages(track_id)
    stages[stage] = {"status": status, "error": error}
    _save_stages(track_id, stages)
    job = _jobs.get(track_id)
    if job is not None:
        job.status = status
        job.error = error
        job.stage = stage


def _dedupe(track_id: int) -> None:
    """Cancel a queued/active job whose row was deleted out from under it."""
    if not db.get_track(track_id):
        job = _jobs.get(track_id)
        if job is not None:
            job.cancel_requested = True
            if job.proc is not None:
                asyncio.ensure_future(_kill_tree(job.proc))


def _reconcile_interrupted() -> None:
    """Report any stage persisted as queued/running as failed. Called once at
    import time; a stage in that state can only be a crash remainder."""
    for row in db.list_tracks(MODEL):
        params = json.loads(row["params_json"] or "{}")
        stages = params.get("stages") or {}
        changed = False
        for stage in STAGES:
            entry = stages.get(stage)
            if entry and entry.get("status") in ("queued", "running"):
                stages[stage] = {"status": "failed", "error": "interrupted"}
                changed = True
        if changed:
            params["stages"] = stages
            db.update_track_params(row["id"], params)


_reconcile_interrupted()


# --- yt-dlp ------------------------------------------------------------------


def _ytdlp_probe(url: str) -> dict:
    """Resolve a URL to its metadata without downloading (runs in a thread)."""
    import yt_dlp

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "ffmpeg_location": ffmpeg_bin(),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


def _ytdlp_download(url: str, out_dir: Path, *, subtitles: dict,
                    cancel_flag: asyncio.Event) -> dict:
    """Download bestaudio as mp3 plus any manual subtitle track (runs in a
    thread). ``cancel_flag`` is set by the event loop to request a cooperative
    stop; yt-dlp calls the hooks often enough to notice promptly."""
    import yt_dlp

    def hook(_d: dict) -> None:
        if cancel_flag.is_set():
            raise _Cancelled()

    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "ffmpeg_location": ffmpeg_bin(),
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "0"},
        ],
        "progress_hooks": [hook],
        "postprocessor_hooks": [hook],
        # Manual subtitles only (writeautomaticsub stays off): a platform's
        # auto-captions are Whisper output, and local Whisper is what we run
        # when the video ships none.
        "writesubtitles": bool(subtitles.get("want")),
        "writeautomaticsub": False,
        "subtitleslangs": subtitles.get("langs") or [],
        "subtitlesformat": "srt/best",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=True)


class _Cancelled(Exception):
    pass


def _manual_subtitle_langs(info: dict) -> list[str]:
    """Language preference for a manual subtitle download: the video's own
    language, then English, then whatever manual track exists."""
    subs = info.get("subtitles") or {}
    if not subs:
        return []
    available = list(subs.keys())
    ordered: list[str] = []
    for lang in (info.get("language"), "en"):
        if lang and lang in subs and lang not in ordered:
            ordered.append(lang)
    for lang in available:
        if lang not in ordered:
            ordered.append(lang)
    return ordered


# --- subprocess helpers ------------------------------------------------------


async def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    # Mirrors stems._kill_tree: on Windows the child needs taskkill /T, on
    # POSIX kill() on the direct child is enough here (no wrapper process).
    if IS_WINDOWS:
        killer = await asyncio.create_subprocess_exec(
            "taskkill", "/PID", str(proc.pid), "/T", "/F",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await killer.wait()
    else:
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def _ffmpeg_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{FFMPEG_BIN_DIR}{os.pathsep}{env.get('PATH', '')}"
    return env


async def _to_wav(src: Path, *, sample_rate: int = 16000, mono: bool = True) -> Path:
    """Convert any audio file to a WAV. Whisper wants 16 kHz mono; callers that
    need the server's own convention pass 44100."""
    bin_path = ffmpeg_bin()
    if not bin_path:
        raise RuntimeError("ffmpeg not found on PATH or in FFMPEG_BIN_DIR")
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    args = [
        bin_path, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src), "-ar", str(sample_rate), "-c:a", "pcm_s16le",
    ]
    if mono:
        args += ["-ac", "1"]
    args.append(out_path)
    proc = await asyncio.create_subprocess_exec(
        *args, env=_ffmpeg_env(),
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        Path(out_path).unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed to convert audio to WAV: {stderr.decode('utf-8', 'replace').strip()[:500]}")
    return Path(out_path)


# --- public API --------------------------------------------------------------


async def start(track_id: int, url: str, info: dict) -> RemixJob:
    """Queue the download+lyrics import for a freshly inserted track."""
    job = RemixJob(status="queued", stage="download")
    _jobs[track_id] = job
    asyncio.create_task(_run_import(track_id, url, info))
    return job


async def start_melody(track_id: int, *, force: bool = False) -> RemixJob:
    """Queue just the melody stage (re-run / first run after YuE2 came up)."""
    job = _jobs.get(track_id)
    if not (job and job.status in ("queued", "running")):
        job = RemixJob(status="queued", stage="melody")
        _jobs[track_id] = job
        asyncio.create_task(_run_melody(track_id, force=force))
    return job


async def start_lyrics_retry(track_id: int) -> RemixJob:
    """Queue just the lyrics stage again (e.g. Whisper was missing before)."""
    job = _jobs.get(track_id)
    if not (job and job.status in ("queued", "running")):
        job = RemixJob(status="queued", stage="lyrics")
        _jobs[track_id] = job
        asyncio.create_task(_run_lyrics_retry(track_id))
    return job


async def cancel(track_id: int) -> dict:
    job = _jobs.get(track_id)
    if job and job.status in ("queued", "running"):
        job.cancel_requested = True
        job.cancel_event.set()
        if job.proc is not None:
            await _kill_tree(job.proc)
    return status(track_id)


def is_active(track_id: int) -> bool:
    job = _jobs.get(track_id)
    return bool(job and job.status in ("queued", "running"))


def forbid(track_id: int) -> None:
    _jobs.pop(track_id, None)


def status(track_id: int) -> dict:
    """Stages from params_json, with a live job's own stage overriding its
    persisted entry (a transition writes params first, the job object last)."""
    stages = get_stages(track_id)
    job = _jobs.get(track_id)
    if job and job.stage and job.status in ("queued", "running", "cancelled", "failed"):
        stages[job.stage] = {"status": job.status, "error": job.error}
    return {"stages": stages, "active": is_active(track_id)}


# --- import flow -------------------------------------------------------------


async def _run_import(track_id: int, url: str, info: dict) -> None:
    try:
        await _stage_download(track_id, info)
        if _cancelled(track_id):
            return
        await _stage_lyrics(track_id, info)
        if _cancelled(track_id):
            return
        if manager.state.models["yue2"].status == ModelStatus.RUNNING:
            await _stage_melody(track_id)
        else:
            _set_stage(track_id, "melody", "waiting_for_yue2")
    except _Cancelled:
        _finish_cancelled(track_id)
    except Exception as exc:  # noqa: BLE001 - any failure must surface to the UI
        _fail_current(track_id, str(exc))


async def _run_lyrics_retry(track_id: int) -> None:
    try:
        await _stage_lyrics(track_id)
    except _Cancelled:
        _finish_cancelled(track_id)
    except Exception as exc:  # noqa: BLE001
        _fail_current(track_id, str(exc))


async def _run_melody(track_id: int, *, force: bool = False) -> None:
    try:
        await _stage_melody(track_id)
    except _Cancelled:
        _finish_cancelled(track_id)
    except Exception as exc:  # noqa: BLE001
        _fail_current(track_id, str(exc))


def _cancelled(track_id: int) -> bool:
    job = _jobs.get(track_id)
    return bool(job and job.cancel_requested)


def _finish_cancelled(track_id: int) -> None:
    job = _jobs.get(track_id)
    if job is None:
        return
    if job.cancel_requested:
        # Mark the stage that was in flight as cancelled; any stage already
        # done keeps its result.
        stage = job.stage or "download"
        stages = get_stages(track_id)
        if stages.get(stage, {}).get("status") not in ("done", "failed", "waiting_for_yue2"):
            _set_stage(track_id, stage, "cancelled")


def _fail_current(track_id: int, error: str) -> None:
    job = _jobs.get(track_id)
    stage = (job.stage if job else None) or "download"
    _set_stage(track_id, stage, "failed", error)


async def _stage_download(track_id: int, info: dict) -> None:
    _set_stage(track_id, "download", "running")
    # The reserved per-source directory comes from the route; fall back to the
    # model dir for a bare retry/restart with no live info.
    out_dir = Path(info["_out_dir"]) if info.get("_out_dir") else db.model_dir(MODEL)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Prefer the original URL (a webpage URL may be rewritten) and the
    # picker's video_id; the resolved webpage_url is the stable citation.
    url = info.get("webpage_url") or info.get("original_url") or info.get("url")
    subs = {"want": bool((info.get("subtitles") or {})), "langs": _manual_subtitle_langs(info)}
    job = _jobs[track_id]
    try:
        await asyncio.to_thread(
            _ytdlp_download, url, out_dir, subtitles=subs, cancel_flag=job.cancel_event)
    except _Cancelled:
        raise
    except Exception as exc:  # noqa: BLE001
        _set_stage(track_id, "download", "failed", str(exc))
        raise

    audio = next((p for p in out_dir.glob("source.*") if p.suffix == ".mp3"), None)
    if audio is None:
        _set_stage(track_id, "download", "failed", "yt-dlp finished but produced no audio file")
        raise RuntimeError("yt-dlp finished but produced no audio file")

    # The row was inserted before the job started; record where the file landed.
    db.update_track_audio_path(track_id, audio)

    if _cancelled(track_id):
        raise _Cancelled()
    _set_stage(track_id, "download", "done")


def _find_subtitle_file(out_dir: Path) -> Optional[Path]:
    for ext in _SUBTITLE_EXTS:
        for path in sorted(out_dir.glob(f"source.*.{ext}")):
            return path
    return None


async def _subtitle_to_srt(path: Path) -> Path:
    """Normalize a downloaded subtitle (vtt or srt) to SRT via ffmpeg."""
    if path.suffix == ".srt":
        return path
    bin_path = ffmpeg_bin()
    if not bin_path:
        raise RuntimeError("ffmpeg not found on PATH or in FFMPEG_BIN_DIR")
    out_path = path.with_suffix(".srt")
    proc = await asyncio.create_subprocess_exec(
        bin_path, "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), str(out_path),
        env=_ffmpeg_env(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"ffmpeg failed to convert subtitles to SRT: {stderr.decode('utf-8', 'replace').strip()[:500]}")
    return out_path


async def _stage_lyrics(track_id: int, info: Optional[dict] = None) -> None:
    _set_stage(track_id, "lyrics", "running")
    row = db.get_track(track_id)
    if row is None:
        raise RuntimeError("track not found")
    # Subtitles are downloaded next to the audio in the source's own dir.
    out_dir = Path(row["audio_path"]).parent
    # Duration comes from the persisted params (survives a retry after a
    # restart); live info, when present, is only a fresher source for it.
    params = load_params(track_id)
    duration = float((info or {}).get("duration") or params.get("duration_s") or 0) or None

    subtitle = _find_subtitle_file(out_dir)
    if subtitle is not None:
        try:
            srt = await _subtitle_to_srt(subtitle)
            lyrics = _format_lyrics(srt.read_text(encoding="utf-8", errors="replace"), duration)
            db.update_track_lyrics(track_id, lyrics)
            _set_lyrics_source(track_id, "subtitles")
            _set_stage(track_id, "lyrics", "done")
            return
        except Exception:  # noqa: BLE001 - fall through to Whisper
            pass

    if not shutil.which(WHISPER_BIN):
        _set_stage(
            track_id, "lyrics", "failed",
            "whisper-cli not found. Install it (./setup_prereqs.sh) or set WHISPER_BIN in backend/.env.",
        )
        return
    if not WHISPER_MODEL_PATH.exists():
        _set_stage(
            track_id, "lyrics", "failed",
            f"Whisper model not found at {WHISPER_MODEL_PATH}. Run ./setup_models.sh or set WHISPER_MODEL_PATH in backend/.env.",
        )
        return

    try:
        srt = await _transcribe_with_whisper(track_id)
    except Exception as exc:  # noqa: BLE001
        _set_stage(track_id, "lyrics", "failed", str(exc))
        return

    try:
        lyrics = _format_lyrics(srt.read_text(encoding="utf-8", errors="replace"), duration)
    except Exception as exc:  # noqa: BLE001
        _set_stage(track_id, "lyrics", "failed", f"could not format transcript: {exc}")
        return
    db.update_track_lyrics(track_id, lyrics)
    _set_lyrics_source(track_id, "whisper")
    _set_stage(track_id, "lyrics", "done")


def _set_lyrics_source(track_id: int, source: str) -> None:
    params = load_params(track_id)
    params["lyrics_source"] = source
    db.update_track_params(track_id, params)


def _format_lyrics(srt_text: str, duration: Optional[float]) -> str:
    """SRT -> sectioned YuE2 lyrics. SheetSage2 structure is not available at
    this point (it runs after lyrics), so when the transcript carries no
    section markers the structurer plans sections by line count."""
    formatter = lyrics_tools.LyricsFormatter()
    structurer = lyrics_tools.LyricsStructurer()
    lyrics, _report = formatter.format(
        subtitles=srt_text, structure="", clean_punct=True, dedupe=True,
        min_line_chars=2, map_sections=True,
    )
    if not any(lyrics_tools._SECTION_TAG_LINE.fullmatch(ln) for ln in lyrics.splitlines()):
        lyrics, _report = structurer.format(lyrics, "", 4, "")
    return lyrics


async def _ensure_vocals(track_id: int) -> Path:
    """Return the Demucs vocals stem, starting separation if needed. Reusing
    the shared stems job means the separation also shows up in the UI."""
    from .. import midi

    existing = midi.source_audio_path(track_id, "vocals")
    if existing and existing.exists():
        return existing
    await stems.start(track_id)
    while stems.is_active(track_id):
        await asyncio.sleep(2)
    st = stems.status(track_id)
    if st["status"] != "done":
        raise RuntimeError(f"stem separation did not finish: {st.get('error') or st['status']}")
    path = midi.source_audio_path(track_id, "vocals")
    if not path or not path.exists():
        raise RuntimeError("stem separation finished but produced no vocals stem")
    return path


async def _transcribe_with_whisper(track_id: int) -> Path:
    vocals = await _ensure_vocals(track_id)
    wav = await _to_wav(vocals, sample_rate=16000, mono=True)
    fd, out_base = tempfile.mkstemp(prefix="remix_whisper_")
    os.close(fd)
    Path(out_base).unlink(missing_ok=True)
    log_name = f"whisper_{track_id}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{log_name}.log"
    env = _ffmpeg_env()
    job = _jobs[track_id]
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
            proc = await asyncio.create_subprocess_exec(
                WHISPER_BIN, "-m", str(WHISPER_MODEL_PATH), "-f", str(wav),
                "-osrt", "-of", out_base, "-np",
                env=env,
                stdout=log_file, stderr=asyncio.subprocess.STDOUT,
            )
            job.proc = proc
            returncode = await proc.wait()
            job.proc = None
        if job.cancel_requested:
            raise _Cancelled()
        if returncode != 0:
            raise RuntimeError(
                f"whisper-cli exited with code {returncode}\n{tail_log(log_name)}")
        srt = Path(out_base + ".srt")
        if not srt.exists():
            raise RuntimeError(f"whisper-cli produced no SRT file\n{tail_log(log_name)}")
        return srt
    finally:
        wav.unlink(missing_ok=True)


async def _stage_melody(track_id: int) -> None:
    if manager.state.models["yue2"].status != ModelStatus.RUNNING:
        _set_stage(track_id, "melody", "waiting_for_yue2")
        return

    _set_stage(track_id, "melody", "running")
    job = _jobs[track_id]
    row = db.get_track(track_id)
    if row is None:
        raise RuntimeError("track not found")

    async with _gpu_lock:
        audio = Path(row["audio_path"])
        if not audio.exists():
            raise RuntimeError("source audio not found")
        wav = await _to_wav(audio, sample_rate=44100, mono=True)
        try:
            base_url = MODELS["yue2"].proxy_target
            spec = yue2_specs()["sheetsage2"]
            await _ensure_loaded(base_url, spec)
            resp = await _client.post(
                f"{base_url}/v1/tasks/run",
                json={
                    "model": spec["id"],
                    "request": {
                        "audio": _server_path(wav),
                        "options": {},
                    },
                },
            )
            resp.raise_for_status()
            abc = _abc_from_result(resp.json())
            if not abc.strip():
                raise RuntimeError("SheetSage2 returned no ABC artifact")
            # Written next to the audio, as the source row promises.
            out_path = audio.with_suffix(".abc")
            out_path.write_text(abc, encoding="utf-8")
            db.update_track_abc(track_id, out_path)
        finally:
            wav.unlink(missing_ok=True)
            if job.cancel_requested:
                raise _Cancelled()
            try:
                await _unload(base_url, spec["id"])
            except Exception:  # noqa: BLE001 - unload is best-effort
                pass

    _set_stage(track_id, "melody", "done")


async def _ensure_loaded(base_url: str, spec: dict) -> None:
    path = Path(spec["path"])
    if not path.exists():
        raise RuntimeError(
            f"SheetSage2 weights not found at {path}. "
            "Install them with: python tools/model_manager_v2.py install sheetsage2_orig")
    listed = await _client.get(f"{base_url}/v1/models")
    listed.raise_for_status()
    for entry in listed.json().get("data") or []:
        if entry.get("id") == spec["id"] and entry.get("loaded"):
            return
    resp = await _client.post(
        f"{base_url}/v1/models/load",
        json={
            "id": spec["id"],
            "path": _server_path(path),
            "family": spec["family"],
            "task": spec["task"],
            "mode": spec["mode"],
            "load_options": {},
            "session_options": {},
        },
    )
    resp.raise_for_status()


async def _unload(base_url: str, model_id: str) -> None:
    resp = await _client.post(f"{base_url}/v1/models/unload", json={"id": model_id})
    resp.raise_for_status()


def _abc_from_result(result: dict) -> str:
    """Mirror of the frontend's abcFromResult: ``text`` first, else the first
    artifact whose format/extension/id names an ABC or score."""
    text = result.get("text")
    if isinstance(text, str) and text.strip():
        return text
    for artifact in result.get("artifacts") or []:
        meta = artifact.get("meta") or {}
        fmt = str(meta.get("format") or meta.get("extension") or artifact.get("id") or "")
        if ("abc" in fmt.lower() or "score" in fmt.lower()):
            payload = artifact.get("payload")
            if isinstance(payload, str):
                try:
                    return base64.b64decode(payload).decode("utf-8", "replace")
                except Exception:  # noqa: BLE001
                    return payload
    return ""
