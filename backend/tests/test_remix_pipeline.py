"""Tests for the remix import pipeline (yt-dlp / Whisper / SheetSage2 stages).

Every external dependency is faked: yt-dlp is a stub YoutubeDL, Whisper is a
small shell script on PATH that writes a fixed SRT, Demucs is faked through
app.stems, and the YuE2 server is an httpx.MockTransport returning an ABC
artifact.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import stat
import struct
import wave
from pathlib import Path

import httpx
import pytest

from app import db, stems
from app.orchestrator.manager import manager
from app.orchestrator.state import ModelStatus
from app.remix import pipeline


# --- fakes -------------------------------------------------------------------


def write_wav(path: Path, seconds: float = 1.0, sample_rate: int = 16000) -> Path:
    """A real (silent) mono PCM WAV, so the pipeline's ffmpeg conversions run
    against something genuine instead of failing on fake bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack("<h", 0) * frames)
    return path


class FakeYDL:
    """Stands in for yt_dlp.YoutubeDL: probe returns ``info``; download writes
    an mp3 (and optionally a subtitle file) and returns the same info."""

    last_opts: dict = {}
    info: dict = {}
    write_subtitle: bool = False

    def __init__(self, opts):
        FakeYDL.last_opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        info = dict(FakeYDL.info)
        if download:
            outtmpl = os.path.expanduser(FakeYDL.last_opts.get("outtmpl", ""))
            out_dir = Path(outtmpl).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            # Named .mp3 to match what the real FFmpegExtractAudio postprocessor
            # produces; the bytes are a real WAV so downstream ffmpeg works.
            audio = write_wav(out_dir / "source.mp3")
            if FakeYDL.write_subtitle and FakeYDL.last_opts.get("writesubtitles"):
                (out_dir / "source.en.srt").write_text(
                    "1\n00:00:00,000 --> 00:00:02,000\nhello world\n\n",
                    encoding="utf-8",
                )
            info["requested_downloads"] = [{"filepath": str(audio)}]
        return info


@pytest.fixture
def fake_ytdlp(monkeypatch):
    import yt_dlp

    FakeYDL.info = {
        "id": "vid123",
        "title": "Test Song",
        "webpage_url": "https://example.com/watch?v=vid123",
        "extractor_key": "Generic",
        "extractor": "generic",
        "uploader": "Tester",
        "duration": 120,
        "thumbnail": "https://example.com/t.jpg",
        "language": "en",
        "subtitles": {},
    }
    FakeYDL.write_subtitle = False
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    return FakeYDL


@pytest.fixture
def fake_whisper(tmp_path, monkeypatch):
    """A shell script standing in for whisper-cli, writing a fixed SRT."""
    script = tmp_path / "whisper-cli"
    script.write_text(
        "#!/bin/sh\n"
        "OUT=\n"
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in -of) OUT="$2"; shift 2;; *) shift;; esac\n'
        "done\n"
        'printf "1\\n00:00:00,000 --> 00:00:02,000\\nwhisper line one\\n\\n'
        '2\\n00:00:02,000 --> 00:00:04,000\\nwhisper line two\\n" > "$OUT.srt"\n',
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    model = tmp_path / "model.bin"
    model.write_bytes(b"fake-model")
    monkeypatch.setattr(pipeline, "WHISPER_BIN", str(script))
    monkeypatch.setattr(pipeline, "WHISPER_MODEL_PATH", model)
    return script


@pytest.fixture
def fake_stems(tmp_path, monkeypatch):
    """Fake Demucs: create a vocals wav and report the job as done."""

    async def fake_start(track_id, *, force=False):
        class J:
            status = "queued"
            error = None

        return J()

    def fake_status(track_id):
        row = db.get_track(track_id)
        vocals = write_wav(Path(row["audio_path"]).parent / "vocals.wav")
        db.update_track_stems(track_id, {"vocals": str(vocals)})
        return {"status": "done", "error": None}

    monkeypatch.setattr(stems, "start", fake_start)
    monkeypatch.setattr(stems, "status", fake_status)
    monkeypatch.setattr(stems, "is_active", lambda track_id: False)
    return fake_status


@pytest.fixture
def fake_yue2(monkeypatch):
    """httpx.MockTransport standing in for the YuE2 server's REST API."""
    abc = "X:1\nT:Fixture\nM:4/4\nL:1/8\nQ:1/4=100\nK:C\nV:Vocal\nC D E F |\n"
    calls = {"load": 0, "run": 0, "unload": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(200, json={"data": []})
        if path == "/v1/models/load":
            calls["load"] += 1
            return httpx.Response(200, json={"ok": True})
        if path == "/v1/models/unload":
            calls["unload"] += 1
            return httpx.Response(200, json={"ok": True})
        if path == "/v1/tasks/run":
            calls["run"] += 1
            return httpx.Response(200, json={
                "artifacts": [{
                    "id": "score",
                    "meta": {"format": "abc"},
                    "payload": base64.b64encode(abc.encode()).decode(),
                }],
            })
        return httpx.Response(404, json={"error": "not found"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(pipeline, "_client", client)
    calls["abc"] = abc
    return calls


# --- probe / validation ------------------------------------------------------


def test_probe_accepts_single_video(fake_ytdlp):
    info = pipeline._ytdlp_probe("https://example.com/watch?v=vid123")
    assert info["id"] == "vid123"
    assert info["duration"] == 120


def test_ytdlp_download_writes_audio(fake_ytdlp, tmp_path):
    out_dir = tmp_path / "out"
    fake_ytdlp.write_subtitle = True
    pipeline._ytdlp_download(
        "https://example.com/watch?v=vid123", out_dir,
        subtitles={"want": True, "langs": ["en"]},
        cancel_flag=asyncio.Event(),
    )
    assert (out_dir / "source.mp3").exists()
    assert (out_dir / "source.en.srt").exists()


def test_manual_subtitle_lang_preference():
    info = {"language": "de", "subtitles": {"fr": [], "en": [], "de": []}}
    assert pipeline._manual_subtitle_langs(info) == ["de", "en", "fr"]
    assert pipeline._manual_subtitle_langs({"subtitles": {}}) == []


# --- stage persistence -------------------------------------------------------


def test_reconcile_marks_running_stages_interrupted():
    tid = db.insert_track(
        model="youtube", title="t", lyrics="", seed=None, duration_ms=None,
        wall_ms=None, params={"stages": {"download": {"status": "done"},
                                         "lyrics": {"status": "running"},
                                         "melody": {"status": "queued"}}},
        audio_path=Path("/tmp/x.mp3"), abc_path=None,
    )
    pipeline._reconcile_interrupted()
    stages = pipeline.get_stages(tid)
    assert stages["download"]["status"] == "done"
    assert stages["lyrics"]["status"] == "failed"
    assert stages["lyrics"]["error"] == "interrupted"
    assert stages["melody"]["status"] == "failed"


def test_default_stages_shape():
    stages = pipeline._default_stages()
    assert list(stages) == list(pipeline.STAGES)
    assert all(v["status"] == "idle" for v in stages.values())


# --- end-to-end import (all fakes) -------------------------------------------


def _make_track(tmp_path, **params):
    audio = write_wav(tmp_path / "src" / "source.mp3")
    base = {"stages": pipeline._default_stages(), "duration_s": 120.0}
    base.update(params)
    return db.insert_track(
        model="youtube", title="Test Song", lyrics="", seed=None,
        duration_ms=120000, wall_ms=None, params=base,
        audio_path=audio, abc_path=None,
    )


@pytest.mark.anyio
async def test_import_download_and_lyrics_via_whisper(
    fake_ytdlp, fake_whisper, fake_stems, tmp_path
):
    tid = _make_track(tmp_path)
    info = {**fake_ytdlp.info, "_out_dir": str(tmp_path / "src")}

    manager.state.models["yue2"].status = ModelStatus.STOPPED
    await pipeline._run_import(tid, "https://example.com/watch?v=vid123", info)

    stages = pipeline.get_stages(tid)
    assert stages["download"]["status"] == "done"
    assert stages["lyrics"]["status"] == "done"
    assert stages["melody"]["status"] == "waiting_for_yue2"
    row = db.get_track(tid)
    assert "whisper line one" in row["lyrics"]
    assert json.loads(row["params_json"])["lyrics_source"] == "whisper"


@pytest.mark.asyncio
async def test_import_uses_manual_subtitles(fake_ytdlp, fake_whisper, fake_stems, tmp_path):
    tid = _make_track(tmp_path)
    fake_ytdlp.write_subtitle = True
    fake_ytdlp.info = {**fake_ytdlp.info, "subtitles": {"en": [{"ext": "srt"}]}}
    info = {**fake_ytdlp.info, "_out_dir": str(tmp_path / "src")}

    manager.state.models["yue2"].status = ModelStatus.STOPPED
    await pipeline._run_import(tid, "https://example.com/watch?v=vid123", info)

    row = db.get_track(tid)
    assert "hello world" in row["lyrics"]
    assert json.loads(row["params_json"])["lyrics_source"] == "subtitles"


@pytest.mark.asyncio
async def test_import_missing_whisper_marks_lyrics_failed(
    fake_ytdlp, fake_stems, tmp_path, monkeypatch
):
    tid = _make_track(tmp_path)
    info = {**fake_ytdlp.info, "_out_dir": str(tmp_path / "src")}
    monkeypatch.setattr(pipeline, "WHISPER_BIN", "definitely-not-a-real-binary-xyz")
    manager.state.models["yue2"].status = ModelStatus.STOPPED
    await pipeline._run_import(tid, "https://example.com/watch?v=vid123", info)

    stages = pipeline.get_stages(tid)
    assert stages["download"]["status"] == "done"
    assert stages["lyrics"]["status"] == "failed"
    assert "WHISPER_BIN" in stages["lyrics"]["error"]
    assert db.get_track(tid)["audio_path"].endswith("source.mp3")


@pytest.mark.asyncio
async def test_import_missing_whisper_model_marks_lyrics_failed(
    fake_ytdlp, fake_whisper, fake_stems, tmp_path, monkeypatch
):
    tid = _make_track(tmp_path)
    info = {**fake_ytdlp.info, "_out_dir": str(tmp_path / "src")}
    monkeypatch.setattr(pipeline, "WHISPER_MODEL_PATH", Path("/nonexistent/model.bin"))
    manager.state.models["yue2"].status = ModelStatus.STOPPED
    await pipeline._run_import(tid, "https://example.com/watch?v=vid123", info)

    stages = pipeline.get_stages(tid)
    assert stages["lyrics"]["status"] == "failed"
    assert "WHISPER_MODEL_PATH" in stages["lyrics"]["error"]


@pytest.mark.asyncio
async def test_melody_waits_for_yue2(fake_ytdlp, fake_whisper, fake_stems, tmp_path):
    tid = _make_track(tmp_path)
    info = {**fake_ytdlp.info, "_out_dir": str(tmp_path / "src")}
    manager.state.models["yue2"].status = ModelStatus.STOPPED
    await pipeline._run_import(tid, "https://example.com/watch?v=vid123", info)
    assert pipeline.get_stages(tid)["melody"]["status"] == "waiting_for_yue2"


@pytest.mark.asyncio
async def test_melody_done_when_yue2_running(
    fake_ytdlp, fake_whisper, fake_stems, fake_yue2, tmp_path, monkeypatch
):
    tid = _make_track(tmp_path)
    info = {**fake_ytdlp.info, "_out_dir": str(tmp_path / "src")}
    manager.state.models["yue2"].status = ModelStatus.RUNNING

    import app.config as config
    spec = config.yue2_specs()["sheetsage2"]
    real_model = tmp_path / "sheetsage2.gguf"
    real_model.write_bytes(b"fake-weights")
    specs = config.yue2_specs()
    specs["sheetsage2"] = {**spec, "path": str(real_model)}
    monkeypatch.setattr(pipeline, "yue2_specs", lambda: specs)

    await pipeline._run_import(tid, "https://example.com/watch?v=vid123", info)

    stages = pipeline.get_stages(tid)
    assert stages["melody"]["status"] == "done"
    row = db.get_track(tid)
    assert row["abc_path"] and Path(row["abc_path"]).exists()
    assert "V:Vocal" in Path(row["abc_path"]).read_text()
    assert fake_yue2["run"] == 1
    assert fake_yue2["unload"] == 1


def test_abc_from_result_prefers_text():
    assert pipeline._abc_from_result({"text": "X:1\n"}) == "X:1\n"
    payload = base64.b64encode(b"X:2\n").decode()
    result = {"artifacts": [{"id": "score", "meta": {"format": "abc"}, "payload": payload}]}
    assert pipeline._abc_from_result(result) == "X:2\n"
    assert pipeline._abc_from_result({}) == ""


@pytest.mark.asyncio
async def test_cancel_marks_stage_cancelled(fake_ytdlp, tmp_path):
    tid = _make_track(tmp_path)
    job = pipeline.RemixJob(status="running", stage="download")
    pipeline._jobs[tid] = job
    await pipeline.cancel(tid)
    assert pipeline.get_stages(tid)["download"]["status"] == "cancelled"
