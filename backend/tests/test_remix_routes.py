"""Tests for the /api/remix routes.

The pipeline's probe and job starters are monkeypatched so these tests focus
on request validation and the shape of the responses, not the stages (covered
by test_remix_pipeline.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db
from app.config import REMIX_MAX_DURATION_S
from app.main import app
from app.orchestrator.manager import manager
from app.orchestrator.state import ModelStatus
from app.remix import pipeline

client = TestClient(app)


@pytest.fixture
def probe(monkeypatch):
    """Replace the yt-dlp probe with a dict that tests can mutate."""
    info = {
        "id": "vid123",
        "title": "Test Song",
        "webpage_url": "https://example.com/watch?v=vid123",
        "extractor_key": "Generic",
        "extractor": "generic",
        "uploader": "Tester",
        "duration": 120,
        "thumbnail": "https://example.com/t.jpg",
        "subtitles": {},
    }
    monkeypatch.setattr(pipeline, "_ytdlp_probe", lambda url: dict(info))
    return info


@pytest.fixture
def no_pipeline_start(monkeypatch):
    """Record that a job was queued without running it."""
    started = {}

    async def fake_start(track_id, url, info):
        started["track_id"] = track_id
        started["url"] = url
        return None

    monkeypatch.setattr(pipeline, "start", fake_start)
    return started


def _insert_track(**params):
    audio = Path("/tmp/remix-test/source.mp3")
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"x")
    base = {"stages": pipeline._default_stages(), "duration_s": 120.0,
            "video_id": "vid123", "extractor": "generic"}
    base.update(params)
    return db.insert_track(
        model="youtube", title="T", lyrics="", seed=None, duration_ms=120000,
        wall_ms=None, params=base, audio_path=audio, abc_path=None,
    )


# --- import validation -------------------------------------------------------


def test_import_returns_track_and_remix(probe, no_pipeline_start):
    r = client.post("/api/remix/import", json={"url": "https://example.com/watch?v=vid123"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "youtube"
    assert body["title"] == "Test Song"
    assert body["remix"]["video_id"] == "vid123"
    assert body["remix"]["duration_s"] == 120
    assert body["remix"]["stages"]["download"]["status"] == "idle"


def test_import_rejects_playlist(probe, no_pipeline_start):
    probe["_type"] = "playlist"
    probe["entries"] = [{"id": "a"}, {"id": "b"}]
    r = client.post("/api/remix/import", json={"url": "https://example.com/list"})
    assert r.status_code == 400
    assert "playlist" in r.json()["detail"].lower()


def test_import_rejects_too_long(probe, no_pipeline_start):
    probe["duration"] = REMIX_MAX_DURATION_S + 1
    r = client.post("/api/remix/import", json={"url": "https://example.com/watch?v=vid123"})
    assert r.status_code == 400
    assert "limit" in r.json()["detail"].lower()


def test_import_rejects_missing_duration(probe, no_pipeline_start):
    probe["duration"] = None
    r = client.post("/api/remix/import", json={"url": "https://example.com/watch?v=vid123"})
    assert r.status_code == 400


def test_import_rejects_empty_url(no_pipeline_start):
    r = client.post("/api/remix/import", json={"url": "   "})
    assert r.status_code == 400


def test_import_dedupes_by_video_id(probe, no_pipeline_start):
    # The stored extractor uses a different case than the probe's
    # "extractor_key" ("Generic"); dedupe must still find the existing row.
    first = db.insert_track(
        model=pipeline.MODEL, title="existing", lyrics="", seed=None,
        duration_ms=120000, wall_ms=None,
        params={"video_id": "vid123", "extractor": "generic"},
        audio_path=None, abc_path=None,
    )
    r = client.post("/api/remix/import", json={"url": "https://example.com/watch?v=vid123"})
    assert r.status_code == 200
    assert r.json()["id"] == first
    # No second row was inserted.
    assert len(db.list_tracks(pipeline.MODEL)) == 1


def test_find_existing_matches_video_id_case_insensitive_extractor():
    from app.api.routes_remix import _find_existing

    db.insert_track(
        model=pipeline.MODEL, title="existing", lyrics="", seed=None,
        duration_ms=120000, wall_ms=None,
        params={"video_id": "vid123", "extractor": "generic"},
        audio_path=None, abc_path=None,
    )
    # yt-dlp may report "Generic" (extractor_key) while the row was stored as
    # "generic" - the same source must still dedupe.
    assert _find_existing("Generic", "vid123") is not None
    # A different video id must not match.
    assert _find_existing("Generic", "other") is None


def test_import_rejects_bad_probe(monkeypatch, no_pipeline_start):
    def boom(url):
        raise RuntimeError("nope")

    monkeypatch.setattr(pipeline, "_ytdlp_probe", boom)
    r = client.post("/api/remix/import", json={"url": "https://example.com/x"})
    assert r.status_code == 400
    assert "could not read" in r.json()["detail"].lower()


# --- get / not-found ---------------------------------------------------------


def test_get_remix_track():
    tid = _insert_track()
    r = client.get(f"/api/remix/{tid}")
    assert r.status_code == 200
    assert r.json()["id"] == tid


def test_get_non_youtube_is_404():
    tid = db.insert_track(
        model="upload", title="u", lyrics="", seed=None, duration_ms=None,
        wall_ms=None, params={}, audio_path=Path("/tmp/u.wav"), abc_path=None)
    assert client.get(f"/api/remix/{tid}").status_code == 404
    assert client.get("/api/remix/999999").status_code == 404


# --- stage endpoints ---------------------------------------------------------


def test_retry_lyrics_queues(monkeypatch):
    tid = _insert_track()
    called = {}

    async def fake_retry(track_id):
        called["id"] = track_id

    monkeypatch.setattr(pipeline, "start_lyrics_retry", fake_retry)
    r = client.post(f"/api/remix/{tid}/lyrics/retry")
    assert r.status_code == 200
    assert called["id"] == tid


def test_melody_409_when_yue2_stopped():
    tid = _insert_track()
    manager.state.models["yue2"].status = ModelStatus.STOPPED
    r = client.post(f"/api/remix/{tid}/melody")
    assert r.status_code == 409
    assert "yue2" in r.json()["detail"].lower()


def test_melody_queues_when_yue2_running(monkeypatch):
    tid = _insert_track()
    manager.state.models["yue2"].status = ModelStatus.RUNNING
    called = {}

    async def fake_melody(track_id, *, force=False):
        called["id"] = track_id

    monkeypatch.setattr(pipeline, "start_melody", fake_melody)
    r = client.post(f"/api/remix/{tid}/melody")
    assert r.status_code == 200
    assert called["id"] == tid


def test_cancel(monkeypatch):
    tid = _insert_track()
    called = {}

    async def fake_cancel(track_id):
        called["id"] = track_id
        return {"stages": {}, "active": False}

    monkeypatch.setattr(pipeline, "cancel", fake_cancel)
    r = client.post(f"/api/remix/{tid}/cancel")
    assert r.status_code == 200
    assert called["id"] == tid


# --- lyrics / abc PUT --------------------------------------------------------


def test_put_lyrics_marks_edited():
    tid = _insert_track()
    r = client.put(f"/api/remix/{tid}/lyrics", json={"lyrics": "[verse]\nhi\n"})
    assert r.status_code == 200
    row = db.get_track(tid)
    assert row["lyrics"].startswith("[verse]")
    assert json.loads(row["params_json"])["lyrics_source"] == "edited"


def test_put_abc_writes_file(tmp_path):
    tid = _insert_track()
    db.update_track_audio_path(tid, tmp_path / "source.mp3")
    r = client.put(f"/api/remix/{tid}/abc", json={"abc": "X:1\n"})
    assert r.status_code == 200
    row = db.get_track(tid)
    assert Path(row["abc_path"]).read_text() == "X:1\n"
    assert r.json()["abc_url"] == f"/api/tracks/{tid}/abc"


# --- ABC tool endpoint -------------------------------------------------------

ABC = "X:1\nM:4/4\nL:1/8\nQ:1/4=100\nK:C\nV:Vocal\nC D E F |\n"


def test_abc_tool_analyze():
    r = client.post("/api/remix/abc/analyze", json={"abc": ABC, "kwargs": {}})
    assert r.status_code == 200
    body = r.json()
    assert body["tool"] == "analyze"
    assert "BPM: 100" in body["result"][0]


def test_abc_tool_modify():
    r = client.post("/api/remix/abc/modify", json={
        "abc": ABC,
        "kwargs": {"tempo_mode": "override", "bpm": 130.0},
    })
    assert r.status_code == 200
    out = r.json()["result"][0]
    assert "Q:1/4=130" in out


def test_abc_tool_clean():
    r = client.post("/api/remix/abc/clean", json={"abc": ABC, "kwargs": {"preset": "off"}})
    assert r.status_code == 200


def test_abc_tool_fit_defaults_lyrics():
    r = client.post("/api/remix/abc/fit", json={"abc": ABC, "kwargs": {"lyrics": "[verse]\nla la\n"}})
    assert r.status_code == 200
    assert "vocal_notes=" in r.json()["result"]


def test_abc_tool_unknown_is_404():
    r = client.post("/api/remix/abc/bogus", json={"abc": ABC})
    assert r.status_code == 404


def test_abc_tool_bad_kwargs_is_400():
    r = client.post("/api/remix/abc/analyze", json={"abc": ABC, "kwargs": {"nonexistent": 1}})
    assert r.status_code == 400


def test_abc_tool_empty_input_is_400():
    r = client.post("/api/remix/abc/analyze", json={"abc": "", "kwargs": {}})
    assert r.status_code == 400


# --- delete ------------------------------------------------------------------


def test_delete_remix(monkeypatch):
    tid = _insert_track()
    monkeypatch.setattr(pipeline, "is_active", lambda track_id: False)
    r = client.delete(f"/api/remix/{tid}")
    assert r.status_code == 200
    assert db.get_track(tid) is None


def test_delete_409_when_importing(monkeypatch):
    tid = _insert_track()
    monkeypatch.setattr(pipeline, "is_active", lambda track_id: True)
    r = client.delete(f"/api/remix/{tid}")
    assert r.status_code == 409
