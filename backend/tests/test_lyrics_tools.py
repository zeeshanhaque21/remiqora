"""Tests for the pure-text lyric tools ported from ComfyUI-YuE2."""
from __future__ import annotations

import pytest

from app.remix import lyrics_tools as lyrics


SRT = """1
00:00:00,000 --> 00:00:02,000
First SRT line

2
00:00:02,000 --> 00:00:04,000
Second SRT line here

3
00:00:04,000 --> 00:00:06,000
Third SRT line text
"""

LRC = """[ti:Fixture]
[00:00.00]Title - Artist
[00:01.00]LRC line one here
[00:03.00]LRC line two here
[00:05.00]LRC line three here
"""

STRUCTURE = "0.0\t3.0\tintro\n3.0\t6.0\tverse\n6.0\t9.0\tchorus\n"
LYRICS_TEXT = "line one\nline two\nline three\nline four\nline five\nline six\n"


def test_parse_structure_spans_tab_and_merge():
    spans = lyrics.parse_structure_spans("0\t3\tverse\n3\t6\tverse\n6\t9\tchorus\n", 10.0)

    # Adjacent same-name spans merge into one.
    assert spans == [(0.0, 6.0, "verse"), (6.0, 9.0, "chorus")]


def test_parse_structure_spans_start_points():
    spans = lyrics.parse_structure_spans("0: intro\n3: verse\n", 9.0)

    assert spans == [(0.0, 3.0, "intro"), (3.0, 9.0, "verse")]


def test_parse_structure_spans_dash_form():
    spans = lyrics.parse_structure_spans("1.5 - 4.5 : chorus\n", 9.0)

    assert spans == [(1.5, 4.5, "chorus")]


def test_parse_subtitles_srt():
    rows = lyrics.LyricsFormatter().parse_subtitles(SRT)

    assert [t for _s, _e, t in rows] == [
        "First SRT line", "Second SRT line here", "Third SRT line text"]
    assert rows[0][:2] == (0.0, 2.0)
    assert rows[2][:2] == (4.0, 6.0)


def test_parse_subtitles_lrc_extends_end_to_next_start():
    rows = lyrics.LyricsFormatter().parse_subtitles(LRC)

    # Metadata and the "title - artist" heading are dropped.
    assert [t for _s, _e, t in rows] == [
        "LRC line one here", "LRC line two here", "LRC line three here"]
    assert rows[0][:2] == (1.0, 3.0)
    assert rows[2][:2] == (5.0, 13.0)  # last row extends by 8s


def test_parse_subtitles_timestamp_form():
    rows = lyrics.LyricsFormatter().parse_subtitles(
        "0.0-2.0: Hello world\n2.0-4.0: second line here\n")

    assert rows == [(0.0, 2.0, "Hello world"), (2.0, 4.0, "second line here")]


def test_format_srt_into_sections():
    out, report = lyrics.LyricsFormatter().format(
        SRT, STRUCTURE, True, True, 2, True)

    assert "[intro]" in out
    assert "[verse]" in out
    assert "[chorus]" in out
    assert "First SRT line" in out
    assert "sections=3" in report


def test_format_lrc_into_sections():
    out, report = lyrics.LyricsFormatter().format(
        LRC, STRUCTURE, True, True, 2, True)

    assert out.count("[") == 3
    assert "LRC line one here" in out
    assert "lines=3 kept=3" in report


def test_format_without_structure_uses_single_verse():
    out, report = lyrics.LyricsFormatter().format(
        SRT, "", True, True, 2, True)

    assert out.startswith("[verse]")
    assert "sections=1" in report


def test_format_clean_punct_and_dedupe():
    subs = "0.0-2.0: Hello, world!\n2.0-4.0: Hello, world!\n"
    out, _report = lyrics.LyricsFormatter().format(subs, "", True, True, 2, True)

    assert out.count("Hello world") == 1


def test_format_empty_raises():
    with pytest.raises(ValueError):
        lyrics.LyricsFormatter().format("", "", True, True, 2, True)


def test_format_min_line_chars_drops_short_lines():
    subs = "0.0-2.0: ab\n2.0-4.0: long enough line\n"
    out, _report = lyrics.LyricsFormatter().format(subs, "", True, True, 5, True)

    assert "ab" not in out.replace("[", "").replace("]", "")
    assert "long enough line" in out


def test_structurer_plain_verse_chunks():
    out, report = lyrics.LyricsStructurer().format(LYRICS_TEXT, "", 4, "")

    assert out == (
        "[verse]\nline one\nline two\nline three\nline four\n\n"
        "[verse]\nline five\nline six\n")
    assert "sections=2" in report


def test_structurer_section_plan():
    out, report = lyrics.LyricsStructurer().format(LYRICS_TEXT, "", 4, "verse:2,chorus:2")

    assert "[verse]\nline one\nline two" in out
    assert "[chorus]\nline three\nline four" in out
    assert "plan=verse:2,chorus:2" in report


def test_structurer_structure_distributes_by_duration():
    out, report = lyrics.LyricsStructurer().format(LYRICS_TEXT, STRUCTURE, 4, "")

    # intro is non-singing and dropped; verse and chorus each take 3 lines.
    assert out.count("[verse]") == 1
    assert out.count("[chorus]") == 1
    assert "[intro]" not in out
    assert "plan=verse:3,chorus:3" in report


def test_structurer_preserves_existing_tags():
    tagged = "[verse]\nline one\nline two\n[chorus]\nline three\n"
    out, report = lyrics.LyricsStructurer().format(tagged, "", 4, "")

    assert out == tagged
    assert "Existing section tags preserved" in report


def test_structurer_empty_raises():
    with pytest.raises(ValueError):
        lyrics.LyricsStructurer().format("", "", 4, "")


def test_parse_plan_defaults_to_verse():
    plan = lyrics.LyricsStructurer._parse_plan("", 4, 9)

    assert plan == [("verse", 4), ("verse", 4), ("verse", 4)]
