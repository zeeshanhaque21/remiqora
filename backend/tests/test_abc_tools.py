"""Tests for the pure-text ABC tools ported from ComfyUI-YuE2."""
from __future__ import annotations

import pytest

from app.remix import abc_tools as abc


# Two voices with a repeated chord line, one section comment per voice.
ABC = """X:1
T:Fixture
M:4/4
L:1/8
Q:1/4=100
K:C
%intro
V:Vocal
C D E F G A B c |
%verse
V:Vocal
c c c c z z z z |
V:Instrumental
"C" C2 E2 G2 c2 | "G" G2 B2 d2 g2 |
"""

# Vocal sits a full octave below the female ranges, so retarget shifts it up.
ABC_VOCAL_LOW = """X:1
M:4/4
L:1/8
Q:1/4=90
K:C
%verse
V:Vocal
C, D, E, F, G, A, B, C |
"""


def test_analyze_abc_reports_bpm_key_duration():
    report, bpm, key, duration = abc.analyze_abc(ABC)

    assert bpm == 100.0
    assert key == "C"
    assert duration > 0
    assert "BPM: 100" in report
    assert "Key: C" in report
    assert "Vocal: notes=12" in report
    assert "Instrumental: notes=8" in report


def test_analyze_abc_empty_raises():
    with pytest.raises(ValueError):
        abc.analyze_abc("   ")


def test_analyze_wraps_empty_for_node():
    assert abc.analyze("") == (
        "ABC input is empty (score planning is disabled).", 0.0, "", 0.0)


def test_score_notes_skips_chords_and_tracks_voice():
    notes = abc.score_notes(ABC)
    voices = {v for v, _s, _p in notes}

    assert voices == {"Vocal", "Instrumental"}
    # Chords are quoted and must not be parsed as notes.
    assert len(notes) == 20
    assert all(0 <= p <= 127 for _v, _s, p in notes)


def test_modify_transposes_whole_score_and_updates_key():
    out, report = abc.modify(ABC, "override", 130.0, 1.0, "whole_score", 2, False, "")

    assert "Q:1/4=130" in out
    assert "K:D" in out
    assert "bpm 100->130" in report
    assert "transpose=2" in report


def test_modify_bpm_multiplier():
    out, report = abc.modify(ABC, "multiplier", 120.0, 1.5, "none", 0, False, "")

    assert "Q:1/4=150" in out
    assert "bpm 100->150" in report


def test_modify_drops_named_sections():
    out, _report = abc.modify(ABC, "keep", 120.0, 1.0, "none", 0, False, "intro")

    assert "%intro" not in out
    assert "%verse" in out


def test_modify_removes_chords():
    out, _report = abc.modify(ABC, "keep", 120.0, 1.0, "whole_score", 2, True, "")

    assert '"' not in out


def test_retarget_chooses_octave_shift_within_safe_modes():
    out, shift, report = abc.retarget(
        ABC_VOCAL_LOW, "female_soprano", "nearest_key_safe", 0)

    assert shift == 12
    assert shift % 12 == 0
    assert out != ABC_VOCAL_LOW
    assert "applied=+12 semitones" in report


def test_retarget_original_is_noop():
    out, shift, _report = abc.retarget(ABC_VOCAL_LOW, "original", "nearest_key_safe", 0)

    assert shift == 0
    assert out == ABC_VOCAL_LOW


def test_retarget_without_vocal_voice_raises():
    no_vocal = "X:1\nM:4/4\nL:1/8\nK:C\nV:Ins\nC D E F |\n"
    with pytest.raises(ValueError):
        abc.retarget(no_vocal, "female_alto", "nearest_key_safe", 0)


def test_clean_medium_replaces_outliers_with_rests_keeping_duration():
    extreme = (
        "X:1\nM:4/4\nL:1/8\nQ:1/4=100\nK:C\nV:Vocal\n"
        "C4 D4 E4 c'4 c'4 c'4 |\n"
    )
    before = abc.analyze_abc(extreme)[3]
    out, report = abc.clean(extreme, "medium", 40.0, True, 12)

    assert "pitch_outliers_removed=" in report
    assert "pitch_outliers_removed=0" not in report  # at least one removed
    # Removed notes become rests of the same duration, so measure count holds.
    assert abc.analyze_abc(out)[3] == pytest.approx(before)


def test_clean_short_notes_replaced_with_rests():
    # L:1/32 default grid: a 1/32 note is 37.5ms at 100bpm, under the 50ms preset.
    short = "X:1\nM:4/4\nL:1/32\nQ:1/4=100\nK:C\nV:Vocal\nC/ D/ E/ F/ G/ A/ B/ c/\n"
    out, report = abc.clean(short, "custom", 50.0, False, 24)

    assert "short_notes_removed=" in report
    assert "short_notes_removed=0" not in report
    assert "z/" in out


def test_clean_off_is_noop():
    out, report = abc.clean(ABC, "off", 40.0, True, 24)

    assert out == ABC
    assert report == "cleanup disabled"


def test_build_style_joins_and_dedupes():
    style = abc.build_style(
        genre="pop, rock", secondary_genres="rock, indie", era="1990s",
        vocal="", instruments="", drums="", mood="", tempo="", language="English")

    assert style == "English, pop, rock, indie, 1990s"


def test_lyric_melody_fit_report_per_section():
    lyrics = "[intro]\nla la la\n[verse]\noh oh oh oh oh oh\n"
    report = abc.lyric_melody_fit(ABC, lyrics, "English")

    assert "vocal_notes=" in report
    assert "estimated_syllables=" in report
    assert "[verse]" in report
    assert "[intro]" in report
