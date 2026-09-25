"""Pure-text lyric formatting: timed subtitles and section tagging.

Ported from ``nodes/sheetsage2.py`` in ScryptHunter/ComfyUI-YuE2 at commit
7ed198838d112df7a2e4f4de7e903d1044632eee (itself a fork of
piscesbody/ComfyUI-YuE2), Apache-2.0; the original license is retained in
``LICENSE-ComfyUI-YuE2``.

The upstream functions/classes ported here are ``parse_structure_spans``,
``LyricsFormatter`` (``parse_subtitles``, ``_fill_missing_times``,
``parse_structure`` and ``format``) and ``LyricsStructurer`` (``format`` and
``_parse_plan``).

Intentionally left out: ``SheetSage2Loader`` / ``SheetSage2Transcribe`` /
``SheetSage2Unload`` (they load torch models and read audio),
``SaveMidiFile``, and ``LoadLyricsFile`` (all filesystem-bound), plus
everything under ``models/``. Remiqora's audio.cpp SheetSage2 returns only an
ABC artifact, so the MIDI-based ABC recovery path upstream never runs here.
"""
from __future__ import annotations

import re

# SheetSage2 structure names mapped to YuE2 section tags (unknown names kept verbatim).
_SECTION_ALIASES = {
    "silence": "intro",
    "instrumental": "interlude",
    "solo": "interlude",
}

# Instrumental sections that get no lyric lines when structuring.
_NON_SINGING = {"intro", "silence", "interlude", "instrumental", "solo"}
_SECTION_TAG_LINE = re.compile(r"^\s*\[[^\]\r\n]+\]\s*$", re.IGNORECASE)


def parse_structure_spans(text: str, duration_hint: float):
    """Parse structure text into ``[(start, end, name)]``, merging adjacent same-name spans.

    Accepts ``start<tab>end<tab>name``, ``start-end: name`` and
    ``start: name`` line formats.
    """
    spans: list[tuple[float, float, str]] = []
    points: list[tuple[float, str]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = re.split(r"\t+", line)
        if len(parts) == 3:
            try:
                spans.append((float(parts[0]), float(parts[1]), parts[2].strip()))
                continue
            except ValueError:
                pass
        m = re.match(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*:\s*(.+)$", line)
        if m:
            spans.append((float(m.group(1)), float(m.group(2)), m.group(3).strip()))
            continue
        m = re.match(r"^(\d+(?:\.\d+)?)\s*:\s*(.+)$", line)
        if m:
            points.append((float(m.group(1)), m.group(2).strip()))

    if not spans and points:
        points.sort()
        for i, (start, name) in enumerate(points):
            end = points[i + 1][0] if i + 1 < len(points) else duration_hint
            spans.append((start, end, name))

    # Merge adjacent same-name spans.
    merged: list[tuple[float, float, str]] = []
    for start, end, name in sorted(spans):
        if merged and merged[-1][2] == name and abs(merged[-1][1] - start) < 1e-3:
            merged[-1] = (merged[-1][0], end, name)
        else:
            merged.append((start, end, name))
    return merged


class LyricsFormatter:
    """Turn timed lyric subtitles into YuE2 sectioned lyrics.

    Inputs (both may be pasted as plain text):
    - ``subtitles``: ``start-end: text`` per line, or standard SRT/LRC
    - ``structure``: ``start<tab>end<tab>section`` per line (SheetSage2 output)

    Processing: parse times -> assign each line to a structure span by its
    midpoint -> insert ``[verse]``/``[chorus]`` tags -> emit empty tags for
    spans with no words (interludes) -> clean punctuation and adjacent
    duplicate lines.
    """

    _PUNCT = str.maketrans("", "", "。！？，、；：,.!?;:\"'`()[]{}<>~…·\u2014“”‘’")

    def parse_subtitles(self, text: str):
        """Parse ``start-end: text``, standard SRT or LRC into ``[(start, end, text)]``.

        Rows whose timestamps are unreliable get ``None`` times: forced
        aligners can emit zero-length ``0.00-0.00`` spans on instrumental or
        breathy passages, and using them directly would drop every such line
        into the first section. :meth:`format` fills those back in by line order.
        """
        rows: list[tuple[float | None, float | None, str]] = []
        pending_start = None
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*:\s*(.+)$", line)
            if m:
                rows.append((float(m.group(1)), float(m.group(2)), m.group(3).strip()))
                continue
            # LRC: [mm:ss.xx]lyric (a line may carry several consecutive time tags).
            # A start with no end records end as None, extended below to the next start.
            lrc = re.match(r"^((?:\[\d+:\d+(?:\.\d+)?\])+)(.*)$", line)
            if lrc:
                text_part = lrc.group(2).strip()
                if text_part:
                    for tag in re.findall(r"\[(\d+):(\d+(?:\.\d+)?)\]", lrc.group(1)):
                        t = int(tag[0]) * 60 + float(tag[1])
                        rows.append((t, None, text_part))
                continue
            m = re.match(
                r"^(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)$", line)
            if m:
                g = [int(x) for x in m.groups()]
                start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
                end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
                rows.append((start, end, ""))
                pending_start = start
                continue
            # SRT body line: attach it to the preceding timing row.
            if pending_start is not None and rows and rows[-1][2] == "":
                s, e, _ = rows[-1]
                rows[-1] = (s, e, line)
                pending_start = None

        out = []
        for start, end, text in rows:
            if not text:
                continue
            # LRC metadata head rows (title/author/...), not lyrics.
            if re.match(r"^(词|曲|编曲|作词|作曲|编|混音|母带|和声|吉他|钢琴|"
                        r"制作|监制|歌词|专辑|歌手|title|artist|album|by)[:：]",
                        text, re.IGNORECASE):
                continue
            # "title - artist" heading lines (LRC convention: time 0 and contains " - ").
            if start == 0 and " - " in text:
                continue
            # Zero-length, inverted or (0,0) sentinel: the aligner failed on
            # singing, so the time is unknown. An LRC row with end=None is a
            # normal start-only row and gets its end on the next loop.
            if start is None or (end is not None and (end <= start or
                                                      (start == 0.0 and end == 0.0))):
                out.append((None, None, text))
            else:
                out.append((start, end, text))
        # LRC start-only rows: extend to the next start (last row +8s) so the
        # span stays valid.
        fixed = []
        for i, (s, e, t) in enumerate(out):
            if s is not None and e is None:
                nxt = next((out[j][0] for j in range(i + 1, len(out))
                            if out[j][0] is not None), None)
                e = nxt if (nxt is not None and nxt > s) else s + 8.0
            fixed.append((s, e, t))
        return fixed

    def _fill_missing_times(self, rows):
        """Interpolate times for unreliable rows by line order.

        Aligners often fail on whole runs of singing (choruses, instrumental
        stretches). The invalid run is spread evenly between its valid
        anchors rather than stepped back from one anchor at a fixed size,
        which would bunch the unaligned lyrics near that anchor.
        """
        n = len(rows)
        valid_idx = [i for i, (s, _, _) in enumerate(rows) if s is not None]

        if not valid_idx:
            # No reliable time at all (aligner failed across an instrumental or
            # long vocal stretch). The lyrics themselves are still correct and
            # must not be dropped: spread them evenly over [0, 1) so they all
            # land in the first section and stay hand-editable.
            total = max(n, 1)
            return [(i / total, (i + 1) / total, t) for i, (_, _, t) in enumerate(rows)]

        # Fallback step: average duration of the valid rows, for the extreme
        # case where a run cannot be spread across anchors.
        fallback = sum(rows[i][1] - rows[i][0] for i in valid_idx) / len(valid_idx)
        fallback = max(float(fallback), 0.05)

        filled = list(rows)
        # Anchor each run of invalid rows on both sides, then spread it evenly.
        bounds = [-1] + valid_idx + [n]
        for left, right in zip(bounds, bounds[1:]):
            lo, hi = left + 1, right          # invalid run [lo, hi)
            if lo >= hi:
                continue
            count = hi - lo
            anchor_l = filled[left][1] if left >= 0 else 0.0
            anchor_r = filled[right][0] if right < n else None
            if anchor_r is None:              # tail: spread from the left anchor at the fallback step
                step = fallback
                base = anchor_l
            else:
                span = max(anchor_r - anchor_l, 0.0)
                step = span / count if span > 0 else fallback
                base = anchor_l

            for k in range(count):
                s = base + step * k
                filled[lo + k] = (s, s + step, filled[lo + k][2])
        return filled

    def parse_structure(self, text: str, duration_hint: float):
        """Compatibility shim: see module-level :func:`parse_structure_spans`."""
        return parse_structure_spans(text, duration_hint)

    def format(self, subtitles, structure, clean_punct, dedupe, min_line_chars,
               map_sections):
        """Port of ``LyricsFormatter.format`` -> ``(lyrics, report)``."""
        sub_rows = self.parse_subtitles(subtitles)
        if not sub_rows:
            raise ValueError(
                "subtitles is empty or unsupported; use LRC, SRT, or 'start-end: text' lines")

        # Filter and clean by text first, then fill times (avoids interpolating
        # rows that are about to be dropped).
        lines = []
        for start, end, text in sub_rows:
            cleaned = text.translate(self._PUNCT).strip() if clean_punct else text.strip()
            if not cleaned or (min_line_chars > 0 and len(cleaned) < min_line_chars):
                continue
            lines.append((start, end, cleaned))
        if dedupe:
            deduped = []
            for row in lines:
                if deduped and deduped[-1][2] == row[2]:
                    continue
                deduped.append(row)
            lines = deduped
        if not lines:
            raise ValueError("No lyric lines remain after cleanup; check min_line_chars")

        # Count failed alignments and interpolated times so the report is honest.
        inferred = sum(1 for s, _, _ in lines if s is None)
        lines = self._fill_missing_times(lines)
        timed = [(s, e, t) for s, e, t in lines if s is not None]
        duration = max((e for _, e, _ in timed), default=float(len(lines)))

        spans = self.parse_structure(structure or "", duration + 1.0)
        if not spans:
            # No structure: everything goes into one section.
            sections = [("verse", [t for _, _, t in lines])]
        else:
            sections = []
            for idx, (start, end, name) in enumerate(spans):
                label = _SECTION_ALIASES.get(name.lower(), name) if map_sections else name
                last = idx == len(spans) - 1
                bucket = []
                for s, e, t in lines:
                    if s is None:
                        continue
                    mid = (s + e) / 2
                    if start <= mid < end or (last and mid >= start):
                        bucket.append(t)
                if sections and sections[-1][0] == label:
                    sections[-1][1].extend(bucket)
                else:
                    sections.append((label, bucket))
            # Lyrics landing in an instrumental span (intro/interlude) mean the
            # boundary is off: move them into the next singing section and leave
            # the instrumental as an empty tag (YuE2 treats empty as instrumental).
            singing = [i for i, (name, _) in enumerate(sections)
                       if (name or "").lower() not in _NON_SINGING]
            if singing:
                def nearest_singing(i):
                    return min(singing, key=lambda j: (abs(j - i), j))
                for i, (name, bucket) in enumerate(sections):
                    if (name or "").lower() in _NON_SINGING and bucket:
                        sections[nearest_singing(i)][1].extend(bucket)
                        bucket.clear()

        out_lines: list[str] = []
        for name, bucket in sections:
            out_lines.append(f"[{(name or 'verse').strip().lower()}]")
            out_lines.extend(bucket)
            out_lines.append("")
        lyrics = "\n".join(out_lines).rstrip() + "\n"

        report = (f"lines={len(sub_rows)} kept={len(lines)} "
                  f"sections={len(sections)} chars={sum(len(t) for _, _, t in lines)}")
        if inferred:
            # When the aligner fails widely, section placement is inferred and
            # the user must be told.
            report += (f" | warning: {inferred}/{len(lines)} lines have unreliable "
                       "timestamps; section placement was inferred and should be reviewed")
        return lyrics, report


class LyricsStructurer:
    """Add ``[verse]``/``[chorus]`` section tags to plain lyrics for YuE2.

    Two modes (with ``structure`` empty, sections are planned by line count):
    - ``lyrics_text`` only: every ``verse_lines`` lines is one section, all ``[verse]``
    - ``section_plan``: comma-separated section sequence, e.g. ``verse,chorus,verse,chorus``
      (optionally with counts ``verse:8,chorus:8``), taking lines in order

    Manually tagged text (containing ``[xxx]`` lines) is preserved verbatim.
    """

    def format(self, lyrics_text, structure, verse_lines, section_plan):
        """Port of ``LyricsStructurer.format`` -> ``(lyrics, report)``."""
        # Text already carrying [tag] lines is preserved as the user wrote it.
        if any(_SECTION_TAG_LINE.fullmatch(line) for line in (lyrics_text or "").splitlines()):
            cleaned = "\n".join(
                ln for ln in (l.strip() for l in lyrics_text.splitlines())
                if ln and not _SECTION_TAG_LINE.fullmatch(ln))
            report = f"Existing section tags preserved ({len(cleaned.splitlines())} lyric lines)"
            return lyrics_text.rstrip() + "\n", report

        lines = [ln.strip() for ln in (lyrics_text or "").splitlines() if ln.strip()]
        if not lines:
            raise ValueError("lyrics_text must not be empty")

        plan = self._parse_plan(section_plan, verse_lines, len(lines))

        # With structure info, distribute lines by singing-span duration ratio.
        spans = parse_structure_spans(structure or "", 0.0) if structure else []
        if spans:
            singing = [(name, end - start) for start, end, name in spans
                       if _SECTION_ALIASES.get(name.lower(), name.lower()) not in _NON_SINGING
                       and (end - start) > 1.0]
            if singing:
                total = sum(d for _, d in singing)
                plan = [(name, max(1, round(len(lines) * dur / total)))
                        for name, dur in singing]
                plan[-1] = (plan[-1][0], len(lines) - sum(n for _, n in plan[:-1]))

        out: list[str] = []
        idx, sections = 0, 0
        for name, count in plan:
            if idx >= len(lines):
                break
            take = lines[idx:idx + count]
            idx += len(take)
            out.append(f"[{name}]")
            out.extend(take)
            out.append("")
            sections += 1
        if idx < len(lines):          # More lines than planned: the rest go into a final section.
            out.append(f"[{plan[-1][0]}]" if not out or out[-1] else f"[{plan[-1][0]}]")
            out.extend(lines[idx:])

        lyrics = "\n".join(out).rstrip() + "\n"
        report = (f"lines={len(lines)} sections={sections} "
                  f"plan={','.join(f'{n}:{c}' for n, c in plan)}")
        return lyrics, report

    @staticmethod
    def _parse_plan(section_plan, verse_lines, n_lines):
        """Parse a section plan into ``[(section_name, line_count)]``."""
        if not section_plan.strip():
            n = max(1, -(-n_lines // max(verse_lines, 1)))
            return [("verse", verse_lines)] * n
        plan = []
        for part in section_plan.split(","):
            part = part.strip().lower()
            if not part:
                continue
            if ":" in part:
                name, cnt = part.split(":", 1)
                plan.append((name.strip() or "verse", max(1, int(cnt))))
            else:
                plan.append((part, verse_lines))
        return plan or [("verse", verse_lines)]
