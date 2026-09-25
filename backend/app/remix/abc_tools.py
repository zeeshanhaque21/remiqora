"""Pure-text ABC score helpers.

Ported from ``nodes/abc_tools.py`` in ScryptHunter/ComfyUI-YuE2 at commit
7ed198838d112df7a2e4f4de7e903d1044632eee (itself a fork of
piscesbody/ComfyUI-YuE2), Apache-2.0; the original license is retained in
``LICENSE-ComfyUI-YuE2``.

The upstream functions/classes ported here are ``score_notes``,
``analyze_abc``, ``ABCAnalyzer.analyze``, ``ABCModifier.modify``,
``VocalRangeRetarget.retarget``, ``MelodyCleanup.clean``,
``YuE2StyleBuilder.build`` and ``LyricsMelodyFit.analyze``, together with the
private helpers they depend on (``_key_parts``, ``_key_accidentals``,
``_pitch``, ``_note_name``, ``_abc_note``, ``_transpose_key``,
``_transpose_chords``, ``_music_segments``).

Each tool keeps the node's ``INPUT_TYPES`` names and defaults as keyword
arguments and returns the node's outputs. ComfyUI node classes, the
``folder_paths``-backed ABC file loader/saver, and everything that touches the
filesystem or torch are intentionally left out: Remiqora only ever needs to
transform ABC text held in memory.
"""
from __future__ import annotations

import re

NOTE_RE = re.compile(r"(?<![A-Za-z])(?P<acc>\^{1,2}|_{1,2}|=)?(?P<letter>[A-Ga-g])(?P<oct>[,']*)(?P<dur>\d+(?:/\d+)?|/\d+|/)?(?P<tie>-)?")
PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_NAMES = ["C", "^C", "D", "^D", "E", "F", "^F", "G", "^G", "A", "^A", "B"]
FLAT_NAMES = ["C", "_D", "D", "_E", "E", "F", "_G", "G", "_A", "A", "_B", "B"]
KEY_PC = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,
          "G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}
FIFTHS_MAJOR = {"C":0,"G":1,"D":2,"A":3,"E":4,"B":5,"F#":6,"C#":7,
                "F":-1,"Bb":-2,"Eb":-3,"Ab":-4,"Db":-5,"Gb":-6,"Cb":-7}
FIFTHS_MINOR = {"A":0,"E":1,"B":2,"F#":3,"C#":4,"G#":5,"D#":6,"A#":7,
                "D":-1,"G":-2,"C":-3,"F":-4,"Bb":-5,"Eb":-6,"Ab":-7}


def _key_parts(value):
    m = re.match(r"\s*([A-Ga-g])([#b]?)(.*)", value or "C")
    if not m: return "C", "", False
    root = m.group(1).upper() + m.group(2)
    suffix = m.group(3)
    minor = suffix.strip().lower().startswith("m") and not suffix.strip().lower().startswith("mix")
    return root, suffix, minor


def _key_accidentals(value):
    root, _, minor = _key_parts(value)
    fifths = (FIFTHS_MINOR if minor else FIFTHS_MAJOR).get(root, 0)
    out = {x: 0 for x in "ABCDEFG"}
    for x in "FCGDAEB"[:max(fifths, 0)]: out[x] = 1
    for x in "BEADGCF"[:max(-fifths, 0)]: out[x] = -1
    return out


def _pitch(match, key_acc, state):
    letter, octs, acc = match.group("letter"), match.group("oct"), match.group("acc")
    octave = 5 if letter.islower() else 4
    octave += octs.count("'") - octs.count(",")
    key = (letter.upper(), octave)
    if acc:
        delta = {"=":0,"^":1,"^^":2,"_":-1,"__":-2}[acc]
        state[key] = delta
    else:
        delta = state.get(key, key_acc.get(letter.upper(), 0))
    return 12 * (octave + 1) + PC[letter.upper()] + delta


def _note_name(midi):
    return f"{['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][midi % 12]}{midi // 12 - 1}"


def _abc_note(midi, duration="", tie="", prefer_flats=False):
    names = FLAT_NAMES if prefer_flats else SHARP_NAMES
    name = names[midi % 12]
    octave = midi // 12 - 1
    accidental, letter = name[:-1], name[-1]
    if octave >= 5:
        letter = letter.lower(); marks = "'" * (octave - 5)
    else:
        marks = "," * (4 - octave)
    return accidental + letter + marks + (duration or "") + (tie or "")


def _transpose_key(value, semitones):
    root, suffix, _ = _key_parts(value)
    pc = (KEY_PC.get(root, 0) + semitones) % 12
    prefer_flats = "b" in root
    names = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"] if prefer_flats else \
            ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    return names[pc] + suffix


def _transpose_chords(text, semitones):
    def chord(m):
        body=m.group(1)
        return '"' + re.sub(r"(?<![A-Za-z])([A-G])([#b]?)(?=[:A-Za-z0-9/(]|$)",
            lambda n: _transpose_key(n.group(1)+n.group(2), semitones), body) + '"'
    return re.sub(r'"([^"]*)"', chord, text)


def _music_segments(line):
    """Yield quoted/nonquoted segments so chord text is never parsed as notes."""
    pos=0
    for m in re.finditer(r'"[^"]*"', line):
        yield False, line[pos:m.start()]; yield True, m.group(0); pos=m.end()
    yield False, line[pos:]


def score_notes(text):
    key="C"; voice="default"; states={}; result=[]; section="unsectioned"
    for raw in (text or "").splitlines():
        line=raw.strip()
        if line.startswith("K:"): key=line[2:].strip(); continue
        if line.startswith("V:"): voice=line[2:].strip().split()[0]; continue
        if line.startswith("%"): section=line[1:].strip().lower() or section; continue
        if re.match(r"^[A-Za-z]:", line): continue
        state=states.setdefault(voice,{})
        for quoted, seg in _music_segments(raw):
            if quoted: continue
            parts=re.split(r"(\|+)",seg)
            for part in parts:
                if part.startswith("|"): state.clear(); continue
                for m in NOTE_RE.finditer(part):
                    result.append((voice, section, _pitch(m,_key_accidentals(key),state)))
    return result


def analyze_abc(text):
    """Return ``(report, bpm, key, duration_seconds)`` for an ABC score."""
    if not (text or "").strip(): raise ValueError("ABC input is empty")
    def header(name, default=""):
        m=re.search(rf"(?m)^{name}:\s*(.+)$",text); return m.group(1).strip() if m else default
    meter=header("M","4/4"); key=header("K","C"); q=header("Q","")
    bpm_match=re.search(r"=\s*(\d+(?:\.\d+)?)",q)
    if not bpm_match: bpm_match=re.search(r"(\d+(?:\.\d+)?)\s*$",q)
    bpm=float(bpm_match.group(1)) if bpm_match else 120.
    try: num,den=map(int,meter.split("/",1))
    except Exception: num,den=4,4
    bars=sum(max(1,len(re.findall(r"\|",l))) for l in text.splitlines() if l and not re.match(r"^[A-Za-z%]:",l))
    # Voices repeat the same measure layout, so use the largest per-voice bar count.
    voice_bars={}; voice="default"
    for l in text.splitlines():
        if l.startswith("V:"): voice=l[2:].strip().split()[0]; continue
        if not re.match(r"^[A-Za-z%]:",l): voice_bars[voice]=voice_bars.get(voice,0)+l.count("|")
    bars=max(voice_bars.values(),default=bars)
    duration=bars*num*(4/den)*60/max(bpm,1e-6)
    notes=score_notes(text); by={}
    for voice,section,pitch in notes: by.setdefault(voice,[]).append(pitch)
    lines=[f"BPM: {bpm:g}",f"Key: {key}",f"Meter: {meter}",f"Measures: ~{bars}",f"Duration: ~{duration:.1f} s"]
    for voice,pitches in sorted(by.items()):
        pitches.sort(); median=pitches[len(pitches)//2]
        lines.append(f"{voice}: notes={len(pitches)} range={_note_name(pitches[0])}..{_note_name(pitches[-1])} median={_note_name(median)}")
        if "vocal" in voice.lower() and median < 57: lines.append("WARNING: Vocal center is low; consider an octave-up test.")
        if "vocal" in voice.lower() and median > 76: lines.append("WARNING: Vocal center is high; consider an octave-down test.")
    return "\n".join(lines), bpm, key, duration


def analyze(abc):
    """Port of ``ABCAnalyzer.analyze`` -> ``(report, bpm, key, duration_seconds)``."""
    if not (abc or "").strip():
        return "ABC input is empty (score planning is disabled).",0.0,"",0.0
    report,bpm,key,duration=analyze_abc(abc); return report,bpm,key,duration


def modify(abc, tempo_mode, bpm=120., tempo_multiplier=1., transpose_scope="whole_score",
           semitones=0, remove_chords=False, drop_sections="",
           vocal_transpose=0, instrumental_transpose=0, octave_shift=0):
    """Port of ``ABCModifier.modify`` -> ``(abc, report)``.

    Defaults mirror the node's ``INPUT_TYPES``. ``tempo_mode`` and
    ``transpose_scope`` are required by the node but given the node's first
    combo option as a default here so callers can omit them.
    """
    if not abc.strip(): raise ValueError("ABC input is empty")
    old_report,old_bpm,key,_=analyze_abc(abc); target=old_bpm
    if tempo_mode=="override": target=bpm
    elif tempo_mode=="multiplier": target=old_bpm*tempo_multiplier
    drop={x.strip().lower() for x in drop_sections.split(",") if x.strip()}
    lines=[]; voice="default"; section=""; skipping=False; original_key=key
    for raw in abc.splitlines():
        if raw.startswith("%"):
            section=raw[1:].strip().lower(); skipping=section in drop
            if skipping: continue
        if skipping: continue
        if raw.startswith("Q:") and tempo_mode!="keep": raw=f"Q:1/4={target:g}"
        if raw.startswith("V:"): voice=raw[2:].strip().split()[0]
        selected=(transpose_scope=="whole_score" or
                  transpose_scope=="vocal_only" and "vocal" in voice.lower() or
                  transpose_scope=="instrumental_only" and ("ins" in voice.lower() or "inst" in voice.lower()))
        offset=(semitones if selected else 0) + 12*octave_shift
        if "vocal" in voice.lower(): offset += vocal_transpose
        elif "ins" in voice.lower() or "inst" in voice.lower(): offset += instrumental_transpose
        if raw.startswith("K:") and transpose_scope=="whole_score" and semitones:
            raw="K:"+_transpose_key(raw[2:].strip(),semitones)
        elif not re.match(r"^[A-Za-z]:",raw) and not raw.startswith("%"):
            pieces=[]; state={}; keyacc=_key_accidentals(original_key)
            for quoted,seg in _music_segments(raw):
                if quoted:
                    pieces.append("" if remove_chords else (_transpose_chords(seg,semitones) if transpose_scope=="whole_score" and semitones else seg)); continue
                if offset:
                    out=[]; pos=0
                    for m in NOTE_RE.finditer(seg):
                        out.append(seg[pos:m.start()]); pitch=_pitch(m,keyacc,state)+offset
                        out.append(_abc_note(pitch,m.group("dur"),m.group("tie"),"b" in original_key)); pos=m.end()
                    out.append(seg[pos:]); seg="".join(out)
                pieces.append(seg)
            raw="".join(pieces)
        lines.append(raw)
    if tempo_mode!="keep" and not any(l.startswith("Q:") for l in lines):
        insert=next((i+1 for i,l in enumerate(lines) if l.startswith("L:")),3); lines.insert(insert,f"Q:1/4={target:g}")
    result="\n".join(lines).rstrip()+"\n"
    report=(f"bpm {old_bpm:g}->{target:g}; transpose={semitones} scope={transpose_scope}; "
            f"vocal={vocal_transpose:+d}; instrumental={instrumental_transpose:+d}; "
            f"octaves={octave_shift:+d}; removed_sections={sorted(drop)}")
    if (transpose_scope in {"vocal_only","instrumental_only"} and semitones%12) or vocal_transpose%12 or instrumental_transpose%12:
        report += " | warning: non-octave voice-only transposition changes its harmonic relationship"
    return result,report


RANGES={"female_contralto":(52,77),"female_alto":(53,77),"female_mezzo":(57,81),
        "female_soprano":(60,84),"male_bass":(40,64),"male_baritone":(45,69),"male_tenor":(48,72)}


def retarget(abc, target_voice="original", mode="nearest_key_safe", manual_semitones=0):
    """Port of ``VocalRangeRetarget.retarget`` -> ``(abc, applied_semitones, report)``."""
    pitches=[p for v,_s,p in score_notes(abc) if "vocal" in v.lower()]
    if not pitches: raise ValueError("No Vocal voice was found in the ABC score")
    pitches.sort(); median=pitches[len(pitches)//2]
    if target_voice=="original": shift=0
    elif mode=="manual": shift=int(manual_semitones)
    else:
        lo,hi=RANGES[target_voice]; center=(lo+hi)/2
        raw=center-median
        # Pitch-class-preserving octave shifts are the only genuinely
        # harmony-safe automatic operation on one voice.
        shift=int(round(raw/12))*12
    out,mod_report=modify(abc,"keep",120,1,"none",0,False,"",
                                         vocal_transpose=shift)
    report=(f"target={target_voice} source={_note_name(pitches[0])}..{_note_name(pitches[-1])} "
            f"median={_note_name(median)} applied={shift:+d} semitones | {mod_report}")
    return out,shift,report


def clean(abc, preset="custom", remove_notes_shorter_than_ms=40.,
          remove_pitch_outliers=True, outlier_semitones=24):
    """Port of ``MelodyCleanup.clean`` -> ``(abc, report)``."""
    if preset=="off": return abc,"cleanup disabled"
    _report,bpm,_key,_dur=analyze_abc(abc)
    if preset=="light": threshold,outliers=30.,False
    elif preset=="medium": threshold,outliers=50.,True
    else: threshold,outliers=remove_notes_shorter_than_ms,remove_pitch_outliers
    pitches=sorted(p for v,_s,p in score_notes(abc) if "vocal" in v.lower())
    median=pitches[len(pitches)//2] if pitches else 60
    base_ms=60000/max(bpm,1)/8 # default L:1/32 used by SheetSage2
    voice="default"; removed_short=removed_outlier=0; lines=[]
    for raw in abc.splitlines():
        if raw.startswith("V:"): voice=raw[2:].strip().split()[0]
        if "vocal" not in voice.lower() or re.match(r"^[A-Za-z]:",raw): lines.append(raw); continue
        state={}; keyacc=_key_accidentals(re.search(r"(?m)^K:\s*(.+)$",abc).group(1) if re.search(r"(?m)^K:\s*(.+)$",abc) else "C")
        def repl(m):
            nonlocal removed_short,removed_outlier
            d=m.group("dur") or "1"
            if "/" in d:
                a,b=(d.split("/",1)+["2"])[:2]; units=(float(a) if a else 1)/(float(b) if b else 2)
            else: units=float(d)
            pitch=_pitch(m,keyacc,state)
            short=units*base_ms < threshold
            extreme=outliers and abs(pitch-median)>outlier_semitones
            if short or extreme:
                removed_short+=int(short); removed_outlier+=int(extreme and not short)
                return "z"+(m.group("dur") or "")
            return m.group(0)
        pieces=[]
        for quoted,seg in _music_segments(raw): pieces.append(seg if quoted else NOTE_RE.sub(repl,seg))
        lines.append("".join(pieces))
    return "\n".join(lines).rstrip()+"\n",f"preset={preset} short_notes_removed={removed_short} pitch_outliers_removed={removed_outlier}"


def build_style(language="English", genre="industrial breakbeat",
                secondary_genres="big beat, rave punk", era="1990s",
                vocal="female, raspy, aggressive, shouted",
                instruments="distorted bass, abrasive synths",
                drums="hard syncopated breakbeats",
                mood="raw, dark, hostile", tempo="fast", extra=""):
    """Port of ``YuE2StyleBuilder.build`` -> the style prompt string.

    The node took ``**kw`` and iterated ``kw.values()`` in insertion order;
    keyword order here matches the node's declared inputs.
    """
    values=(language,genre,secondary_genres,era,vocal,instruments,drums,mood,tempo,extra)
    parts=[]
    for value in values:
        for x in str(value).split(","):
            x=x.strip()
            if x and x.lower() not in {p.lower() for p in parts}: parts.append(x)
    return ", ".join(parts)


def lyric_melody_fit(abc, lyrics, language="English"):
    """Port of ``LyricsMelodyFit.analyze`` -> the fit report string."""
    lyric_sections={}; sec="unsectioned"
    for line in lyrics.splitlines():
        m=re.match(r"\s*\[([^]]+)\]\s*$",line)
        if m: sec=m.group(1).lower(); lyric_sections.setdefault(sec,[])
        elif line.strip(): lyric_sections.setdefault(sec,[]).append(line.strip())
    note_sections={}
    for voice,section,p in score_notes(abc):
        if "vocal" in voice.lower(): note_sections[section]=note_sections.get(section,0)+1
    def syllables(s):
        if language!="English": return len(re.findall(r"\w+",s))
        total=0
        for word in re.findall(r"[A-Za-z]+",s.lower()):
            n=len(re.findall(r"[aeiouy]+",word)); n-=int(word.endswith("e") and n>1); total+=max(1,n)
        return total
    out=["Heuristic only: melisma and sustained notes can make a good fit differ from 1:1."]
    keys=list(dict.fromkeys([*lyric_sections,*note_sections]))
    for s in keys:
        sy=syllables(" ".join(lyric_sections.get(s,[]))); no=note_sections.get(s,0); ratio=sy/max(no,1)
        verdict="good" if .65<=ratio<=1.5 else "lyrics may be dense" if ratio>1.5 else "melody may require melisma/sustains"
        out.append(f"[{s}] vocal_notes={no} estimated_syllables={sy} ratio={ratio:.2f} | {verdict}")
    return "\n".join(out)
