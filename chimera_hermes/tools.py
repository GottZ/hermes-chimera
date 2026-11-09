"""Tools of the toolset ``chimera``: schemas and handlers.

Handlers follow Hermes' contract ``handler(args: dict, **kwargs) -> str``: always a JSON string, never an exception.
Renders run as jobs (``jobs.py``); a call waits at most ``wait_budget_s`` and otherwise returns ``status: running``
with a ``job_id`` for ``chimera_job``. Hermes attaches MEDIA lines from tool results automatically only for its own
speech and image tools, so every result carries the ``media`` line and tells the model to put it into its reply.
Schemas avoid ``oneOf``/``$ref``: several model providers reject them.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
from typing import Any, Callable

from . import _vocal, media, paths, sessions, settings

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------ schemas

_NOTE_ITEM = {"type": "object", "description": "One item: a note {pitch, beats, lyric?, dyn?}, a rest {rest} or spoken words {say, beats, rate?, hz?}",
  "properties": {
    "pitch": {"type": "string", "description": "Note name like C4, F#3, Bb4 (C4 = middle C) or a MIDI number as text like \"60\""},
    "beats": {"type": "number", "description": "Length in beats (note or spoken slot)"},
    "lyric": {"type": "string", "description": "Syllable sung on this note; 'twin-' continues the word on the next note; '-' holds the previous vowel; end with , or . to breathe"},
    "dyn": {"type": "string", "enum": ["pp", "p", "mp", "mf", "f", "ff"]},
    "rest": {"type": "number", "description": "Rest length in beats"},
    "say": {"type": "string", "description": "English words spoken (not sung) inside a slot of `beats`"},
    "rate": {"type": "number", "description": "Speaking rate for `say`, 0.5-1.8"},
    "hz": {"type": "number", "description": "Speaking pitch for `say` in Hz"}}}
_VOICE = {"type": "object", "properties": {
    "name": {"type": "string"},
    "body": {"type": "string", "description": "male | female | child | blend like 'female=0.5,child=0.2'"},
    "start": {"type": "number", "description": "Beat where this voice enters (canon entries)"},
    "volume": {"type": "number", "description": "Mix gain in dB (-40..12)"},
    "pan": {"type": "number", "description": "-1 left .. 1 right"},
    "breath": {"type": "number"}, "tension": {"type": "number"}, "vibrato": {"type": "number"},
    "vibrato_rate": {"type": "number"}, "breathiness": {"type": "number"}, "brightness": {"type": "number"},
    "ring": {"type": "number"}, "roughness": {"type": "number"}, "seed": {"type": "integer"},
    "lyrics": {"type": "string", "description": "Alternative to per-note lyric: whole lyric text, one syllable per sounding note"},
    "notes": {"type": "array", "items": _NOTE_ITEM}},
  "required": ["notes"]}
_SCORE = {"type": "object", "description": "Score — format, ranges and examples: skill chimera:music",
  "properties": {"title": {"type": "string"}, "bpm": {"type": "number"}, "lead_in": {"type": "number"},
                 "normalize": {"type": "boolean"}, "stereo": {"type": "string", "enum": ["auto", "true", "false"]},
                 "format": {"type": "string", "enum": ["auto", "wav", "mp3"]},
                 "voices": {"type": "array", "items": _VOICE}},
  "required": ["bpm", "voices"]}
_MEDIA_RULE = ("When it succeeds, copy the returned `media` value (a line starting with MEDIA:) exactly, on its own line, "
               "into your reply — outside any code block. That line is how the audio reaches the user.")
CHIMERA_RENDER = {"name": "chimera_render",
  "description": ("Sing a score with physically modelled voices (1-8 voices: choir, canon, duet, chant, sung or spoken lines on "
                  "a shared beat grid) and return an audio file. Load skill chimera:music first if you have not. " + _MEDIA_RULE +
                  " If `status` is 'running', call chimera_job with `job_id`."),
  "parameters": {"type": "object", "properties": {
      "score": _SCORE,
      "filename": {"type": "string", "description": "Optional short file name (letters, digits, hyphens)"},
      "wait_s": {"type": "number", "description": "Optional: max seconds to wait before returning a job id (0 = return at once)"}},
    "required": ["score"]}}
CHIMERA_CHECK = {"name": "chimera_check",
  "description": "Validate a chimera score without rendering: errors with fixes, syllable-per-note alignment, length and estimated render time.",
  "parameters": {"type": "object", "properties": {"score": _SCORE}, "required": ["score"]}}
MAX_TEXT_CHARS = 4000            # = Hermes' default text per TTS call; about 4 minutes of speech
CHIMERA_SPEAK = {"name": "chimera_speak",
  "description": ("Speak English text in a specific chimera voice (male, female, child, androgynous, teen, male-deep, female-bright, "
                  "narrator, whisper, male-whisper, robot, elder, or a spec like 'female;pitch=230') and return an audio file. "
                  f"At most {MAX_TEXT_CHARS} characters per call; split longer text. "
                  "For your normal spoken replies use text_to_speech. " + _MEDIA_RULE),
  "parameters": {"type": "object", "properties": {
      "text": {"type": "string"}, "voice": {"type": "string"}, "speed": {"type": "number", "description": "0.5-1.8"},
      "faithful": {"type": "boolean", "description": "true: render like the original chimera's command line `chimera render text` (no number/abbreviation expansion, one piece, pitch glides in at the start); the original web app uses other default pitches"},
      "wait_s": {"type": "number", "description": "Optional: max seconds to wait before returning a job id (0 = return at once)"}},
    "required": ["text"]}}
CHIMERA_JOB = {"name": "chimera_job",
  "description": ("Wait for a chimera render that returned status 'running' and get its media line, or stop it with "
                  "cancel: true. " + _MEDIA_RULE),
  "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}, "wait_s": {"type": "number"},
                                                  "cancel": {"type": "boolean", "description": "true: stop this render"}},
                 "required": ["job_id"]}}

SCHEMAS: dict[str, dict] = {s["name"]: s for s in (CHIMERA_RENDER, CHIMERA_CHECK, CHIMERA_SPEAK, CHIMERA_JOB)}

INSTRUCTION_DONE = "Put the media line on its own line in your reply, exactly as given, outside code blocks."
INSTRUCTION_RUNNING = "Call chimera_job with this job_id to wait for the audio."
MAX_SCORE_JSON_CHARS = 2_000_000

_numpy_cache: bool | None = None


def check_available() -> bool:
    """check_fn of every chimera tool: numpy importable in this process (never imports it)."""
    global _numpy_cache
    if _numpy_cache is None:
        _numpy_cache = _vocal.numpy_ok()
    return _numpy_cache


# ------------------------------------------------------------------------------------------------ argument checks

class ArgError(Exception):
    """A tool argument the model has to fix; the message says what to send instead."""


def _args(args: Any) -> dict:
    if not isinstance(args, dict):
        raise ArgError("arguments must be a JSON object")
    return args


def _score(a: dict) -> dict:
    if "score" not in a or a["score"] is None:
        raise ArgError("`score` is required (an object with bpm and voices; see skill chimera:music)")
    score = a["score"]
    if isinstance(score, str):
        if len(score) > MAX_SCORE_JSON_CHARS:
            raise ArgError("`score` is too large")
        try:
            score = json.loads(score)
        except (ValueError, RecursionError) as e:
            raise ArgError(f"`score` is a string but not valid JSON ({e}); send the score object itself") from None
    if not isinstance(score, dict):
        raise ArgError("`score` must be an object with bpm and voices")
    return score


def _opt_str(a: dict, key: str, *, max_len: int = 200) -> str | None:
    v = a.get(key)
    if v is None:
        return None
    if not isinstance(v, str):
        raise ArgError(f"`{key}` must be a string")
    if len(v) > max_len:
        raise ArgError(f"`{key}` is longer than {max_len} characters")
    return v


def _opt_number(a: dict, key: str, *, lo: float = 0.0, positive: bool = False) -> float | None:
    v = a.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise ArgError(f"`{key}` must be a finite number")
    v = float(v)
    if v < lo or (positive and v <= 0):
        raise ArgError(f"`{key}` must be {'greater than' if positive else 'at least'} {lo:g}")
    return v


def _owner(kwargs: dict) -> frozenset | None:
    """Who may wait for a job: the conversation (``conversation_id``, passed by newer Hermes) and the session
    (``session_id``, passed by every supported Hermes). A later call matches when it shares one of them or, after a
    context compression, when its session continues the job's session (``sessions.py``)."""
    return sessions.owner_ids(kwargs)


def _opt_bool(a: dict, key: str) -> bool | None:
    v = a.get(key)
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    raise ArgError(f"`{key}` must be true or false")


def _ok(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _fail(error: str, **extra: Any) -> str:
    return json.dumps({"success": False, "error": error, **extra}, ensure_ascii=False, default=str)


def _issues(issues: Any) -> list[dict]:
    out = []
    for i in issues or []:
        if dataclasses.is_dataclass(i):
            out.append(dataclasses.asdict(i))
        elif isinstance(i, dict):
            out.append(dict(i))
        else:
            out.append({"path": "", "severity": "error", "message": str(i)})
    return out


def _guard(name: str, fn: Callable[..., str]) -> Callable[..., str]:
    def handler(args: Any = None, **kwargs: Any) -> str:
        try:
            if not check_available():
                return _fail(_vocal.NUMPY_MISSING)
            return fn(_args(args), **kwargs)
        except ArgError as e:
            return _fail(str(e))
        except Exception as e:  # noqa: BLE001 - tool handlers never raise
            logger.warning("chimera: %s failed: %s", name, e, exc_info=True)
            if _vocal.is_error(e, "ScoreError"):
                issues = _issues(getattr(e, "errors", []))
                return _fail(f"score has {len(issues) or 1} error(s)", issues=issues)
            return _fail(f"{name} failed: {type(e).__name__}: {e}")
    handler.__name__ = name
    handler.__qualname__ = name
    return handler


# ------------------------------------------------------------------------------------------------ job plumbing

def _want_mp3(fmt: Any) -> tuple[bool, str | None]:
    """(want mp3, warning) from score "format" > plugin setting music_format (auto = mp3 iff ffmpeg exists)."""
    choice = fmt if fmt in ("auto", "wav", "mp3") else settings.setting(_vocal.ctx(), "music_format", "auto")
    if choice not in ("auto", "wav", "mp3"):
        choice = "auto"
    if choice == "wav":
        return False, None
    have = media.ffmpeg_path() is not None
    if choice == "mp3" and not have:
        return False, "MP3 was requested but ffmpeg is not installed; delivering WAV"
    return have, None


def _finish(wav: str, seconds: float, want_mp3: bool, warnings: list[str]) -> tuple[str, str]:
    paths.private_file(wav)
    if want_mp3:
        mp3 = wav[:-4] + ".mp3" if wav.lower().endswith(".wav") else wav + ".mp3"
        if media.to_mp3(wav, mp3, seconds=seconds):
            paths.private_file(mp3)
            return mp3, "mp3"
        warnings.append("ffmpeg could not make an MP3; delivering WAV")
    return wav, "wav"


def _remove_empty(path: Any) -> None:
    """Drop the reserved (still empty) output file of a render that failed."""
    try:
        if os.path.getsize(path) == 0:
            os.remove(path)
    except OSError:
        pass


def _keep_days() -> float:
    v = settings.setting(_vocal.ctx(), "keep_days", 7)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 7.0


def _cleanup(eng: Any) -> None:
    """Housekeeping before a render. The numba cache folder the engine uses is always kept (numba does not touch it
    on a cache hit, so its age says nothing); without that folder's name the numba cache is left alone."""
    keep = None
    try:
        d = eng.numba_cache_dir()
        if d:
            keep = {os.path.basename(os.path.normpath(d))}
    except Exception:  # noqa: BLE001 - an engine without the method (or a failure) keeps the whole cache
        keep = None
    paths.cleanup(_keep_days(), keep_cache=keep)


def _tmp_dir() -> str | None:
    try:
        return str(paths.tmp_dir(create=True))
    except Exception:  # noqa: BLE001 - the engine then uses the output folder
        return None


def _respond(job_id: str, owner: Any, wait_s: float | None, related: Any = None) -> str:
    from .jobs import wait_budget_s

    reg = _vocal.jobs()
    st = reg.wait(job_id, owner, wait_budget_s(wait_s), related=related)
    if st.status == "done" and st.result is not None:
        return _ok({**st.result, "job_id": st.id})
    if st.status in ("failed", "cancelled"):
        return _fail(st.error or st.status, job_id=st.id, status=st.status)
    return _ok({"success": True, "status": "running", "job_id": st.id, "progress": round(st.progress, 3),
                "eta_s": round(reg.eta_s(st.id)), "instruction": INSTRUCTION_RUNNING})


def _deadline_warning(est: dict, kernel: str) -> str | None:
    """Warn when the estimated render time reaches half of Hermes' tool time limit (the call will then usually end
    with a job id instead of the audio)."""
    d = settings.tool_deadline_s()
    try:
        e = float(est.get(kernel, 0.0))
    except (TypeError, ValueError, AttributeError):
        return None
    if d is None or not math.isfinite(e) or e < 0.5 * d:
        return None
    return (f"estimated render time {e:.0f} s ({kernel} backend) is at least half of Hermes' tool time limit "
            f"({d:.0f} s): expect status 'running' and wait with chimera_job"
            + ("; installing the companion plugin chimera-numba makes rendering about 19x faster"
               if kernel == "numpy" else ""))


def _speak_eta(eng: Any, text: str) -> float:
    """Expected seconds for speaking ``text``: about 15 characters per second of speech, times the engine's measured
    real-time factor (EMA) of the kernel in use, spread over its workers (phrase windows render in parallel)."""
    kernel = _kernel_now(eng)
    try:
        rtf = float(eng.status().rtf_estimate[kernel])
        par = max(1, int(getattr(eng, "workers", 1)))
    except Exception:
        rtf, par = (0.45 if kernel == "numba" else 10.0), 1
    return len(text) / 15.0 * rtf / par


def _kernel_now(eng: Any) -> str:
    try:
        return "numba" if eng.status().numba_state == "ready" else "numpy"
    except Exception:
        return "numpy"


# ------------------------------------------------------------------------------------------------ handlers

def _render(a: dict, **kwargs: Any) -> str:
    score = _score(a)
    filename = _opt_str(a, "filename")
    wait_s = _opt_number(a, "wait_s")
    eng, cfg = _vocal.engine_and_config()
    opts = _vocal.score_options(cfg)
    report = eng.check_score(score, max_voices=opts["max_voices"])
    if not report.ok:
        issues = _issues(report.issues)
        n = sum(1 for i in issues if i.get("severity") == "error") or len(issues)
        return _fail(f"score has {n} error(s)", issues=issues)
    kernel = _kernel_now(eng)
    eta = float((report.estimate_s or {}).get(kernel, 0.0))
    raw_name = filename if filename else score.get("title") if isinstance(score.get("title"), str) else None
    want_mp3, fmt_warning = _want_mp3(score.get("format"))
    time_warning = _deadline_warning(dict(report.estimate_s or {}), kernel)

    def job(progress, cancel) -> dict:
        _cleanup(eng)
        out_dir, stem = paths.new_output_stem(raw_name, "score")
        try:
            res = eng.render_score(score, str(out_dir), stem, want_mp3=False, progress=progress, cancel=cancel,
                                   tmp_dir=_tmp_dir(), **opts)
        except BaseException:
            _remove_empty(out_dir / f"{stem}.wav")
            raise
        warnings = [w for w in [fmt_warning, time_warning, *list(res.warnings or [])] if w]
        file, fmt = _finish(res.path_wav, res.seconds, want_mp3, warnings)
        return {"success": True, "status": "done", "media": media.media_line(file), "file": file,
                "wav": res.path_wav, "format": fmt, "seconds": round(float(res.seconds), 3),
                "sample_rate": res.sample_rate, "channels": res.channels, "voices": res.voices,
                "backend": res.kernel, "render_s": round(float(res.render_s), 3), "warnings": warnings,
                "instruction": INSTRUCTION_DONE}

    st = _vocal.jobs().start("render", _owner(kwargs), job, eta_s=eta)
    return _respond(st.id, _owner(kwargs), wait_s)


def _check(a: dict, **kwargs: Any) -> str:
    from .jobs import wait_budget_s

    score = _score(a)
    eng, cfg = _vocal.engine_and_config()
    report = eng.check_score(score, max_voices=_vocal.score_options(cfg)["max_voices"])
    kernel = _kernel_now(eng)
    est = dict(report.estimate_s or {})
    budget = wait_budget_s(None)
    warnings = [w for w in [_deadline_warning(est, kernel)] if w]
    return _ok({"success": True, "ok": bool(report.ok), "issues": _issues(report.issues), "warnings": warnings,
                "voices": [dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v for v in report.voices],
                "seconds": report.seconds, "voice_seconds": report.voice_seconds, "estimate_s": est,
                "backend": kernel, "fits_one_call": float(est.get(kernel, 0.0)) <= budget,
                "lead_in": report.lead_in, "memory_mb": report.memory_mb, "disk_mb": report.disk_mb})


def _speak(a: dict, **kwargs: Any) -> str:
    text = a.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ArgError("`text` must be non-empty English text")
    if len(text) > MAX_TEXT_CHARS:
        raise ArgError(f"`text` has {len(text)} characters; chimera_speak takes at most {MAX_TEXT_CHARS} (about four "
                       "minutes of speech) per call - split the text and call chimera_speak once per part")
    voice = _opt_str(a, "voice")
    speed = _opt_number(a, "speed", positive=True)
    wait_s = _opt_number(a, "wait_s")
    faithful = _opt_bool(a, "faithful")
    if faithful is None:
        faithful = settings.faithful(_vocal.ctx())
    presets = _vocal.api().presets
    if voice is not None and voice.strip():
        spec = voice.strip()
        preset = presets.parse_voice_spec(spec)
        if preset is None:
            raise ArgError(f"unknown voice {voice!r}; use one of {', '.join(presets.PRESETS)} or a spec like "
                           "'female;pitch=230'")
    else:
        tts_voice = settings.core_tts().get("voice")
        spec, preset = settings.resolve_voice_spec(_vocal.ctx(), tts_voice if isinstance(tts_voice, str) else None)
    spd = settings.resolve_speed(speed, preset)
    eng, cfg = _vocal.engine_and_config()
    opts = _vocal.speech_options(cfg)
    want_mp3, fmt_warning = _want_mp3(None)
    eta = _speak_eta(eng, text)

    def job(progress, cancel) -> dict:
        _cleanup(eng)
        out_dir, stem = paths.new_output_stem(text[:60], "speech")
        wav = str(out_dir / f"{stem}.wav")
        try:
            res = eng.speak_to_wav(text, wav, spec, speed=spd, faithful=faithful, tmp_dir=_tmp_dir(),
                                   cancel=cancel, **opts)
        except Exception as e:
            _remove_empty(wav)
            if _vocal.is_error(e, "TextError"):
                raise RuntimeError(str(e) or "chimera reads English text; nothing speakable was left after "
                                   "cleanup") from e
            raise
        seconds = float(res.seconds)
        warnings = [w for w in [fmt_warning] if w]
        file, fmt = _finish(wav, seconds, want_mp3, warnings)
        return {"success": True, "status": "done", "media": media.media_line(file), "file": file, "wav": wav,
                "format": fmt, "seconds": round(seconds, 3), "sample_rate": res.sample_rate, "channels": 1,
                "voices": 1, "voice": getattr(preset, "id", str(preset)), "backend": res.kernel,
                "faithful": bool(faithful), "warnings": warnings, "instruction": INSTRUCTION_DONE}

    st = _vocal.jobs().start("speak", _owner(kwargs), job, eta_s=eta)
    return _respond(st.id, _owner(kwargs), wait_s)


def _job(a: dict, **kwargs: Any) -> str:
    job_id = a.get("job_id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise ArgError("`job_id` must be the job id string a chimera tool returned")
    wait_s = _opt_number(a, "wait_s")
    cancel = _opt_bool(a, "cancel")
    related = sessions.related_lookup(kwargs)
    try:
        if cancel:
            _vocal.jobs().cancel(job_id.strip(), _owner(kwargs), related=related)
            return _respond(job_id.strip(), _owner(kwargs), 10.0 if wait_s is None else wait_s, related)
        return _respond(job_id.strip(), _owner(kwargs), wait_s, related)
    except KeyError:
        return _fail(f"unknown job {job_id!r}: jobs live in this Hermes process for 2 hours and belong to the "
                     "conversation that started them")


HANDLERS: dict[str, Callable[..., str]] = {
    "chimera_render": _guard("chimera_render", _render),
    "chimera_check": _guard("chimera_check", _check),
    "chimera_speak": _guard("chimera_speak", _speak),
    "chimera_job": _guard("chimera_job", _job),
}
