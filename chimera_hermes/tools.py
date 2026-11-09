"""Tools of the toolset ``chimera`` (INTERFACES §11): schemas and handlers.

Handlers follow the Hermes contract ``handler(args: dict, **kwargs) -> str``: always a JSON string, never raise
(website/docs/developer-guide/plugins/index.md:886-899). Renders run as jobs (``jobs.py``); a call waits at most
``wait_budget_s`` and otherwise returns ``status: running`` + ``job_id`` for ``chimera_job``.

S1: schemas complete (interface data, INTERFACES §11.1); handlers are signatures only.
"""

from __future__ import annotations

from typing import Any, Callable

# ------------------------------------------------------------------------------------------------ schemas (§11.1)

_NOTE_ITEM = {"type": "object", "description": "One item: a note {pitch, beats, lyric?, dyn?}, a rest {rest} or spoken words {say, beats, rate?, hz?}",
  "properties": {
    "pitch": {"type": "string", "description": "Note name like C4, F#3, Bb4 (C4 = middle C) or a MIDI number as text"},
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
      "wait_s": {"type": "number", "description": "Optional: max seconds to wait before returning a job id"}},
    "required": ["score"]}}
CHIMERA_CHECK = {"name": "chimera_check",
  "description": "Validate a chimera score without rendering: errors with fixes, syllable-per-note alignment, length and estimated render time.",
  "parameters": {"type": "object", "properties": {"score": _SCORE}, "required": ["score"]}}
CHIMERA_SPEAK = {"name": "chimera_speak",
  "description": ("Speak English text in a specific chimera voice (male, female, child, androgynous, teen, male-deep, female-bright, "
                  "narrator, whisper, male-whisper, robot, elder, or a spec like 'female;pitch=230') and return an audio file. "
                  "For your normal spoken replies use text_to_speech. " + _MEDIA_RULE),
  "parameters": {"type": "object", "properties": {
      "text": {"type": "string"}, "voice": {"type": "string"}, "speed": {"type": "number", "description": "0.5-1.8"},
      "wait_s": {"type": "number"}}, "required": ["text"]}}
CHIMERA_JOB = {"name": "chimera_job",
  "description": "Wait for a chimera render that returned status 'running' and get its media line. " + _MEDIA_RULE,
  "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}, "wait_s": {"type": "number"}},
                 "required": ["job_id"]}}

SCHEMAS: dict[str, dict] = {s["name"]: s for s in (CHIMERA_RENDER, CHIMERA_CHECK, CHIMERA_SPEAK, CHIMERA_JOB)}

INSTRUCTION_DONE = "Put the media line on its own line in your reply, exactly as given, outside code blocks."
INSTRUCTION_RUNNING = "Call chimera_job with this job_id to wait for the audio."


def check_available() -> bool:
    """check_fn of every chimera tool: numpy importable in this process (never imports it)."""
    raise NotImplementedError


def chimera_render(args: Any = None, **kwargs: Any) -> str:
    """Validate (engine.check_score), start a render job, wait wait_budget_s(wait_s); §11.2 done/running/failure."""
    raise NotImplementedError


def chimera_check(args: Any = None, **kwargs: Any) -> str:
    """Validation report without rendering (§11.2 chimera_check)."""
    raise NotImplementedError


def chimera_speak(args: Any = None, **kwargs: Any) -> str:
    """Speak text in an explicit chimera voice into an audio file (job, like chimera_render)."""
    raise NotImplementedError


def chimera_job(args: Any = None, **kwargs: Any) -> str:
    """Wait for a running job of the same conversation; unknown id or other conversation -> "unknown job"."""
    raise NotImplementedError


HANDLERS: dict[str, Callable[..., str]] = {
    "chimera_render": chimera_render,
    "chimera_check": chimera_check,
    "chimera_speak": chimera_speak,
    "chimera_job": chimera_job,
}
