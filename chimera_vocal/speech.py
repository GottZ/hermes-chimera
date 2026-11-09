"""TTS path: text -> plan -> phrase windows -> RenderJobs.

The text is planned once; it is split only at boundaries with a pause >= 0.18 s, each window is rendered by a fresh
voice from the window's control block (clock exactly as in the continuous voice) with its own gestures, and the
pieces are overlap-added. Windows k >= 1 start with the pitch glide settled: a fresh voice otherwise glides up from
110 Hz at every phrase start. Measured against the template, the assembled result stays within the spread the
template shows against itself with another random phase (envelope, f0, voicing, third-octave spectrum)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Sequence

from .core.jsmath import js_round
from .core.types import PhraseInput, PlanOptions, Seeds

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from .music.compile import RenderJob
    from .presets import VoicePreset

SPEECH_T0 = 0.05          # render.rs:178
SPLIT_MIN_PAUSE = 0.18    # split only at boundaries with pause >= 0.18 s (english.rs:155-161)
WINDOW_PRE = 0.05         # W_k = 64·floor((t0_k − 0.05)·fs/64)
WINDOW_POST = 0.15        # E_k = ceil((end_k + 0.15)·fs)
TEXT_TAIL = 0.3           # last window ends at js_round(fs·(end + 0.3)) (render.rs:184)
FS = 48000


def plan_speech(text: str, preset: "VoicePreset", *, speed: float | None, g2p: Any,
                text_prep: bool = True, split: bool = True,
                settle_first: bool = True) -> "tuple[list[RenderJob], int]":
    """-> (jobs, n_total). ``split=False, settle_first=False`` reproduces ``chimera render text`` exactly."""
    from . import TextError
    from .music.compile import RenderJob
    from .music.planning import planner_api
    from .presets import speaking_f0

    api = planner_api()
    prepared = api.prepare_text(text) if text_prep else text
    phrases = g2p.to_phrases(prepared) if prepared and prepared.strip() else []
    phrases = [p for p in phrases if p.segments]
    if not phrases:
        raise TextError(f"chimera reads English text; nothing speakable in {text[:80]!r}")
    f0 = speaking_f0(preset)
    rate = speed if speed is not None else preset.speech.rate
    opts = PlanOptions(preset.body, SPEECH_T0, None if rate == 1.0 else float(rate), f0,
                       accent=preset.speech.accent)
    inputs = [PhraseInput(p.segments, p.intonation, p.pause) for p in phrases]
    whole = api.plan_phrases(inputs, opts)
    n_total = int(js_round(FS * (whole.end + TEXT_TAIL)))
    base = Seeds.for_voice(0, preset.seed)
    source = preset.source.patched(f0=f0)
    if not split or len(phrases) == 1:
        return [RenderJob(id="w0", body=preset.body, seeds=base, source=source, gestures=list(whole.gestures),
                          n_out=n_total, start_block=0, settle_f0=bool(settle_first))], n_total
    # plan_phrases chains phrase plans (t = end + pause, planner.rs:519-530): plan each phrase on that chain to know
    # which gestures belong to it
    per: list[tuple[float, float, list]] = []
    t = SPEECH_T0
    for p in inputs:
        o = PlanOptions(opts.body, t, opts.rate, opts.f0, opts.intonation, opts.accent, opts.open_end, opts.sung)
        plan = api.plan_phrases([p], o)
        per.append((t, plan.end, list(plan.gestures)))
        t = plan.end + (p.pause if p.pause is not None else 0.25)
    if [g for _, _, gs in per for g in gs] != list(whole.gestures):
        raise RuntimeError("per-phrase plans differ from the whole plan (planner chain assumption broken)")
    groups: list[list[int]] = [[0]]
    for k in range(1, len(per)):
        pause = inputs[k - 1].pause if inputs[k - 1].pause is not None else 0.25
        if pause >= SPLIT_MIN_PAUSE:
            groups.append([k])
        else:
            groups[-1].append(k)
    jobs = []
    for gi, grp in enumerate(groups):
        t0, end = per[grp[0]][0], per[grp[-1]][1]
        w = 64 * math.floor((t0 - WINDOW_PRE) * FS / 64)
        e = n_total if gi == len(groups) - 1 else math.ceil((end + WINDOW_POST) * FS)
        jobs.append(RenderJob(id=f"w{gi}", body=preset.body, seeds=Seeds.for_window(base, gi), source=source,
                              gestures=[g for k in grp for g in per[k][2]], n_out=e - w, start_block=w // 64,
                              settle_f0=gi > 0 or bool(settle_first)))
    return jobs, n_total


def assemble(pieces: "Sequence[np.ndarray]", jobs: "Sequence[RenderJob]", n_total: int) -> "np.ndarray":
    """Overlap-add at 64·job.start_block in float32, in job order; truncate/pad to n_total."""
    import numpy as np
    out = np.zeros(int(n_total), np.float32)
    for piece, job in zip(pieces, jobs):
        o = 64 * job.start_block
        k = max(0, min(len(piece), n_total - o))
        if k:
            out[o:o + k] = out[o:o + k] + np.asarray(piece[:k], dtype=np.float32)
    return out
