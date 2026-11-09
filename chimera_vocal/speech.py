"""TTS path: text -> plan -> phrase windows -> RenderJobs (INTERFACES §9.2, DESIGN §3.3 = E1-A rule)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from .music.compile import RenderJob
    from .presets import VoicePreset

SPEECH_T0 = 0.05          # render.rs:178
SPLIT_MIN_PAUSE = 0.18    # split only at boundaries with pause >= 0.18 s (english.rs:155-161)
WINDOW_PRE = 0.05         # W_k = 64·floor((t0_k − 0.05)·fs/64)
WINDOW_POST = 0.15        # E_k = ceil((end_k + 0.15)·fs)
TEXT_TAIL = 0.3           # last window ends at js_round(fs·(end + 0.3)) (render.rs:184)


def plan_speech(text: str, preset: "VoicePreset", *, speed: float | None, g2p: Any,
                text_prep: bool = True, split: bool = True,
                settle_first: bool = True) -> "tuple[list[RenderJob], int]":
    """-> (jobs, n_total). ``split=False, settle_first=False`` reproduces ``chimera render text`` exactly."""
    raise NotImplementedError


def assemble(pieces: "Sequence[np.ndarray]", jobs: "Sequence[RenderJob]", n_total: int) -> "np.ndarray":
    """Overlap-add at 64·job.start_block in float32, in job order; truncate/pad to n_total."""
    raise NotImplementedError
