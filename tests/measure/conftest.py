"""C4 measurement fixtures: the render pipeline (validate -> compile -> render_job -> mix) with FakeVoice ("fake",
always) or the real engine ("real", once C1–C3 are merged, S3). Fixtures of tests/support/fakes.py re-exported."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from support.fakes import (fake_voice, fake_workers, make_engine, make_scheduler, real_engine_available,  # noqa: F401
                           stub_planner)


@dataclass
class Pipeline:
    kind: str                 # "fake" | "real"
    g2p: Any
    planner: Any              # StubPlanner or None (real C1)

    def compile(self, score: dict):
        from chimera_vocal.music.compile import compile_score
        from chimera_vocal.music.schema import validate_score
        report = validate_score(score, g2p=self.g2p)
        assert report.ok, [i for i in report.issues if i.severity == "error"]
        jobs, mixes, n_mix = compile_score(report, g2p=self.g2p)
        return report, jobs, mixes, n_mix

    def render(self, job) -> np.ndarray:
        """One job in-process with the numpy kernel (FakeVoice or the real HumanVoice defaults)."""
        from chimera_vocal.music.compile import render_job
        return render_job(job, None, None)

    def tracks(self, score: dict):
        report, jobs, mixes, n_mix = self.compile(score)
        return report, jobs, mixes, n_mix, [self.render(j) for j in jobs]

    def placed(self, track: np.ndarray, vm, n_mix: int) -> np.ndarray:
        """One voice alone on the mix timeline (through mix(), no normalisation)."""
        from chimera_vocal.music.mixer import mix
        from chimera_vocal.music.compile import VoiceMix
        out, _ = mix([track], [VoiceMix(vm.offset, 0.0, 0.0, vm.gain_env)], n_mix, stereo=False, normalize=False)
        return out


@pytest.fixture(params=["fake", "real"])
def pipeline(request):
    if request.param == "fake":
        planner = request.getfixturevalue("stub_planner")
        request.getfixturevalue("fake_voice")
        return Pipeline("fake", planner.g2p, planner)
    if not real_engine_available():
        pytest.skip("real engine (C1–C3) not merged yet - runs in S3")
    from chimera_vocal.music.compile import set_voice_factory
    from chimera_vocal.music.planning import planner_api, set_planner_api
    set_planner_api(None)
    set_voice_factory(None)
    return Pipeline("real", planner_api().default_g2p(), None)
