"""Measurement fixtures: the render pipeline (validate -> compile -> render_job -> mix) with FakeVoice ("fake",
always) or the real engine ("real", when the text, plan, control and dsp packages are importable; marked ``slow``:
each case renders seconds of audio in-process on the numpy kernel, run them with ``--run-slow``). Fixtures of
tests/support/fakes.py re-exported."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from support.fakes import (fake_voice, fake_workers, make_engine, make_scheduler, real_engine_available,  # noqa: F401
                           stub_planner)


F0_GLIDE_REL = 1e-12   # glide f0 of a voice that ran from block 0 vs a settled one: within ulps (4.8e-15 measured)


@dataclass
class Pipeline:
    kind: str                 # "fake" | "real"
    g2p: Any
    planner: Any              # StubPlanner or None (real planner)

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

    def control_rows(self, job, n_blocks: int, skip: int = 0):
        """The control track the production path gives ``job`` (real engine): the voice built by
        ``compile.voice_factory()`` (``_human_voice`` -> ``HumanVoice(start_block=job.start_block)``) and set up by
        ``render_job`` (source, articulation, gestures, ``settle_pitch``), then ``skip`` + ``n_blocks`` control steps;
        returns the TrackChunk of the last ``n_blocks`` (row r = block ``job.start_block + skip + r`` of the mix)."""
        import dataclasses

        from chimera_vocal.music.compile import render_job, set_voice_factory, voice_factory
        assert self.kind == "real" and not job.commands, "control rows: real engine, jobs without timed commands"
        prod = voice_factory()
        built = []

        def capture(j, kernel, geometry):
            v = prod(j, kernel, geometry)
            built.append(v)
            return v
        prev = set_voice_factory(capture)
        try:
            render_job(dataclasses.replace(job, n_out=0), None, None)      # set-up only, no samples
        finally:
            set_voice_factory(prev)
        (v,) = built
        if skip:
            v.control.advance(skip)
        return v.control.advance(n_blocks)

    def late_track_mismatches(self, job) -> list[str]:
        """Control track of a late job (start_block > 0, settled glide) against the same job built from block 0
        (no settle, what ``render notes`` does) at the same absolute blocks, over the late job's whole length."""
        import dataclasses
        n = -(-job.n_out // 64)
        late = self.control_rows(job, n)
        ref = self.control_rows(dataclasses.replace(job, start_block=0, settle_f0=False), n, skip=job.start_block)
        return self.track_mismatches(late, ref)

    @staticmethod
    def spec_start_block(job, fs: int = 48000) -> int:
        """Start block of a score voice: max(0, floor((t_first - 0.30)·48000/64)), t_first = earliest
        t0 - ramp_in of the voice (the 0.30 s pre-roll is the design value, not the module constant)."""
        import math
        t_first = min(g.t0 - g.ramp_in for g in job.gestures)
        return max(0, math.floor((t_first - 0.30) * fs / 64))

    @staticmethod
    def track_mismatches(late, ref) -> list[str]:
        """Row-for-row comparison of two TrackChunks: every field bit-equal, except the glide f0 column of ``glot_par``
        (relative <= F0_GLIDE_REL). Returns readable mismatches (empty = equal)."""
        from chimera_vocal.core.glottis_params import G_F0
        out = []
        assert late.n_blocks == ref.n_blocks
        for name in late._fields[1:]:
            a, b = np.asarray(getattr(late, name)), np.asarray(getattr(ref, name))
            if name == "glot_par":
                a, b = a.copy(), b.copy()
                fa, fb = a[:, G_F0].copy(), b[:, G_F0].copy()
                rel = np.abs(fa - fb) / np.maximum(np.maximum(np.abs(fa), np.abs(fb)), 1e-300)
                bad = np.flatnonzero(rel > F0_GLIDE_REL)
                if bad.size:
                    out.append(f"glot_par f0: {bad.size} rows over {F0_GLIDE_REL:g} rel, first row {int(bad[0])}: "
                               f"{fa[bad[0]]!r} vs {fb[bad[0]]!r}")
                a[:, G_F0] = b[:, G_F0] = 0.0
            rows = np.flatnonzero((a != b).reshape(len(a), -1).any(axis=1))
            if rows.size:
                out.append(f"{name}: {rows.size} rows differ, first row {int(rows[0])}")
        return out


@pytest.fixture(params=["fake", pytest.param("real", marks=pytest.mark.slow)])
def pipeline(request):
    if request.param == "fake":
        planner = request.getfixturevalue("stub_planner")
        request.getfixturevalue("fake_voice")
        return Pipeline("fake", planner.g2p, planner)
    if not real_engine_available():
        pytest.skip("needs the real engine (text, plan, control and dsp packages)")
    from chimera_vocal.music.compile import set_voice_factory
    from chimera_vocal.music.planning import planner_api, set_planner_api
    set_planner_api(None)
    set_voice_factory(None)
    return Pipeline("real", planner_api().default_g2p(), None)
