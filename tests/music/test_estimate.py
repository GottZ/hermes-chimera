"""K-5.6 test_estimate (TESTPLAN L5, INTERFACES §8.5): longest-processing-time assignment on workers − 1 music
slots, RENDERED voice lengths (not the mix length), compile surcharge only while the numba marker is missing."""

from __future__ import annotations

import pytest

from chimera_vocal.music.estimate import (DEFAULT_COMPILE_S, DEFAULT_RTF, estimate_kernels, estimate_seconds,
                                          music_slots)


@pytest.mark.parametrize("jobs,rtf,par,expected", [
    ([10, 10, 10], 1.0, 2, 20.5),           # 10 | 10 -> third on the first free slot
    ([3, 5, 7, 2], 2.0, 2, 18.5),           # 14 | 10, 6 -> B (16), 4 -> A (18)
    ([3, 5, 7, 2], 2.0, 1, 34.5),           # one slot: sum
    ([3, 5, 7, 2], 2.0, 8, 14.5),           # more slots than jobs: the longest job
    ([9, 8, 7, 6, 5], 1.0, 3, 13.5),        # 9 | 8 | 7, 6 -> C (13), 5 -> B (13)
    ([], 1.0, 2, 0.5),
])
def test_lpt(jobs, rtf, par, expected):
    assert estimate_seconds(jobs, rtf=rtf, parallelism=par) == pytest.approx(expected, abs=1e-12)


def test_compile_and_overhead():
    assert estimate_seconds([1.0], rtf=1.0, parallelism=1, compile_s=30.0) == pytest.approx(31.5)
    assert estimate_seconds([1.0], rtf=1.0, parallelism=1, overhead_s=0.0) == pytest.approx(1.0)


@pytest.mark.parametrize("workers,slots", [(1, 1), (2, 1), (3, 2), (4, 3), (8, 7)])
def test_music_slots(workers, slots):
    assert music_slots(workers) == slots


def test_defaults():
    assert DEFAULT_RTF == {"numba": 0.45, "numpy": 5.0}
    assert DEFAULT_COMPILE_S == 30.0


@pytest.mark.parametrize("state,compile_added", [("ready", False), ("cold", True), ("warming", True),
                                                 ("absent", True), ("failed", True)])
def test_estimate_kernels_compile_only_without_marker(state, compile_added):
    rtf = {"numba": 0.5, "numpy": 5.0}
    est = estimate_kernels([60.0, 50.0, 40.0], workers=3, rtf=rtf, numba_state=state, compile_s=30.0)
    # 2 music slots: numba 30 | 25, 20 -> 45; numpy 300 | 250, 200 -> 450
    assert est["numba"] == pytest.approx(45.5 + (30.0 if compile_added else 0.0))
    assert est["numpy"] == pytest.approx(450.5), "the numpy kernel never compiles"


def test_estimate_kernels_default_rtf():
    est = estimate_kernels([10.0], workers=2, numba_state="ready")
    assert est == pytest.approx({"numba": 10.0 * 0.45 + 0.5, "numpy": 10.0 * 5.0 + 0.5})


def canon_score() -> dict:
    notes = [{"pitch": p, "beats": 1, "lyric": "ah"} for p in (60, 62, 64, 60, 60, 62, 64, 60)]
    return {"bpm": 120, "voices": [{"body": b, "start": s, "notes": notes}
                                   for b, s in (("female", 0), ("male", 8), ("child", 16))]}


def test_rendered_not_mix_lengths(stub_planner, fake_workers, make_engine, monkeypatch):
    eng = make_engine(workers=3)
    report = eng.check_score(canon_score())
    assert report.ok, report.issues
    rendered = [v.seconds for v in report.voices]
    assert report.voice_seconds == pytest.approx(sum(rendered), rel=1e-12)
    # later voices start late (start_block > 0) and are rendered shorter than the mix
    assert rendered[2] < rendered[1] < rendered[0] <= report.seconds + 1e-9
    assert rendered[2] <= report.seconds - 8.0 + 0.31
    st = eng.status()
    assert st.numba_state == "absent"
    assert report.estimate_s == pytest.approx(estimate_kernels(rendered, workers=3, rtf=st.rtf_estimate,
                                                               numba_state=st.numba_state))
    assert report.estimate_s["numpy"] < estimate_kernels([report.seconds] * 3, workers=3, rtf=st.rtf_estimate,
                                                         numba_state=st.numba_state)["numpy"]


def test_engine_estimate_marker(stub_planner, fake_workers, make_engine, monkeypatch):
    import chimera_vocal.scheduler as sched
    monkeypatch.setattr(sched, "_numba_installed", lambda: True)
    marker = {"content_hash": "x", "selftest": {"passed": True}}
    monkeypatch.setattr(sched, "_marker_ok", lambda base: marker)
    eng = make_engine(workers=3)
    report = eng.check_score(canon_score())
    rendered = [v.seconds for v in report.voices]
    assert eng.status().numba_state == "ready"
    no_compile = estimate_kernels(rendered, workers=3, rtf=eng.status().rtf_estimate, numba_state="ready")
    assert report.estimate_s["numba"] == pytest.approx(no_compile["numba"])
    monkeypatch.setattr(sched, "_marker_ok", lambda base: None)
    eng2 = make_engine(workers=3)
    report2 = eng2.check_score(canon_score())
    assert eng2.status().numba_state == "cold"
    assert report2.estimate_s["numba"] == pytest.approx(no_compile["numba"] + DEFAULT_COMPILE_S)
