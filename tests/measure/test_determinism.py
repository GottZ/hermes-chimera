"""K-4.6 test_determinism (T6): the same score renders bit-identically twice, with 1 and with 4
workers, and after cancel + repeat; ONE kernel per request even when the warm-up marker appears during the request;
voice 0 with ``seed: null`` uses the template seeds. Red proof: kernel chosen per job."""

from __future__ import annotations

import threading

import pytest

from chimera_vocal import RenderCancelled
from chimera_vocal.core.types import Seeds
from chimera_vocal.music.compile import compile_score
from chimera_vocal.music.schema import validate_score


def score(seed_last=None) -> dict:
    voices = [{"body": b, "start": s, "notes": [{"pitch": 55 + 5 * i + k, "beats": 1, "lyric": "la"}
                                                for k in range(6)]}
              for i, (b, s) in enumerate((("male", 0), ("female", 2), ("child", 4), ("female", 1)))]
    voices[-1]["seed"] = seed_last
    voices[-1]["pan"] = 0.4
    return {"bpm": 110, "voices": voices}


def render_bytes(eng, tmp_path, name, **kw) -> tuple[bytes, object]:
    res = eng.render_score(score(), str(tmp_path / name), name, want_mp3=False, **kw)
    with open(res.path_wav, "rb") as f:
        return f.read(), res


def test_repeat_and_worker_count(stub_planner, fake_workers, make_engine, tmp_path):
    for d in ("a", "b", "c"):
        (tmp_path / d).mkdir()
    a, ra = render_bytes(make_engine(workers=1), tmp_path, "a")
    b, rb = render_bytes(make_engine(workers=4), tmp_path, "b")
    eng = make_engine(workers=4)
    c, rc = render_bytes(eng, tmp_path, "c")
    assert a == b == c, "bit-identical with 1 and 4 workers and on repeat"
    assert ra.kernel == rb.kernel == rc.kernel == "numpy"


def test_cancel_then_repeat(stub_planner, fake_workers, make_engine, tmp_path, monkeypatch):
    (tmp_path / "ref").mkdir()
    (tmp_path / "x").mkdir()
    ref, _ = render_bytes(make_engine(workers=3), tmp_path, "ref")
    eng = make_engine(workers=3)
    cancel = threading.Event()
    with pytest.raises(RenderCancelled):
        eng.render_score(score(), str(tmp_path / "x"), "x", want_mp3=False, cancel=cancel,
                         progress=lambda p: cancel.set() if p > 0 else None)
    again, _ = render_bytes(eng, tmp_path, "x")
    assert again == ref


def test_one_kernel_per_request(stub_planner, fake_workers, make_engine, tmp_path, monkeypatch):
    import chimera_vocal.scheduler as sched
    (tmp_path / "ref").mkdir()
    (tmp_path / "flip").mkdir()
    ref, _ = render_bytes(make_engine(workers=3), tmp_path, "ref")
    state = {"ready": False}
    original = sched.RenderScheduler.choose_kernel

    def choose_then_marker_appears(self):
        k = original(self)
        state["ready"] = True          # the warm-up finishes right after the request chose its kernel
        return k
    monkeypatch.setattr(sched, "_numba_installed", lambda: True)
    monkeypatch.setattr(sched, "_marker_ok", lambda base: {"content_hash": "x", "selftest": {"passed": True}}
                        if state["ready"] else None)
    monkeypatch.setattr(sched.RenderScheduler, "choose_kernel", choose_then_marker_appears)
    got, res = render_bytes(make_engine(workers=3), tmp_path, "flip")
    assert res.kernel == "numpy", "the kernel chosen at submit holds for the whole request"
    assert got == ref, "no job of the request switched to numba when the marker appeared"


def test_voice_seeds(stub_planner):
    for seed_last, expect_last in ((None, Seeds.for_voice(3)), (0, Seeds()), (77, Seeds.for_voice(3, 77))):
        report = validate_score(score(seed_last), g2p=stub_planner.g2p)
        jobs, _, _ = compile_score(report, g2p=stub_planner.g2p)
        assert jobs[0].seeds == Seeds(), "voice 0 with seed null = template seeds"
        assert jobs[1].seeds == Seeds.for_voice(1) and jobs[2].seeds == Seeds.for_voice(2)
        assert jobs[3].seeds == expect_last
        assert len({j.seeds for j in jobs[:3]}) == 3
