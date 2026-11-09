"""K-3.11 test_selftest: the self-test of both kernels passes; a falsified kernel (every radiated sample × 1.01)
fails (INTERFACES §6.6). Without C2 the self-test's control model is the recorded /ɑ/ track of the same scenario."""

from __future__ import annotations

import pytest

import recorded_control as rc
import support
from chimera_vocal.dsp import backend


def test_selftest_expected_data():
    d = support.load_json(support.PACKAGE_DIR / "dsp" / backend.SELFTEST_FILE)
    assert len(d["expected"]) == 4800
    assert d["tolerance"]["rel"] == backend.SELFTEST_TOL == 1e-6
    assert d["expected"] == support.audio_head("voice_male_110hz_aa-back")


def test_selftest_scenario_is_the_recorded_one():
    """The self-test drives exactly the R1 held-vowel scenario the expected samples come from (ops of
    track-sustain-aa-male: set_source f0 110, set_articulation vowel_artic("ɑ", male), set_gate 1, blocks of 128)."""
    sc = backend.selftest_scenario()
    ops = rc.track("sustain-aa-male").scenario["ops"]
    want_artic = next(o["patch"] for o in ops if o["op"] == "set_articulation")
    assert sc["source"] == {"f0": 110.0}
    assert sc["artic"] == want_artic
    assert sc["gate"] == 1.0 and sc["samples"] == 4800 and sc["block"] == 128
    assert sc["body"] == "male" and sc["fs_out"] == 48000 and sc["oversample"] == 2


def test_selftest_passes(kernel_name, run):
    r = run(kernel_name, "selftest")
    assert bool(r["passed"]), f"selftest failed: max_rel_err {float(r['max_rel_err']):.3e}"
    assert float(r["max_rel_err"]) <= 1e-6
    assert str(r["backend"]) == kernel_name
    assert float(r["seconds"]) > 0.0


def test_selftest_detects_falsified_kernel(kernel_name, run):
    r = run(kernel_name, "selftest", falsify=1.01)
    assert not bool(r["passed"]), "a kernel with a constant off by 1 % passed the self-test"
    assert float(r["max_rel_err"]) > 1e-3
