"""Render cases for the acoustics tests, written once and run with either kernel.

The numpy kernel runs them in the test process; the numba kernel runs them in a fresh interpreter
(``dsp_numba.run_case``) because the test process never imports numba. Each case takes
kernel NAMES (str) and plain arguments and returns a dict of numpy arrays (written to an .npz by the subprocess).
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

import recorded_control as rc
import support
from chimera_vocal.core.body import BodyMix
from chimera_vocal.core.types import Seeds
from chimera_vocal.dsp import layout as L
from chimera_vocal.dsp import network, voice
from chimera_vocal.dsp.kernel_numpy import KERNEL_NUMPY

_LOADED: dict = {}          # filled in the numba subprocess: {"numba": KernelImpl, "geometry": GeometryImpl | None}
STATE_FLOAT = ("nf", "bf", "rf", "noise_f", "glot", "sc", "dec_buf")
STATE_INT = ("ni", "bi", "ri", "noise_i", "glot_i", "si", "nbr_start", "nbr")


def control_available() -> bool:
    """The control model (``chimera_vocal.control.track.ControlModel``) is importable."""
    try:
        return importlib.util.find_spec("chimera_vocal.control.track") is not None
    except ModuleNotFoundError:
        return False


def kernel(name: str):
    if name == "numpy":
        return KERNEL_NUMPY
    if name == "numba":
        assert "numba" in _LOADED, "the numba kernel exists only inside a dsp_numba subprocess"
        return _LOADED["numba"]
    raise ValueError(name)


def geometry():
    """GeometryImpl for real-control cases: ref-numba inside a numba subprocess, else GEOMETRY_VEC."""
    if _LOADED.get("geometry") is not None:
        return _LOADED["geometry"]
    from chimera_vocal.control.geometry import GEOMETRY_VEC
    return GEOMETRY_VEC


@contextlib.contextmanager
def control_factory(factory):
    old = voice.CONTROL_FACTORY
    voice.CONTROL_FACTORY = factory
    try:
        yield
    finally:
        voice.CONTROL_FACTORY = old


def body_of(scenario: dict):
    bm = scenario.get("body_mix")
    if bm is not None:
        return BodyMix(bm.get("female", 0.0), bm.get("child", 0.0)).body()
    return BodyMix.parse(scenario["body"]).body()


def state_arrays(st, prefix: str = "") -> dict:
    return {prefix + k: np.array(getattr(st, k)) for k in STATE_FLOAT + STATE_INT}


def fresh_state(trk: rc.Track, *, seeds: Seeds = Seeds()):
    """human_net + output chain + the recorded constructor row (= HumanVoice::new with the recorded control)."""
    m = trk.scenario["oversample"]
    fs_out = float(trk.scenario["fs_out"])
    st, idx = network.human_net(body_of(trk.scenario), fs_out * m, seeds)
    network.configure_output(st, fs_out, m)
    network.apply_initial(st, idx, trk.initial)
    return st, idx


def drive(kernels, st, idx, trk: rc.Track, n_out: int, call_out: int, *, inherit: str = "track", pos: int = 0):
    """Render n_out output samples straight through the kernel wrapper in calls of call_out output samples;
    kernels is a list used round-robin per call. Returns (audio float32[n_out], next track block)."""
    m = int(st.si[L.SI_M])
    out = np.zeros(n_out, dtype=np.float32)
    done = 0
    call = 0
    while done < n_out:
        n = min(call_out, n_out - done)
        n_int = n * m
        k = kernels[call % len(kernels)]
        steps = voice.control_steps(st, n_int)
        chunk = trk.chunk(pos, pos + steps, inherit=inherit)
        pos += steps
        raw = np.zeros(n_int, dtype=np.float64)
        voice.run(k, st, idx, chunk, n_int, raw)
        got = k.output_chain(st.sc, st.si, st.dec_c, st.dec_buf, raw, n_int, out[done:done + n])
        assert got == n, (got, n)
        done += n
        call += 1
    return out, pos


# ------------------------------------------------------------------------------------------------- cases

def case_track_audio(kernel_name: str, scenario: str, n_out: int, *, full: bool = False, inherit: str = "track",
                     call_out: int = 64 * 750, flush: bool = True) -> dict:
    """V3 track -> kernel -> output chain (K-3.4, K-3.5, K-3.8, K-3.10, K-3.10r)."""
    trk = rc.track(scenario, full=full)
    st, idx = fresh_state(trk)
    if not flush:
        st.si[L.SI_FLAGS] &= ~L.F_FLUSH_SUBNORMALS
    t0 = time.perf_counter()
    audio, _ = drive([kernel(kernel_name)], st, idx, trk, n_out, call_out, inherit=inherit)
    return {"audio": audio, "seconds": np.array(time.perf_counter() - t0)} | state_arrays(st)


def case_state_after(kernel_name: str, scenario: str, n_internal: int) -> dict:
    """State after exactly n_internal internal samples from the constructor state, one run() call (K-3.5)."""
    trk = rc.track(scenario)
    st, idx = fresh_state(trk)
    steps = voice.control_steps(st, n_internal)
    raw = np.zeros(n_internal, dtype=np.float64)
    voice.run(kernel(kernel_name), st, idx, trk.chunk(0, steps), n_internal, raw)
    return {"raw": raw} | state_arrays(st)


def case_swap(kernel_names: list, scenario: str, n_out: int, call_out: int = 64) -> dict:
    """Kernels alternate per call of one control block (K-3.6)."""
    trk = rc.track(scenario)
    st, idx = fresh_state(trk)
    audio, _ = drive([kernel(k) for k in kernel_names], st, idx, trk, n_out, call_out)
    return {"audio": audio} | state_arrays(st)


def glottis_module(kernel_name: str):
    if kernel_name == "numba":
        assert "numba" in _LOADED
        return sys.modules["_chimera_kernel_numba"]
    from chimera_vocal.dsp import kernel_numpy
    return kernel_numpy


def case_glottis(kernel_name: str, set_index: int, n: int = 2000) -> dict:
    """V5: ParametricGlottis::new(96000), params = set, next(dp_i) for i < n (K-3.2)."""
    from chimera_vocal.dsp import glottis
    v5 = support.vector("glottis")
    s = v5["next"][set_index]
    glot, glot_i = glottis.init_glottis(v5["next_fs"], Seeds())
    glottis.set_params(glot, s["glottis"])
    dp = dp_series(s["dp_kind"], n)
    area = np.zeros(n)
    f0_now = np.zeros(n)
    voicing = np.zeros(n)
    glottis_module(kernel_name).glottis_block(glot, glot_i, dp, area, f0_now, voicing)
    return {"area": area, "f0_now": f0_now, "voicing": voicing, "glot": glot, "glot_i": glot_i}


def dp_series(kind: str, n: int = 2000, n_total: int = 2000) -> np.ndarray:
    cm = 980.665
    if kind == "const8":
        return np.full(n, 8.0 * cm)
    if kind == "ramp6":
        return np.array([((6.0 * cm) * i) / float(n_total - 1) for i in range(n)])
    raise ValueError(kind)


def case_output_chain(kernel_name: str, raw: list, m: int, *, output_gain: float = 1.0) -> dict:
    """Push raw through the output chain of a fresh (fs_out, m) state (K-3.3)."""
    from chimera_vocal.core.body import BODY_MALE
    st, idx = network.human_net(BODY_MALE, 96000.0, Seeds())
    network.configure_output(st, 96000.0 / m, m)
    st.sc[L.S_OUTPUT_GAIN] = output_gain
    x = np.asarray(raw, dtype=np.float64)
    out = np.zeros(len(x), dtype=np.float32)
    counts = []
    for i in range(len(x)):         # one push per call: the count must follow the decimator phase
        counts.append(kernel(kernel_name).output_chain(st.sc, st.si, st.dec_c, st.dec_buf, x[i:i + 1], 1,
                                                       out[sum(counts):]))
    return {"out": out[:sum(counts)], "counts": np.array(counts), "dec_c": st.dec_c.copy(),
            "dc_coef": np.array(st.sc[L.S_DC_COEF]), "lim_release": np.array(st.sc[L.S_LIM_RELEASE]),
            "lim_hold_n": np.array(st.si[L.SI_LIM_HOLD_N])}


def make_voice(kernel_name: str, control: dict, **kw):
    """HumanVoice with a test control model: {"kind": "recorded", "scenario": s, "full": bool, "inherit": str} or
    {"kind": "steady", "scenario": s, "block": b} or {"kind": "real"} (the real control model)."""
    kind = control["kind"]
    if kind == "recorded":
        fac = rc.recorded_factory(control["scenario"], full=control.get("full", False),
                                  inherit=control.get("inherit", "track"))
    elif kind == "steady":
        fac = rc.steady_factory(rc.steady_posture(control.get("scenario", "sustain-aa-male"),
                                                  control.get("block", 79)))
    elif kind == "real":
        fac = None
    else:
        raise ValueError(kind)
    body = kw.pop("body", None)
    if body is None:
        body = BodyMix.parse(control.get("body", "male"))
    with control_factory(fac or voice.CONTROL_FACTORY):
        if kind == "real":
            kw.setdefault("geometry", geometry())
        return voice.HumanVoice(kw.pop("fs_out", 48000.0), body, kw.pop("oversample", 2),
                                kernel=kernel(kernel_name), **kw)


_SOURCE_KEYS = (("f0", "f0"), ("glide", "glide"), ("pressure", "pressure"), ("breathiness", "breathiness"),
                ("tension", "tension"), ("fry", "fry"), ("roughness", "roughness"), ("vibrato_depth", "vibratoDepth"),
                ("vibrato_rate", "vibratoRate"), ("drift", "drift"), ("voiced", "voiced"), ("brightness", "brightness"))


def snake_source(patch: dict) -> dict:
    """Template (camelCase) source patch -> SourceParams field names (the form HumanVoice.set_source takes)."""
    return {p: patch[j] for p, j in _SOURCE_KEYS if j in patch}


def apply_ops(v, ops: list, at: int) -> None:
    """Scenario ops of a V3 track (snake op names) or a golden scenario (template op names) due at sample `at`."""
    from chimera_vocal.core.types import Gesture
    for o in ops:
        if o["at"] != at:
            continue
        op = o["op"]
        if op in ("set_source", "source"):
            patch = o.get("patch", o.get("values"))
            v.set_source(snake_source(patch))
        elif op in ("set_gate", "gate"):
            v.set_gate(float(o["level"]))
        elif op in ("set_articulation", "artic"):
            v.set_articulation(dict(o.get("patch", o.get("values"))), o.get("tip_free", o.get("tipFree")))
        elif op == "schedule":
            gs = [Gesture.from_json(g) for g in o["gestures"]]
            off = v.clock + float(o.get("offset", 0.0))
            for g in gs:
                g.t0 += off
                g.t1 += off
            v.schedule(gs)
        elif op == "release":
            v.release(float(o["tag"]))
        elif op == "cancel":
            v.cancel_score()
        else:
            raise ValueError(op)


def render_ops(v, n: int, ops: list, block: int) -> np.ndarray:
    """golden_voice.rs `run`: ops due at or before i, then render up to the next op or block end."""
    ops = sorted(ops, key=lambda o: o["at"])
    out = np.zeros(n, dtype=np.float32)
    i = oi = 0
    while i < n:
        while oi < len(ops) and ops[oi]["at"] <= i:
            apply_ops(v, [ops[oi]], ops[oi]["at"])
            oi += 1
        nxt = max(ops[oi]["at"], i + 1) if oi < len(ops) else n
        end = min(n, i + block, nxt)
        v.render(end - i, out[i:end])
        i = end
    return out


def case_voice_track(kernel_name: str, scenario: str, n: int, block: int, *, full: bool = False,
                     inherit: str = "track") -> dict:
    """HumanVoice + recorded control, the track's ops, host blocks of `block` (K-3.4 via the voice, K-3.7)."""
    trk = rc.track(scenario, full=full)
    v = make_voice(kernel_name, {"kind": "recorded", "scenario": scenario, "full": full, "inherit": inherit},
                   body=body_of(trk.scenario))
    out = render_ops(v, n, trk.scenario["ops"], block)
    return {"audio": out, "clock": np.array(v.clock)} | state_arrays(v.state)


def case_voice_steady(kernel_name: str, *, body: str = "male", f0: float = 110.0, n: int = 9600, block: int = 64,
                      settle: bool = False, start_block: int = 0, probe: bool = True, control: str = "steady",
                      posture: tuple = ("sustain-aa-male", 79)) -> dict:
    """HumanVoice + a held posture (SteadyControl from a recorded row, or the real control model's default
    posture): set_source(f0), set_gate(1), optional settle_pitch, render in blocks of `block` and probe the glottis
    after every call (K-3.14, K-3.15)."""
    ctl = {"kind": "steady", "scenario": posture[0], "block": posture[1]} if control == "steady" else {"kind": "real"}
    v = make_voice(kernel_name, ctl, body=BodyMix.parse(body).body(), start_block=start_block)
    v.set_source({"f0": f0})
    v.set_gate(1.0)
    if settle:
        v.settle_pitch()
    clocks0 = v.clock
    out = np.zeros(n, dtype=np.float32)
    f0_now, voicing, clocks = [], [], []
    for i in range(0, n, block):
        v.render(min(block, n - i), out[i:i + block])
        if probe:
            f0_now.append(float(v.state.glot[L.G_F0_NOW]))
            voicing.append(float(v.state.glot[L.G_VOICING]))
            clocks.append(v.clock)
    calls = getattr(v.control, "calls", [])
    return {"audio": out, "f0_now": np.array(f0_now), "voicing": np.array(voicing), "clocks": np.array(clocks),
            "clock0": np.array(clocks0), "calls": np.array(json.dumps([[str(x) for x in c] for c in calls]))}


def case_flush(kernel_name: str, *, flush: bool, voiced_blocks: int = 80, pause_blocks: int = 600,
               scenario: str = "hello-male") -> dict:
    """Recorded speech onset, then a long pause (lung pressure target 0, glottis amplitude 0, tract held), with or
    without the per-block flush (K-3.8). Reports the subnormal count of the float state after every block."""
    trk = rc.track(scenario)
    st, idx = fresh_state(trk)
    if not flush:
        st.si[L.SI_FLAGS] &= ~L.F_FLUSH_SUBNORMALS
    voiced = trk.chunk(0, voiced_blocks)
    last = trk.chunk(voiced_blocks - 1, voiced_blocks)
    if pause_blocks:
        pause = rc.concat_chunks([last] * pause_blocks)
        pause.p_lung[:] = 0.0
        pause.glot_par[:, L.G_AMPLITUDE] = 0.0
        pause.noise_inherit[:] = -1
        for j in range(int(last.noise_n[0])):
            pause.noise_inherit[:, j] = j
        track = rc.concat_chunks([voiced, pause])
    else:
        track = voiced
    m = int(st.si[L.SI_M])
    every = int(st.si[L.SI_CTRL_EVERY])
    n_blocks = track.n_blocks
    out = np.zeros(n_blocks * every // m, dtype=np.float32)
    sub_counts = np.zeros(n_blocks, dtype=np.int64)
    k = kernel(kernel_name)
    tiny = np.finfo(np.float64).tiny
    for b in range(n_blocks):
        raw = np.zeros(every)
        voice.run(k, st, idx, rc.TrackChunk(1, *[np.ascontiguousarray(getattr(track, f)[b:b + 1])
                                                 for f in rc.TrackChunk._fields[1:]]), every, raw)
        k.output_chain(st.sc, st.si, st.dec_c, st.dec_buf, raw, every, out[b * every // m:(b + 1) * every // m])
        sub_counts[b] = sum(int(np.count_nonzero((a != 0.0) & (np.abs(a) < tiny)))
                            for a in (st.nf, st.bf, st.rf, st.noise_f, st.glot, st.sc))
    return {"audio": out, "subnormals": sub_counts} | state_arrays(st)


def case_flush_unit(kernel_name: str) -> dict:
    """flush_subnormals on a state seeded with subnormal, tiny and normal values in every float array."""
    from chimera_vocal.core.body import BODY_MALE
    st, idx = network.human_net(BODY_MALE, 96000.0, Seeds())
    rng = np.random.default_rng(5)
    vals = np.array([5e-324, -2e-310, 1e-31, -9.9e-31, 1.0000001e-30, -3e-30, 0.5, -7.0])
    seeded = {}
    for name in ("nf", "bf", "rf", "noise_f", "glot", "sc"):
        a = getattr(st, name)
        a[...] = rng.choice(vals, size=a.shape)
        seeded[name] = a.copy()
    kernel(kernel_name).flush_subnormals(st.nf, st.bf, st.rf, st.noise_f, st.glot, st.sc)
    return {f"before_{k}": v for k, v in seeded.items()} | {f"after_{k}": getattr(st, k).copy() for k in seeded}


def case_selftest(kernel_name: str, *, falsify: float = 1.0) -> dict:
    """backend.selftest with the given kernel; falsify != 1 scales every raw sample (a broken constant)."""
    from chimera_vocal.dsp import backend
    k = kernel(kernel_name)
    fac = None if control_available() else rc.recorded_factory("sustain-aa-male")
    if falsify != 1.0:
        inner = k.run_samples

        def bad_run(*args):
            inner(*args)
            args[-1][:] *= falsify

        k = k._replace(run_samples=bad_run)
    geo = _LOADED.get("geometry") if kernel_name == "numba" else None
    if geo is None and control_available():
        geo = geometry()
    with control_factory(fac or voice.CONTROL_FACTORY):
        r = backend.selftest(k, geo)
    return {"passed": np.array(r.passed), "max_rel_err": np.array(r.max_rel_err), "backend": np.array(r.backend),
            "seconds": np.array(r.seconds)}


def case_golden(kernel_name: str, name: str, *, block: int = 128, n: int | None = None) -> dict:
    """A golden voice scenario of the template (tests/golden/voice.json) with the REAL control model."""
    s = next(x for x in support.golden("voice") if x["name"] == name)
    v = make_voice(kernel_name, {"kind": "real"}, body=BodyMix.parse(s["body"]).body(), fs_out=float(s["fsOut"]),
                   oversample=int(s["oversample"]))
    n = s["samples"] if n is None else n
    return {"audio": render_ops(v, n, s["ops"], block)}


def case_voice_ops_real(kernel_name: str, scenario: str, *, block: int = 128) -> dict:
    """A V3 scenario's complete input (ops incl. planned gestures) with the REAL control model (K-3.10)."""
    trk = rc.track(scenario, full=True)
    v = make_voice(kernel_name, {"kind": "real"}, body=body_of(trk.scenario))
    return {"audio": render_ops(v, trk.n_samples, trk.scenario["ops"], block)}


CASES = {name[5:]: fn for name, fn in globals().items() if name.startswith("case_")}


def run_local(case: str, **kw) -> dict:
    return CASES[case](**kw)


def subprocess_main(req: dict) -> int:
    """Entry of the numba subprocess: load the numba kernel (worker-style, fixed module names), run one case."""
    import dsp_numba
    _LOADED.update(dsp_numba.load_for_tests(req["cache_dir"]))
    res = CASES[req["case"]](**req["args"])
    np.savez(req["out"], **res)
    Path(req["out"] + ".info.json").write_text(json.dumps(dsp_numba.dispatch_stats()), encoding="utf-8")
    return 0
