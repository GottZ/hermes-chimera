"""K-4.11 test_consonant_cost (TESTPLAN L4, ``slow``, S3, report only, P1-24): RTF of both kernels on the
consonant-rich P1 probe sentences against a held vowel. Writes ``consonant_cost.json`` to $CHIMERA_REPORT_DIR (or
the test's tmp dir) and prints it."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path

import pytest

from support.fakes import real_engine_available

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not real_engine_available(), reason="real engine - S3")]
PROBES = ["Stop at the top, Pat, and take the tea to the cat.", "Six sharp spikes split the stiff straps."]


def test_consonant_cost(kernel_name, make_scheduler, tmp_path, monkeypatch):
    from chimera_vocal.music.compile import RenderJob, set_voice_factory
    from chimera_vocal.music.planning import planner_api, set_planner_api
    from chimera_vocal.presets import PRESETS
    from chimera_vocal.scheduler import MemorySink, Request
    from chimera_vocal.speech import plan_speech
    from chimera_vocal.core.types import Gesture, Seeds, SourceParams
    from chimera_vocal.core.body import BodyMix
    monkeypatch.delenv("CHIMERA_WORKER_VOICE_FACTORY", raising=False)
    set_planner_api(None)
    set_voice_factory(None)
    sched = make_scheduler(workers=2)
    g2p = planner_api().default_g2p()

    def timed(job):
        sink = MemorySink(job.n_out)
        t = time.perf_counter()
        sched.submit(Request(job.id, "music", kernel_name), job, sink=sink, cancel=threading.Event()).result(3600)
        return (time.perf_counter() - t) / (job.n_out / 48000)

    vowel = RenderJob(id="vowel", body=BodyMix(), seeds=Seeds(), source=SourceParams().patched(f0=110.0),
                      gestures=[Gesture(0.05, 3.9, 0.04, 0.06, 1.0, {"breath": 1.0}, label="breath")],
                      n_out=4 * 48000)
    timed(vowel)    # warm the worker (and the numba cache)
    rows = {"kernel": kernel_name, "vowel_rtf": timed(vowel), "probes": {}}
    for text in PROBES:
        jobs, _ = plan_speech(text, PRESETS["male"], speed=None, g2p=g2p, text_prep=False, split=False,
                              settle_first=False)
        rows["probes"][text] = timed(jobs[0])
    out = Path(os.environ.get("CHIMERA_REPORT_DIR", tmp_path)) / f"consonant_cost-{kernel_name}.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(json.dumps(rows))
    assert all(math.isfinite(v) and v > 0 for v in [rows["vowel_rtf"], *rows["probes"].values()])
