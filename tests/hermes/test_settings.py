"""Configuration resolution: voice, speed, engine config, tool deadline and wait budget (K-6.5, K-6.9)."""

from __future__ import annotations

import logging
import math
import sys

import pytest

import c5fakes


def _ids(p):
    return p.id


# K-6.5: plugin setting > tts.voice > male; a foreign voice warns once and falls back (red proof: order swapped)
def test_voice_resolution(plugin, ctx, caplog):
    s = plugin.mod("settings")
    assert _ids(s.resolve_voice(ctx, None)) == "male"
    assert _ids(s.resolve_voice(ctx, "child")) == "child"
    ctx.settings["voice"] = "female"
    assert _ids(s.resolve_voice(ctx, "child")) == "female"
    assert _ids(s.resolve_voice(ctx, None)) == "female"
    ctx.settings["voice"] = ""
    assert _ids(s.resolve_voice(ctx, "elder")) == "elder"
    ctx.settings["voice"] = "female;pitch=230;rate=1.1"
    p = s.resolve_voice(ctx, "child")
    assert p.id == "female" and p.speech.rate == pytest.approx(1.1)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ctx.settings["voice"] = "alloy"
        assert _ids(s.resolve_voice(ctx, "female")) == "female"
        assert _ids(s.resolve_voice(ctx, "female")) == "female"
        assert _ids(s.resolve_voice(ctx, "nova")) == "male"
        assert _ids(s.resolve_voice(ctx, "nova")) == "male"
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert sum("alloy" in m for m in msgs) == 1, msgs
    assert sum("nova" in m for m in msgs) == 1, msgs
    ctx.settings["voice"] = 42
    assert _ids(s.resolve_voice(ctx, None)) == "male"


def test_resolve_speed(plugin):
    s = plugin.mod("settings")
    P = c5fakes.PRESETS
    assert s.resolve_speed(None, P["male"]) == pytest.approx(1.0)
    assert s.resolve_speed(None, P["narrator"]) == pytest.approx(0.92)
    assert s.resolve_speed(1.25, P["narrator"]) == pytest.approx(1.15)
    assert s.resolve_speed(3.0, P["male"]) == pytest.approx(1.8)
    assert s.resolve_speed(0.1, P["male"]) == pytest.approx(0.5)
    for bad in (float("nan"), float("inf"), -1.0, 0, True, "fast"):
        assert s.resolve_speed(bad, P["teen"]) == pytest.approx(1.05), bad


# engine_config: plugin settings, env wins, numba cache under <HERMES_HOME>/cache/chimera/numba
def test_engine_config(plugin, ctx, monkeypatch):
    s = plugin.mod("settings")
    cfg = s.engine_config(ctx)
    assert isinstance(cfg, plugin.vocal.EngineConfig)
    assert (cfg.backend, cfg.workers, cfg.text_prep, cfg.max_voices) == ("auto", 0, True, 8)
    assert cfg.cache_dir == str(plugin.hermes.home / "cache" / "chimera" / "numba")
    ctx.settings.update(backend="numpy", workers=3, text_prep=False, max_voices=4)
    cfg = s.engine_config(ctx)
    assert (cfg.backend, cfg.workers, cfg.text_prep, cfg.max_voices) == ("numpy", 3, False, 4)
    monkeypatch.setenv("CHIMERA_BACKEND", "numba")
    monkeypatch.setenv("CHIMERA_WORKERS", "2")
    cfg = s.engine_config(ctx)
    assert (cfg.backend, cfg.workers) == ("numba", 2)
    monkeypatch.delenv("CHIMERA_BACKEND")
    monkeypatch.delenv("CHIMERA_WORKERS")
    ctx.settings.update(backend="gpu", workers="many")
    cfg = s.engine_config(ctx)
    assert (cfg.backend, cfg.workers) == ("auto", 0)


def _deadline(plugin, config=None, env=None, monkeypatch=None):
    plugin.hermes.config.clear()
    plugin.hermes.config.update(config or {})
    if env is None:
        monkeypatch.delenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", raising=False)
    else:
        monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", env)
    return plugin.mod("settings").tool_deadline_s(), plugin.mod("jobs").wait_budget_s(None)


def _tools(conc=None, seq=None):
    t = {}
    if conc is not None:
        t["concurrent_batch"] = conc
    if seq is not None:
        t["sequential_call"] = seq
    return {"timeouts": {"tools": t}}


# K-6.9: deadline mirrors agent.deadline.resolve_timeout; budget = max(1, min(d - 45, 375)), d < 60 -> 0,
# unbounded -> 1800 (red proof: env var ignored)
def test_deadline(plugin, monkeypatch):
    assert _deadline(plugin, monkeypatch=monkeypatch) == (420.0, 375.0)
    assert _deadline(plugin, _tools(120), monkeypatch=monkeypatch) == (120.0, 75.0)
    assert _deadline(plugin, env="90", monkeypatch=monkeypatch) == (90.0, 45.0)
    assert _deadline(plugin, _tools(200), env="90", monkeypatch=monkeypatch) == (200.0, 155.0)  # config > env
    assert _deadline(plugin, _tools(50), monkeypatch=monkeypatch) == (50.0, 0.0)
    assert _deadline(plugin, _tools(0), monkeypatch=monkeypatch) == (None, 1800.0)
    assert _deadline(plugin, _tools(-5), monkeypatch=monkeypatch) == (None, 1800.0)
    assert _deadline(plugin, env="0", monkeypatch=monkeypatch) == (None, 1800.0)
    assert _deadline(plugin, _tools(600, 200), monkeypatch=monkeypatch) == (200.0, 155.0)
    assert _deadline(plugin, _tools(150, 900), monkeypatch=monkeypatch) == (150.0, 105.0)
    assert _deadline(plugin, _tools(None, 0), monkeypatch=monkeypatch) == (420.0, 375.0)
    assert _deadline(plugin, _tools(True), env="90", monkeypatch=monkeypatch) == (90.0, 45.0)   # bool falls through
    assert _deadline(plugin, _tools("abc"), monkeypatch=monkeypatch) == (420.0, 375.0)
    assert _deadline(plugin, _tools(float("nan")), env="100", monkeypatch=monkeypatch) == (100.0, 55.0)
    assert _deadline(plugin, _tools("300"), monkeypatch=monkeypatch) == (300.0, 255.0)
    assert _deadline(plugin, env="junk", monkeypatch=monkeypatch) == (420.0, 375.0)
    assert _deadline(plugin, {"timeouts": "broken"}, monkeypatch=monkeypatch) == (420.0, 375.0)
    assert _deadline(plugin, _tools(61), monkeypatch=monkeypatch) == (61.0, 16.0)
    assert _deadline(plugin, _tools(45.5), monkeypatch=monkeypatch) == (45.5, 0.0)
    d, b = _deadline(plugin, _tools(10 ** 12), monkeypatch=monkeypatch)
    assert d == ((0xFFFFFFFF - 1000) / 1000.0 if sys.platform == "win32" else 31_536_000.0) and b == 375.0


def test_wait_budget_requested(plugin, monkeypatch):
    jobs = plugin.mod("jobs")
    _deadline(plugin, monkeypatch=monkeypatch)
    assert jobs.wait_budget_s(30) == 30.0
    assert jobs.wait_budget_s(0.2) == 1.0
    assert jobs.wait_budget_s(1000) == 375.0
    _deadline(plugin, _tools(0), monkeypatch=monkeypatch)
    assert jobs.wait_budget_s(30) == 30.0
    assert jobs.wait_budget_s(5000) == 1800.0
    _deadline(plugin, _tools(50), monkeypatch=monkeypatch)
    assert jobs.wait_budget_s(30) == 0.0
    assert not math.isnan(jobs.wait_budget_s(None))
