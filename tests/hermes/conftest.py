"""Fixtures for the C5 tests (Hermes binding; TESTPLAN L6).

Unit tests run against fakes: the public Hermes modules come from ``c5_support/fakehermes`` (put on sys.path only
inside the ``fake_hermes`` fixture), the engine package from ``c5fakes.make_fake_vocal`` (patched into the C5 seam
``chimera_hermes._vocal.api``). Tests marked ``hermes`` run Hermes itself in a subprocess (``hermes_lab``).
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

SUPPORT = Path(__file__).resolve().parent / "c5_support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

import c5fakes  # noqa: E402

_HERMES_TOPS = ("agent", "gateway", "hermes_cli", "hermes_constants")
_ENV_CLEAN = ("HERMES_PLUGIN_HOST_PROCESS", "HERMES_CONCURRENT_TOOL_TIMEOUT_S", "HERMES_SESSION_PLATFORM",
              "CHIMERA_BACKEND", "CHIMERA_WORKERS", "CHIMERA_THREADS")


@pytest.fixture
def fake_hermes(tmp_path, monkeypatch):
    """Fake public Hermes modules with a fresh HERMES_HOME; yields a namespace with live handles:
    ``home`` (Path), ``config`` (dict behind load_config_readonly), ``session`` (dict behind get_session_env),
    ``writes`` (core-config writes, must stay empty)."""
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] in _HERMES_TOPS}
    for k in saved:
        del sys.modules[k]
    monkeypatch.syspath_prepend(str(c5fakes.FAKEHERMES_DIR))
    for name in _ENV_CLEAN:
        monkeypatch.delenv(name, raising=False)
    import gateway.session_context as session_context
    import hermes_cli.config as hconfig
    import hermes_constants

    home = tmp_path / "hermes-home"
    home.mkdir()
    hermes_constants.HOME = home
    hconfig.CONFIG = {}
    hconfig.WRITES = []
    session_context.ENV = {}
    yield types.SimpleNamespace(home=home, config=hconfig.CONFIG, session=session_context.ENV, writes=hconfig.WRITES,
                                modules=types.SimpleNamespace(config=hconfig, session_context=session_context,
                                                              constants=hermes_constants))
    for k in [k for k in sys.modules if k.split(".")[0] in _HERMES_TOPS]:
        del sys.modules[k]
    sys.modules.update(saved)


@pytest.fixture
def fake_engine():
    return c5fakes.FakeEngine()


@pytest.fixture
def vocal(fake_engine):
    return c5fakes.make_fake_vocal(fake_engine)


@pytest.fixture
def plugin(fake_hermes, vocal, monkeypatch):
    """The plugin imported as ``hermes_plugins.chimera_t<n>`` with the engine seam patched to the fakes.

    ``plugin.mod("tools")`` returns ``<pkg>.chimera_hermes.tools``."""
    pkg = c5fakes.load_plugin()
    seam = c5fakes.submodule(pkg, "chimera_hermes._vocal")
    monkeypatch.setattr(seam, "api", lambda: vocal)
    ns = types.SimpleNamespace(pkg=pkg, seam=seam, vocal=vocal, engine=vocal.engine, hermes=fake_hermes,
                               mod=lambda rel: c5fakes.submodule(pkg, f"chimera_hermes.{rel}"))
    yield ns
    c5fakes.unload_plugin(pkg)


@pytest.fixture
def ctx():
    return c5fakes.FakeCtx()


@pytest.fixture
def registered(plugin, ctx):
    """``register(ctx)`` done; returns the FakeCtx with all registrations."""
    plugin.pkg.register(ctx)
    return ctx


def call_tool(ctx: c5fakes.FakeCtx, name: str, args, **kwargs) -> dict:
    """Call a registered tool handler like Hermes does and parse its JSON string."""
    import json

    out = ctx.tools[name]["handler"](args, **kwargs)
    assert isinstance(out, str), f"{name} returned {type(out).__name__}, not a JSON string"
    data = json.loads(out)
    assert isinstance(data, dict), out
    return data


@pytest.fixture
def tool():
    return call_tool


# ------------------------------------------------------------------------------------------------ Hermes lab

def _hermes_python(upstream: Path) -> Path | None:
    env = os.environ.get("HERMES_PYTHON")
    candidates = [Path(env)] if env else []
    candidates += [upstream.parent / "venv" / "bin" / "python", upstream / ".venv" / "bin" / "python",
                   upstream.parent / "venv" / "Scripts" / "python.exe", upstream / ".venv" / "Scripts" / "python.exe"]
    for c in candidates:
        if c.is_file():
            return c
    return None


@pytest.fixture
def hermes_lab(hermes_upstream, tmp_path):
    """A Hermes interpreter from HERMES_PYTHON (or the venv next to / inside HERMES_UPSTREAM) plus an isolated
    environment: fresh HERMES_HOME and HOME under tmp_path, no inherited HERMES_* variables or secrets."""
    py = _hermes_python(hermes_upstream)
    if py is None:
        pytest.skip("no Hermes interpreter: set HERMES_PYTHON")
    home = tmp_path / "hermes-home"
    fakehome = tmp_path / "home"
    tmpdir = tmp_path / "tmp"
    for d in (home, fakehome, tmpdir):
        d.mkdir()
    (home / "config.yaml").write_text(
        "_config_version: 53\nupdates:\n  check: false\ntelemetry:\n  shared_metrics:\n    enabled: false\n"
        "    send: false\n    offer_version: 2\nauxiliary:\n  title_generation:\n    enabled: false\n",
        encoding="utf-8")
    env = {"PATH": f"{py.parent}{os.pathsep}/usr/bin{os.pathsep}/bin", "HOME": str(fakehome),
           "HERMES_HOME": str(home), "TMPDIR": str(tmpdir), "LANG": "C.UTF-8", "TZ": "UTC", "NO_COLOR": "1",
           "COLUMNS": "200", "TERM": "dumb", "PYTHONDONTWRITEBYTECODE": "1"}
    hermes_bin = py.parent / ("hermes.exe" if os.name == "nt" else "hermes")
    return types.SimpleNamespace(python=py, hermes=hermes_bin if hermes_bin.exists() else None, env=env, home=home,
                                 tmp=tmpdir, upstream=hermes_upstream)
