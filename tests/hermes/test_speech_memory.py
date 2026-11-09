"""Speech has a memory bound in the Hermes process (REVIEW-2A R2A-02).

A fresh subprocess plays the Hermes host: fake public Hermes modules, the plugin loaded like Hermes loads it, the real
engine package, render workers with FakeVoice (the host-side path - windows, assembly, WAV writing - is the real one).
It speaks the longest text ``chimera_speak`` accepts (4 000 characters, about 250 s of audio) through the tool and
through the TTS provider; the peak RSS of the host (Linux VmHWM, else ru_maxrss) may rise by at most 64 MB - the
same bound as the music mixer (K-5.10). Measured on 9a616be: +454 MB (tool) and +457 MB (provider); with 20 000
characters +2.3 GB. Red proof: the in-memory path of 9a616be (``speak`` + ``Audio.to_wav``).
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from support import REPO_ROOT

pytest.importorskip("resource", reason="peak RSS needs the POSIX resource module")

LIMIT_MB = 64.0

HOST = textwrap.dedent(r"""
    import json, os, resource, sys
    root, path, n_chars, home = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    sys.path[:0] = [os.path.join(root, "tests"), os.path.join(root, "tests", "hermes", "c5_support"),
                    os.path.join(root, "tests", "hermes", "c5_support", "fakehermes")]
    os.environ.update(CHIMERA_TESTING="1", CHIMERA_WORKER_VOICE_FACTORY="support.fakes:fake_voice_factory",
                      CHIMERA_WORKERS="4", CHIMERA_BACKEND="numpy")
    import hermes_constants
    hermes_constants.HOME = home
    import c5fakes
    from pathlib import Path

    def peak_mb():
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1]) / 1024.0
        except OSError:
            pass
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0)

    pkg = c5fakes.load_plugin(root=Path(root))
    ctx = c5fakes.FakeCtx()
    pkg.register(ctx)
    para = ("The quick brown fox jumps over the lazy dog while the orchestra tunes its instruments, "
            "and everyone waits patiently for the concert to begin. ")
    text = (para * (n_chars // len(para) + 1))[:n_chars]
    warm = json.loads(ctx.tools["chimera_speak"]["handler"]({"text": "Hello there.", "wait_s": 600}))
    assert warm["status"] == "done", warm
    base = peak_mb()
    if path == "tool":
        res = json.loads(ctx.tools["chimera_speak"]["handler"]({"text": text, "wait_s": 1800}))
        assert res.get("status") == "done", res
        size = os.path.getsize(res["wav"])
    else:
        out = ctx.tts_providers[0].synthesize(text, os.path.join(home, "reply.mp3"))
        size = os.path.getsize(out)
    print(json.dumps({"base_mb": base, "peak_mb": peak_mb(), "wav_bytes": size}))
    c5fakes.submodule(pkg, "chimera_hermes._vocal").api().get_engine().shutdown()
""")


@pytest.mark.parametrize("path", ["tool", "provider"])
def test_speech_host_memory(tmp_path, path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    r = subprocess.run([sys.executable, "-c", HOST, str(REPO_ROOT), path, "4000", str(home)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    out = json.loads(r.stdout.splitlines()[-1])
    assert out["wav_bytes"] > 44 + 2 * 48000 * 200, out           # really about 250 s of audio
    rise = out["peak_mb"] - out["base_mb"]
    assert rise <= LIMIT_MB, f"host peak RSS rose by {rise:.0f} MB speaking 4000 characters ({out})"
    leftovers = [p for p in home.rglob(".chimera-tmp-*")]
    assert leftovers == [], leftovers


def test_speak_text_limit(registered, tool):
    from c5fakes import FakeEngine  # noqa: F401  (registered uses the fake engine)
    r = tool(registered, "chimera_speak", {"text": "a" * 4001, "wait_s": 30})
    assert r["success"] is False
    assert "4001 characters" in r["error"] and "at most 4000" in r["error"] and "split" in r["error"], r
    ok = tool(registered, "chimera_speak", {"text": ("word " * 800)[:4000], "wait_s": 30})
    assert ok["success"] is True, ok
