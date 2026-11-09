"""Two Hermes profiles in one process: each plugin copy speaks, and each renders with its own settings.

A fresh subprocess plays a Hermes process with two profiles: two copies of the plugin (Hermes imports a directory
plugin once per profile scope, the second as ``hermes_plugins.<slug>__home_<digest>``) with their own settings share
the engine of the first copy; render workers use FakeVoice, the rest is the real engine (real text preparation and
score validation).

* The second copy speaks (tool and TTS provider). Red proof: its voice preset (a class of the second copy) reached
  the first copy's engine, which did not recognise it - "unknown voice VoicePreset(id='male', ...)".
* Profile B's request is placed inside profile A's job - after A's tool call returned or A's provider resolved its
  engine, before the engine plans the render. A's result must be what A gets alone. Red proof (R2A-23): the engine
  read the settings applied last at render time - A's speech came out with B's text preparation (B's length) and
  A's two-voice score failed with B's limit of one voice.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from support import REPO_ROOT

HOST = textwrap.dedent(r"""
    import json, os, sys
    root, home = sys.argv[1], sys.argv[2]
    sys.path[:0] = [os.path.join(root, "tests"), os.path.join(root, "tests", "hermes", "c5_support"),
                    os.path.join(root, "tests", "hermes", "c5_support", "fakehermes")]
    os.environ.update(CHIMERA_TESTING="1", CHIMERA_WORKER_VOICE_FACTORY="support.fakes:fake_voice_factory",
                      CHIMERA_WORKERS="2")
    import hermes_constants
    hermes_constants.HOME = home
    import c5fakes
    from pathlib import Path

    TEXT = "Call me at 555 1234 on the 3rd of May, Dr. Smith."
    DUET = {"bpm": 120, "voices": [{"body": "male", "notes": [{"pitch": "C3", "beats": 1, "lyric": "la"}]},
                                   {"body": "female", "notes": [{"pitch": "E4", "beats": 1, "lyric": "la"}]}]}
    SOLO = {"bpm": 120, "voices": [{"notes": [{"pitch": "C4", "beats": 1, "lyric": "la"}]}]}
    a_pkg = c5fakes.load_plugin("chimera", root=Path(root))
    b_pkg = c5fakes.load_plugin("chimera__home_0123456789ab", root=Path(root))
    a_ctx = c5fakes.FakeCtx({"backend": "numpy", "text_prep": True, "max_voices": 8, "music_format": "wav"})
    b_ctx = c5fakes.FakeCtx({"backend": "numpy", "text_prep": False, "max_voices": 1, "music_format": "wav"})
    a_pkg.register(a_ctx)
    b_pkg.register(b_ctx)
    tools_a = c5fakes.submodule(a_pkg, "chimera_hermes.tools")
    prov_a = c5fakes.submodule(a_pkg, "chimera_hermes.provider")

    def call(ctx, name, args):
        return json.loads(ctx.tools[name]["handler"](args, session_id="s-" + str(id(ctx))))

    def b_asks():                      # any request of profile B applies B's settings to the shared engine
        r = call(b_ctx, "chimera_check", {"score": SOLO})
        assert r["success"] is True and r["ok"] is True, r

    def wav_seconds(path):
        return (os.path.getsize(path) - 44) / 2 / 48000.0

    def synth(ctx, name):
        try:
            return {"seconds": wav_seconds(ctx.tts_providers[0].synthesize(TEXT, os.path.join(home, name)))}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    out = {"a_alone": call(a_ctx, "chimera_speak", {"text": TEXT, "wait_s": 300})}
    b_asks()
    out["b_alone"] = call(b_ctx, "chimera_speak", {"text": TEXT, "wait_s": 300})
    out["b_provider"] = synth(b_ctx, "b.wav")

    orig_cleanup = tools_a._cleanup
    tools_a._cleanup = lambda eng: (b_asks(), orig_cleanup(eng))       # runs first in A's job thread
    out["a_speak"] = call(a_ctx, "chimera_speak", {"text": TEXT, "wait_s": 300})
    out["a_duet"] = call(a_ctx, "chimera_render", {"score": DUET, "wait_s": 300})
    tools_a._cleanup = orig_cleanup

    orig_tmp = prov_a._tmp_dir
    prov_a._tmp_dir = lambda: (b_asks(), orig_tmp())[1]               # after A resolved the engine, before it renders
    out["a_provider"] = synth(a_ctx, "a.wav")
    prov_a._tmp_dir = orig_tmp
    print(json.dumps(out))
    c5fakes.submodule(a_pkg, "chimera_hermes._vocal").api().get_engine().shutdown()
""")


@pytest.fixture(scope="module")
def two_profiles(tmp_path_factory):
    home = tmp_path_factory.mktemp("two-profiles") / "hermes-home"
    home.mkdir()
    r = subprocess.run([sys.executable, "-c", HOST, str(REPO_ROOT), str(home)], capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout.splitlines()[-1])


def test_second_profile_copy_speaks(two_profiles):
    b = two_profiles["b_alone"]
    assert b.get("status") == "done", f"the second plugin copy cannot speak: {b}"
    assert "seconds" in two_profiles["b_provider"], f"the second copy's TTS provider failed: {two_profiles}"


def test_each_profile_renders_with_its_own_settings(two_profiles):
    out = two_profiles
    alone, b_alone = out["a_alone"].get("seconds"), out["b_alone"].get("seconds")
    assert alone is not None and b_alone is not None and alone != b_alone, \
        f"probe needs settings that change the result: {out}"
    speak = out["a_speak"]
    assert speak.get("status") == "done", speak
    assert speak["seconds"] == alone, f"A's speech used B's text preparation: {out}"
    duet = out["a_duet"]
    assert duet.get("status") == "done" and duet.get("voices") == 2, f"A's score was held to B's voice limit: {duet}"
    # the tool rounds to 1 ms; A and B differ by 0.6 s
    assert abs(out["a_provider"].get("seconds", -1.0) - alone) <= 0.0005, f"A's spoken reply used B's settings: {out}"
