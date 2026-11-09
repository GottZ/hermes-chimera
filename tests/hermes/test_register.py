"""K-6.1 register(ctx) + tool schemas + manifest consistency (TESTPLAN L6; INTERFACES §2, §3a, §11.1, §13, §15)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import c5fakes

ROOT = c5fakes.REPO_ROOT
TOOL_NAMES = {"chimera_render", "chimera_check", "chimera_speak", "chimera_job"}


def _yaml_list(text: str, key: str) -> list[str]:
    m = re.search(rf"^{key}:\s*\[([^\]]*)\]", text, re.M)
    assert m, f"{key} missing"
    return [x.strip() for x in m.group(1).split(",") if x.strip()]


# K-6.1 (in-process part): every registration with the expected shape
def test_register_fake_ctx(registered, plugin):
    ctx = registered
    assert len(ctx.tts_providers) == 1
    prov = ctx.tts_providers[0]
    import agent.tts_provider as tp

    assert isinstance(prov, tp.TTSProvider)
    assert prov.name == "chimera"
    assert prov.display_name == "Chimera (local vocal synth)"
    assert prov.stream_sample_rate == 48000
    assert prov.get_setup_schema() == {"name": "Chimera (local vocal synth)", "badge": "local · free",
                                       "tag": "Physically modelled English voices on your CPU", "env_vars": []}

    assert set(ctx.tools) == TOOL_NAMES
    tools_mod = plugin.mod("tools")
    for name, reg in ctx.tools.items():
        assert reg["toolset"] == "chimera"
        assert reg["schema"]["name"] == name
        assert callable(reg["handler"])
        assert reg["check_fn"] is tools_mod.check_available
        assert reg["override"] is False
    manifest = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
    assert set(_yaml_list(manifest, "provides_tools")) == set(ctx.tools)

    assert set(ctx.commands) == {"chimera"}
    cmd = ctx.commands["chimera"]
    assert cmd["args_hint"] == "[status|voices|voice <id>|warmup|test]"
    assert cmd["description"]
    assert isinstance(cmd["handler"]("voices"), str)

    assert set(ctx.cli_commands) == {"chimera"}
    cli = ctx.cli_commands["chimera"]
    assert callable(cli["setup_fn"]) and callable(cli["handler_fn"]) and cli["help"]

    assert set(ctx.skills) == {"music"}
    skill = ctx.skills["music"]
    assert Path(skill["path"]).resolve() == (ROOT / "skills" / "music" / "SKILL.md").resolve()
    assert skill["description"].startswith("Write scores for chimera_render")
    fm = skill["frontmatter"]
    assert fm["name"] == "music" and fm["version"] == "0.1.0" and fm["description"] == skill["description"]
    assert fm["metadata"]["hermes"]["requires_tools"] == ["chimera_render"]
    assert set(fm["metadata"]["hermes"]["tags"]) >= {"music", "singing", "audio"}

    assert ctx.config_writes == [] and plugin.hermes.writes == []
    assert plugin.engine.calls == [], "register must not touch the engine"


REGISTER_PROBE = r"""
import json, sys, threading
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import c5fakes
before = set(sys.modules)
threads = threading.active_count()
pkg = c5fakes.load_plugin("chimera")
ctx = c5fakes.FakeCtx()
pkg.register(ctx)
new = sorted(set(sys.modules) - before)
print(json.dumps({"numpy": "numpy" in sys.modules, "numba": "numba" in sys.modules,
                  "threads_delta": threading.active_count() - threads, "new_modules": new,
                  "tools": sorted(ctx.tools), "providers": [p.name for p in ctx.tts_providers]}))
"""


# K-6.1 (fresh interpreter): no numpy/numba import and no threads during register (rot: `import numpy` at module level)
def test_register_imports_nothing_heavy():
    r = subprocess.run([sys.executable, "-I", "-c", REGISTER_PROBE, str(c5fakes.HERE), str(c5fakes.FAKEHERMES_DIR)],
                       capture_output=True, text=True, timeout=120, cwd=str(c5fakes.HERE))
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["numpy"] is False, "register imported numpy"
    assert out["numba"] is False, "register imported numba"
    assert out["threads_delta"] == 0, "register started a thread"
    assert out["tools"] == sorted(TOOL_NAMES) and out["providers"] == ["chimera"]
    heavy = [m for m in out["new_modules"]
             if m.split(".")[0] in ("numpy", "numba", "llvmlite") or "chimera_vocal" in m]
    assert heavy == [], heavy


def _walk(obj, path="$"):
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


# INTERFACES §11.1: schema shape (no oneOf/$ref, required fields, MEDIA rule in the descriptions)
def test_tool_schemas(plugin):
    tools = plugin.mod("tools")
    assert set(tools.SCHEMAS) == TOOL_NAMES and set(tools.HANDLERS) == TOOL_NAMES
    for name, schema in tools.SCHEMAS.items():
        json.dumps(schema)
        assert schema["parameters"]["type"] == "object"
        for p, node in _walk(schema):
            if isinstance(node, dict):
                assert not ({"oneOf", "anyOf", "allOf", "$ref"} & set(node)), f"{name} {p}"
    req = {n: s["parameters"]["required"] for n, s in tools.SCHEMAS.items()}
    assert req == {"chimera_render": ["score"], "chimera_check": ["score"], "chimera_speak": ["text"],
                   "chimera_job": ["job_id"]}
    score = tools.SCHEMAS["chimera_render"]["parameters"]["properties"]["score"]
    assert score["required"] == ["bpm", "voices"]
    assert score["properties"]["voices"]["items"]["required"] == ["notes"]
    note = score["properties"]["voices"]["items"]["properties"]["notes"]["items"]["properties"]
    assert set(note) == {"pitch", "beats", "lyric", "dyn", "rest", "say", "rate", "hz"}
    assert note["dyn"]["enum"] == ["pp", "p", "mp", "mf", "f", "ff"]
    assert tools.SCHEMAS["chimera_check"]["parameters"]["properties"]["score"] == score
    for name in ("chimera_render", "chimera_speak", "chimera_job"):
        d = tools.SCHEMAS[name]["description"]
        assert "MEDIA:" in d and "outside any code block" in d, name
    assert "chimera_job" in tools.SCHEMAS["chimera_render"]["description"]
    assert "skill chimera:music" in tools.SCHEMAS["chimera_render"]["description"]
    for v in ("male", "female", "child", "androgynous", "teen", "male-deep", "female-bright", "narrator", "whisper",
              "male-whisper", "robot", "elder"):
        assert v in tools.SCHEMAS["chimera_speak"]["description"]


# INTERFACES §2/§3/§3a: manifests and dependency declarations
def test_manifests():
    main = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
    assert re.search(r"^name: chimera$", main, re.M)
    assert re.search(r"^manifest_version: 2$", main, re.M)
    assert re.search(r'^requires_hermes: ">=0\.21\.6"$', main, re.M)
    assert set(_yaml_list(main, "platforms")) == {"linux", "macos", "windows"}
    assert set(_yaml_list(main, "config_keys")) == {"tts.voice", "tts.chimera"}
    voices = re.search(r"choices: \[\"\", ([^\]]*)\]", re.sub(r"\s+", " ", main))
    assert voices, "voice choices missing"
    listed = [v.strip() for v in voices.group(1).split(",")]
    assert listed == list(c5fakes.PRESET_TABLE), listed
    for key in ("voice:", "backend:", "workers:", "text_prep:", "music_format:", "keep_days:", "max_voices:"):
        assert re.search(rf"^  {key}", main, re.M), key

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    import tomllib

    proj = tomllib.loads(pyproject)
    assert proj["project"]["dependencies"] == ["numpy>=2.3.2,<3"]
    assert "build-system" not in proj
    assert "numba" not in pyproject.split("[tool.pytest")[0]

    accel = (ROOT / "accel" / "plugin.yaml").read_text(encoding="utf-8")
    assert re.search(r"^name: chimera-numba$", accel, re.M)
    assert _yaml_list(accel, "requires_plugins") == ["chimera"]
    accel_proj = tomllib.loads((ROOT / "accel" / "pyproject.toml").read_text(encoding="utf-8"))
    deps = accel_proj["project"]["dependencies"]
    assert len(deps) == 1 and deps[0].startswith("numba>=0.67,<1;")
    assert not re.search(r"\bnot\s*\(", deps[0]), "PEP 508 has no `not (...)`; Hermes rejects it"
    from packaging.requirements import Requirement

    marker = Requirement(deps[0]).marker
    env = {"sys_platform": "linux", "platform_machine": "x86_64", "platform_release": "6.1.0"}
    assert marker.evaluate(env)
    assert not marker.evaluate({**env, "platform_release": "5.10.0-android12"})
    assert marker.evaluate({**env, "sys_platform": "darwin", "platform_machine": "arm64"})
    assert not marker.evaluate({**env, "sys_platform": "darwin", "platform_machine": "x86_64"})
    assert marker.evaluate({**env, "sys_platform": "win32", "platform_machine": "ARM64"})
    accel_init = (ROOT / "accel" / "__init__.py").read_text(encoding="utf-8")
    assert "def register(ctx)" in accel_init and "import numba" not in accel_init
