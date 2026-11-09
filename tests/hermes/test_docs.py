"""C5 documents: README (DESIGN §4.7/§7, catalog rule 13), LICENSE (D-05), THIRD_PARTY_NOTICES (INTERFACES §1),
skill chimera:music (DESIGN §9, INTERFACES §13). Written in S2; these tests state what they must contain."""

from __future__ import annotations

import hashlib
import json
import re

import pytest

import c5fakes

ROOT = c5fakes.REPO_ROOT
TEMPLATE_COMMIT = "092b16ce12b9"


def _read(rel: str) -> str:
    p = ROOT / rel
    assert p.is_file(), f"{rel} missing"
    return p.read_text(encoding="utf-8")


def test_license():
    text = _read("LICENSE")
    assert text.startswith("MIT License")
    assert "Copyright (c) 2026 Jameson Hodge" in text and "Copyright (c) 2026 Jan-Stefan Janetzky" in text
    assert "WITHOUT WARRANTY OF ANY KIND" in text


def test_third_party_notices():
    text = _read("THIRD_PARTY_NOTICES.md")
    assert "jamesondh/chimera-vocal-synth" in text and TEMPLATE_COMMIT in text
    assert "Copyright (c) 2026 Jameson Hodge" in text and "MIT" in text
    assert "Carnegie Mellon University" in text and "CMUdict" in text
    assert "Redistributions of source code must retain" in text, "CMUdict licence text must be verbatim"
    for name in ("cmudict.txt", "vowel_targets.json", "consonant_targets.json"):
        assert name in text, name
        p = ROOT / "chimera_vocal" / "data" / name
        if p.is_file():  # data files belong to C1; checked once they are in the tree (S3)
            assert hashlib.sha256(p.read_bytes()).hexdigest() in text, f"sha256 of {name} not in the notices"


def test_readme():
    text = _read("README.md")
    low = text.lower()
    heads = [h.lower() for h in re.findall(r"^#{1,3} (.+)$", text, re.M)]
    for words in (["install"], ["restart"], ["text-to-speech", "read", "speak"], ["voice"], ["chimera-numba", "faster"],
                  ["music", "sing"], ["performance", "speed"], ["settings", "configuration"],
                  ["disclosure", "what it does"], ["limit"], ["licen"]):
        assert any(any(w in h for w in words) for h in heads), f"README needs a section about {words[0]}"
    assert re.search(r"^# chimera\b", text, re.M), "README title is the plugin name"
    assert "hermes plugins install GottZ/hermes-chimera --yes-deps" in text
    assert "hermes plugins install 'GottZ/hermes-chimera#accel' --yes-deps" in text, \
        "the companion id must be quoted (zsh treats # as a glob character)"
    assert "<owner>" not in text and "<repo>" not in text
    assert not re.search(r"plugins install\s+/", text), "a local path does not install (S0-4a b2)"
    assert "hermes config set tts.provider chimera" in text
    assert "Capabilities" in text and "Text-to-Speech" in text
    assert "/chimera voice" in text and "/chimera status" in text
    for v in c5fakes.PRESET_TABLE:
        assert f"`{v}`" in text, v
    for needle in ("ffmpeg", "cache/audio/chimera", "cache/chimera", "subprocess", "oversized_file",
                   "plugins.isolation: host", "english", "jnorthrup/hermes-plugin-chimera", "network"):
        assert needle.lower() in low, needle
    assert "macos x86_64" in low and "musl" in low


def _frontmatter(text: str) -> dict:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m, "SKILL.md needs YAML frontmatter"
    out = {}
    for line in m.group(1).splitlines():
        k, sep, v = line.partition(":")
        if sep and not line.startswith(" "):
            out[k.strip()] = v.strip().strip('"')
    out["_raw"] = m.group(1)
    return out


def test_skill(registered):
    text = _read("skills/music/SKILL.md")
    assert len(text.splitlines()) <= 250 and len(text.encode("utf-8")) <= 12_000
    fm = _frontmatter(text)
    reg = registered.skills["music"]["frontmatter"]
    assert fm["name"] == reg["name"] == "music"
    assert fm["version"] == reg["version"]
    assert fm["description"] == reg["description"]
    assert "requires_tools" in fm["_raw"] and "chimera_render" in fm["_raw"]
    for needle in ("chimera_check", "chimera_render", "chimera_job", "MEDIA:", "lead_in", "say", "dyn", "start",
                   "pan", "Row, Row, Row Your Boat", "Capabilities", "/chimera voice", "chimera-numba",
                   "restart"):
        assert needle in text, needle
    for rng in ("D2", "C6", "A3", "A5"):
        assert rng in text, rng
    blocks = re.findall(r"```json\n(.*?)\n```", text, re.S)
    scores = [json.loads(b) for b in blocks]
    assert len(scores) >= 3, "three worked examples (SATB, canon, spoken + sung)"
    assert any(len(s["voices"]) == 4 for s in scores), "SATB example"
    assert any(sorted(v.get("start", 0) for v in s["voices"]) == [0, 8, 16] for s in scores), "canon example"
    assert any(any("say" in n for v in s["voices"] for n in v["notes"]) for s in scores), "spoken slot example"
    for s in scores:
        assert set(s) >= {"bpm", "voices"} and all("notes" in v for v in s["voices"])


def test_skill_examples_validate_with_the_real_engine():
    """S3 hook: once the engine package is merged, every skill example must pass the real validation."""
    try:
        from chimera_vocal import get_engine  # noqa: F401 - absent in the C5 tree until S3
    except ImportError:
        pytest.skip("engine API not in this tree (S3)")
    from chimera_vocal import get_engine

    text = _read("skills/music/SKILL.md")
    eng = get_engine()
    for block in re.findall(r"```json\n(.*?)\n```", text, re.S):
        report = eng.check_score(json.loads(block))
        assert report.ok, [i.message for i in report.issues]
