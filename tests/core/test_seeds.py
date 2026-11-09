"""K-0.9: per-voice and per-window seeds (voice 0 and window 0 keep the template seeds, the others derive theirs)."""

import pytest

from chimera_vocal.core.types import Seeds


def test_template_seeds():
    assert Seeds() == Seeds(12345, 777, 4242)
    assert Seeds.for_voice(0) == Seeds() and Seeds.for_voice(0, None) == Seeds()
    assert Seeds.for_voice(3, 0) == Seeds()          # seed 0 reproduces the template
    base = Seeds.for_voice(2)
    assert Seeds.for_window(base, 0) is base


def test_derived_seeds_distinct_and_nonzero():
    for comp in ("net", "glottis", "drift"):
        vals = [getattr(Seeds.for_window(Seeds(), k), comp) for k in range(1, 1001)]
        assert all(0 < v <= 0xFFFFFFFF for v in vals), comp
        assert len(set(vals)) == len(vals), comp
        assert getattr(Seeds(), comp) not in vals, comp
    voices = [Seeds.for_voice(i) for i in range(1, 1001)]
    assert len(set(voices)) == len(voices)
    assert all(v.net != v.glottis != v.drift for v in voices)


def test_deterministic_across_processes():
    """Fixed values: derivation must not depend on hash() / PYTHONHASHSEED."""
    import subprocess
    import sys
    from support import REPO_ROOT

    code = "from chimera_vocal.core.types import Seeds; print(Seeds.for_voice(5), Seeds.for_window(Seeds(), 7))"
    outs = set()
    for hs in ("1", "2"):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT,
                           env={"PYTHONHASHSEED": hs, "PYTHONPATH": str(REPO_ROOT)})
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout)
    assert len(outs) == 1
    assert outs.pop().strip() == f"{Seeds.for_voice(5)} {Seeds.for_window(Seeds(), 7)}"


def test_seed_override_and_errors():
    assert Seeds.for_voice(1, 9) == Seeds.for_voice(9) == Seeds.for_voice(4, 9)
    with pytest.raises(ValueError):
        Seeds.for_voice(-1)
    with pytest.raises(ValueError):
        Seeds.for_window(Seeds(), -1)
