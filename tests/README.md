# Tests

```sh
python -m pytest                      # fast suite (numpy kernel; numba tests run when numba is installed)
python -m pytest --run-slow           # adds the long renders (same as CHIMERA_RUN_SLOW=1)
```

Run from the repository root; `pyproject.toml` puts the root and `tests/` on the import path.

## Layers

Each test directory covers one layer of the engine, from the arithmetic helpers up to the Hermes binding:

| Layer | Directory | What it checks |
|---|---|---|
| 0 | `core/` | JavaScript-compatible math, random numbers, seeds, WAV writing, reference files |
| 1 | `text/`, `plan/` | text normalisation, G2P, syllables, phrase and song planning |
| 2 | `control/` | articulators, gesture score, vocal-tract geometry, control tracks |
| 3 | `dsp/` | acoustic network, glottis, decimator, both kernels (numpy and numba), voice |
| 4 | `measure/` | measurements on rendered audio: pitch, formants, levels, onsets, determinism |
| 5 | `engine/`, `music/` | engine API, worker processes, scheduler, score validation, mixer |
| 6 | `hermes/` | plugin registration, tools, TTS provider, slash command, settings, jobs and sessions, several profiles in one process, docs |

## Test IDs

Docstrings name tests with stable IDs such as `K-3.10` or `K-4.4`: the first number is the layer above, the
second numbers the test within it. A letter suffix marks a part or variant of the same check (`K-0.3n` = numba
version, `K-3.10r` = red proof of `K-3.10`). The IDs do not change when a test file is renamed or moved, so they
are safe to quote in issues and commit messages.

A "Red proof" line in a docstring names a deliberate fault that the test was shown to catch.

## Tolerances

Comparisons against the template (jamesondh/chimera-vocal-synth) use these classes; the helpers live in
`tests/support/__init__.py` and `tests/dsp/dsp_assert.py`:

| Class | Applies to | Tolerance |
|---|---|---|
| T1 | text, G2P, syllables, plan structure | exact |
| T1n | numbers in plans | <= 1e-12 relative (floor 1e-12) |
| T2 | geometry, areas, resampling, control-track floats | <= 1e-9 relative (floor 1e-6) |
| T2i | integer fields of control tracks | exact, per block |
| T3 | audio against the template or the other kernel | max abs difference <= 1e-6 x peak, length exact |
| T4 | measured values (f0, formants, levels) | stated per test |
| T5 | numba kernel against numpy kernel | state <= 1e-9 relative; audio as T3 |
| T6 | repeated renders within one kernel | bit-identical |

## Markers

| Marker | Runs when | Otherwise |
|---|---|---|
| `slow` | `--run-slow` or `CHIMERA_RUN_SLOW=1` | skipped |
| `numba` | numba is installed (only worker subprocesses import it; the test process never does) | skipped |
| `lab` | `CHIMERA_LAB_REFS` names a directory with the full-length references | skipped |
| `hermes` | `HERMES_UPSTREAM` names a Hermes checkout (interpreter: `HERMES_PYTHON`, or a venv next to or inside it) | skipped |

Select or exclude them as usual, e.g. `python -m pytest -m "not numba"`.

## Reference data

- `golden/`: fixtures copied from the template's own test suite (provenance and checksums in `golden/README.md`).
- `vectors/`: values exported from the template by `tools/vector_export` (format in `vectors/README.md`).
- Full-length references (template renders, complete control tracks, audio) are too large for the repository and
  contain audio, so they live in a separate directory named by `CHIMERA_LAB_REFS`. Its layout is described in
  `golden/README.md` ("Full-length references") and `vectors/README.md`; `tools/vector_export` regenerates the
  exported part. Without the variable the `lab` tests are skipped and the rest of the suite is unaffected.
