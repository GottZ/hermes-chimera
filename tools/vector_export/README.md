# Vector exporters (template engine → `tests/vectors`)

Tools that export the conformance vectors in `tests/vectors/` from the Rust engine of the template
<https://github.com/jamesondh/chimera-vocal-synth> at commit `092b16ce12b927d473c884ca2dfae53ce1b995df`
(crate `crates/chimera`). The template is MIT-licensed, "Copyright (c) 2026 Jameson Hodge"; `crate-chimera.patch`
modifies a copy of its source and the exporters link against it, so that notice applies to them as well (full
text in the plugin's [`LICENSE`](../../LICENSE)). Nothing here is part of the plugin at run time.

| Path | What |
|---|---|
| `exporter/` | Rust binary `chimera-s0-exporter`, linked against the **unmodified** template crate, public API only: `rng`, `decimator`, `glottis`, public part of `topology`, the four track scenarios (descriptions and the template's audio) |
| `track-exporter/` | Rust binary `chimera-s0-track-exporter`, linked against the patched crate copy: private state (RNG states, network/glottis/voice internals, `apply_source` mask), the V3 control tracks with `noise_inherit`, and the self-check (kernel emulation from `noise_inherit`, the REVIEW-1 P1 counter-check) |
| `common/scenarios.rs` | shared by both binaries (compiled against each one's `chimera`): scenarios, source sets, bodies |
| `crate-chimera.patch` | the instrumentation of the crate copy, as a diff against `crates/chimera/{Cargo.toml,src}` of the template: read-only accessor modules `src/chimera_ext/*.rs`, one `#[path]` line per touched file, four hook calls in `HumanVoice::control`, standalone `Cargo.toml`. With no recorder started the copy renders bit-identically to the template |
| `prepare.py` | creates a work directory: copy of this directory, `upstream` (symlink to the template checkout), `crate-chimera` (template crate + patch, applied by a strict built-in diff reader) |
| `assemble.py` | selects, cross-checks and writes `tests/vectors/{rng,decimator,glottis,topology,track-*}.json` and `manifest-rust.json` |
| `verify.py` | re-reads the vectors as a consumer would and compares them with the full tracks |
| `compare_shift.py` | evidence for the S1-V3fix correction: compares corrected full tracks with pre-fix ones (needs the old files) |
| `r1/` | Rust binary `chimera-vectors` (public API): G2P, phrase and song plans, geometry, held vowels, "Hello world.", golden self-check, libm probe — the raw material of `audio-head.json`, `g2p.json`, `plan_phrases.json`, `plan_song.json`, `geometry9.json` and `chimera_vocal/dsp/selftest.json` |

## Re-creating the vectors

Requirements: Rust (tested with rustc 1.99.1; the release profile is the template's: opt-level 3, LTO fat,
codegen-units 1, panic abort), Python 3 (standard library), git. Run everything **outside** this repository; build
output must not land in it (`CARGO_TARGET_DIR`).

```sh
git clone https://github.com/jamesondh/chimera-vocal-synth "$UPSTREAM"
git -C "$UPSTREAM" checkout 092b16ce12b927d473c884ca2dfae53ce1b995df
python3 -I tools/vector_export/prepare.py --upstream "$UPSTREAM" --work "$WORK"
cd "$WORK"
export CARGO_TARGET_DIR="$WORK/target"
cargo build --release --locked --manifest-path exporter/Cargo.toml
cargo build --release --locked --manifest-path track-exporter/Cargo.toml
cargo build --release --locked --manifest-path r1/Cargo.toml
mkdir -p out "$CHIMERA_LAB_REFS/audio" "$CHIMERA_LAB_REFS/tracks"
"$CARGO_TARGET_DIR/release/chimera-s0-exporter" upstream out "$CHIMERA_LAB_REFS/audio"
"$CARGO_TARGET_DIR/release/chimera-s0-track-exporter" upstream out "$CHIMERA_LAB_REFS/audio" "$CHIMERA_LAB_REFS/tracks"
python3 -I assemble.py . "$CHIMERA_LAB_REFS" <plugin>/tests/vectors upstream
python3 -I verify.py <plugin>/tests/vectors "$CHIMERA_LAB_REFS"
CHIMERA_UPSTREAM_HEAD=092b16ce12b927d473c884ca2dfae53ce1b995df "$CARGO_TARGET_DIR/release/chimera-vectors" upstream r1-out
```

- `$CHIMERA_LAB_REFS` is the directory of the full-length lab references (the same variable the `lab` tests read).
  The exporters write the template's renders (`audio/track_<scenario>.f32`, `audio/stop_male_text.f32`) and the full
  tracks (`tracks/track-<scenario>.full.json`, `glottis-next.full.json`, `topology.full.json`) there.
  `assemble.py` compares three of the renders with independent references when they exist
  (`r1/voice_male_110hz_aa-back_1s.f32`, `r1/speech_male_hello_world.f32` — written by `r1/` — and the template
  CLI's `ref-audio/song-female/twinkle-twinkle-female.wav`); missing references are skipped and noted in the manifest.
- `assemble.py` always writes the four `track-*.json`; `decimator.json`, `glottis.json` and `topology.json` are
  written when their numbers equal the export (text fields may change). `rng.json` is kept trimmed in the
  repository (see `tests/vectors/README.md`, "Repository trims"): only its `source` text is updated in place, and
  the script checks that every stored array is a prefix of the export.
- The output is deterministic: two runs from the same template commit produce byte-identical files.
- `r1-out/` holds the raw R1 files (`*.json` pretty-printed, `*.f32` raw little-endian float32); the repository files
  listed in the table were derived from them (compact JSON, 4 800-sample audio heads); their `what`/`format` keys
  describe the conversion.
