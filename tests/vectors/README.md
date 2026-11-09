# Test vectors (Rust export of the template engine)

Conformance vectors exported from the template's Rust engine for the port's tests. This README covers the files
written by the S0 vector export: `rng.json`, `decimator.json`, `glottis.json`, `topology.json` and
`track-<scenario>.json`. The other files in this directory (`audio-head.json`, `g2p.json`, `plan_phrases.json`,
`plan_song.json`, `geometry9.json`) come from `tools/vector_export/r1` and describe themselves (`what`/`format` keys).

## Provenance and license

- Template: <https://github.com/jamesondh/chimera-vocal-synth>, commit `092b16ce12b927d473c884ca2dfae53ce1b995df`,
  crate `crates/chimera`. MIT License, "Copyright (c) 2026 Jameson Hodge"; these numbers are derived from it under
  that license (full text in the plugin's [`LICENSE`](../../LICENSE)).
- Exporter: [`tools/vector_export/`](../../tools/vector_export/README.md) (how to re-create the files is described there):
  - `exporter/` builds against the **unmodified** template crate and uses its public API only;
  - `track-exporter/` builds against a copy of the template crate patched with `crate-chimera.patch`: read-only
    accessors (`src/chimera_ext/`) and a control-step recorder (four hook calls in `HumanVoice::control`). The copy
    renders all four track scenarios bit-identically to the unmodified crate (see "Self-check");
  - `assemble.py` selects, cross-checks (public API vs. copy, 199 checks) and writes these files (it keeps the
    integrator's trim of `rng.json`, see the last section); `verify.py` re-reads them as a consumer would.
- Build profile as the template workspace (release, opt-level 3, LTO fat, codegen-units 1), rustc 1.99.1.
- Full-length versions live in the lab directory `$CHIMERA_LAB_REFS` (not part of the repository):
  `tracks/track-<scenario>.full.json`, `tracks/glottis-next.full.json`, `tracks/topology.full.json`,
  `audio/track_<scenario>.f32`, `audio/stop_male_text.f32`.

## Conventions

- Compact JSON, UTF-8. f64 values are written as Python `repr` (shortest decimal that reads back to the same
  double), integers as JSON integers, booleans as JSON booleans. Parse with any correctly rounding JSON parser.
- All indices are 0-based. Node and branch indices are those of the template's `AirwayNetwork` (creation order).
- Units: CGS (cm, cm², dyn/cm², s) and Hz, as in the template (`physics.rs`; 1 cmH2O = 980.665 dyn/cm²).
- Glottis parameter vectors always have 17 entries in `GlottisParams` field order (`sources/glottis.rs:13-49`):
  `f0, amplitude, abduction, chink, open_quotient, skew, closure_sharpness, effort, jitter, shimmer,
  vibrato_depth, vibrato_rate, drift, fry, diplophonia, closure, chaos` (= `core.glottis_params.G_*`).
- Bodies: `male`, `female`, `child` (the presets) and the mixes `{"female": 0.5, "child": 0}` and
  `{"female": 0.5, "child": 0.5}` (`BodyMix::body()`, `anatomy/human.rs:322-383`).

## `rng.json` (V1) — `math.rs:175-260`

```
{"source": str,
 "rng":   [{"seed": u32, "next": [1000 f64], "states": [1000 u32], "gauss": [250 f64], "bipolar": [100 f64]}, ...],
 "drift": [{"control_rate": f64, "base_hz": f64, "seed": u32, "steps": [1000 f64]}, ...]}
```
- Seeds 12345 (network noise), 777 (glottis jitter), 4242 (glottis drift), 1, 0, 4294967295, 2654435769.
- `next[i]`: the i-th `Rng::next()` of `Rng::new(seed)`; `states[i]`: the internal state `s` right after that call.
  `Rng::new` maps seed 0 to 1 (`math.rs:188-190`), so the state sequence is `(s0 + (i+1)·0x6d2b79f5) mod 2^32`
  with `s0 = seed or 1` (checked for all 7 000 states).
- `gauss` and `bipolar` each start from a fresh `Rng::new(seed)`.
- `drift`: `Drift::new(control_rate, base_hz, seed).step()` 1 000 times. `(3000, 0.3, 4242)` is the glottis drift
  at an internal rate of 96 kHz (`Drift::new(fs / 32, 0.3, 4242)`, `glottis.rs:119`); `(1500, 0.3, 4242)` is the
  same at 48 kHz internal (oversample 1); `(750, 1.0, 1)` is an extra configuration.

## `decimator.json` (V2d) — `airway/decimator.rs`

- `m2.coeffs`: the 49 taps of `Decimator::new(2)` (= `design_lowpass(49, 0.46/2, 8.5)`, normalised to sum 1; read
  from the decimator itself and checked equal). `initial_pos` 0, `initial_count` 0, `buf_len` 98 (two copies of the
  ring).
- `impulse_out`: the first 8 outputs for the input 1.0 at push 0 followed by zeros; `impulse_push_index`: the push
  (0-based) at which `push()` returned true. With m = 2 the first output appears at push 1 (not 0) and output k
  equals `coeffs[2k+1]` (`decimator.rs:63-80`: the write position starts at 0 and the phase counter must reach m).
- `m1`: `Decimator::new(1)` is a 1-tap pass-through (`coeffs [1.0]`), output at every push.

## `glottis.json` (V5) — `sources/glottis.rs`, `voice/human_voice.rs:203-341`

Part `params` (12 source sets that cover every branch of `apply_source`):
```
{"field_order": [17 str], "default_glottis": [17 f64],
 "params": [{"name": str, "body": "male|female|child|mix", "body_mix": {"female": f64, "child": f64},
             "body_scale": f64, "source": {VoiceSource, serde camelCase, 12 fields},
             "glottis": [17 f64], "written": [17 bool]}, ...], ...}
```
- `glottis` = `voice.glottis.params` after `HumanVoice::new(48000, body_mix.body(), 2)` followed by
  `set_source(patch)`; `source` is the voice's `VoiceSource` after the patch. `body_scale` = `airway.body.scale`.
- `written[i]` is true when `apply_source()` writes field i: measured by running it on parameters that were all set
  to a sentinel. It is the same for all sets: everything except `f0`, `closure` and `chaos` (those keep their
  previous value: `f0` and `closure` are set per control step, `chaos` never by the voice).
- Sets: `default` (male), `voiced-false` (female), `breathiness-0.6` (child), `tension-0.9` (mix .5/0, branch
  tension > 0.6), `tension-0.1` (mix .5/.5), `brightness-0.9` (male), `brightness-0.0` (female), `roughness-0.8`
  (child, diplophonia branch), `pressure-neg6` (mix .5/0, |pressure|), `pressure-1` (mix .5/.5, effort clamps to 0),
  `pressure-20` (male, effort clamps to 1), `fry-vibrato` (female; fry 0.5, vibratoDepth 50, vibratoRate 5.5, drift 0).

Part `next` (`ParametricGlottis::next(&mut self, dp: f64) -> f64`, `glottis.rs:148`):
```
{"next_fs": 96000.0, "next_n": 2000, "next_dp_kinds": {...}, "next_rule": str, "next_index_rule": str,
 "next_index": [307 int],
 "next": [{"name": str, "from": <params name>, "overrides": [[field, f64], ...], "glottis": [17 f64],
           "dp_kind": "const8|ramp6", "threshold": f64, "dp": [f64 per next_index], "area": [...],
           "f0_now": [...], "voicing": [...]}, ...]}
```
- Procedure: `g = ParametricGlottis::new(96000)`; `g.params = glottis`; for i in 0..2000: `area_i = g.next(dp_i)`.
  `area` is the return value (equal to the telemetry field `g.area`), `f0_now` and `voicing` are read after the call.
- `dp`: `const8` = 8·980.665 for every sample; `ramp6` = ((6·980.665)·i)/1999, evaluated left to right.
- Sets: `voiced-constant` (from `fry-vibrato`, f0 330; fry > 0.3 double pulse, vibrato), `threshold-ramp` (from
  `roughness-0.8`, f0 440, chaos 0.7; diplophonia and logistic map, dp crosses the phonation threshold),
  `voiceless` (from `voiced-false`, f0 205, fry 0.5, vibrato 50 c at 5.5 Hz, closure 0.3).
- **Truncated:** only samples 0..63, every 8th from 64 and sample 1999 are stored (`next_index`). Run all 2000 samples
  and compare at those indices. All 2000 samples and the private glottis state (phase, cycle index, RNG states, drift
  filter, smoothers) after 1000 and 2000 samples: `tracks/glottis-next.full.json` in the lab.

## `topology.json` (V2) — `voice/human_airway.rs:69-154`, `airway/network.rs:426-617`

One entry per body in `bodies` (labels `male`, `female`, `child`, `mix-f050`, `mix-f050-c050`), all at an internal
rate of 96 000 Hz (`HumanVoice::new(48000, body, 2)`):
- counts `n_ph`, `n_or`, `n_trachea`, `n_nasal`, `n_fossa`, `n_nodes`, `n_branches`, `n_rad`;
  `ph_nodes`/`or_nodes` (node per section), `glottis_port`, `velum_port` (branch indices), `lungs`,
  `sub_node` (= trachea last node), `supra_node` (= pharynx first node), `rad_lips`, `rad_nose` (radiator indices);
- `tubes`: name, index, first node, n, branch_base and the `TubeSpec` (length, `area` as `{"uniform": a}` or
  `{"profile": [...]}`, `wall` [mass, resistance, stiffness] or null, perimeter/viscous factor, wall radiation);
  tube order trachea, pharynx, oral, nasal, piriform, lungs, sinus;
- `branch_of`: `HumanAirway::branch_of(tube, i)` for every section i of trachea, pharynx and oral (index clamped to
  0..n-2, `human_airway.rs:156-159`);
- `branches`: `a`, `b`, `kind` (0 series, 1 port) per branch; `ports`: per port branch its spec and current/target area;
  `radiators`: radiator nodes;
- `nodes`: `fixed`, `junction` (≥ 3 connections), `has_wall` as node lists; per node `node_tube`, `rad_of`
  (radiator index or -1), `wall_m`, `wall_r`, `wall_k`, `wall_perim`, `wall_rad`, `visc_f`; neighbour lists in CSR
  form (`nbr_start` [n_nodes+1], `nbr`);
- `constants`: network and airway constants (closed_area, min_node_area, squeeze_area, viscous_omega, vol_lp_coef,
  noise gains, slew-limit rates, output_gain, …);
- `initial`: the state right after the constructor (rule in `initial_rule`): node `area`, `amin`, `dx`,
  `tube_length`, the airway's `ph_areas`/`or_areas`/`ph_min`/`or_min`/`ph_len`/`or_len`, frame velum port and
  teeth position, the initial noise list `[src, fwd, bwd, gain, lp_coef, hp_coef]`, network RNG state, glottis
  parameters and complete private glottis state, voice scalars (ctrl_every 128, output_gain, dc_coef, limiter
  constants, …), decimator position, articulator state and the `VoiceSource`. Every field in `zero_fields` is 0.
  `vol` and `inv_a` are not stored (derived, see `initial_rule`). The complete public and private dumps:
  `tracks/topology.full.json` in the lab.

## `track-<scenario>.json` (V3) — control track per control step (`voice/human_voice.rs:426-509`)

**Block:** block b is the b-th call of `HumanVoice::control()`. It runs before internal sample `b·128`, i.e. before
output sample `64·b` (oversample 2, `ctrl_every` = 64·m). **Row b = state after the complete control step b, as the
kernel consumes it during block b** — every field, including the breath and pitch part of the step (gate, `p_lung`,
effort, f0 glide, `human_voice.rs:477-501`); the recorder takes the row as the last statement of `control()`, and
`render()` changes none of these values between `control()` and the block's first sample (`human_voice.rs:354-366`).
The row `initial` stays the state after the constructor. `n_blocks` = ceil(n_samples·2/128).

*Correction:* track files exported before this fix took the row after the contact re-apply but before the
breath/pitch part, so `p_lung`, `glot_par[f0]`, `glot_par[effort]` (and `gate`, `f0_glide` in the full tracks)
carried the values of step b−1. The current files need no shift; the new row b equals the old row b+1 in those
fields and the old row b in every other field (checked for every block, `tools/vector_export/compare_shift.py`).

| key | content |
|---|---|
| `scenario` | complete input: body, body_mix, fs_out 48000, oversample 2, host_block 128, `n_samples`, `ops` (each `{"at": sample, "op": "set_source" \| "set_articulation" \| "set_gate" \| "schedule", ...}` with the patch / gestures in the template's serde form), `ops_rule`, `provenance` (which CLI path it reproduces) |
| `n_ph`, `n_or`, `ctrl_every`, `block_rule`, `n_blocks` | as above |
| `initial` | one row for the state after `HumanVoice::new` (before the ops at sample 0 and before the first control step): all float fields below plus `noise_n`, `noise_idx`, `noise_inherit` (all -1) |
| `int_runs` | `[[first_block, noise_idx, noise_inherit], ...]`: the integer fields of **every** block, run-length coded; a run holds from `first_block` up to the next run's first block (last run: up to `n_blocks`). `noise_idx` = list of `[src, fwd, bwd]` branch indices of the block's final noise list (`noise_n` = its length, 1..3); `noise_inherit` = one int per source |
| `reapply_blocks` | blocks in which the contact constraint moved an articulator (`enforce_contacts()` true → `airway.reapply`, a second `set_noise_sources` call) |
| `list_a` | `{block: [[src, fwd, bwd], ...]}`: the intermediate list A of the first `set_noise_sources` call, for the re-apply blocks whose A differs from the final list in its (src, fwd) pairs |
| `p1_critical_blocks`, `noise_inherit_p1_at_critical` | blocks where `noise_inherit` differs from "final list against the previous block" (REVIEW-1 P1-01), and that wrong value |
| `float_blocks` | the blocks (ascending) whose float fields are stored; `float_selection` says why |
| `float` | per float block: `tube_area` [n_ph + n_or] (pharynx then oral, after the slew limit), `tube_min` [n_ph + n_or] (sub-section minima), `tube_len` [ph_len, or_len] (cm), `velum_area` (velum port target, cm²; never negative here, the network stores js_max(0, ·)), `noise_par` [[gain, lowpass_hz, highpass_hz] per source], `glot_par` [17] (the glottis parameters the block runs with, incl. f0 and effort of this step), `p_lung` (lung-pressure target of this step, dyn/cm²; `p_lung = pressure·980.665·gate`), `clock` (voice clock after the step, s) |

**`noise_inherit`** (P1-01): for source j of the block's final list, the index k in the **previous block's final
list** whose filter state (hp, lp1, lp2) source j continues with; -1 means it starts at 0. It composes both
`set_noise_sources` calls of a block with contact re-application: list A inherits from the previous list P, the
final list B from A, each by the **first** source with the same (src, fwd) (`network.rs:727-745`; the new list
replaces the old one at once, `noise_next` is only scratch space). Block 0 refers to the list of `initial`.
To fill a `TrackChunk` (INTERFACES §6.3): pad `noise_idx` with -1 and `noise_par` with 0 to three sources,
`noise_inherit` with -1. `tube_area`/`tube_min` are the raw airway arrays; the network applies
`area_target = js_max(0, a)`, `amin_target = js_max(0, js_min(a, m))` (`network.rs:633-648`). Noise filter
coefficients follow from the Hz values: `lp_coef = 1 - exp(-2π·lowpass_hz/fs)`, `hp_coef = exp(-2π·highpass_hz/fs)`
with fs = 96 000 (checked bitwise against the network for every block).

| scenario | provenance | samples | blocks | float rows in the repo | re-apply blocks | P1-critical blocks |
|---|---|---|---|---|---|---|
| `sustain-aa-male` | `held_vowel("ɑ")` of `tools/vector_export/r1` (= its `voice_male_110hz_aa-back_1s.f32`, bit-identical) | 48 000 | 750 | 0–79 | 0 | none |
| `hello-male` | `render text "Hello world." male` (= `speech_male_hello_world.f32` of `tools/vector_export/r1`, bit-identical) | 57 845 | 904 | 0–79, 625–627 | 170 | 626 |
| `stop-male` | `render text "Stop at the top, Pat, and take the tea to the cat." male` | 186 134 | 2 909 | every 49th, 1856–1858 | 749 | 1857 |
| `song-female` | `render song female`, demo song "Twinkle, twinkle" (byte-identical to the template CLI's output `twinkle-twinkle-female.wav`) | 494 400 | 7 725 | every 123rd | 725 | 158–160, 609, 629, 635, 642, 1055, 1057, 1058, 1060, 1509, 1529, 1535, 1542, 2905, 6030 |

**Truncated:** the integer fields are complete for every block of every scenario; float fields only for
`float_blocks` (blocks 0–79 cover the 4 800-sample audio heads). Every field of every block, plus clock, gate,
glide f0, `p_lung_now` (lung pressure applied before the block's first sample) and `p_lung_step` (= (p_lung −
p_lung_now)/128, the per-sample ramp), articulator position/velocity, contact depths, `velum_target`, noise
coefficients and the list A of every re-apply block: `$CHIMERA_LAB_REFS/tracks/track-<scenario>.full.json` (same
key names under `blocks`, one entry per block).
The audio the tracks belong to: `audio/track_<scenario>.f32` (raw little-endian float32, mono, 48 kHz).

## Self-check of the track export

In the crate copy a mode sets, after every control step, the filter states of the final noise list **only** from
`noise_inherit` and the saved states of the previous block's final list (what a port kernel does with the track).
Full-length audio of all four scenarios is bit-identical to the unmodified template (first bit difference: none),
and in record mode every recorded `noise_inherit` predicts the actual filter states bitwise. Counter-check with the
REVIEW-1 emulation "final list against the previous block": `hello-male` first exceeds 1e-6 × peak at sample
40 066 (first bit difference 40 064 = start of block 626), max |Δ| = 1.27e-2 × peak, 10-ms envelope ≤ 0.0006 dB;
`song-female` from sample 67 523, max 8.4e-3 × peak, envelope ≤ 0.057 dB; `stop-male` stays bit-identical even
though block 1857 is P1-critical. Numbers and checksums: `manifest-rust.json`, which `tools/vector_export/assemble.py`
writes into its work directory.

Playback check: a port kernel (CPython reference of the acoustic kernel) driven by these tracks **without any shift**
reproduces both 4 800-sample heads bit-identically, and all four scenarios over their full length including the last
block with no sample deviating by more than 1e-6 × peak (`sustain-aa-male` bit-identical; `hello-male`, `stop-male`,
`song-female`: 1, 6 and 8 single samples differ by at most 2.5e-14, 6.9e-10 and 2.0e-8 × peak — most likely
last-bit differences of the platform math library against the template's `libm` crate: the identical deviations
appear with a different kernel form).

## Repository trims by the integrator (S0)

To keep the JSON test data inside the TESTPLAN §2.1 budget (≤ 3 MB), I0 shortened three files after export; the
full versions stay in the lab (`$CHIMERA_LAB_REFS`), sha256 of each full file in the `note` field / `manifest.json`:

- `rng.json`: per seed the first 250 `next`, 50 `states`, 60 `gauss`, 20 `bipolar`; per drift the first 200 `steps`
  (`states` follow `s_n = s_0 + n·0x6d2b79f5 mod 2^32`, checked for all 7 000 in the full file). Full:
  `tracks/rng.full.json`.
- `geometry9.json`: the array members of `frame` (grid, outlines, teeth, …) are removed, the scalar members kept;
  `resample18` and `airway96k` are complete. Grids and outlines are covered by the 114 postures of
  `tests/golden/geometry.json`. Full: `r1/geometry.json`.
- `audio-head.json`: the `voice_male_110hz_aa-back` case carries `head_ref` instead of `head`; the identical 4 800
  samples are `chimera_vocal/dsp/selftest.json` → `expected`. Use `support.audio_head(name)`.
