# Golden fixtures

Text copies of the conformance fixtures of the template engine, used by the port's golden tests.

## Provenance and license

- Template: <https://github.com/jamesondh/chimera-vocal-synth>, commit `092b16ce12b927d473c884ca2dfae53ce1b995df`,
  directory `crates/chimera/tests/golden/`.
- The template is licensed under the MIT License, "Copyright (c) 2026 Jameson Hodge". These fixtures are
  copied or derived from it under that license. The full license text, including the template's copyright
  notice, is in this plugin's [`LICENSE`](../../LICENSE).
- The template's own fixture README says what they are: the `voice-*` audio, `geometry.json` and the plans in
  `planning.json` are snapshots of the template's Rust engine; the phonetics text fixtures record its former
  TypeScript engine. The template checks numbers to 1e-9 relative and audio sample for sample.

### CMU Pronouncing Dictionary

The dictionary pronunciations in the G2P fixtures — the `dict` cases of `phonetics-g2p.json` and
`phonetics-phrases.json` (and of `tests/vectors/g2p.json`), and every plan built from them in `planning.json` — come from
the CMU Pronouncing Dictionary as shipped by the template (`public/data/cmudict.txt.gz`):
"Copyright (C) 1993-2015 Carnegie Mellon University. All rights reserved." It is distributed under a BSD-style
(2-clause-like) license that permits redistribution with the copyright notice, conditions and disclaimer retained; the
full text is the template's `public/data/CMUDICT-LICENSE.txt` and is reproduced in this plugin's
`THIRD_PARTY_NOTICES.md`.

## Formats

- JSON is compact (no whitespace between tokens), UTF-8, no trailing newline. f64 numbers are written as Python `repr`
  (shortest round-trip), so parsing gives the exact value.
- `voice-<name>.head.json` hold float32 samples as the **shortest decimal that reads back to the same float32**:
  parse as f64 (any JSON parser), then convert to float32 (`numpy.asarray(head, dtype=numpy.float32)`) — the
  result is bit-identical to the template's `.f32` file. `peak_full` and `peak_head` are max |x| over the whole
  file and over the head (f64 repr of the float32 value).
- Fields of a head file: `name` (scenario in `voice.json`), `source`, `source_sha256`, `total_samples`, `fs`
  (output rate), `oversample`, `peak_full`, `peak_head`, `head_samples` (4 800), `head`.
- Audio tolerance (T3): max |Δ| ≤ 1e-6 × peak, length exact. On a head, `peak_head` is the peak of the compared
  samples (the template's `compare_audio` rule, `crates/chimera/tests/common/mod.rs`); `peak_full` gives the
  template's whole-file criterion restricted to the head.

## Files

| File | Source (template) | sha256 source | sha256 file | Conversion | Missing here / full version |
|---|---|---|---|---|---|
| `phonetics-ipa.json` | `crates/chimera/tests/golden/phonetics-ipa.json` | `7266a3ce987991ae9c573d422c69fa1c49b134b9c61043b5c368762f36132871` | `7266a3ce987991ae9c573d422c69fa1c49b134b9c61043b5c368762f36132871` | byte-identical copy | nothing |
| `phonetics-g2p.json` | `crates/chimera/tests/golden/phonetics-g2p.json` | `6e14b2abc0eaa64036044bc163852a45ca9ec35a26c9992e2dd973dd5583359d` | `6e14b2abc0eaa64036044bc163852a45ca9ec35a26c9992e2dd973dd5583359d` | byte-identical copy | nothing |
| `phonetics-phrases.json` | `crates/chimera/tests/golden/phonetics-phrases.json` | `b2b51caaba0e6bfaf4881704744190806ddade145ea49700e0bffcc7fa5c7a5d` | `b2b51caaba0e6bfaf4881704744190806ddade145ea49700e0bffcc7fa5c7a5d` | byte-identical copy | nothing |
| `phonetics-syllables.json` | `crates/chimera/tests/golden/phonetics-syllables.json` | `c4369c16b06420b4d5849caf0c1c4efbdcf2e805a0a84d2adfd1638ba18905da` | `c4369c16b06420b4d5849caf0c1c4efbdcf2e805a0a84d2adfd1638ba18905da` | byte-identical copy | nothing |
| `phonetics-data.json` | `crates/chimera/tests/golden/phonetics-data.json` | `81c49f3e33d73b99599be40ab59f3dd55c6a61ad6d5ff18731232802ccf382f9` | `81c49f3e33d73b99599be40ab59f3dd55c6a61ad6d5ff18731232802ccf382f9` | byte-identical copy | nothing |
| `geometry.json` | `crates/chimera/tests/golden/geometry.json` | `c0ee688025e5fa56c056f1dca4f3589086bb1f81d3e81e2208fa483e32ffec99` | `c0ee688025e5fa56c056f1dca4f3589086bb1f81d3e81e2208fa483e32ffec99` | byte-identical copy | nothing |
| `voice.json` | `crates/chimera/tests/golden/voice.json` | `5386f4f93b3beacfb1b353166e71bd2468bd1cb254348665a23e66b684cf62c6` | `5386f4f93b3beacfb1b353166e71bd2468bd1cb254348665a23e66b684cf62c6` | byte-identical copy | nothing |
| `planning.json` | `crates/chimera/tests/golden/planning.json` | `e34fb5534c279b53c1b9d365c6a6bbb8f0ed1e23f9ab1c503d33fa99d95b22fe` | `e59e0090a0eb479bf66bb76dd0f9b01e504c3553d0955b637f24e1dff9ba426b` | compact JSON with the beatbox members removed; every other member byte-identical | `sounds`, `ready`, `notation`, `player`, `audio` (beatbox sounds, ready posture, beat notation, beat player, beatbox bar); full file: template repository at the commit above |
| `voice-sustain-male.head.json` | `crates/chimera/tests/golden/voice-sustain-male.f32` | `37d2f0fe46b775cda96d604fbccf6e1cec33953a50a06bbf06282352ee7e01d8` | `b07506a16822a181e793a2eed46b6df9a2cb1e32d94cab937f53e79f23d130c6` | first 4 800 float32 samples (of 28800) as shortest round-trip decimals, plus peaks and scenario keys | samples 4 800–28799; full file: `$CHIMERA_LAB_REFS/golden/voice-sustain-male.f32` |
| `voice-artic-sweep.head.json` | `crates/chimera/tests/golden/voice-artic-sweep.f32` | `955b66af24305f8340e710a57385a38fc0e755c2328deaea516ddf0d659e6a1b` | `960312acb379928cf74fe0cf77e031a37e6d0da4f4a48a0fa3a2a503404ea001` | first 4 800 float32 samples (of 38400) as shortest round-trip decimals, plus peaks and scenario keys | samples 4 800–38399; full file: `$CHIMERA_LAB_REFS/golden/voice-artic-sweep.f32` |
| `voice-speech-male.head.json` | `crates/chimera/tests/golden/voice-speech-male.f32` | `b5c2c12a3a7f94a7f7ba8b709fa4e9651dd858134c8ec9927a700ac7fd7a2899` | `45d23b55b62f74fd1c77f01c7d2de4044090eb1696b645a3fa0551b2d9e6112f` | first 4 800 float32 samples (of 76800) as shortest round-trip decimals, plus peaks and scenario keys | samples 4 800–76799; full file: `$CHIMERA_LAB_REFS/golden/voice-speech-male.f32` |
| `voice-speech-female-96k.head.json` | `crates/chimera/tests/golden/voice-speech-female-96k.f32` | `126a71d879ec53988122f21ffd14107156a061da31472b8342f47be514dad1de` | `ed34a36d2f9ae53ae0c831d18454bf0b2aef8738e2d931a586dfeabd7583bfe0` | first 4 800 float32 samples (of 96000) as shortest round-trip decimals, plus peaks and scenario keys | samples 4 800–95999; full file: `$CHIMERA_LAB_REFS/golden/voice-speech-female-96k.f32` |
| `voice-speech-child.head.json` | `crates/chimera/tests/golden/voice-speech-child.f32` | `27bf999d070da0ceea87258ea73270805f5896f0506ef5825bbc8ac29b9a7fb4` | `a8f36f867538dd874fbddc3559d2e4e784772c98679fd60ac2279fefb6a04840` | first 4 800 float32 samples (of 48000) as shortest round-trip decimals, plus peaks and scenario keys | samples 4 800–47999; full file: `$CHIMERA_LAB_REFS/golden/voice-speech-child.f32` |
| `voice-release-cancel.head.json` | `crates/chimera/tests/golden/voice-release-cancel.f32` | `4a602b36fe8e0bd50dbd064dc2d1768e93ba77a3d0c810a3f6a28677ec04f452` | `2d99fbf9b08080eeed676170cafee7eaee9c9ea8c17bd3af012b62f4c44d3a1e` | first 4 800 float32 samples (of 57600) as shortest round-trip decimals, plus peaks and scenario keys | samples 4 800–57599; full file: `$CHIMERA_LAB_REFS/golden/voice-release-cancel.f32` |

Not copied (outside the port's scope): `analysis.json`, `analysis-voice-frame.f32`, `beatbox-bar.f32`,
`creatures.json`, `creatures.f32`, `creature-units.json`.

## Full-length references

The repository keeps text only. Full-length renders (all six `voice-*.f32`, the template's CLI renders, the
vector exporter's renders, RMS envelopes and telemetry f0 tracks) live outside it in a directory named by the
environment variable `CHIMERA_LAB_REFS` (layout and checksums in its `README.md` and `SHA256SUMS`). Tests that
need them carry the `lab` marker and are skipped when the variable is unset.

Related: `chimera_vocal/dsp/selftest.json` (selftest expectation) and `tests/vectors/audio-head.json` (V7) hold
4 800-sample heads of the vector exporter's renders in the same number format.
