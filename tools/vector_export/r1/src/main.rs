//! Export conformance vectors from the upstream chimera engine for a Python port.
//! Public API only; nothing in the upstream tree is modified.
//!
//! Usage: chimera-vectors <upstream-repo-root> <out-dir>
//!
//! Writes (out-dir):
//!   g2p.json            English G2P: text -> phrases (with and without CMUdict)
//!   plan_phrases.json   phrase planner (male, as `chimera render text` calls it)
//!   plan_song.json      sung melody: 4 notes with syllables (male)
//!   geometry.json       vocal-tract frame, resampled areas and airway areas for a/ɑ/i per body
//!   voice_*.f32         raw little-endian f32 mono samples of a held vowel (male, 110 Hz, 48 kHz)
//!   speech_male_hello_world.f32   "Hello world." end to end, as `chimera render text` renders it
//!   golden_sustain_male.f32       this build's render of the golden scenario voice-sustain-male
//!   manifest.json       parameters, sample counts, determinism and golden-fixture checks
//!   libm_probe.f64      per function N records (x, y, f(x, y)) as little-endian f64, see libm_probe.json

use chimera::anatomy::human::{BODY_MALE, BodyMix, BodyName, HumanVocalTract, NEUTRAL};
use chimera::phonetics::english::EnglishG2P;
use chimera::phonetics::planner::{PhraseInput, PlanOptions, plan_phrases, vowel_artic, vowel_table};
use chimera::phonetics::sing::{SongOptions, SungNote, plan_song};
use chimera::phonetics::syllables::lyrics_to_syllables;
use chimera::voice::human_airway::HumanAirway;
use chimera::voice::human_voice::{HumanVoice, SourcePatch};
use serde_json::{Value, json};
use std::io::Read;

const SENTENCES: [&str; 3] = ["Hello world.", "Can you hear me?", "The quick brown fox jumps over 42 lazy dogs."];
const FS: f64 = 48000.0;
const BLOCK: usize = 128;

fn load_g2p(root: &str) -> EnglishG2P {
    // same as crates/chimera-cli/src/render.rs `g2p()`
    let path = format!("{root}/public/data/cmudict.txt.gz");
    let gz = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut text = String::new();
    flate2::read::GzDecoder::new(&gz[..]).read_to_string(&mut text).unwrap();
    let mut g = EnglishG2P::new();
    g.load_dict(&text);
    g
}

fn write_json(dir: &str, name: &str, v: &Value) {
    let path = format!("{dir}/{name}");
    std::fs::write(&path, serde_json::to_string_pretty(v).unwrap() + "\n").unwrap();
    println!("wrote {path}");
}

fn write_f32(dir: &str, name: &str, data: &[f32]) {
    let path = format!("{dir}/{name}");
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(&path, bytes).unwrap();
    println!("wrote {path} ({} samples)", data.len());
}

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn to_value<T: serde::Serialize>(x: &T) -> Value {
    serde_json::to_value(x).unwrap()
}

/// Render `n` samples in host blocks of `block`, applying `ops` (sample index, closure) first.
fn render(v: &mut HumanVoice, n: usize, block: usize, ops: &mut [(usize, &mut dyn FnMut(&mut HumanVoice))]) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let (mut i, mut oi) = (0usize, 0usize);
    while i < n {
        while oi < ops.len() && ops[oi].0 <= i {
            (ops[oi].1)(v);
            oi += 1;
        }
        let next = if oi < ops.len() { ops[oi].0.max(i + 1) } else { usize::MAX };
        let end = n.min(i + block).min(next);
        v.render(&mut out[i..end]);
        i = end;
    }
    out
}

/// A held vowel: male body, 2x oversampling, f0 110 Hz, the calibrated vowel posture, gate on at 0.
fn held_vowel(ipa: &str, n: usize, block: usize) -> Vec<f32> {
    let mut v = HumanVoice::new(FS, BODY_MALE, 2);
    v.set_source(&SourcePatch::f0(110.0));
    v.set_articulation(&vowel_artic(ipa, BodyMix::MALE).expect("vowel"), None);
    v.set_gate(1.0);
    render(&mut v, n, block, &mut [])
}

/// Inputs and outputs of the engine's transcendental functions (chimera::math, i.e. the `libm`
/// crate), so a port can measure how far its own math library is from the engine's bit for bit.
fn libm_probe(out: &str) {
    use chimera::math as m;
    const N: usize = 100_000;
    type F = fn(f64, f64) -> f64;
    let funcs: [(&str, F, (f64, f64), (f64, f64)); 12] = [
        ("sin", |x, _| m::sin(x), (-200.0, 200.0), (0.0, 0.0)),
        ("cos", |x, _| m::cos(x), (-200.0, 200.0), (0.0, 0.0)),
        ("exp", |x, _| m::exp(x), (-60.0, 10.0), (0.0, 0.0)),
        ("pow", |x, y| m::pow(x, y), (1e-3, 20.0), (-4.0, 4.0)),
        ("ln", |x, _| m::ln(x), (1e-6, 1e4), (0.0, 0.0)),
        ("log10", |x, _| m::log10(x), (1e-9, 1e4), (0.0, 0.0)),
        ("log2", |x, _| m::log2(x), (1e-6, 1e4), (0.0, 0.0)),
        ("tanh", |x, _| m::tanh(x), (-6.0, 6.0), (0.0, 0.0)),
        ("atan2", |x, y| m::atan2(x, y), (-10.0, 10.0), (-10.0, 10.0)),
        ("acos", |x, _| m::acos(x), (-1.0, 1.0), (0.0, 0.0)),
        ("cbrt", |x, _| m::cbrt(x), (-1e3, 1e3), (0.0, 0.0)),
        ("hypot", |x, y| m::hypot(x, y), (-50.0, 50.0), (-50.0, 50.0)),
    ];
    let mut rng = m::Rng::new(12345);
    let mut bytes: Vec<u8> = Vec::with_capacity(funcs.len() * N * 24);
    let mut index = Vec::new();
    for (name, f, (xa, xb), (ya, yb)) in funcs {
        index.push(json!({ "fn": name, "offsetRecords": index.len() * N, "n": N, "x": [xa, xb], "y": [ya, yb] }));
        for _ in 0..N {
            let x = xa + (xb - xa) * rng.next();
            let y = ya + (yb - ya) * rng.next();
            for v in [x, y, f(x, y)] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    std::fs::write(format!("{out}/libm_probe.f64"), &bytes).unwrap();
    write_json(out, "libm_probe.json", &json!({ "record": "x f64, y f64, result f64 (little-endian), 24 bytes", "functions": index, "source": "chimera::math (libm crate =0.2.16)" }));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = args.get(1).expect("usage: chimera-vectors <upstream-root> <out-dir>").clone();
    let out = args.get(2).expect("usage: chimera-vectors <upstream-root> <out-dir>").clone();
    std::fs::create_dir_all(&out).unwrap();
    chimera::phonetics::planner::init();
    let g = load_g2p(&root);
    let nd = EnglishG2P::new();

    // (a) G2P
    let g2p: Vec<Value> = SENTENCES
        .iter()
        .map(|t| {
            let ph = g.to_phrases(t);
            let ipa: Vec<String> = ph.iter().map(|p| p.segments.iter().map(|s| format!("{}{}", if s.stress == Some(1) { "ˈ" } else { "" }, s.ipa)).collect()).collect();
            json!({ "text": t, "ipa": ipa, "dict": to_value(&ph), "noDict": to_value(&nd.to_phrases(t)) })
        })
        .collect();
    write_json(&out, "g2p.json", &json!({ "source": "EnglishG2P::to_phrases (CMUdict loaded / not loaded)", "cases": g2p }));

    // (b) phrase plans, as `render text` (crates/chimera-cli/src/render.rs `say`): t0 0.05, f0 112 (male)
    let plans: Vec<Value> = SENTENCES
        .iter()
        .map(|t| {
            let inputs: Vec<PhraseInput> = g.to_phrases(t).iter().map(|p| PhraseInput { segments: p.segments.clone(), intonation: Some(p.intonation), pause: Some(p.pause) }).collect();
            let opts = PlanOptions { body: "male".into(), t0: 0.05, f0: Some(112.0), ..Default::default() };
            let plan = plan_phrases(&inputs, &opts);
            json!({ "text": t, "phrases": to_value(&inputs), "options": to_value(&opts), "plan": to_value(&plan) })
        })
        .collect();
    write_json(&out, "plan_phrases.json", &json!({ "source": "planner::plan_phrases", "cases": plans }));

    // (c) a sung melody: 4 notes, syllables from lyrics_to_syllables (as `render song` does)
    let syl = lyrics_to_syllables("Twinkle little", &g);
    assert_eq!(syl.len(), 4, "expected 4 syllables");
    let midi = [48.0, 48.0, 55.0, 55.0];
    let notes: Vec<SungNote> = syl.iter().zip(midi).map(|(s, m)| SungNote { midi: m, beats: 1.0, syl: s.clone() }).collect();
    let sopts = SongOptions { body: "male".into(), bpm: 100.0, t0: 0.3, vibrato: 35.0 };
    let splan = plan_song(&notes, &sopts);
    write_json(&out, "plan_song.json", &json!({ "source": "sing::plan_song", "lyrics": "Twinkle little", "syllables": to_value(&syl), "notes": to_value(&notes), "options": to_value(&sopts), "plan": to_value(&splan) }));

    // (d) geometry per body and vowel: NEUTRAL + calibrated vowel patch
    let mut geo = Vec::new();
    for body in [BodyName::Male, BodyName::Female, BodyName::Child] {
        let mut tract = HumanVocalTract::new(body.body());
        let mut airway = HumanAirway::new(FS * 2.0, body.body());
        for ipa in ["a", "ɑ", "i"] {
            let patch = vowel_artic(ipa, body.into()).expect("vowel");
            let artic = patch.apply_to(&NEUTRAL);
            let frame = tract.compute_new(&artic);
            let (n_ph, n_or) = (18usize, 18usize);
            let (mut ph, mut or, mut ph_min, mut or_min) = (vec![0.0; n_ph], vec![0.0; n_or], vec![0.0; n_ph], vec![0.0; n_or]);
            let (ph_len, or_len) = tract.resample(&artic, &frame, &mut ph, &mut or, Some(&mut ph_min), Some(&mut or_min));
            airway.apply(&artic, 0.0);
            let formants = vowel_table().get(body.as_str()).and_then(|t| t.get(ipa)).map(|v| v.formants.clone());
            geo.push(json!({
                "body": body.as_str(),
                "ipa": ipa,
                "patch": to_value(&patch),
                "artic": to_value(&artic),
                "calibratedFormants": formants,
                "frame": to_value(&frame),
                "resample18": { "phLen": ph_len, "orLen": or_len, "ph": ph, "or": or, "phMin": ph_min, "orMin": or_min },
                "airway96k": { "fs": FS * 2.0, "nPh": airway.n_ph, "nOr": airway.n_or, "phLen": airway.ph_len, "orLen": airway.or_len, "phAreas": airway.ph_areas, "orAreas": airway.or_areas },
            }));
        }
    }
    write_json(&out, "geometry.json", &json!({ "source": "HumanVocalTract::compute_new + resample(18,18); HumanAirway::new(96000).apply(artic, 0)", "cases": geo }));

    // (e) held vowels, male, 110 Hz, 48 kHz (oversample 2), host blocks of 128
    let mut voice_meta = Vec::new();
    for (ipa, tag) in [("ɑ", "aa-back"), ("a", "a-front")] {
        let full = held_vowel(ipa, 48000, BLOCK);
        write_f32(&out, &format!("voice_male_110hz_{tag}_first4800.f32"), &full[..4800]);
        write_f32(&out, &format!("voice_male_110hz_{tag}_1s.f32"), &full);
        let mut same_blocks = Vec::new();
        for b in [1usize, 77, 4096] {
            let other = held_vowel(ipa, 48000, b);
            let first = full.iter().zip(&other).position(|(x, y)| x.to_bits() != y.to_bits());
            same_blocks.push(json!({ "block": b, "firstBitDifference": first }));
        }
        let peak = full.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        voice_meta.push(json!({ "ipa": ipa, "tag": tag, "samples": full.len(), "peak": peak, "blockSizeChecks": same_blocks }));
    }

    // self-check: reproduce the upstream golden fixture voice-sustain-male.f32 bit for bit
    // (scenario from tests/golden/voice.json: male, 48 kHz, oversample 2, 28800 samples, gate 1 at 0, gate 0 at 21600)
    let fixture = read_f32(&format!("{root}/crates/chimera/tests/golden/voice-sustain-male.f32"));
    let mut v = HumanVoice::new(FS, BODY_MALE, 2);
    let mut on = |v: &mut HumanVoice| v.set_gate(1.0);
    let mut off = |v: &mut HumanVoice| v.set_gate(0.0);
    let got = render(&mut v, 28800, BLOCK, &mut [(0, &mut on), (21600, &mut off)]);
    let first = fixture.iter().zip(&got).position(|(x, y)| x.to_bits() != y.to_bits());
    let golden = json!({ "fixture": "crates/chimera/tests/golden/voice-sustain-male.f32", "samples": got.len(), "fixtureSamples": fixture.len(), "firstBitDifference": first });
    println!("golden voice-sustain-male: {} samples, first bit difference {:?}", got.len(), first);
    write_f32(&out, "golden_sustain_male.f32", &got);

    // end-to-end speech, as `render text` (crates/chimera-cli/src/render.rs `say`): male, f0 112, t0 0.05, 0.3 s tail
    let inputs: Vec<PhraseInput> = g.to_phrases(SENTENCES[0]).iter().map(|p| PhraseInput { segments: p.segments.clone(), intonation: Some(p.intonation), pause: Some(p.pause) }).collect();
    let plan = plan_phrases(&inputs, &PlanOptions { body: "male".into(), t0: 0.05, f0: Some(112.0), ..Default::default() });
    let mut sv = HumanVoice::new(FS, BODY_MALE, 2);
    sv.set_source(&SourcePatch::f0(112.0));
    sv.schedule(&plan.gestures);
    let total = chimera::math::js_round(FS * (plan.end + 0.3)) as usize;
    let speech = render(&mut sv, total, BLOCK, &mut []);
    write_f32(&out, "speech_male_hello_world.f32", &speech);

    write_json(&out, "manifest.json", &json!({
        "generator": "tools/vector_export/r1 (chimera-vectors), public API of crates/chimera",
        "upstreamHead": std::env::var("CHIMERA_UPSTREAM_HEAD").unwrap_or_default(),
        "f32Format": "raw little-endian IEEE-754 float32, mono, no header",
        "voice": { "fsOut": FS, "oversample": 2, "body": "male", "f0": 110.0, "source": "DEFAULT_SOURCE + f0 110", "artic": "HumanVoice default + set_articulation(vowel_artic(ipa, male))", "gate": "set_gate(1.0) before the first sample", "hostBlock": BLOCK, "renders": voice_meta },
        "goldenSelfCheck": golden,
        "sentences": SENTENCES,
    }));
    libm_probe(&out);
}
