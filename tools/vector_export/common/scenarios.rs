//! Shared by `exporter` (upstream crate, public API) and `track-exporter` (crate copy): both
//! compile this file against their own `chimera` dependency, so both build the same scenarios,
//! glottis parameter sets and bodies from the same source text. Public API only.
#![allow(dead_code)]

use chimera::anatomy::human::{ArticPatch, BodyMix, HumanBody};
use chimera::control::score::Gesture;
use chimera::math::js_round;
use chimera::phonetics::english::EnglishG2P;
use chimera::phonetics::planner::{PhraseInput, PlanOptions, plan_phrases, vowel_artic};
use chimera::phonetics::sing::{SongOptions, SungNote, midi_to_hz, plan_song};
use chimera::phonetics::songs::DEMO_SONGS;
use chimera::phonetics::syllables::lyrics_to_syllables;
use chimera::sources::glottis::GlottisParams;
use chimera::voice::human_voice::{HumanVoice, SourcePatch};
use serde_json::{Value, json};
use std::io::Read;

pub const FS_OUT: f64 = 48000.0;
pub const OVERSAMPLE: usize = 2;
pub const HOST_BLOCK: usize = 128;
pub const UPSTREAM_HEAD: &str = "092b16ce12b927d473c884ca2dfae53ce1b995df";

pub fn load_g2p(root: &str) -> EnglishG2P {
    // same as crates/chimera-cli/src/render.rs `g2p()`
    let path = format!("{root}/public/data/cmudict.txt.gz");
    let gz = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut text = String::new();
    flate2::read::GzDecoder::new(&gz[..]).read_to_string(&mut text).unwrap();
    let mut g = EnglishG2P::new();
    g.load_dict(&text);
    g
}

pub fn to_value<T: serde::Serialize>(x: &T) -> Value {
    serde_json::to_value(x).unwrap()
}

/// The five bodies of the topology vector: (label, kind, mix).
pub fn bodies() -> Vec<(&'static str, &'static str, BodyMix)> {
    vec![
        ("male", "male", BodyMix::MALE),
        ("female", "female", BodyMix::FEMALE),
        ("child", "child", BodyMix::CHILD),
        ("mix-f050", "mix", BodyMix::new(0.5, 0.0)),
        ("mix-f050-c050", "mix", BodyMix::new(0.5, 0.5)),
    ]
}

pub fn body_json(b: &HumanBody) -> Value {
    json!({ "name": b.name, "scale": b.scale, "pharynx": b.pharynx, "tongue_radius": b.tongue_radius, "lip_width": b.lip_width })
}

pub fn glottis_array(g: &GlottisParams) -> [f64; 17] {
    [g.f0, g.amplitude, g.abduction, g.chink, g.open_quotient, g.skew, g.closure_sharpness, g.effort, g.jitter, g.shimmer, g.vibrato_depth, g.vibrato_rate, g.drift, g.fry, g.diplophonia, g.closure, g.chaos]
}

pub fn glottis_from(v: &[f64; 17]) -> GlottisParams {
    GlottisParams { f0: v[0], amplitude: v[1], abduction: v[2], chink: v[3], open_quotient: v[4], skew: v[5], closure_sharpness: v[6], effort: v[7], jitter: v[8], shimmer: v[9], vibrato_depth: v[10], vibrato_rate: v[11], drift: v[12], fry: v[13], diplophonia: v[14], closure: v[15], chaos: v[16] }
}

pub const GLOTTIS_FIELDS: [&str; 17] = ["f0", "amplitude", "abduction", "chink", "open_quotient", "skew", "closure_sharpness", "effort", "jitter", "shimmer", "vibrato_depth", "vibrato_rate", "drift", "fry", "diplophonia", "closure", "chaos"];

/// The 12 source sets of V5 "params": (name, body index into `bodies()`, patch).
pub fn source_sets() -> Vec<(&'static str, usize, SourcePatch)> {
    let p = SourcePatch::default;
    vec![
        ("default", 0, p()),
        ("voiced-false", 1, SourcePatch { voiced: Some(false), ..p() }),
        ("breathiness-0.6", 2, SourcePatch { breathiness: Some(0.6), ..p() }),
        ("tension-0.9", 3, SourcePatch { tension: Some(0.9), ..p() }),
        ("tension-0.1", 4, SourcePatch { tension: Some(0.1), ..p() }),
        ("brightness-0.9", 0, SourcePatch { brightness: Some(0.9), ..p() }),
        ("brightness-0.0", 1, SourcePatch { brightness: Some(0.0), ..p() }),
        ("roughness-0.8", 2, SourcePatch { roughness: Some(0.8), ..p() }),
        ("pressure-neg6", 3, SourcePatch { pressure: Some(-6.0), ..p() }),
        ("pressure-1", 4, SourcePatch { pressure: Some(1.0), ..p() }),
        ("pressure-20", 0, SourcePatch { pressure: Some(20.0), ..p() }),
        ("fry-vibrato", 1, SourcePatch { fry: Some(0.5), vibrato_depth: Some(50.0), vibrato_rate: Some(5.5), drift: Some(0.0), ..p() }),
    ]
}

/// V5 "next": (name, source set, overrides on the 17 params, dp description). fs = 96 000.
pub const NEXT_FS: f64 = 96000.0;
pub const NEXT_N: usize = 2000;
pub fn next_sets() -> Vec<(&'static str, &'static str, Vec<(usize, f64)>, &'static str)> {
    vec![
        // fry 0.5 (> 0.3: second pulse), vibrato 50 c at 5.5 Hz, drift 0; f0 raised so 2000 samples hold ~4 cycles
        ("voiced-constant", "fry-vibrato", vec![(0, 330.0)], "const8"),
        // diplophonia 0.42 (roughness 0.8) and chaos 0.7 (logistic map), dp ramps through the phonation threshold
        ("threshold-ramp", "roughness-0.8", vec![(0, 440.0), (16, 0.7)], "ramp6"),
        // voiceless: amplitude 0, wide abduction; plus fry/vibrato (f0 path) and closure 0.3 (statics path)
        ("voiceless", "voiced-false", vec![(0, 205.0), (13, 0.5), (10, 50.0), (11, 5.5), (15, 0.3)], "const8"),
    ]
}

/// dp (dyn/cm^2) at sample i; CM_H2O = 980.665 (physics.rs)
pub fn next_dp(kind: &str, i: usize) -> f64 {
    let cm = chimera::physics::CM_H2O;
    match kind {
        "const8" => 8.0 * cm,
        "ramp6" => 6.0 * cm * (i as f64) / 1999.0,
        _ => panic!("dp kind"),
    }
}

pub fn dp_doc() -> Value {
    json!({
        "const8": "dp[i] = 8.0 * 980.665 (evaluated as 8.0 * CM_H2O)",
        "ramp6": "dp[i] = 6.0 * 980.665 * i / 1999.0 (left to right: ((6.0 * CM_H2O) * i) / 1999.0), i = 0..1999",
    })
}

pub enum OpKind {
    SetSource(SourcePatch),
    SetArticulation(ArticPatch, Option<bool>),
    SetGate(f64),
    Schedule(Vec<Gesture>),
}

pub struct Op {
    pub at: usize,
    pub kind: OpKind,
}

pub struct Scenario {
    pub name: &'static str,
    pub body_label: &'static str,
    pub body_mix: BodyMix,
    pub n_samples: usize,
    pub ops: Vec<Op>,
    pub provenance: String,
    pub info: Value,
}

pub const SCENARIOS: [&str; 4] = ["sustain-aa-male", "hello-male", "stop-male", "song-female"];

fn say(g: &EnglishG2P, name: &'static str, text: &str) -> Scenario {
    // crates/chimera-cli/src/render.rs `say` (render text), male: f0 112, t0 0.05, 0.3 s tail
    let f0 = 112.0;
    let phrases = g.to_phrases(text);
    let inputs: Vec<PhraseInput> = phrases.iter().map(|p| PhraseInput { segments: p.segments.clone(), intonation: Some(p.intonation), pause: Some(p.pause) }).collect();
    let opts = PlanOptions { body: "male".into(), t0: 0.05, f0: Some(f0), ..Default::default() };
    let plan = plan_phrases(&inputs, &opts);
    let n = js_round(FS_OUT * (plan.end + 0.3)) as usize;
    Scenario {
        name,
        body_label: "male",
        body_mix: BodyMix::MALE,
        n_samples: n,
        ops: vec![Op { at: 0, kind: OpKind::SetSource(SourcePatch::f0(f0)) }, Op { at: 0, kind: OpKind::Schedule(plan.gestures.clone()) }],
        provenance: format!("as `chimera render text <out> \"{text}\" male` (crates/chimera-cli/src/render.rs `say`, render.rs:145-188): HumanVoice::new(48000, BODY_MALE, 2), set_source(f0 112), schedule(plan_phrases(G2P(text), PlanOptions{{body: male, t0: 0.05, f0: 112}}).gestures), js_round(48000*(plan.end+0.3)) samples in host blocks of 128"),
        info: json!({ "text": text, "phrases": to_value(&inputs), "plan_options": to_value(&opts), "plan_end": plan.end, "plan_segments": to_value(&plan.segments) }),
    }
}

pub fn build(name: &str, g: &EnglishG2P) -> Scenario {
    match name {
        "sustain-aa-male" => {
            // tools/vector_export/r1/src/main.rs `held_vowel("ɑ", 48000, 128)`
            let artic = vowel_artic("ɑ", BodyMix::MALE).expect("vowel");
            Scenario {
                name: "sustain-aa-male",
                body_label: "male",
                body_mix: BodyMix::MALE,
                n_samples: 48000,
                ops: vec![
                    Op { at: 0, kind: OpKind::SetSource(SourcePatch::f0(110.0)) },
                    Op { at: 0, kind: OpKind::SetArticulation(artic, None) },
                    Op { at: 0, kind: OpKind::SetGate(1.0) },
                ],
                provenance: "held vowel as tools/vector_export/r1 `held_vowel(\"ɑ\")` (byte-identical to its voice_male_110hz_aa-back_1s.f32): HumanVoice::new(48000, BODY_MALE, 2), set_source(f0 110), set_articulation(vowel_artic(\"ɑ\", male), None), set_gate(1.0), 48000 samples in host blocks of 128".to_string(),
                info: json!({ "ipa": "ɑ" }),
            }
        }
        "hello-male" => say(g, "hello-male", "Hello world."),
        "stop-male" => say(g, "stop-male", "Stop at the top, Pat, and take the tea to the cat."),
        "song-female" => {
            // crates/chimera-cli/src/render.rs `song` (render song <dir> female), second demo song
            let song = &DEMO_SONGS[1];
            assert_eq!(song.name, "Twinkle, twinkle");
            let syl = lyrics_to_syllables(song.lyrics, g);
            let shift = song.octave.female * 12.0;
            let notes: Vec<SungNote> = song.notes.iter().enumerate().map(|(i, n)| SungNote { midi: n[0] + shift, beats: n[1], syl: syl[i % syl.len()].clone() }).collect();
            let src = SourcePatch { f0: Some(midi_to_hz(notes[0].midi)), voiced: Some(true), pressure: Some(9.0), tension: Some(0.55), ..Default::default() };
            let opts = SongOptions { body: "female".into(), bpm: song.bpm, t0: 0.3, vibrato: 35.0 };
            let plan = plan_song(&notes, &opts);
            let n = js_round(FS_OUT * (plan.end + 0.4)) as usize;
            Scenario {
                name: "song-female",
                body_label: "female",
                body_mix: BodyMix::FEMALE,
                n_samples: n,
                ops: vec![Op { at: 0, kind: OpKind::SetSource(src) }, Op { at: 0, kind: OpKind::Schedule(plan.gestures.clone()) }],
                provenance: "as `chimera render song <dir> female` for DEMO_SONGS[1] \"Twinkle, twinkle\" (crates/chimera-cli/src/render.rs:211-272; byte-identical to the template's `chimera render song <dir> female` output twinkle-twinkle-female.wav): HumanVoice::new(48000, BODY_FEMALE, 2), set_source{f0: midi_to_hz(first note), voiced, pressure 9, tension 0.55}, schedule(plan_song(notes, SongOptions{female, bpm 100, t0 0.3, vibrato 35}).gestures), js_round(48000*(plan.end+0.4)) samples in host blocks of 128".to_string(),
                info: json!({ "song": song.name, "lyrics": song.lyrics, "syllables": to_value(&syl), "notes": to_value(&notes), "song_options": to_value(&opts), "plan_end": plan.end, "plan_segments": to_value(&plan.segments) }),
            }
        }
        _ => panic!("unknown scenario {name}"),
    }
}

pub fn new_voice(sc: &Scenario) -> HumanVoice {
    HumanVoice::new(FS_OUT, sc.body_mix.body(), OVERSAMPLE)
}

fn apply_op(v: &mut HumanVoice, op: &Op) {
    match &op.kind {
        OpKind::SetSource(p) => v.set_source(p),
        OpKind::SetArticulation(a, t) => v.set_articulation(a, *t),
        OpKind::SetGate(x) => v.set_gate(*x),
        OpKind::Schedule(gs) => v.schedule(gs),
    }
}

/// Render the scenario: ops due at sample i are applied before sample i; host blocks of 128
/// (an op splits a block), as tools/vector_export/r1 `render`.
pub fn render(sc: &Scenario, v: &mut HumanVoice) -> Vec<f32> {
    let n = sc.n_samples;
    let mut out = vec![0.0f32; n];
    let (mut i, mut oi) = (0usize, 0usize);
    while i < n {
        while oi < sc.ops.len() && sc.ops[oi].at <= i {
            apply_op(v, &sc.ops[oi]);
            oi += 1;
        }
        let next = if oi < sc.ops.len() { sc.ops[oi].at.max(i + 1) } else { usize::MAX };
        let end = n.min(i + HOST_BLOCK).min(next);
        v.render(&mut out[i..end]);
        i = end;
    }
    out
}

pub fn describe(sc: &Scenario) -> Value {
    let ops: Vec<Value> = sc
        .ops
        .iter()
        .map(|o| match &o.kind {
            OpKind::SetSource(p) => json!({ "at": o.at, "op": "set_source", "patch": to_value(p) }),
            OpKind::SetArticulation(a, t) => json!({ "at": o.at, "op": "set_articulation", "patch": to_value(a), "tip_free": t }),
            OpKind::SetGate(x) => json!({ "at": o.at, "op": "set_gate", "level": x }),
            OpKind::Schedule(gs) => json!({ "at": o.at, "op": "schedule", "gestures": to_value(gs) }),
        })
        .collect();
    let b = sc.body_mix.body();
    json!({
        "name": sc.name,
        "body": sc.body_label,
        "body_mix": { "female": sc.body_mix.female, "child": sc.body_mix.child },
        "body_params": body_json(&b),
        "fs_out": FS_OUT,
        "oversample": OVERSAMPLE,
        "fs_internal": FS_OUT * OVERSAMPLE as f64,
        "host_block": HOST_BLOCK,
        "n_samples": sc.n_samples,
        "ops": ops,
        "ops_rule": "ops with at <= i are applied in list order before output sample i is rendered; all ops here have at = 0, i.e. they run after HumanVoice::new and before the first render() call (so before the first control step)",
        "provenance": sc.provenance,
        "info": sc.info,
    })
}

pub fn write_f32(path: &str, data: &[f32]) {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("{path}: {e}"));
}

pub fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub fn write_json(path: &str, v: &Value) {
    std::fs::write(path, serde_json::to_string(v).unwrap()).unwrap_or_else(|e| panic!("{path}: {e}"));
    println!("wrote {path}");
}
