//! S0 export through the crate copy (`../crate-chimera`, upstream 092b16ce12b9 + read-only accessors
//! and the control-step recorder): private state for V1/V2/V2d/V5, the V3 control tracks with
//! `noise_inherit`, and the self-checks against the unmodified upstream renders.
//!
//! Usage: chimera-s0-track-exporter <upstream-root> <out-dir> <audio-dir> <tracks-dir>
#![recursion_limit = "512"]

#[path = "../../common/scenarios.rs"]
mod scenarios;

use chimera::airway::decimator::Decimator;
use chimera::math::Rng;
use chimera::sources::glottis::ParametricGlottis;
use chimera::voice::human_voice::HumanVoice;
use chimera::voice::human_voice::chimera_ext::{self as rec, Mode, NoiseRow, Row};
use scenarios::*;
use serde_json::{Value, json};

const SEEDS: [u32; 7] = [12345, 777, 4242, 1, 0, 4294967295, 2654435769];

fn rng_states() -> Value {
    let v: Vec<Value> = SEEDS
        .iter()
        .map(|&s| {
            let mut r = Rng::new(s);
            let initial = r.chimera_state();
            let (mut next, mut states) = (Vec::new(), Vec::new());
            for _ in 0..1000 {
                next.push(r.next());
                states.push(r.chimera_state());
            }
            json!({ "seed": s, "initial_state": initial, "next": next, "states": states })
        })
        .collect();
    json!({ "rng": v })
}

fn topology_private() -> Value {
    let v: Vec<Value> = bodies()
        .iter()
        .map(|(label, _, mix)| {
            let voice = HumanVoice::new(48000.0, mix.body(), 2);
            json!({ "label": label, "network": voice.airway.net.chimera_topology(), "voice": voice.chimera_private() })
        })
        .collect();
    json!({ "bodies": v })
}

fn glottis_private() -> Value {
    let bodies = bodies();
    let mut masks = Vec::new();
    let mut by_name = std::collections::HashMap::new();
    for (name, bi, patch) in source_sets() {
        let mut v = HumanVoice::new(48000.0, bodies[bi].2.body(), 2);
        v.set_source(&patch);
        let g = glottis_array(&v.glottis.params);
        by_name.insert(name, g);
        masks.push(json!({ "name": name, "written": v.chimera_apply_source_mask(), "glottis": g }));
    }
    let mut next = Vec::new();
    for (name, from, overrides, dp_kind) in next_sets() {
        let mut p = by_name[from];
        for (k, x) in &overrides {
            p[*k] = *x;
        }
        let mut g = ParametricGlottis::new(NEXT_FS);
        let initial = g.chimera_state();
        g.params = glottis_from(&p);
        let mut out = Vec::new();
        let mut mid = Value::Null;
        for i in 0..NEXT_N {
            out.push(g.next(next_dp(dp_kind, i)));
            if i == 999 {
                mid = g.chimera_state();
            }
        }
        next.push(json!({ "name": name, "out": out, "state_initial": initial, "state_after_1000": mid, "state_after_2000": g.chimera_state() }));
    }
    json!({ "params": masks, "next": next })
}

fn decimator_private() -> Value {
    let d2 = Decimator::new(2);
    let d1 = Decimator::new(1);
    json!({ "m2": { "coeffs": d2.chimera_coeffs(), "pos": d2.chimera_pos() }, "m1": { "coeffs": d1.chimera_coeffs(), "pos": d1.chimera_pos() } })
}

fn noise_idx(l: &[NoiseRow]) -> Vec<[usize; 3]> {
    l.iter().map(|s| [s.src, s.fwd, s.bwd]).collect()
}
fn noise_par(l: &[NoiseRow]) -> Vec<[f64; 3]> {
    l.iter().map(|s| [s.gain, s.lowpass_hz, s.highpass_hz]).collect()
}
fn noise_coef(l: &[NoiseRow]) -> Vec<[f64; 2]> {
    l.iter().map(|s| [s.lp_coef, s.hp_coef]).collect()
}
fn tube(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().chain(b).copied().collect()
}

fn row_json(r: &Row) -> Value {
    json!({
        "tube_area": tube(&r.ph_areas, &r.or_areas),
        "tube_min": tube(&r.ph_min, &r.or_min),
        "tube_len": [r.ph_len, r.or_len],
        "velum_area": r.velum,
        "velum_target": r.velum_target,
        "noise_n": r.noise.len(),
        "noise_idx": noise_idx(&r.noise),
        "noise_par": noise_par(&r.noise),
        "noise_coef": noise_coef(&r.noise),
        "glot_par": r.glot,
        "p_lung": r.p_lung,
        "p_lung_now": r.p_lung_now,
        "p_lung_step": r.p_lung_step,
        "clock": r.clock,
        "gate": r.gate,
        "f0_glide": r.f0_glide,
        "artic_x": r.artic_x,
        "artic_v": r.artic_v,
        "contact_tip": r.contact_tip,
        "contact_body": r.contact_body,
    })
}

fn track_json(desc: &Value, initial: &Row, rows: &[Row], n_ph: usize, n_or: usize) -> Value {
    let list_a: serde_json::Map<String, Value> = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| r.reapply)
        .map(|(b, r)| (b.to_string(), json!({ "noise_idx": noise_idx(&r.list_a), "noise_par": noise_par(&r.list_a) })))
        .collect();
    json!({
        "scenario": desc,
        "n_ph": n_ph,
        "n_or": n_or,
        "ctrl_every": 64 * OVERSAMPLE,
        "block_rule": "block b = the b-th control() call; it runs before internal sample b*ctrl_every, i.e. before output sample b*ctrl_every/oversample = 64*b; row b = state after the complete control step b, as the kernel consumes it during block b (taken as the last statement of control(); initial = state after the constructor)",
        "n_blocks": rows.len(),
        "initial": row_json(initial),
        "blocks": {
            "tube_area": rows.iter().map(|r| tube(&r.ph_areas, &r.or_areas)).collect::<Vec<_>>(),
            "tube_min": rows.iter().map(|r| tube(&r.ph_min, &r.or_min)).collect::<Vec<_>>(),
            "tube_len": rows.iter().map(|r| [r.ph_len, r.or_len]).collect::<Vec<_>>(),
            "velum_area": rows.iter().map(|r| r.velum).collect::<Vec<_>>(),
            "velum_target": rows.iter().map(|r| r.velum_target).collect::<Vec<_>>(),
            "noise_n": rows.iter().map(|r| r.noise.len()).collect::<Vec<_>>(),
            "noise_idx": rows.iter().map(|r| noise_idx(&r.noise)).collect::<Vec<_>>(),
            "noise_par": rows.iter().map(|r| noise_par(&r.noise)).collect::<Vec<_>>(),
            "noise_coef": rows.iter().map(|r| noise_coef(&r.noise)).collect::<Vec<_>>(),
            "noise_inherit": rows.iter().map(|r| r.inherit.clone()).collect::<Vec<_>>(),
            "noise_inherit_p1": rows.iter().map(|r| r.inherit_p1.clone()).collect::<Vec<_>>(),
            "reapply": rows.iter().map(|r| r.reapply as u8).collect::<Vec<_>>(),
            "glot_par": rows.iter().map(|r| r.glot).collect::<Vec<_>>(),
            "p_lung": rows.iter().map(|r| r.p_lung).collect::<Vec<_>>(),
            "p_lung_now": rows.iter().map(|r| r.p_lung_now).collect::<Vec<_>>(),
            "p_lung_step": rows.iter().map(|r| r.p_lung_step).collect::<Vec<_>>(),
            "clock": rows.iter().map(|r| r.clock).collect::<Vec<_>>(),
            "gate": rows.iter().map(|r| r.gate).collect::<Vec<_>>(),
            "f0_glide": rows.iter().map(|r| r.f0_glide).collect::<Vec<_>>(),
            "artic_x": rows.iter().map(|r| r.artic_x).collect::<Vec<_>>(),
            "artic_v": rows.iter().map(|r| r.artic_v).collect::<Vec<_>>(),
            "contact_tip": rows.iter().map(|r| r.contact_tip).collect::<Vec<_>>(),
            "contact_body": rows.iter().map(|r| r.contact_body).collect::<Vec<_>>(),
        },
        "list_a": list_a,
    })
}

/// Compare `got` with the upstream render `want`.
fn compare(want: &[f32], got: &[f32]) -> Value {
    let peak = want.iter().fold(0.0f64, |m, x| m.max((*x as f64).abs()));
    let mut first_bit: Option<usize> = None;
    let mut first_tol: Option<usize> = None;
    let mut n_bit = 0usize;
    let mut maxd = 0.0f64;
    let mut max_at: Option<usize> = None;
    for i in 0..want.len().min(got.len()) {
        if want[i].to_bits() != got[i].to_bits() {
            n_bit += 1;
            first_bit.get_or_insert(i);
        }
        let d = (want[i] as f64 - got[i] as f64).abs();
        if d > maxd {
            maxd = d;
            max_at = Some(i);
        }
        if first_tol.is_none() && d > 1e-6 * peak {
            first_tol = Some(i);
        }
    }
    json!({ "len_want": want.len(), "len_got": got.len(), "first_bit_difference": first_bit, "n_bit_differences": n_bit, "first_over_1e-6_peak": first_tol, "max_abs_diff": maxd, "max_abs_diff_over_peak": maxd / peak, "max_at": max_at, "peak": peak })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: chimera-s0-track-exporter <upstream-root> <out-dir> <audio-dir> <tracks-dir>";
    let root = args.get(1).expect(usage).clone();
    let out = args.get(2).expect(usage).clone();
    let audio = args.get(3).expect(usage).clone();
    let tracks = args.get(4).expect(usage).clone();
    std::fs::create_dir_all(&out).unwrap();
    std::fs::create_dir_all(&tracks).unwrap();
    chimera::init();
    let g = load_g2p(&root);

    write_json(&format!("{out}/rng_states.json"), &rng_states());
    write_json(&format!("{out}/topology_private.json"), &topology_private());
    write_json(&format!("{out}/glottis_private.json"), &glottis_private());
    write_json(&format!("{out}/decimator_private.json"), &decimator_private());

    let mut checks = Vec::new();
    for name in SCENARIOS {
        let sc = build(name, &g);
        let want = read_f32(&format!("{audio}/track_{name}.f32"));
        // (a) the copy with no recorder started
        let mut v = new_voice(&sc);
        let plain = render(&sc, &mut v);
        // (b) record mode
        let mut v = new_voice(&sc);
        let initial = v.chimera_snapshot();
        rec::start(Mode::Record);
        let recorded = render(&sc, &mut v);
        let rows = rec::take().unwrap().rows;
        // (c) kernel emulation: filter states set only from noise_inherit + previous final list
        let mut v = new_voice(&sc);
        rec::start(Mode::EmulateInherit);
        let emulated = render(&sc, &mut v);
        let rows_e = rec::take().unwrap().rows;
        // (d) counter-check: REVIEW-1 P1 emulation "final list against the previous block"
        let mut v = new_voice(&sc);
        rec::start(Mode::P1Rebase);
        let p1 = render(&sc, &mut v);
        rec::take();
        write_f32(&format!("{out}/p1rebase_{name}.f32"), &p1);

        let n_ph = v.airway.n_ph;
        let n_or = v.airway.n_or;
        let desc = describe(&sc);
        let full = track_json(&desc, &initial, &rows, n_ph, n_or);
        write_json(&format!("{tracks}/track-{name}.full.json"), &full);

        let same_inherit = rows.iter().zip(&rows_e).all(|(a, b)| a.inherit == b.inherit && a.noise.len() == b.noise.len());
        let reapply = rows.iter().filter(|r| r.reapply).count();
        let a_ne_b: Vec<usize> = rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.reapply && r.list_a.iter().map(|s| (s.src, s.fwd)).collect::<Vec<_>>() != r.noise.iter().map(|s| (s.src, s.fwd)).collect::<Vec<_>>())
            .map(|(b, _)| b)
            .collect();
        let critical: Vec<usize> = rows.iter().enumerate().filter(|(_, r)| r.inherit != r.inherit_p1).map(|(b, _)| b).collect();
        let fails: u32 = rows.iter().map(|r| r.inherit_check_failures).sum();
        let coef_fails: u32 = rows.iter().map(|r| r.coef_check_failures).sum::<u32>() + initial.coef_check_failures;
        let c = json!({
            "scenario": name,
            "n_samples": sc.n_samples,
            "n_blocks": rows.len(),
            "reapply_blocks": reapply,
            "a_ne_b_blocks": a_ne_b,
            "p1_critical_blocks": critical,
            "inherit_check_failures": fails,
            "coef_check_failures": coef_fails,
            "emulation_rows_same_inherit": same_inherit,
            "copy_plain_vs_upstream": compare(&want, &plain),
            "copy_record_vs_upstream": compare(&want, &recorded),
            "copy_emulate_inherit_vs_upstream": compare(&want, &emulated),
            "copy_p1_rebase_vs_upstream": compare(&want, &p1),
        });
        println!("{}", serde_json::to_string(&c).unwrap());
        checks.push(c);
    }
    write_json(&format!("{out}/selfcheck.json"), &json!({ "checks": checks }));
}
