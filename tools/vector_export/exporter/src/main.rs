//! Vector export, public API of the unmodified template crate (jamesondh/chimera-vocal-synth @ 092b16ce12b9).
//!
//! Usage: chimera-s0-exporter <upstream-root> <out-dir> <audio-dir>
//! Writes to <out-dir>: rng_public.json, glottis_public.json, topology_public.json,
//! decimator_public.json, scenario_<name>.json; to <audio-dir>: track_<name>.f32 (the upstream
//! renders the tracks are valid for) and stop_male_text.f32.

#![recursion_limit = "512"]

#[path = "../../common/scenarios.rs"]
mod scenarios;

use chimera::airway::decimator::{Decimator, design_lowpass};
use chimera::math::{Drift, Rng};
use chimera::sources::glottis::{DEFAULT_GLOTTIS, ParametricGlottis};
use chimera::voice::human_voice::HumanVoice;
use scenarios::*;
use serde_json::{Value, json};

const SEEDS: [u32; 7] = [12345, 777, 4242, 1, 0, 4294967295, 2654435769];
const DRIFTS: [(f64, f64, u32); 3] = [(3000.0, 0.3, 4242), (1500.0, 0.3, 4242), (750.0, 1.0, 1)];

fn rng_public() -> Value {
    let rng: Vec<Value> = SEEDS
        .iter()
        .map(|&s| {
            let mut r = Rng::new(s);
            let next: Vec<f64> = (0..1000).map(|_| r.next()).collect();
            let mut r = Rng::new(s);
            let gauss: Vec<f64> = (0..250).map(|_| r.gauss()).collect();
            let mut r = Rng::new(s);
            let bipolar: Vec<f64> = (0..100).map(|_| r.bipolar()).collect();
            json!({ "seed": s, "next": next, "gauss": gauss, "bipolar": bipolar })
        })
        .collect();
    let drift: Vec<Value> = DRIFTS
        .iter()
        .map(|&(cr, hz, s)| {
            let mut d = Drift::new(cr, hz, s);
            let steps: Vec<f64> = (0..1000).map(|_| d.step()).collect();
            json!({ "control_rate": cr, "base_hz": hz, "seed": s, "steps": steps })
        })
        .collect();
    json!({ "rng": rng, "drift": drift })
}

fn glottis_public() -> Value {
    let bodies = bodies();
    let mut params = Vec::new();
    let mut by_name = std::collections::HashMap::new();
    for (name, bi, patch) in source_sets() {
        let (_, kind, mix) = bodies[bi];
        let mut v = HumanVoice::new(48000.0, mix.body(), 2);
        v.set_source(&patch);
        let g = glottis_array(&v.glottis.params);
        by_name.insert(name, g);
        params.push(json!({
            "name": name,
            "body": kind,
            "body_mix": { "female": mix.female, "child": mix.child },
            "body_scale": v.airway.body.scale,
            "patch": to_value(&patch),
            "source": to_value(&v.source),
            "glottis": g,
        }));
    }
    let mut next = Vec::new();
    for (name, from, overrides, dp_kind) in next_sets() {
        let mut p = by_name[from];
        for (k, x) in &overrides {
            p[*k] = *x;
        }
        let mut g = ParametricGlottis::new(NEXT_FS);
        let threshold = g.threshold;
        g.params = glottis_from(&p);
        let (mut out, mut area, mut f0_now, mut voicing, mut dp) = (vec![], vec![], vec![], vec![], vec![]);
        for i in 0..NEXT_N {
            let d = next_dp(dp_kind, i);
            out.push(g.next(d));
            area.push(g.area);
            f0_now.push(g.f0_now);
            voicing.push(g.voicing);
            dp.push(d);
        }
        let ov: Vec<Value> = overrides.iter().map(|(k, x)| json!([GLOTTIS_FIELDS[*k], x])).collect();
        next.push(json!({ "name": name, "from": from, "overrides": ov, "glottis": p, "dp_kind": dp_kind, "threshold": threshold, "dp": dp, "out": out, "area": area, "f0_now": f0_now, "voicing": voicing, "cycle_phase_end": g.cycle_phase() }));
    }
    json!({ "field_order": GLOTTIS_FIELDS, "default_glottis": glottis_array(&DEFAULT_GLOTTIS), "new_glottis_params": glottis_array(&ParametricGlottis::new(NEXT_FS).params), "params": params, "next_fs": NEXT_FS, "next_n": NEXT_N, "dp_kinds": dp_doc(), "next": next })
}

fn topology_public() -> Value {
    let mut out = Vec::new();
    for (label, kind, mix) in bodies() {
        let body = mix.body();
        let v = HumanVoice::new(48000.0, body, 2);
        let aw = &v.airway;
        let net = &aw.net;
        let le = net.linear_elements();
        let tubes: Vec<Value> = net.tubes.iter().zip(&net.tube_names).map(|(t, n)| json!({ "name": n, "index": t.index, "first": t.first, "n": t.n, "branch_base": t.branch_base })).collect();
        let branch_of = |t: chimera::airway::network::Tube| -> Vec<usize> { (0..t.n).map(|i| aw.branch_of(t, i as isize)).collect() };
        let noise: Vec<Value> = net.noise_sources().iter().map(|s| json!([s.src, s.fwd, s.bwd, s.gain, s.level])).collect();
        let fr = aw.frame.as_ref().unwrap();
        out.push(json!({
            "label": label,
            "body": kind,
            "body_mix": { "female": mix.female, "child": mix.child },
            "body_params": body_json(&body),
            "fs_internal": v.fs,
            "n_ph": aw.n_ph, "n_or": aw.n_or,
            "n_trachea": aw.trachea.n, "n_nasal": aw.nasal.n, "n_fossa": aw.fossa.n,
            "n_nodes": net.n_nodes, "n_branches": net.n_branches,
            "tubes": tubes,
            "tube_length": net.tube_length,
            "trachea": [aw.trachea.first, aw.trachea.n, aw.trachea.branch_base],
            "pharynx": [aw.pharynx.first, aw.pharynx.n, aw.pharynx.branch_base],
            "oral": [aw.oral.first, aw.oral.n, aw.oral.branch_base],
            "nasal": [aw.nasal.first, aw.nasal.n, aw.nasal.branch_base],
            "fossa": [aw.fossa.first, aw.fossa.n, aw.fossa.branch_base],
            "lungs": aw.lungs,
            "glottis_port": aw.glottis_port,
            "velum_port": aw.velum_port,
            "sub_node": aw.trachea.last_node(),
            "supra_node": aw.pharynx.first_node(),
            "rad_lips": aw.rad_lips,
            "rad_nose": aw.rad_nose,
            "branch_of": { "trachea": branch_of(aw.trachea), "pharynx": branch_of(aw.pharynx), "oral": branch_of(aw.oral) },
            "linear_elements": { "a": le.a, "b": le.b, "fixed": le.fixed, "rad_nodes": le.rad_nodes, "wall_m": le.wall_m, "wall_r": le.wall_r, "wall_k": le.wall_k, "wall_rad": le.wall_rad },
            "constants": {
                "fs": net.fs, "t": net.t, "closed_area": net.closed_area, "min_node_area": net.min_node_area, "squeeze_area": net.squeeze_area,
                "volume_source_hz": net.volume_source_hz, "viscous_scale": net.viscous_scale, "viscous_omega": net.viscous_omega, "noise_scale": net.noise_scale,
                "enable_volume_source": net.enable_volume_source, "enable_junction_floor": net.enable_junction_floor, "energy_safe": net.energy_safe, "symmetric_kinetic": net.symmetric_kinetic,
                "noise_gain": [aw.noise_gain.glottis, aw.noise_gain.constriction, aw.noise_gain.teeth],
                "max_area_rate": aw.max_area_rate, "area_rate_gain": aw.area_rate_gain, "seal_rate": aw.seal_rate, "seal_lo": aw.seal_lo, "seal_hi": aw.seal_hi, "seal_ease": aw.seal_ease,
                "output_gain": v.output_gain, "max_penetration": v.max_penetration,
            },
            "state": {
                "area": net.area, "amin": net.amin, "dx": net.dx, "p": net.p, "u": net.u, "port_area": net.port_area, "branch_area": net.branch_area,
                "wall_y": net.wall_y, "wall_u": net.wall_u, "rad_u": net.rad_u, "psrc": net.psrc, "out": net.out, "ext_flow": net.ext_flow,
                "ph_areas": aw.ph_areas, "or_areas": aw.or_areas, "ph_min": aw.ph_min, "or_min": aw.or_min, "ph_len": aw.ph_len, "or_len": aw.or_len,
                "frame_velum_port": fr.velum_port, "frame_teeth_s": fr.teeth_s,
                "noise": noise,
                "glottis_params": glottis_array(&v.glottis.params),
                "glottis_public": { "fs": v.glottis.fs, "voicing": v.glottis.voicing, "area": v.glottis.area, "f0_now": v.glottis.f0_now, "threshold": v.glottis.threshold, "cycle_phase": v.glottis.cycle_phase() },
                "source": to_value(&v.source),
                "gate": v.gate, "clock": v.clock, "fs_out": v.fs_out, "m": v.m, "fs": v.fs,
                "artic_x": v.artic.x.to_array(), "artic_v": v.artic.v, "artic_target": v.artic.target.to_array(), "tip_free": v.artic.tip_free,
            },
        }));
    }
    json!({ "bodies": out })
}

fn impulse(m: usize) -> (Vec<f64>, Vec<usize>) {
    let mut d = Decimator::new(m);
    let (mut outs, mut at) = (Vec::new(), Vec::new());
    let mut i = 0usize;
    while outs.len() < 8 {
        let x = if i == 0 { 1.0 } else { 0.0 };
        if d.push(x) {
            outs.push(d.value);
            at.push(i);
        }
        i += 1;
    }
    (outs, at)
}

fn decimator_public() -> Value {
    let (o2, a2) = impulse(2);
    let (o1, a1) = impulse(1);
    json!({
        "m2": { "factor": 2, "taps": 2 * 24 + 1, "design": "design_lowpass(49, 0.46 / 2.0, 8.5)", "coeffs": design_lowpass(49, 0.46 / 2.0, 8.5), "impulse_out": o2, "impulse_push_index": a2 },
        "m1": { "factor": 1, "taps": 1, "coeffs": [1.0], "impulse_out": o1, "impulse_push_index": a1 },
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: chimera-s0-exporter <upstream-root> <out-dir> <audio-dir>";
    let root = args.get(1).expect(usage).clone();
    let out = args.get(2).expect(usage).clone();
    let audio = args.get(3).expect(usage).clone();
    std::fs::create_dir_all(&out).unwrap();
    std::fs::create_dir_all(&audio).unwrap();
    chimera::init();
    let g = load_g2p(&root);

    write_json(&format!("{out}/rng_public.json"), &rng_public());
    write_json(&format!("{out}/glottis_public.json"), &glottis_public());
    write_json(&format!("{out}/topology_public.json"), &topology_public());
    write_json(&format!("{out}/decimator_public.json"), &decimator_public());

    for name in SCENARIOS {
        let sc = build(name, &g);
        let mut v = new_voice(&sc);
        let t = std::time::Instant::now();
        let a = render(&sc, &mut v);
        let peak = a.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        println!("{name}: {} samples, peak {peak}, {:.2} s", a.len(), t.elapsed().as_secs_f64());
        write_f32(&format!("{audio}/track_{name}.f32"), &a);
        if name == "stop-male" {
            write_f32(&format!("{audio}/stop_male_text.f32"), &a);
        }
        let mut d = describe(&sc);
        d["peak"] = json!(peak);
        write_json(&format!("{out}/scenario_{name}.json"), &d);
    }
}
