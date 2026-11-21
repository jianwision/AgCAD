//! Reference designs and goals (tests, benchmarks, docs).

use crate::abi::*;
use crate::studio::Design;
use bytemuck::Zeroable;

pub fn feature(kind: u32, op: u32, vals: [(u32, f64); 6]) -> Node {
    let mut n = Node::zeroed();
    n.kind = kind;
    n.op = op;
    n.flags = nflags::ACTIVE;
    n.pref = [NO_PARAM; NVALS];
    for (j, (p, v)) in vals.iter().enumerate() { n.pref[j] = *p; n.val[j] = *v; }
    n
}

/// Bolted flange: disc (radius $r, thickness $t) minus a pattern of $n holes on a bolt circle
/// at r − 15 mm (a parameter + offset value).
pub fn flange(r: f64, t: f64, n: f64) -> Design {
    let disc = feature(profile::CIRCLE, fop::ADD, [(NO_PARAM, 0.0), (NO_PARAM, 0.0), (0, 0.0), (NO_PARAM, 0.0), (NO_PARAM, 0.0), (1, 0.0)]);
    let mut hole = feature(profile::CIRCLE, fop::REMOVE, [(0, -15.0), (NO_PARAM, 0.0), (NO_PARAM, 4.5), (NO_PARAM, 0.0), (NO_PARAM, -1.0), (1, 1.0)]);
    hole.flags |= nflags::PATTERN;
    hole.pref[6] = 2;
    hole.val[6] = 0.0;
    hole.val[7] = core::f64::consts::TAU;
    Design { nodes: vec![disc, hole], params: vec![r, t, n], names: vec!["r".into(), "t".into(), "n".into()] }
}

pub fn goal(vol_lo: f64, vol_hi: f64) -> Goal {
    let mut g = Goal::zeroed();
    g.domain_lo = [-90.0, -90.0, -4.0];
    g.domain_hi = [90.0, 90.0, 26.0];
    g.vol_lo = vol_lo;
    g.vol_hi = vol_hi;
    g.max_ext = [170.0, 170.0, 25.0];
    g.place = [-20.0, -20.0, 20.0, 20.0];
    g.hole_r = 3.0;
    g.hole_z0 = -1.0;
    g.hole_z1 = 23.0;
    g.gain = 0.5;
    g.param_lo[..3].copy_from_slice(&[50.0, 6.0, 3.0]);
    g.param_hi[..3].copy_from_slice(&[80.0, 20.0, 12.0]);
    g.param_sign[..3].copy_from_slice(&[1.0, 1.0, -1.0]);
    g.n_params = 3;
    g.seed = 0xA9CAD;
    g
}


/// Machined plate: 100 × 60 × 10 plate ($w width), a 3 × 2 rectangular pattern of rounded
/// pockets ($r corner radius, 6 deep) and a chamfered square boss.
pub fn pocketed_plate(w: f64, r: f64) -> Design {
    let plate = feature(profile::RECT, fop::ADD, [(NO_PARAM, 0.0), (NO_PARAM, 0.0), (0, 0.0), (NO_PARAM, 60.0), (NO_PARAM, 0.0), (NO_PARAM, 10.0)]);
    let mut pocket = feature(profile::RECT, fop::REMOVE, [(NO_PARAM, 8.0), (NO_PARAM, 8.0), (NO_PARAM, 22.0), (NO_PARAM, 22.0), (NO_PARAM, 4.0), (NO_PARAM, 11.0)]);
    pocket.flags |= nflags::PATTERN_RECT | nflags::FILLET;
    pocket.val[6] = 3.0;
    pocket.val[7] = 26.0;
    pocket.val[8] = 2.0;
    pocket.val[9] = 26.0;
    pocket.pref[10] = 1;
    let mut boss = feature(profile::RECT, fop::ADD, [(NO_PARAM, 84.0), (NO_PARAM, 20.0), (NO_PARAM, 96.0), (NO_PARAM, 40.0), (NO_PARAM, 10.0), (NO_PARAM, 18.0)]);
    boss.flags |= nflags::CHAMFER;
    boss.val[10] = 2.5;
    Design { nodes: vec![plate, pocket, boss], params: vec![w, r], names: vec!["w".into(), "r".into()] }
}

pub fn plate_goal() -> Goal {
    let mut g = Goal::zeroed();
    g.domain_lo = [-5.0, -5.0, -2.0];
    g.domain_hi = [125.0, 65.0, 22.0];
    // reachable: V ≈ 600 w − pockets + boss ∈ [52.5k, 67k] over w ∈ [96, 120]
    g.vol_lo = 58_000.0;
    g.vol_hi = 62_000.0;
    g.max_ext = [130.0, 70.0, 25.0];
    g.gain = 0.5;
    g.param_lo[..2].copy_from_slice(&[96.0, 0.5]);
    g.param_hi[..2].copy_from_slice(&[120.0, 6.5]);
    g.param_sign[..2].copy_from_slice(&[1.0, 1.0]);
    g.n_params = 2;
    g.seed = 7;
    g
}

/// Two bodies: a plate (body 0) and a separate cylinder body (body 1) combined by `op`
/// (boolean.add / remove / intersect), cylinder radius $rc.
pub fn two_bodies(op: u32, rc: f64) -> Design {
    let plate = feature(profile::RECT, fop::ADD, [(NO_PARAM, 0.0), (NO_PARAM, 0.0), (NO_PARAM, 80.0), (NO_PARAM, 50.0), (NO_PARAM, 0.0), (NO_PARAM, 10.0)]);
    let body = Node { kind: profile::BODY, op: fop::ADD, flags: nflags::ACTIVE, src: 0, pref: [NO_PARAM; NVALS], val: [0.0; NVALS] };
    let cyl = feature(profile::CIRCLE, fop::ADD, [(NO_PARAM, 80.0), (NO_PARAM, 25.0), (0, 0.0), (NO_PARAM, 0.0), (NO_PARAM, -2.0), (NO_PARAM, 14.0)]);
    let combine = Node { kind: profile::COMBINE, op, flags: nflags::ACTIVE, src: 0, pref: [NO_PARAM; NVALS], val: [0.0; NVALS] };
    Design { nodes: vec![plate, body, cyl, combine], params: vec![rc], names: vec!["rc".into()] }
}

/// A closed box hollowed by part.shell ($t wall thickness).
pub fn shelled_box(t: f64) -> Design {
    let b = feature(profile::RECT, fop::ADD, [(NO_PARAM, 0.0), (NO_PARAM, 0.0), (NO_PARAM, 100.0), (NO_PARAM, 60.0), (NO_PARAM, 0.0), (NO_PARAM, 10.0)]);
    let mut sh = Node { kind: profile::SHELL, op: fop::ADD, flags: nflags::ACTIVE, src: 0, pref: [NO_PARAM; NVALS], val: [0.0; NVALS] };
    sh.pref[0] = 0;
    Design { nodes: vec![b, sh], params: vec![t], names: vec!["t".into()] }
}

pub fn box_goal() -> Goal {
    let mut g = Goal::zeroed();
    g.domain_lo = [-6.0, -6.0, -4.0];
    g.domain_hi = [106.0, 66.0, 16.0];
    g.vol_lo = 20_000.0;
    g.vol_hi = 40_000.0;
    g.max_ext = [200.0, 200.0, 200.0];
    g.gain = 0.5;
    g.n_params = 1;
    g.param_lo[0] = 0.5;
    g.param_hi[0] = 4.0;
    g.param_sign[0] = -1.0;
    g
}
