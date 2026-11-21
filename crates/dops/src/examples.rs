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
    Design { nodes: vec![disc, hole], params: vec![r, t, n], names: vec!["r".into(), "t".into(), "n".into()], profiles: vec![] }
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
    Design { nodes: vec![plate, pocket, boss], params: vec![w, r], names: vec!["w".into(), "r".into()], profiles: vec![] }
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
    Design { nodes: vec![plate, body, cyl, combine], params: vec![rc], names: vec!["rc".into()], profiles: vec![] }
}

/// A closed box hollowed by part.shell ($t wall thickness).
pub fn shelled_box(t: f64) -> Design {
    let b = feature(profile::RECT, fop::ADD, [(NO_PARAM, 0.0), (NO_PARAM, 0.0), (NO_PARAM, 100.0), (NO_PARAM, 60.0), (NO_PARAM, 0.0), (NO_PARAM, 10.0)]);
    let mut sh = Node { kind: profile::SHELL, op: fop::ADD, flags: nflags::ACTIVE, src: 0, pref: [NO_PARAM; NVALS], val: [0.0; NVALS] };
    sh.pref[0] = 0;
    Design { nodes: vec![b, sh], params: vec![t], names: vec!["t".into()], profiles: vec![] }
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

fn blank(kind: u32, op: u32) -> Node { Node { kind, op, flags: nflags::ACTIVE, src: 0, pref: [NO_PARAM; NVALS], val: [0.0; NVALS] } }

/// Goal with a domain around the operation examples below.
pub fn ops_goal() -> Goal {
    let mut g = goal(1.0, 2.0);
    g.domain_lo = [-62.0, -62.0, -6.0];
    g.domain_hi = [106.0, 106.0, 46.0];
    g.max_ext = [300.0; 3];
    g
}

/// 100 × 60 × 10 plate with a through slot (stadium (30, 30)–(70, 30), width 10) as a
/// general sketch pocket. Volume 60 000 − (400 + 25π)·10.
pub fn slotted_plate() -> Design {
    let plate = feature(profile::RECT, fop::ADD, [(NO_PARAM, 0.0), (NO_PARAM, 0.0), (NO_PARAM, 100.0), (NO_PARAM, 60.0), (NO_PARAM, 0.0), (NO_PARAM, 10.0)]);
    let mut slot = blank(profile::SKETCH, fop::REMOVE);
    slot.val[4] = -1.0;
    slot.val[5] = 11.0;
    slot.val[15] = 0.0;
    let hp = core::f64::consts::FRAC_PI_2;
    let loopv = vec![
        Seg::Line { a: [30.0, 25.0], b: [70.0, 25.0] },
        Seg::Arc { c: [70.0, 30.0], r: 5.0, a0: -hp, a1: hp },
        Seg::Line { a: [70.0, 35.0], b: [30.0, 35.0] },
        Seg::Arc { c: [30.0, 30.0], r: 5.0, a0: hp, a1: 3.0 * hp },
    ];
    Design { nodes: vec![plate, slot], params: vec![], names: vec![], profiles: vec![vec![loopv]] }
}

/// L-bracket: an L outline with a concave fillet (r 6) at the inner corner and a round hole,
/// padded 0..$t (parameter t = 12).
pub fn l_bracket() -> Design {
    let mut pad = blank(profile::SKETCH, fop::ADD);
    pad.val[5] = 0.0;
    pad.pref[5] = 0;
    pad.val[15] = 0.0;
    let hp = core::f64::consts::FRAC_PI_2;
    let outer = vec![
        Seg::Line { a: [0.0, 0.0], b: [60.0, 0.0] },
        Seg::Line { a: [60.0, 0.0], b: [60.0, 15.0] },
        Seg::Line { a: [60.0, 15.0], b: [21.0, 15.0] },
        Seg::Arc { c: [21.0, 21.0], r: 6.0, a0: -hp, a1: -2.0 * hp }, // concave: clockwise
        Seg::Line { a: [15.0, 21.0], b: [15.0, 50.0] },
        Seg::Line { a: [15.0, 50.0], b: [0.0, 50.0] },
        Seg::Line { a: [0.0, 50.0], b: [0.0, 0.0] },
    ];
    let hole = vec![Seg::Arc { c: [45.0, 7.5], r: 3.0, a0: 0.0, a1: -core::f64::consts::TAU }];
    Design { nodes: vec![pad], params: vec![12.0], names: vec!["t".into()], profiles: vec![vec![outer, hole]] }
}

/// Stepped shaft about z: Ø40 × 10 flange and Ø24 × 30 journal (rectilinear meridian: exact
/// layered). Volume π(400·10 + 144·30).
pub fn stepped_shaft() -> Design {
    let mut s = blank(profile::REVOLVE, fop::ADD);
    s.val[15] = 0.0;
    let pts = [[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [12.0, 10.0], [12.0, 40.0], [0.0, 40.0]];
    let lp = (0..pts.len()).map(|k| Seg::Line { a: pts[k], b: pts[(k + 1) % pts.len()] }).collect();
    Design { nodes: vec![s], params: vec![], names: vec![], profiles: vec![vec![lp]] }
}

/// Coned shaft (sloped meridian: device-only). Frustum r 20 → 10 over 30: π·30·(400 + 200 + 100)/3.
pub fn coned_shaft() -> Design {
    let mut s = blank(profile::REVOLVE, fop::ADD);
    s.val[15] = 0.0;
    let pts = [[0.0, 0.0], [20.0, 0.0], [10.0, 30.0], [0.0, 30.0]];
    let lp = (0..pts.len()).map(|k| Seg::Line { a: pts[k], b: pts[(k + 1) % pts.len()] }).collect();
    Design { nodes: vec![s], params: vec![], names: vec![], profiles: vec![vec![lp]] }
}

/// Bracket pad x ∈ [10, 40] with a hole, mirrored across x = 0. Volume 2·(30·40 − 25π)·8.
pub fn mirrored_bracket() -> Design {
    let pad = feature(profile::RECT, fop::ADD, [(NO_PARAM, 10.0), (NO_PARAM, -20.0), (NO_PARAM, 40.0), (NO_PARAM, 20.0), (NO_PARAM, 0.0), (NO_PARAM, 8.0)]);
    let hole = feature(profile::CIRCLE, fop::REMOVE, [(NO_PARAM, 25.0), (NO_PARAM, 0.0), (NO_PARAM, 5.0), (NO_PARAM, 0.0), (NO_PARAM, -1.0), (NO_PARAM, 9.0)]);
    let mut m = blank(profile::MIRROR, fop::ADD);
    m.val[0] = 1.0;
    Design { nodes: vec![pad, hole, m], params: vec![], names: vec![], profiles: vec![] }
}

/// 40 × 40 × 20 boss with a draft of tan θ = $d (parameter d = 0.1): a frustum.
pub fn drafted_boss() -> Design {
    let mut b = feature(profile::RECT, fop::ADD, [(NO_PARAM, -20.0), (NO_PARAM, -20.0), (NO_PARAM, 20.0), (NO_PARAM, 20.0), (NO_PARAM, 0.0), (NO_PARAM, 20.0)]);
    b.flags |= nflags::DRAFT;
    b.pref[12] = 0;
    Design { nodes: vec![b], params: vec![0.1], names: vec!["d".into()], profiles: vec![] }
}

/// 60 × 40 × 20 block with rounded (r = $r, 4) top and bottom edges.
pub fn rounded_block() -> Design {
    let mut b = feature(profile::RECT, fop::ADD, [(NO_PARAM, -30.0), (NO_PARAM, -20.0), (NO_PARAM, 30.0), (NO_PARAM, 20.0), (NO_PARAM, 0.0), (NO_PARAM, 20.0)]);
    b.flags |= nflags::ROUND_Z;
    b.pref[13] = 0;
    Design { nodes: vec![b], params: vec![4.0], names: vec!["r".into()], profiles: vec![] }
}

/// Plate with a square pocket repeated at user-pattern positions and a two-island sketch
/// pocket (two circles in one sketch).
pub fn user_pattern_plate() -> Design {
    let plate = feature(profile::RECT, fop::ADD, [(NO_PARAM, 0.0), (NO_PARAM, 0.0), (NO_PARAM, 100.0), (NO_PARAM, 60.0), (NO_PARAM, 0.0), (NO_PARAM, 10.0)]);
    let mut sq = feature(profile::RECT, fop::REMOVE, [(NO_PARAM, 10.0), (NO_PARAM, 10.0), (NO_PARAM, 20.0), (NO_PARAM, 20.0), (NO_PARAM, 4.0), (NO_PARAM, 11.0)]);
    sq.flags |= nflags::PATTERN_USER;
    sq.val[6] = 0.0;
    let pts = [[0.0, 0.0], [60.0, 0.0], [25.0, 30.0], [70.0, 32.0]];
    let entry = pts.iter().map(|t| Seg::Line { a: *t, b: *t }).collect();
    let mut two = blank(profile::SKETCH, fop::REMOVE);
    two.val[4] = -1.0;
    two.val[5] = 11.0;
    two.val[15] = 1.0;
    let tau = core::f64::consts::TAU;
    let islands = vec![vec![Seg::Arc { c: [50.0, 30.0], r: 4.0, a0: 0.0, a1: tau }], vec![Seg::Arc { c: [85.0, 15.0], r: 3.0, a0: 0.0, a1: tau }]];
    Design { nodes: vec![plate, sq, two], params: vec![], names: vec![], profiles: vec![vec![entry], islands] }
}
