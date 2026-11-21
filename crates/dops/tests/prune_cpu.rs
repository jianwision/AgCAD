//! Per-cell tape pruning is exact: inside a cell, point values of the pruned tape are
//! bit-identical to the full tape.

use agcad_dops::abi::{fop, CNode, TABLE};
use agcad_dops::examples::{flange, pocketed_plate, shelled_box, two_bodies};
use agcad_dops::math::{compile_node, hash3, iv, tape_iv_prune, tape_sdf_c, tape_sdf_p, Prune};
use agcad_dops::Design;

fn compile(d: &Design) -> (Vec<CNode>, Vec<f32>) {
    let mut tabs = vec![0.0f32; d.nodes.len() * TABLE];
    let prm = if d.params.is_empty() { vec![0.0] } else { d.params.clone() };
    let c = d.nodes.iter().enumerate().map(|(i, n)| compile_node(n, &prm, &mut tabs[i * TABLE..(i + 1) * TABLE])).collect();
    (c, tabs)
}

fn u(h: u32) -> f32 { (h >> 8) as f32 / 16_777_216.0 }

#[test]
fn pruned_point_values_are_bit_identical_and_pruning_happens() {
    let designs = [flange(60.0, 12.0, 12.0), flange(67.3, 9.1, 7.9), pocketed_plate(100.0, 3.0), two_bodies(fop::REMOVE, 12.0), shelled_box(2.0)];
    let (mut cells, mut pruned_terms) = (0u64, 0u64);
    for (di, d) in designs.iter().enumerate() {
        let (t, tb) = compile(d);
        for k in 0..4000u32 {
            let h = |j: u32| u(hash3(di as u32, k, j));
            let s = 0.3 + 6.0 * h(9);
            let lo = [-80.0 + 180.0 * h(0), -80.0 + 160.0 * h(1), -4.0 + 26.0 * h(2)];
            let b = [iv(lo[0], lo[0] + s), iv(lo[1], lo[1] + s), iv(lo[2], lo[2] + s * 0.5)];
            let mut pr = Prune::NONE;
            tape_iv_prune(&t, &tb, b[0], b[1], b[2], 1e-3, &mut pr);
            cells += 1;
            pruned_terms += pr.skip.count_ones() as u64 + pr.inst.iter().map(|m| m.count_zeros() as u64).sum::<u64>();
            for q in 0..8u32 {
                let p = [b[0].lo + s * h(10 + 3 * q), b[1].lo + s * h(11 + 3 * q), b[2].lo + 0.5 * s * h(12 + 3 * q)];
                let (a, c) = (tape_sdf_c(&t, &tb, p[0], p[1], p[2]), tape_sdf_p(&t, &tb, &pr, p[0], p[1], p[2]));
                assert_eq!(a.to_bits(), c.to_bits(), "design {di} cell {k}: {a} vs pruned {c}");
            }
        }
    }
    assert!(pruned_terms > cells, "pruning should remove terms ({pruned_terms} over {cells} cells)");
}
