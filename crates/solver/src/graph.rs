//! Constraint-graph analysis (spec 14 § Constraint solving; `constraint.analyze`):
//! structural rank (Hopcroft–Karp), DOF, Dulmage–Mendelsohn decomposition into over-,
//! well- and under-determined parts, conflict sets, and a block-triangular solve plan
//! (Tarjan SCC) so large sketches/assemblies solve as many small dense blocks.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Bipartite incidence: constraint `i` depends on variables `cons[i]`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConstraintGraph {
    pub n_vars: usize,
    pub cons: Vec<Vec<u32>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Block {
    pub vars: Vec<u32>,
    pub cons: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphReport {
    pub n_vars: usize,
    pub n_cons: usize,
    pub structural_rank: usize,
    /// Remaining structural degrees of freedom (`n_vars - rank`).
    pub dof: usize,
    /// Constraints in the over-determined part (conflict candidates; `OVER_CONSTRAINED`).
    pub over_constraints: Vec<u32>,
    /// Variables in the under-determined part (free directions; `UNDER_CONSTRAINED`).
    pub under_vars: Vec<u32>,
    /// Well-determined blocks in solve order (block triangular form).
    pub blocks: Vec<Block>,
}

const NIL: u32 = u32::MAX;

/// Maximum matching constraint→variable (Hopcroft–Karp). Returns (match_c, match_v).
pub fn max_matching(g: &ConstraintGraph) -> (Vec<u32>, Vec<u32>) {
    let nc = g.cons.len();
    let mut mc = vec![NIL; nc];
    let mut mv = vec![NIL; g.n_vars];
    let mut dist = vec![0u32; nc];
    loop {
        // BFS from free constraints
        let mut q = VecDeque::new();
        for c in 0..nc {
            if mc[c] == NIL { dist[c] = 0; q.push_back(c); } else { dist[c] = u32::MAX; }
        }
        let mut found = false;
        while let Some(c) = q.pop_front() {
            for &v in &g.cons[c] {
                let w = mv[v as usize];
                if w == NIL { found = true; } else if dist[w as usize] == u32::MAX { dist[w as usize] = dist[c] + 1; q.push_back(w as usize); }
            }
        }
        if !found { break; }
        fn dfs(c: usize, g: &ConstraintGraph, mc: &mut [u32], mv: &mut [u32], dist: &mut [u32]) -> bool {
            for i in 0..g.cons[c].len() {
                let v = g.cons[c][i] as usize;
                let w = mv[v];
                if w == NIL || (dist[w as usize] == dist[c] + 1 && dfs(w as usize, g, mc, mv, dist)) {
                    mc[c] = v as u32;
                    mv[v] = c as u32;
                    return true;
                }
            }
            dist[c] = u32::MAX;
            false
        }
        let mut aug = false;
        for c in 0..nc {
            if mc[c] == NIL && dfs(c, g, &mut mc, &mut mv, &mut dist) { aug = true; }
        }
        if !aug { break; }
    }
    (mc, mv)
}

pub fn analyze(g: &ConstraintGraph) -> GraphReport {
    let nc = g.cons.len();
    let (mc, mv) = max_matching(g);
    let rank = mc.iter().filter(|m| **m != NIL).count();
    // var -> constraints adjacency
    let mut vcons: Vec<Vec<u32>> = vec![Vec::new(); g.n_vars];
    for (c, vs) in g.cons.iter().enumerate() { for &v in vs { vcons[v as usize].push(c as u32); } }
    // Over-determined: constraints reachable from unmatched constraints by alternating paths
    // (constraint -edge-> var -matching-> constraint).
    let mut over_c = vec![false; nc];
    let mut q: VecDeque<usize> = (0..nc).filter(|&c| mc[c] == NIL).collect();
    for &c in &q { over_c[c] = true; }
    let mut over_v = vec![false; g.n_vars];
    while let Some(c) = q.pop_front() {
        for &v in &g.cons[c] {
            if over_v[v as usize] { continue; }
            over_v[v as usize] = true;
            let w = mv[v as usize];
            if w != NIL && !over_c[w as usize] { over_c[w as usize] = true; q.push_back(w as usize); }
        }
    }
    // Under-determined: variables reachable from unmatched variables (var -edge-> cons -matching-> var).
    let mut under_v = vec![false; g.n_vars];
    let mut under_c = vec![false; nc];
    let mut qv: VecDeque<usize> = (0..g.n_vars).filter(|&v| mv[v] == NIL).collect();
    for &v in &qv { under_v[v] = true; }
    while let Some(v) = qv.pop_front() {
        for &c in &vcons[v] {
            if under_c[c as usize] { continue; }
            under_c[c as usize] = true;
            let w = mc[c as usize];
            if w != NIL && !under_v[w as usize] { under_v[w as usize] = true; qv.push_back(w as usize); }
        }
    }
    // Well-determined part: matched constraints not in over/under; SCCs of the directed graph
    // c -> c' if c uses var v matched to c' (Tarjan), emitted in dependency order.
    let well: Vec<usize> = (0..nc).filter(|&c| !over_c[c] && !under_c[c] && mc[c] != NIL).collect();
    let mut index = vec![u32::MAX; nc];
    let mut low = vec![0u32; nc];
    let mut on = vec![false; nc];
    let mut stack: Vec<usize> = Vec::new();
    let mut counter = 0u32;
    let mut blocks: Vec<Block> = Vec::new();
    let in_well: Vec<bool> = (0..nc).map(|c| well.contains(&c)).collect();
    fn strong(c: usize, g: &ConstraintGraph, mc: &[u32], mv: &[u32], in_well: &[bool], index: &mut [u32], low: &mut [u32], on: &mut [bool], stack: &mut Vec<usize>, counter: &mut u32, blocks: &mut Vec<Block>) {
        index[c] = *counter; low[c] = *counter; *counter += 1;
        stack.push(c); on[c] = true;
        for &v in &g.cons[c] {
            let d = mv[v as usize];
            if d == NIL || d as usize == c || !in_well[d as usize] { continue; }
            let d = d as usize;
            if index[d] == u32::MAX {
                strong(d, g, mc, mv, in_well, index, low, on, stack, counter, blocks);
                low[c] = low[c].min(low[d]);
            } else if on[d] {
                low[c] = low[c].min(index[d]);
            }
        }
        if low[c] == index[c] {
            let mut b = Block { vars: Vec::new(), cons: Vec::new() };
            loop {
                let w = stack.pop().unwrap();
                on[w] = false;
                b.cons.push(w as u32);
                b.vars.push(mc[w]);
                if w == c { break; }
            }
            b.cons.sort_unstable();
            b.vars.sort_unstable();
            blocks.push(b); // Tarjan emits SCCs in reverse topological order of dependencies = solve order
        }
    }
    for &c in &well {
        if index[c] == u32::MAX {
            strong(c, g, &mc, &mv, &in_well, &mut index, &mut low, &mut on, &mut stack, &mut counter, &mut blocks);
        }
    }
    GraphReport {
        n_vars: g.n_vars,
        n_cons: nc,
        structural_rank: rank,
        dof: g.n_vars - rank,
        over_constraints: (0..nc as u32).filter(|&c| over_c[c as usize]).collect(),
        under_vars: (0..g.n_vars as u32).filter(|&v| under_v[v as usize]).collect(),
        blocks,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn detects_under_over_and_blocks() {
        // vars: 0..4. c0: x0 ; c1: x0,x1 ; c2: x1 (redundant with c0,c1 structurally → over) ; x2,x3 free-ish
        let g = ConstraintGraph { n_vars: 4, cons: vec![vec![0], vec![0, 1], vec![1], vec![2, 3]] };
        let r = analyze(&g);
        assert_eq!(r.structural_rank, 3);
        assert_eq!(r.dof, 1);
        assert!(!r.over_constraints.is_empty());
        assert!(r.under_vars.contains(&2) || r.under_vars.contains(&3));
    }
    #[test]
    fn triangular_plan_orders_dependencies() {
        // c0: x0 ; c1: x0, x1 ; c2: x1, x2  → blocks {c0},{c1},{c2} in order
        let g = ConstraintGraph { n_vars: 3, cons: vec![vec![0], vec![0, 1], vec![1, 2]] };
        let r = analyze(&g);
        assert_eq!(r.dof, 0);
        assert_eq!(r.blocks.len(), 3);
        assert_eq!(r.blocks[0].cons, vec![0]);
        assert_eq!(r.blocks[2].cons, vec![2]);
    }
}
