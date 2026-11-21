//! Device operation registry: the single table that drives the allow-list, kernel dispatch
//! (via [`OpTable`]), documentation, fuzzing and lowering. Every entry is a catalog op
//! (CATIA ontology); opcodes are resolved from the catalog, never hard-coded.

use crate::abi::OpTable;
use agcad_ir::Catalog;

#[derive(Clone, Copy, Debug)]
pub struct DeviceOp {
    pub name: &'static str,
    /// What the device implementation does with the command record.
    pub semantics: &'static str,
    /// Lowerable to the exact B-rep path (layered prisms) without loss.
    pub exact: bool,
}

pub const DEVICE_OPS: &[DeviceOp] = &[
    DeviceOp { name: "document.update", semantics: "no-op slot filler (rebuild happens every round)", exact: true },
    DeviceOp { name: "knowledge.parameter", semantics: "args[0] = parameter index, args[1] = 0 set / 1 add, imm[0] = value", exact: true },
    DeviceOp { name: "part.pad", semantics: "append an add-material feature: args[0] = profile (rect/circle/polygon), arena payload = values", exact: true },
    DeviceOp { name: "part.pocket", semantics: "append a remove-material feature (same payload as part.pad)", exact: true },
    DeviceOp { name: "part.hole", semantics: "append a remove-material circular feature (profile circle)", exact: true },
    DeviceOp { name: "transform.circular_pattern", semantics: "pattern feature args[0] (u32::MAX = last) about z: arena v6 = count, v7 = total angle", exact: true },
    DeviceOp { name: "document.deactivate", semantics: "args[0] = feature index (u32::MAX = last), args[1] = 1 activate / 0 deactivate", exact: true },
    DeviceOp { name: "transform.rectangular_pattern", semantics: "pattern feature args[0] along x / y: arena v6 = count₁, v7 = spacing₁, v8 = count₂, v9 = spacing₂ (≤ 64 instances)", exact: true },
    DeviceOp { name: "part.edge_fillet", semantics: "round the vertical edges of rectangle feature args[0] (arena v10 = radius; exact), or with args[1] = EDGES_Z the top / bottom edges of any extrusion (v13; device-only → certified implicit)", exact: true },
    DeviceOp { name: "part.chamfer", semantics: "chamfer the vertical edges of rectangle feature args[0] (v10; exact), or the top / bottom edges with args[1] = EDGES_Z (v13; device-only)", exact: true },
    DeviceOp { name: "part.draft", semantics: "draft the walls of extrusion args[0]: arena v12 = tan θ about its z0 plane, |θ| ≤ 30° (device-only → certified implicit)", exact: false },
    DeviceOp { name: "transform.mirror", semantics: "mirror the finished body across x = 0 (arena v0 = 1) and / or y = 0 (v1 = 1); exact (layered reflection + union)", exact: true },
    DeviceOp { name: "part.shaft", semantics: "revolve pool sketch v15 (ρ, z) about the vertical axis through (v0, v1); exact for axis-aligned meridians, otherwise device-only", exact: true },
    DeviceOp { name: "part.groove", semantics: "remove a revolution (same payload as part.shaft)", exact: true },
    DeviceOp { name: "part.shell", semantics: "hollow the current body inward (closed): arena v0 = wall thickness. Exact (2D inward offset) for a single convex pad minus cuts; otherwise device-only → certified implicit body", exact: false },
    DeviceOp { name: "document.body", semantics: "start a new body (the current body is pushed; depth ≤ 4)", exact: true },
    DeviceOp { name: "boolean.add", semantics: "combine the pushed body (target) with the current body (tool): union", exact: true },
    DeviceOp { name: "boolean.remove", semantics: "target minus tool", exact: true },
    DeviceOp { name: "boolean.intersect", semantics: "target ∩ tool", exact: true },
];

/// Resolve the device dispatch table from the catalog. Panics if the catalog lost an op the
/// device layer implements (caught by the registry test on every build).
pub fn op_table(c: &Catalog) -> OpTable {
    let op = |n: &str| c.get(n).unwrap_or_else(|| panic!("device op {n} missing from the catalog")).opcode as u32;
    OpTable {
        update: op("document.update"),
        parameter: op("knowledge.parameter"),
        pad: op("part.pad"),
        pocket: op("part.pocket"),
        hole: op("part.hole"),
        circular_pattern: op("transform.circular_pattern"),
        deactivate: op("document.deactivate"),
        rect_pattern: op("transform.rectangular_pattern"),
        fillet: op("part.edge_fillet"),
        chamfer: op("part.chamfer"),
        shell: op("part.shell"),
        body: op("document.body"),
        add: op("boolean.add"),
        remove: op("boolean.remove"),
        intersect: op("boolean.intersect"),
        draft: op("part.draft"),
        mirror: op("transform.mirror"),
        shaft: op("part.shaft"),
        groove: op("part.groove"),
        _pad: [0; 1],
    }
}

/// Opcode allow-list (65 536-bit bitmap) of the device policy's capability manifest.
pub fn allow_list(c: &Catalog) -> Vec<u32> {
    let ops: Vec<u16> = DEVICE_OPS.iter().map(|d| c.get(d.name).expect("catalog").opcode).collect();
    agcad_harness::wavefront::allow_bitmap(&ops)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn registry_matches_catalog() {
        let c = Catalog::global();
        let t = op_table(c);
        assert_eq!(t.pad, c.get("part.pad").unwrap().opcode as u32);
        let allow = allow_list(c);
        for d in DEVICE_OPS {
            let o = c.get(d.name).unwrap().opcode as usize;
            assert!(allow[o >> 5] >> (o & 31) & 1 == 1, "{}", d.name);
        }
        // a catalog op that is not a device op stays disallowed
        let o = c.get("part.thread").unwrap().opcode as usize;
        assert_eq!(allow[o >> 5] >> (o & 31) & 1, 0);
    }
}
