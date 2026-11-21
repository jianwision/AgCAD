//! # agcad-std — standard features
//!
//! CATIA-canonical standard features implementing `agcad_runtime::FeatureImpl` against the
//! catalog manifests. Ops without an implementation report `UNSUPPORTED_OP` (AG-07-002).
//! Pass P7 coverage: sketch + constraints (LM-solved), Part Design (pad, pocket, shaft,
//! groove, hole, primitive, implicit body, lattice), booleans (exact layered + implicit
//! smooth), patterns/transforms, mesh tessellation + implicit meshing, evaluations incl.
//! DFM, queries, Knowledgeware parameters, datums, STEP/STL/3MF export.

pub mod assembly;
pub mod boolean;
pub mod evaluate;
pub mod file;
pub mod knowledge;
pub mod mesh;
pub mod part;
pub mod physics;
pub mod sketch;
pub mod transform;
pub mod util;

use agcad_runtime::rebuild::Registry;
use std::sync::Arc;

pub fn standard_registry() -> Registry {
    let mut r = Registry::default();
    for op in ["sketch.create", "sketch.point", "sketch.line", "sketch.circle", "sketch.arc", "sketch.rectangle", "sketch.profile", "sketch.polygon", "sketch.elongated_hole", "sketch.solve", "sketch.regions"] {
        r.register(Arc::new(sketch::SketchOp(op)));
    }
    for op in ["constraint.coincidence", "constraint.horizontal", "constraint.vertical", "constraint.parallelism", "constraint.perpendicularity", "constraint.equal", "constraint.distance", "constraint.radius", "constraint.diameter", "constraint.fix", "constraint.concentricity", "constraint.tangency"] {
        r.register(Arc::new(sketch::ConstraintOp(op)));
    }
    for op in ["part.pad", "part.pocket", "part.shaft", "part.groove", "part.primitive", "part.hole", "part.implicit_body", "part.lattice", "part.edge_fillet", "part.chamfer"] {
        r.register(Arc::new(part::PartOp(op)));
    }
    for op in ["boolean.add", "boolean.remove", "boolean.intersect", "boolean.smooth_boolean"] {
        r.register(Arc::new(boolean::BooleanOp(op)));
    }
    for op in ["transform.circular_pattern", "transform.rectangular_pattern", "transform.user_pattern", "transform.translation", "transform.rotation", "transform.symmetry", "transform.mirror"] {
        r.register(Arc::new(transform::TransformOp(op)));
    }
    for op in ["mesh.tessellate", "convert.implicit_to_mesh"] { r.register(Arc::new(mesh::MeshOp(op))); }
    for op in ["evaluate.measure_inertia", "evaluate.bounding_box", "evaluate.check", "evaluate.thickness_analysis", "evaluate.overhang_analysis", "evaluate.draft_analysis", "evaluate.internal_radius_analysis", "evaluate.sample", "evaluate.clash", "evaluate.measure_item"] {
        r.register(Arc::new(evaluate::EvalOp(op)));
    }
    for op in ["assembly.component", "assembly.fix", "assembly.coincidence", "assembly.contact", "assembly.offset", "assembly.angle", "assembly.update"] { r.register(Arc::new(assembly::AssemblyOp(op))); }
    for op in ["evaluate.stiffness", "evaluate.modal", "study.topology_optimization"] { r.register(Arc::new(physics::PhysicsOp(op))); }
    for op in ["query.parallel_to", "query.element_type", "query.geometry", "query.created_by", "query.named"] { r.register(Arc::new(evaluate::QueryOp(op))); }
    for op in ["knowledge.parameter", "datum.plane", "datum.axis_system", "datum.point"] { r.register(Arc::new(knowledge::KnowledgeOp(op))); }
    for op in ["file.export_step", "file.export_mesh", "file.export_3mf", "file.export_usd", "file.export_feature_script"] { r.register(Arc::new(file::FileOp(op))); }
    r.register(Arc::new(file::ImportStep));
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use agcad_geo::kernel::NativeKernel;
    use agcad_ir::program::Value;
    use agcad_ir::Quantity;
    use agcad_runtime::document::PartState;
    use agcad_runtime::rebuild::{update, Product};
    use std::collections::BTreeMap;

    fn m(kv: &[(&str, Value)]) -> BTreeMap<String, Value> { kv.iter().map(|(k, v)| (k.to_string(), v.clone())).collect() }
    fn r(id: u32) -> Value { Value::Ref { id: format!("f{id}"), field: None } }
    fn mm(x: f64) -> Value { Value::Quantity(Quantity::mm(x)) }
    fn p2(x: f64, y: f64) -> Value { Value::List(vec![mm(x), mm(y)]) }

    #[test]
    fn bolted_flange_end_to_end() {
        let mut s = PartState::new("flange");
        let sk = s.add_feature("sketch.create", 1, m(&[("support", Value::String("xy".into()))]), 0, None);
        let c = s.add_feature("sketch.circle", 1, m(&[("sketch", r(sk)), ("mode", Value::String("center_radius".into())), ("pts", Value::List(vec![p2(0.0, 0.0)])), ("r", mm(60.0))]), 0, None);
        let reg = s.add_feature("sketch.regions", 1, m(&[("sketch", r(c))]), 0, None);
        let _pad = s.add_feature("part.pad", 1, m(&[("profile", r(reg)), ("limits", mm(12.0))]), 0, Some("disc".into()));
        let mut spec = BTreeMap::new();
        spec.insert("d".to_string(), mm(9.0));
        spec.insert("through".to_string(), Value::Bool(true));
        let hole = s.add_feature("part.hole", 1, m(&[("support", Value::String("cap_end".into())), ("positions", Value::List(vec![p2(50.0, 0.0)])), ("spec", Value::Map(spec))]), 0, None);
        let pat = s.add_feature("transform.circular_pattern", 1, m(&[("items", Value::List(vec![r(hole)])), ("axis", Value::String("z".into())), ("count", Value::Int(6)), ("angle", Value::Float(core::f64::consts::TAU))]), 0, None);
        let chk = s.add_feature("evaluate.check", 1, m(&[("body", r(pat))]), 0, None);
        let inert = s.add_feature("evaluate.measure_inertia", 1, m(&[("items", Value::List(vec![r(pat)]))]), 0, None);
        let step = s.add_feature("file.export_step", 1, m(&[("items", Value::List(vec![r(pat)])), ("path", Value::String("flange.stp".into()))]), 0, None);
        let reg_ = standard_registry();
        let res = update(&s, &reg_, &NativeKernel, BTreeMap::new(), vec![]);
        for o in &res.outcomes { assert!(o.status.is_ok(), "{} failed: {:?}", o.feature_ref, o.error); }
        let Product::Json(j) = &res.products[&chk] else { panic!() };
        assert_eq!(j["valid"], true, "{j}");
        let Product::Json(mi) = &res.products[&inert] else { panic!() };
        let exact = core::f64::consts::PI * (60.0f64.powi(2) - 6.0 * 4.5f64.powi(2)) * 12.0;
        let vol = mi["volume"].as_f64().unwrap();
        assert!((vol - exact).abs() / exact < 1e-4, "{vol} vs {exact}");
        let Product::Other(o) = &res.products[&step] else { panic!() };
        let fb = o.downcast_ref::<file::FileBytes>().unwrap();
        let txt = String::from_utf8(fb.bytes.clone()).unwrap();
        let (bodies, _) = agcad_geo::step_read::read_step(&txt, &Default::default()).unwrap();
        assert!(bodies[0].validate().valid);
    }

    #[test]
    fn gyroid_block_meshes_watertight() {
        let mut s = PartState::new("lattice");
        let src = "#[implicit] fn blk(p: Vec3) -> Length { let q = vec3(p.x.abs(), p.y.abs(), p.z.abs()) - vec3(10.0*mm, 10.0*mm, 10.0*mm); q.x.max(q.y).max(q.z) }";
        let b = s.add_feature("part.implicit_body", 1, m(&[("source", Value::String(src.into())), ("bounds", Value::List(vec![Value::List(vec![mm(-11.0), mm(-11.0), mm(-11.0)]), Value::List(vec![mm(11.0), mm(11.0), mm(11.0)])]))]), 0, None);
        let lat = s.add_feature("part.lattice", 1, m(&[("region", r(b)), ("cell", Value::String("gyroid".into())), ("size", mm(5.0)), ("wall", mm(0.8))]), 0, None);
        let mesh = s.add_feature("convert.implicit_to_mesh", 1, m(&[("body", r(lat)), ("res", mm(0.5))]), 0, None);
        let res = update(&s, &standard_registry(), &NativeKernel, BTreeMap::new(), vec![]);
        for o in &res.outcomes { assert!(o.status.is_ok(), "{} failed: {:?}", o.feature_ref, o.error); }
        let Product::Mesh(mm_) = &res.products[&mesh] else { panic!() };
        assert!(mm_.triangles.len() > 1000);
        assert!(mm_.is_watertight());
    }
}
