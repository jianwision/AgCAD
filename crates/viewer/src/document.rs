use super::media::{Frame, Vertex};
use agcad_geo::{
    math::{V3, v3},
    surface::Surface,
    tessellate::{Mesh, try_tessellate_with_edges},
};
use agcad_harness::Requirements;
use agcad_model::Observation;
use agcad_raytrace::{CpuTracer, Ray};
use agcad_server::Server;
use serde_json::{Value, json};
use std::sync::Arc;

mod picking;
use picking::BodyTopology;

pub struct Scene {
    pub id: u64,
    pub revision: u32,
    pub vertices: Vec<Vertex>,
    center: V3,
    radius: f64,
    tracer: CpuTracer,
    face_of: Vec<u32>,
    topology: Vec<BodyTopology>,
    pub faces: Vec<Value>,
    pub description: Value,
    pub tree: Value,
    pub report: Value,
}
fn f3(v: V3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

impl Scene {
    pub fn empty() -> Arc<Self> {
        Arc::new(Self {
            id: 0,
            revision: 0,
            vertices: vec![],
            center: V3::ZERO,
            radius: 1.,
            tracer: CpuTracer::new(&Mesh::default()),
            face_of: vec![],
            topology: vec![],
            faces: vec![],
            description: json!({"bodies":0,"units":"mm"}),
            tree: json!({"revision":0,"features":[]}),
            report: Value::Null,
        })
    }
}

#[derive(Clone)]
pub struct Source {
    pub requirements: Requirements,
    pub base: Value,
    pub edits: Vec<Value>,
}
impl Source {
    pub fn build(&self, id: u64, fit: Option<&Scene>) -> Result<Arc<Scene>, String> {
        let mut s = Server::new(self.requirements.clone(), "part");
        let mut report = Value::Null;
        for program in (!self.base.is_null())
            .then_some(&self.base)
            .into_iter()
            .chain(&self.edits)
        {
            report = s.call_tool("run_program", &json!({"program":program}))?["report"].clone();
            if report["status"] != "ACCEPTED" {
                if let Some(failed) =
                    report["features"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .find(|f| {
                            f["status"] != "ACCEPTED" && f["message"]["untrusted_text"].is_string()
                        })
                {
                    return Err(format!(
                        "{}: {}",
                        failed["feature"]["untrusted_text"]
                            .as_str()
                            .unwrap_or("Rebuild"),
                        failed["message"]["untrusted_text"].as_str().unwrap()
                    ));
                }
                return Err(format!("Rebuild failed: {report}"));
            }
        }
        let tree = s.call_tool("specification_tree", &json!({}))?;
        let mut description = s.call_tool("describe", &json!({}))?;
        let product = s
            .session
            .accepted_bodies
            .get("main")
            .and_then(|accepted| accepted.exact.as_ref());
        let layered = match product {
            Some(agcad_runtime::rebuild::Product::Solid(agcad_geo::kernel::Solid::Layered(l))) => {
                Some(l)
            }
            _ => None,
        };
        description["vertical_fillets"] =
            json!(layered.is_some_and(|l| l.layers.iter().any(|slab| {
                slab.regions
                    .iter()
                    .any(|p| p.loops.iter().any(|lp| lp.len() > 1))
            })));
        let body = product
            .and_then(|p| p.exact_body())
            .ok_or("The viewport needs an exact solid or sheet")?;
        let (mesh, paths) =
            try_tessellate_with_edges(&body, &Default::default()).map_err(|e| e.to_string())?;
        let topology = BodyTopology::new("edited".into(), &body, &mesh, paths);
        let mut faces = face_info(&body, layered);
        for (index, face) in faces.iter_mut().enumerate() {
            face["entity"] = json!(format!(
                "r{}-{}",
                if self.base.is_null() {
                    self.edits.len() as u64 + 1
                } else {
                    id
                },
                index
            ));
            face["body"] = json!("edited");
        }
        Scene::from_mesh(
            tree["revision"].as_u64().unwrap() as u32,
            mesh,
            vec![topology],
            faces,
            description,
            tree,
            report,
            fit,
        )
    }
    pub fn observation(
        &self,
        scene: &Scene,
        prompt: &str,
        selections: Vec<Value>,
        history: &Value,
    ) -> Observation {
        let turns = history.as_array().map(Vec::as_slice).unwrap_or_default();
        let history: Vec<_> = turns[turns.len().saturating_sub(12)..]
            .iter()
            .filter_map(|turn| {
                let role = turn.get("role")?.as_str()?;
                if !matches!(role, "user" | "assistant") {
                    return None;
                }
                let text: String = turn.get("text")?.as_str()?.chars().take(2500).collect();
                Some(json!({"role":role,"text":text}))
            })
            .collect();
        let selection = selections.last().unwrap_or(&Value::Null);
        let face = selections.iter().rev().find(|v| v["kind"] == "face");
        let extrusion_face = if scene.description["bodies"].as_u64().unwrap_or(1) > 1 {
            None
        } else if selections.len() > 1 {
            None
        } else if !selection.is_null() {
            face.and_then(|f| f["index"].as_u64())
                .map(|i| i as usize)
                .filter(|&i| scene.faces.get(i).is_some_and(|f| f["can_extrude"] == true))
        } else {
            let mut ends = scene
                .faces
                .iter()
                .enumerate()
                .filter(|(_, f)| f["name"] == "cap_end" && f["can_extrude"] == true);
            ends.next()
                .map(|(i, _)| i)
                .filter(|_| ends.next().is_none())
        };
        Observation {
            requirements: json!(self.requirements),
            revision: scene.revision,
            requirements_rev: self.requirements.rev,
            report: json!({"prompt":prompt,"conversation":history,"empty":scene.faces.is_empty(),"selected_entity":selection,"selections":selections,"selected_face":face,"specification_tree":scene.tree,"body":scene.description,
                "extrusion_face":extrusion_face,"block_edges":scene.block_edges(),"all_edge_fillet":scene.topology.len()==1 && !scene.block_edges().is_empty(),"vertical_corner_fillet":scene.description["vertical_fillets"]==true,"accepted_edits":self.edits,"initial_feature":(!self.base.is_null()).then_some("imported"),"modeling_operations":viewport_operations(),
                "starter_shapes": ["For a new block use primitive: {shape: 'block', width: mm, depth: mm, height: mm}. For a new cylinder use primitive: {shape: 'cylinder', diameter: mm, height: mm}."],
                "response_formats":["{answer: text}","{answer: text, scope: face or body, finish: {color: #rrggbb, roughness: 0..1, metalness: 0..1, designation?: string, density_kg_m3?: number}}","{program: AgIR program envelope}"]}),
            ..Default::default()
        }
    }
    pub fn check_edit(&self, raw: &Value) -> Result<Value, String> {
        let program = agcad_ir::json::decode_program(&raw.to_string(), agcad_ir::Catalog::global())
            .map_err(|e| e.to_string())?;
        if program.statements.is_empty() {
            return Err("The model returned an empty edit".into());
        }
        for stmt in &program.statements {
            if viewport_operations().binary_search(&stmt.op).is_err() {
                return Err(format!(
                    "{} is outside the viewport's geometry operations",
                    stmt.op
                ));
            }
        }
        serde_json::from_str(&agcad_ir::json::encode_program(&program)).map_err(|e| e.to_string())
    }
}

pub fn viewport_operations() -> &'static [String] {
    static OPS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    OPS.get_or_init(|| {
        agcad_std::standard_registry()
            .implemented()
            .into_iter()
            .filter(|op| {
                op == "direct.push_pull"
                    || matches!(
                        op.split('.').next(),
                        Some(
                            "knowledge"
                                | "sketch"
                                | "part"
                                | "transform"
                                | "shape"
                                | "wireframe"
                                | "surface"
                                | "datum"
                                | "gsd"
                        )
                    )
            })
            .collect()
    })
}

fn face_info(
    body: &agcad_geo::Brep,
    layered: Option<&agcad_geo::layered::LayeredSolid>,
) -> Vec<Value> {
    body.faces
        .iter()
        .enumerate().map(|(index,f)| {
            let surface = &body.surfaces[f.surface as usize];
            let mut info = json!({"name":f.name,"same_sense":f.same_sense,"can_extrude":body.kind==agcad_geo::brep::BodyKind::Solid && agcad_std::push_pull::supported(body,layered,index)});
            info["surface"] = match surface {
                Surface::Cylinder { frame, r } => {
                    json!({"kind":"Cylinder","diameter_mm":2.*r,"frame":frame})
                }
                Surface::Plane { frame } => json!({"kind":"Plane","frame":frame}),
                Surface::Sphere { frame, r } => {
                    json!({"kind":"Sphere","radius_mm":r,"frame":frame})
                }
                Surface::Cone { .. } => json!({"kind":"Cone"}),
                Surface::Torus { .. } => json!({"kind":"Torus"}),
                Surface::Extrusion { .. } => json!({"kind":"Extrusion"}),
                Surface::Revolution { .. } => json!({"kind":"Revolution"}),
                Surface::Nurbs(_) => json!({"kind":"NURBS"}),
                Surface::Offset { .. } => json!({"kind":"Offset"}),
            };
            info
        })
        .collect()
}

impl Scene {
    #[allow(clippy::too_many_arguments)]
    fn from_mesh(
        revision: u32,
        mesh: Mesh,
        topology: Vec<BodyTopology>,
        faces: Vec<Value>,
        description: Value,
        tree: Value,
        report: Value,
        fit: Option<&Scene>,
    ) -> Result<Arc<Self>, String> {
        if mesh.triangles.is_empty()
            || mesh.triangles.len() > u32::MAX as usize / (3 * size_of::<Vertex>())
        {
            return Err("Empty or oversized display mesh".into());
        }
        let (center, radius) = fit
            .filter(|s| !s.vertices.is_empty())
            .map(|s| (s.center, s.radius))
            .unwrap_or_else(|| {
                let mut bb = agcad_geo::math::Aabb::EMPTY;
                for p in &mesh.positions {
                    bb.add(*p);
                }
                let center = (bb.min + bb.max) * 0.5;
                let r = mesh
                    .positions
                    .iter()
                    .map(|p| p.dist(center))
                    .fold(0., f64::max);
                (center, (1.15 * r).max(0.001))
            });
        let mut vertices = Vec::with_capacity(mesh.triangles.len() * 3);
        for (tri, face) in mesh.triangles.iter().zip(&mesh.face_of) {
            let [a, b, c] = tri.map(|i| mesh.positions[i as usize]);
            let normal = f3((b - a).cross(c - a).normalized());
            for p in [a, b, c] {
                vertices.push(Vertex {
                    xyz: f3((p - center) * (1. / radius)),
                    normal,
                    face: *face,
                });
            }
        }
        // Distinct candidates can share a document revision after discard or a part switch.
        static NEXT_SCENE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Ok(Arc::new(Scene {
            id: NEXT_SCENE.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            revision,
            vertices,
            center,
            radius,
            tracer: CpuTracer::new(&mesh),
            face_of: mesh.face_of,
            topology,
            faces,
            description,
            tree,
            report,
        }))
    }

    pub fn step(src: &str) -> Result<Arc<Self>, String> {
        let (bodies, report, styles) =
            agcad_geo::step_read::read_styled_step(src, &Default::default())?;
        let mut mesh = Mesh::default();
        let mut faces = Vec::new();
        let mut topology = Vec::new();
        for (body, style) in bodies.iter().zip(styles) {
            let (part, paths) =
                try_tessellate_with_edges(body, &Default::default()).map_err(|e| e.to_string())?;
            topology.push(BodyTopology::new(
                style.entity.to_string(),
                body,
                &part,
                paths,
            ));
            mesh.append(&part, faces.len() as u32);
            for (mut meta, (entity, color)) in face_info(body, None).into_iter().zip(style.faces) {
                meta["entity"] = json!(entity.to_string());
                meta["body"] = json!(style.entity.to_string());
                meta["body_name"] = json!(style.name);
                meta["material"] = json!(style.material);
                meta["body_color"] = json!(style.color);
                meta["face_color"] = json!(color);
                faces.push(meta);
            }
        }
        if bodies.is_empty() {
            return Err("STEP contains no supported bodies".into());
        }
        Self::from_mesh(
            1,
            mesh,
            topology,
            faces,
            json!({"bodies":bodies.len(),"schema":report.schema,"kind":format!("{:?}",bodies[0].kind)}),
            Value::Null,
            json!(report.warnings),
            None,
        )
    }
}
