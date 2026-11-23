use agcad_geo::tessellate::{TessOptions, tessellate};
use agcad_harness::Requirements;
use agcad_server::Server;
use serde_json::{Value, json};

fn server(kind: &str, limit: Option<f64>) -> Server {
    let mut req = json!({"id":"r","rev":1,"deliver":{"body_type":kind}});
    if let Some(limit) = limit {
        req["hard"] = json!([{"id":"size","measure":"evaluate.bounding_box","field":"max_extent","le":limit}]);
        req["phases"] = json!({"update":["size"]});
    }
    Server::new(Requirements::parse(&req.to_string()).unwrap(), "regression")
}

fn feature(id: &str, op: &str, args: Value) -> Value {
    json!({"id":id,"op":op,"args":args})
}

fn run(s: &mut Server, steps: Vec<Value>) -> Value {
    let rev = s.session.doc.workspace("main").unwrap().revision;
    s.call_tool(
        "run_program",
        &json!({"program":{"agir":"0.2","expected_revision":rev,
        "requirements_rev":1,"program":steps}}),
    )
    .unwrap()
}

fn radius(r: f64) -> Value {
    feature(
        "radius",
        "knowledge.parameter",
        json!({"name":"r","value":format!("{r} mm")}),
    )
}

fn exports(s: &mut Server) -> Vec<Vec<u8>> {
    ["step_ap242", "stl", "3mf"]
        .iter()
        .map(|fmt| {
            let out = s.call_tool("export", &json!({"format":fmt})).unwrap();
            std::fs::read(out["path"].as_str().unwrap()).unwrap()
        })
        .collect()
}

#[test]
fn exports_keep_the_accepted_revision_and_belong_to_the_session() {
    let mut s = server("SolidBody", Some(7.));
    let pad = vec![
        radius(2.),
        feature("s", "sketch.create", json!({"support":"xy"})),
        feature(
            "c",
            "sketch.circle",
            json!({"sketch":"%s","mode":"center_radius","pts":[[0,0]],"r":"$r"}),
        ),
        feature("g", "sketch.regions", json!({"sketch":"%c"})),
        feature("pad", "part.pad", json!({"profile":"%g","limits":"2 mm"})),
    ];
    assert_eq!(run(&mut s, pad.clone())["report"]["status"], "ACCEPTED");
    let baseline = exports(&mut s);
    for (r, status) in [(6., "REQUIREMENT_FAILED"), (-1., "FEATURE_FAILED")] {
        assert_eq!(run(&mut s, vec![radius(r)])["report"]["status"], status);
        assert_eq!(exports(&mut s), baseline);
        assert_eq!(
            s.call_tool("describe", &json!({})).unwrap()["parameters"]["r"],
            2.
        );
    }
    assert_eq!(
        run(&mut s, vec![radius(3.)])["report"]["status"],
        "ACCEPTED"
    );
    assert!(exports(&mut s).iter().zip(&baseline).all(|(a, b)| a != b));
    let mut other = server("SurfaceBody", None);
    assert_ne!(s.session.out_dir, other.session.out_dir);
    let mut mismatch = pad;
    mismatch.push(feature("finish", "document.finish", json!({})));
    let result = run(&mut other, mismatch);
    assert_eq!(result["report"]["status"], "REQUIREMENT_FAILED");
    assert_eq!(result["delivery"]["requirements_met"], false);
    assert_eq!(result["delivery"]["delivery_verified"], false);
    assert!(other.call_tool("export", &json!({})).is_err());
}

fn circles(kind: &str, prefix: &str, sections: &[(f64, f64)]) -> Vec<Value> {
    let mut steps = Vec::new();
    let mut profiles = Vec::new();
    for (i, (z, r)) in sections.iter().enumerate() {
        let label = format!("{prefix}{i}");
        steps.extend([
            feature(&format!("p{label}"),"datum.plane",json!({"mode":"through","refs":["xy"],"value":[0,0,z]})),
            feature(&format!("s{label}"),"sketch.create",json!({"support":format!("%p{label}")})),
            feature(&format!("c{label}"),"sketch.circle",json!({"sketch":format!("%s{label}"),"mode":"center_radius","pts":[[0,0]],"r":format!("{r} mm")})),
            feature(&format!("g{label}"),"sketch.regions",json!({"sketch":format!("%c{label}")})),
        ]);
        profiles.push(format!("%g{label}"));
    }
    steps.push(feature(
        prefix,
        kind,
        json!({"coupling":"ratio","sections":profiles}),
    ));
    steps
}

#[test]
#[ignore = "requires CUDA or AGCAD_ALLOW_CPU_REFERENCE=1"]
fn loft_volume_converges_and_interior_pinch_is_rejected() {
    let op = "part.multi_sections_solid";
    for (sections, exact) in [
        (vec![(0., 10.), (20., 10.)], 2000. * std::f64::consts::PI),
        (
            vec![(0., 10.), (10., 20.), (20., 10.)],
            20. * std::f64::consts::PI * (400. - 400. / 3. + 20.),
        ),
    ] {
        let mut s = server("SolidBody", None);
        let built = run(&mut s, circles(op, "loft", &sections));
        assert_eq!(built["report"]["status"], "ACCEPTED", "{built}");
        let body = s.session.accepted_bodies["main"]
            .exact
            .as_ref()
            .unwrap()
            .exact_body()
            .unwrap();
        let mut error = f64::INFINITY;
        for sag in [0.1, 0.01] {
            let mesh = tessellate(
                &body,
                &TessOptions {
                    sag,
                    ..Default::default()
                },
            );
            assert!(mesh.is_watertight());
            let relative = (mesh.mass_properties().0 / exact - 1.).abs();
            assert!(relative < error, "volume error {relative} at sag {sag}");
            error = relative;
        }
        assert!(error < 0.002, "volume error {error}");
    }
    let mut s = server("SolidBody", None);
    let result = run(
        &mut s,
        circles(op, "pinch", &[(0., 25.), (100., 1.), (200., 1.)]),
    );
    assert_eq!(result["report"]["status"], "FEATURE_FAILED");
    assert!(
        result["report"]["features"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f["status"] == "TOPOLOGY_FAILED"
                && f["message"]["untrusted_text"]
                    .as_str()
                    .unwrap()
                    .contains("sections 2 and 3"))
    );
}

#[test]
#[ignore = "requires CUDA or AGCAD_ALLOW_CPU_REFERENCE=1"]
fn sheet_inspection_mesh_and_open_blend_chaining() {
    let mut s = server("SurfaceBody", Some(100.));
    let mut steps = Vec::new();
    for (i, id) in ["a", "b", "c", "d"].iter().enumerate() {
        let z = 20. * i as f64;
        steps.extend(circles(
            "surface.multi_sections_surface",
            id,
            &[(z, 10.), (z + 5., 10.)],
        ));
    }
    for (id, scope, name) in [
        ("ae", "a", "side.0/end"),
        ("bs", "b", "side.0/start"),
        ("ce", "c", "side.0/end"),
        ("ds", "d", "side.0/start"),
    ] {
        steps.push(feature(
            id,
            "query.named",
            json!({"scope":format!("%{scope}"),"names":[name]}),
        ));
    }
    for (id, a, b) in [("ab", "ae", "bs"), ("cd", "ce", "ds")] {
        steps.push(feature(
            id,
            "surface.blend",
            json!({"a":format!("%{a}"),"b":format!("%{b}"),"order":[1,1]}),
        ));
    }
    steps.extend([
        feature(
            "end",
            "query.named",
            json!({"scope":"%ab","names":["section.end"]}),
        ),
        feature(
            "start",
            "query.named",
            json!({"scope":"%cd","names":["section.start"]}),
        ),
        feature(
            "bridge",
            "surface.blend",
            json!({"a":"%end","b":"%start","order":[1,1]}),
        ),
        feature(
            "stl",
            "file.export_mesh",
            json!({"items":["%bridge"],"format":"stl","path":"bridge.stl"}),
        ),
        feature("finish", "document.finish", json!({})),
    ]);
    let result = run(&mut s, steps);
    assert_eq!(result["report"]["status"], "ACCEPTED", "{result}");
    assert_eq!(result["delivery"]["delivery_verified"], true);
    assert!(
        s.session
            .out_dir
            .join("bridge.stl")
            .metadata()
            .unwrap()
            .len()
            > 84
    );
    assert_eq!(
        s.call_tool("query", &json!({})).unwrap()["elements"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let description = s.call_tool("describe", &json!({})).unwrap();
    assert!(description["volume"].is_null());
    assert!(description["area"].as_f64().unwrap() > 0.);
}

/// Run each backend in a fresh process so other tests and CUDA sessions keep their policy.
fn translated_patch(t: f64) -> agcad_geo::brep::Brep {
    use agcad_geo::{
        brep::BodyKind,
        curve::Curve,
        math::v3,
        nurbs::{NurbsCurve, NurbsSurface},
        surface::Surface,
    };
    let k = vec![0., 0., 0., 0., 1., 2., 3., 4., 5., 6., 6., 6., 6.];
    let mut s = NurbsSurface {
        du: 3,
        dv: 3,
        nu: 9,
        nv: 9,
        ku: k.clone(),
        kv: k.clone(),
        poles: vec![],
        weights: vec![1.; 81],
    };
    for i in 0..9 {
        for j in 0..9 {
            if i > 0 && j > 0 && i < 8 && j < 8 {
                s.weights[9 * i + j] = 0.65 + ((i * 7 + j * 5) % 7) as f64 * 0.23;
            }
            let h = if i * j == 0 || i == 8 || j == 8 {
                0.
            } else {
                ((3 * i + j) as f64).sin() * 1.7
            };
            s.poles
                .push(v3(t + i as f64 * 2., -2. * t + j as f64, 3. * t + h));
        }
    }
    let mut b = agcad_geo::brep::Brep::new(BodyKind::Sheet);
    let sh = b.add_shell(false);
    let pts = [0, 72, 80, 8].map(|idx| b.add_vertex(s.poles[idx]));
    let mut edges = vec![];
    for side in 0..4 {
        let ids: Vec<usize> = (0..9)
            .map(|i| match side {
                0 => 9 * i,
                1 => 72 + i,
                2 => (8 - i) * 9 + 8,
                _ => 8 - i,
            })
            .collect();
        let c = NurbsCurve {
            degree: 3,
            knots: k.clone(),
            poles: ids.iter().map(|&i| s.poles[i]).collect(),
            weights: vec![1.; 9],
        };
        let x = b.add_curve(Curve::Nurbs(c));
        edges.push((
            b.add_edge(
                x,
                pts[side],
                pts[(side + 1) % 4],
                0.,
                6.,
                format!("e{side}"),
            ),
            true,
        ));
    }
    let id = b.add_surface(Surface::Nurbs(s));
    b.add_face(id, true, sh, &edges, &[], "patch", [0., 6., 0., 6.]);
    b
}

#[test]
#[ignore = "requires a CUDA device"]
fn geometry_batches_match_host() {
    use agcad_geo::{accelerator, math::V3, tessellate::Mesh};
    use agcad_runtime::rebuild::Product;
    use std::{fs, process::Command};
    if let Some(output) = std::env::var_os("AGCAD_PARITY_OUTPUT") {
        let backend = std::env::var("AGCAD_GEOMETRY_BACKEND").unwrap();
        let mut s = server("SolidBody", None);
        let mut ops = circles(
            "part.multi_sections_solid",
            "loft",
            &[(0., 3.), (5., 7.), (10., 3.)],
        );
        ops.push(feature(
            "inertia",
            "evaluate.measure_inertia",
            json!({"items":["%loft"]}),
        ));
        ops.push(feature(
            "wall",
            "evaluate.thickness_analysis",
            json!({"body":"%loft","min":"0.5 mm"}),
        ));
        assert_eq!(run(&mut s, ops)["report"]["status"], "ACCEPTED");
        let events = &s.session.trace.events;
        for operation in [
            "loft_rail_pairs",
            "nurbs_curvature",
            "nurbs_interior_points",
            "mesh_mass_properties",
            "wall_thickness",
        ] {
            assert!(events.iter().any(|e| matches!(e, agcad_trace::TraceEvent::GeometryWork {operation: op,backend: b, fallback:None,..} if op==operation && (backend=="auto" || b==&backend))), "{operation}: {events:?}");
        }
        let products = &s.session.last_update.as_ref().unwrap().products;
        let report = |id: &str| match &products[&s.session.ids["main"][id]] {
            Product::Json(v) => v.clone(),
            _ => panic!("missing measure"),
        };
        let body = s.session.accepted_bodies["main"]
            .exact
            .as_ref()
            .unwrap()
            .exact_body()
            .unwrap();
        let mesh = tessellate(
            &body,
            &TessOptions {
                sag: 0.03,
                ..Default::default()
            },
        );
        assert!(mesh.is_watertight());
        let mut inverted = mesh.clone();
        for tri in &mut inverted.triangles {
            tri.swap(0, 1);
        }
        let properties = accelerator::mass_properties(&[
            mesh.clone().into(),
            Mesh::default().into(),
            inverted.into(),
        ])
        .unwrap();
        assert_eq!(properties[1], (0., 0., V3::ZERO));
        assert!((properties[0].0 + properties[2].0).abs() < 1e-8);
        assert!(properties[0].2.dist(properties[2].2) < 1e-10);
        // Parallel faces far from the origin exercise the FP64 ray bias and a no-hit ray.
        let origin = agcad_geo::math::v3(100_000., -200_000., 30_000.);
        let lower = [origin, origin + V3::X, origin + V3::Y];
        let mut plate = Mesh::default();
        plate.positions.extend(lower);
        plate
            .positions
            .extend(lower.map(|point| point + V3::Z * 0.125));
        plate.triangles.extend([[0, 2, 1], [3, 4, 5]]);
        assert!((accelerator::wall_minimum(&plate, 1).unwrap() - (0.125 - 1e-7)).abs() < 5e-12);
        plate.triangles.pop();
        assert!(accelerator::wall_minimum(&plate, 1).unwrap().is_infinite());
        if backend == "cuda" {
            let rail = |x| {
                agcad_geo::nurbs::NurbsCurve::bezier(vec![
                    agcad_geo::math::v3(x, 0., 0.),
                    agcad_geo::math::v3(x, 0., 1.),
                ])
            };
            let tol = 1e-6;
            let flags = accelerator::rail_candidates(&[rail(0.), rail(1.), rail(tol)], tol)
                .unwrap()
                .unwrap();
            assert_eq!(flags, [0, 1, 0]);
        }
        let options = TessOptions {
            sag: 1.0959940203,
            ..Default::default()
        };
        let patch = translated_patch(10_000_000.);
        let (translated, dispatch) = accelerator::capture(|| tessellate(&patch, &options));
        assert!(dispatch.iter().all(|event| event.error.is_none()));
        if backend != "auto" {
            assert!(
                dispatch
                    .iter()
                    .any(|e| e.operation == "nurbs_curvature" && e.backend == backend)
            );
        }
        let mut cube = tessellate(
            &agcad_geo::builders::block(V3::ZERO, V3::X + V3::Y + V3::Z),
            &Default::default(),
        );
        let shift = agcad_geo::math::v3(1e8, -2e8, 3e8);
        for point in &mut cube.positions {
            *point = *point + shift;
        }
        let (batch, dispatch) = accelerator::capture(|| {
            accelerator::mass_properties(&[Mesh::default().into(), cube.clone().into()]).unwrap()
        });
        assert!(
            dispatch
                .iter()
                .any(|e| e.operation == "mesh_mass_properties"
                    && e.error.is_none()
                    && (backend == "auto" || e.backend == backend))
        );
        assert_eq!(batch[1], (1., 6., shift + (V3::X + V3::Y + V3::Z) * 0.5));
        assert_eq!(batch[1], cube.mass_properties());
        cube.triangles[0][0] = u32::MAX;
        assert!(accelerator::wall_minimum(&cube, 1).is_err());
        assert!(accelerator::mass_properties(&[cube.into()]).is_err());
        let result = json!({"patch_triangles":translated.triangles, "mesh":mesh,"inertia":report("inertia"),"wall":report("wall"),"batch":properties});
        fs::write(output, serde_json::to_vec(&result).unwrap()).unwrap();
        return;
    }
    let reference = |backend| {
        let path = std::env::temp_dir().join(format!(
            "agcad-geometry-{}-{backend}.json",
            std::process::id()
        ));
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "geometry_batches_match_host", "--ignored"])
            .env("AGCAD_GEOMETRY_BACKEND", backend)
            .env("AGCAD_PARITY_OUTPUT", &path)
            .status()
            .unwrap();
        assert!(status.success());
        let result: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        fs::remove_file(path).unwrap();
        result
    };
    let host = reference("host");
    let cuda = reference("cuda");
    let auto = reference("auto");
    for result in [&cuda, &auto] {
        assert_eq!(host["patch_triangles"], result["patch_triangles"]);
        let masses = |v: &Value| {
            serde_json::from_value::<Vec<accelerator::MassProperties>>(v["batch"].clone()).unwrap()
        };
        for ((hv, ha, hc), (v, a, c)) in masses(&host).into_iter().zip(masses(result)) {
            assert!((hv - v).abs() < 1e-8 && (ha - a).abs() < 1e-8 && hc.dist(c) < 1e-10);
        }
    }
    let as_mesh = |v: &Value| serde_json::from_value::<Mesh>(v["mesh"].clone()).unwrap();
    let (a, b) = (as_mesh(&host), as_mesh(&cuda));
    assert_eq!(a.triangles, b.triangles);
    assert_eq!(a.face_of, b.face_of);
    assert_eq!(a.positions.len(), b.positions.len());
    assert!(
        a.positions
            .iter()
            .zip(b.positions)
            .all(|(p, q)| p.dist(q) < 1e-10)
    );
    for (op, field) in [
        ("wall", "min"),
        ("inertia", "area"),
        ("inertia", "volume"),
        ("inertia", "mass"),
    ] {
        let value = host[op][field].as_f64().unwrap();
        assert!((value - cuda[op][field].as_f64().unwrap()).abs() < 1e-10 * value.abs().max(1.));
    }
    assert_eq!(host["wall"]["samples"], cuda["wall"]["samples"]);
    assert_eq!(host["wall"]["passes"], cuda["wall"]["passes"]);
    for k in 0..3 {
        assert!(
            (host["inertia"]["centroid"][k].as_f64().unwrap()
                - cuda["inertia"]["centroid"][k].as_f64().unwrap())
            .abs()
                < 1e-10
        );
    }
}
