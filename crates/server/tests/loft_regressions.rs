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
