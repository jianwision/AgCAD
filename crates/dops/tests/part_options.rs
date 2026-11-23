use agcad_runtime::document::PartState;
use serde_json::json;

#[test]
fn gpu_raise_rejects_options_it_cannot_represent() {
    for (op, args, diagnostic) in [
        ("part.shaft", json!({"angles":["0 deg","90 deg"]}), "angles"),
        ("part.hole", json!({"support":"side.0"}), "support"),
        (
            "part.hole",
            json!({"support":"cap_end","spec":{"countersink_d":8}}),
            "countersink",
        ),
        ("part.pad", json!({"thick":1}), "thick"),
    ] {
        let mut part = PartState::new("options");
        let args = args
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), agcad_ir::json::decode_value(v, k).unwrap()))
            .collect();
        part.add_feature(op, 1, args, 0, None);
        let error = agcad_dops::lower::raise(&part)
            .err()
            .expect("unsupported GPU operation succeeded");
        assert!(error.contains(diagnostic), "{error}");
    }
}
