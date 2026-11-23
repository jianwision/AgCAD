use super::*;
use crate::document::viewport_operations;
use agcad_ir::Catalog;

fn function(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    json!({"type":"function","function":{"name":name,"description":description,"parameters":{
        "type":"object","properties":properties,"required":required,"additionalProperties":false
    }}})
}

pub(super) fn available(obs: &Observation) -> Vec<Value> {
    let report = &obs.report;
    let mut tools = Vec::new();
    tools.push(function("preview_sphere", "Create a sphere immediately. Omit optional dimensions for a 20 mm diameter sphere centered at the origin. Default to a separate part for an occupied viewport. Join only on explicit request.",
        json!({"diameter":{"type":"number","exclusiveMinimum":0},"center":{"type":"array","minItems":3,"maxItems":3,"items":{"type":"number"}},"join":{"type":"boolean"}}), &[]));
    tools.push(function("preview_extrusion","Extend the selected planar face or the unique native end profile by a signed displacement along its outward normal. Copy the user's distance including units. If no target exists, this tool explains what to select.",
        json!({"distance":{"type":"string","description":"Signed displacement with units, e.g. 3 cm or -2 mm"}}), &["distance"]));
    tools.push(function("preview_edge_round","For smooth, round or fillet edges. Use scope all for all edges (also when a face is picked). Use selected for a single picked edge. Scope vertical requires an explicit request or acceptance of the narrower operation.",
        json!({"scope":{"type":"string","enum":["all","vertical","selected"]},"radius_mm":{"type":"number","exclusiveMinimum":0}}), &["scope"]));
    let selected = report["selections"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default();
    if selected.iter().any(|v| v["kind"] == "vertex") {
        tools.push(function(
            "preview_vertex_bevel",
            "Cut off the selected block corners with an equal setback in mm.",
            json!({"distance_mm":{"type":"number","exclusiveMinimum":0}}),
            &["distance_mm"],
        ));
        if selected.iter().filter(|v| v["kind"] == "vertex").count() == 2 {
            tools.push(function(
                "measure_vertices",
                "Measure the exact distance between the two selected model vertices.",
                json!({}),
                &[],
            ));
        }
    }
    if selected.iter().any(|v| v["kind"] == "face") {
        tools.push(function("preview_face_offset","Offset all selected planar block faces along their respective outward normals. Use a negative distance for inward.",json!({"distance_mm":{"type":"number"}}),&["distance_mm"]));
    }
    if selected.iter().any(|v| v["kind"] == "edge") {
        tools.push(function(
            "preview_edge_chamfer",
            "Chamfer the picked straight block edge by an equal setback in mm.",
            json!({"distance_mm":{"type":"number","exclusiveMinimum":0}}),
            &["distance_mm"],
        ));
    }
    if report["empty"] == true {
        let mm = json!({"type":"number","exclusiveMinimum":0,"maximum":1000000});
        tools.push(function(
            "preview_block",
            "First rectangular solid, centered in XY with its bottom at Z=0. All dimensions in mm.",
            json!({"width":mm,"depth":mm,"height":mm}),
            &["width", "depth", "height"],
        ));
        tools.push(function("preview_cylinder", "First solid circular cylinder along +Z with its bottom at the XY origin. Supply diameter and height in mm.",
            json!({"diameter":mm,"height":mm}), &["diameter","height"]));
    }
    if !report["selected_face"].is_null() {
        let fraction = json!({"type":"number","minimum":0,"maximum":1});
        tools.push(function("preview_finish", "Change the finish of the selected face or its whole body. Include only properties the user wants changed.",
            json!({"scope":{"type":"string","enum":["face","body"]},"color":{"type":"string","pattern":"^#[0-9a-fA-F]{6}$"},
                "roughness":fraction,"metalness":fraction,"designation":{"type":"string","maxLength":160},
                "density_kg_m3":{"type":"number","exclusiveMinimum":0}}), &["scope"]));
    }
    if report["body"]["bodies"].as_u64().unwrap_or(1) <= 1 {
        tools.push(function("describe_operations", "Look up exact inputs, outputs and examples for a batch of operation names from modeling_operations.",
            json!({"names":{"type":"array","minItems":1,"maxItems":12,"items":{"type":"string"}}}), &["names"]));
        tools.push(function("preview_geometry", "Preview one AgIR edit containing all requested steps. Use fresh statement IDs, %id references and the catalog signatures. The server supplies revision fields.",
            json!({"statements":{"type":"array","minItems":1,"items":{"type":"object","required":["id","op","args"],
                "properties":{"id":{"type":"string"},"op":{"type":"string"},"args":{"type":"object"},"over":{"type":"string"}},"additionalProperties":false}}}),
            &["statements"]));
    }
    tools
}

fn selected<'a>(report: &'a Value, kind: &str) -> Vec<&'a Value> {
    report["selections"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|v| v["kind"] == kind)
        .collect()
}
fn xyz(p: &Value) -> Result<[f64; 3], String> {
    let v: agcad_geo::V3 =
        serde_json::from_value(p.clone()).map_err(|_| "Missing model coordinates")?;
    Ok([v.x, v.y, v.z])
}
pub(crate) fn plane_target(face: &Value) -> Result<Value, String> {
    if face["surface"]["kind"] != "Plane" {
        return Err(format!("{} is not planar", face["label"]));
    }
    let frame = &face["surface"]["frame"];
    let mut normal = xyz(&frame["z"])?;
    if face["same_sense"] == false {
        normal = normal.map(|x| -x);
    }
    Ok(json!({"body":face["body"],"plane_origin":xyz(&frame["origin"])?,"normal":normal}))
}
pub(super) enum Invocation {
    Information(Value),
    Preview(Value),
}

pub(super) fn invoke(
    call: &Value,
    tools: &[Value],
    obs: &Observation,
) -> Result<Invocation, String> {
    let name = call["function"]["name"]
        .as_str()
        .ok_or("Missing tool name")?;
    let tool = tools
        .iter()
        .find(|t| t["function"]["name"] == name)
        .ok_or_else(|| format!("{name} is unavailable for the current part or selection"))?;
    let raw = call["function"]["arguments"]
        .as_str()
        .ok_or("Missing function arguments")?;
    let mut args: serde_json::Map<String, Value> =
        serde_json::from_str(raw).map_err(|e| format!("Invalid {name} arguments: {e}"))?;
    let schema = &tool["function"]["parameters"];
    for key in args.keys() {
        if schema["properties"].get(key).is_none() {
            return Err(format!("Unknown {name} argument {key}"));
        }
    }
    for key in schema["required"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
    {
        if !args.contains_key(key) {
            return Err(format!("{name} requires {key}"));
        }
    }
    let result = match name {
        "describe_operations" => {
            let names: Vec<String> =
                serde_json::from_value(args.remove("names").unwrap()).map_err(|e| e.to_string())?;
            if names.is_empty() || names.len() > 12 {
                return Err("Choose 1 to 12 operation names".into());
            }
            let mut signatures = Vec::new();
            for name in names {
                if viewport_operations().binary_search(&name).is_err() {
                    return Err(format!("Unknown viewport operation {name}"));
                }
                let op = Catalog::global()
                    .get(&name)
                    .ok_or("Missing operation signature")?;
                signatures.push(json!({"op":op.name,"description":op.summary,"inputs":op.inputs,"outputs":op.outputs,
                    "example":agcad_server::example_statement(op)}));
            }
            return Ok(Invocation::Information(json!(signatures)));
        }
        "preview_sphere" => {
            let diameter = args.remove("diameter").unwrap_or(json!(20.0));
            let center = args.remove("center").unwrap_or(json!([0, 0, 0]));
            let join = args
                .remove("join")
                .map(serde_json::from_value::<bool>)
                .transpose()
                .map_err(|e| e.to_string())?
                .unwrap_or(false);
            json!({"sphere":{"shape":"sphere","diameter":diameter,"center":center},"join":join,
                "answer":format!("Sphere preview with diameter {diameter} mm and center {center} mm.")})
        }
        "preview_extrusion" => {
            let Some(face) = obs.report["extrusion_face"].as_u64() else {
                return Ok(Invocation::Preview(
                    json!({"answer":if obs.report["empty"]==true {"Choose a sketch profile for the extrusion."} else if !obs.report["selected_face"].is_null() {"This face does not support extrusion. Select a planar prism end face."} else {"Select the planar face to extrude."}}),
                ));
            };
            let distance = agcad_ir::units::Quantity::parse(
                args["distance"]
                    .as_str()
                    .ok_or("Supply distance with units")?,
            )?;
            if distance.dim != agcad_ir::units::Dimension::Length
                || !distance.value.is_finite()
                || distance.value == 0.0
            {
                return Err("Supply a nonzero length with units".into());
            }
            json!({"program":{"expected_revision":obs.revision,"requirements_rev":obs.requirements_rev,"program":[
                {"id":format!("extend_{}",obs.revision),"op":"direct.push_pull","args":{"faces":[face],"d":distance.value}}
            ]},"answer":format!("Extrusion preview with {} mm displacement from the face.",distance.value)})
        }
        "measure_vertices" => {
            let points = selected(&obs.report, "vertex");
            if points.len() != 2 {
                return Err("Pick exactly two vertices".into());
            }
            let read = |v: &Value| {
                serde_json::from_value::<agcad_geo::V3>(v["point_mm"].clone())
                    .map_err(|e| e.to_string())
            };
            let distance = read(points[0])?.dist(read(points[1])?);
            json!({"answer":format!("Distance between the selected vertices: {distance:.6} mm.")})
        }
        "preview_vertex_bevel"
        | "preview_face_offset"
        | "preview_edge_chamfer"
        | "preview_edge_round"
            if name != "preview_edge_round" || args["scope"] == "selected" || (args["scope"]=="all" && obs.report["all_edge_fillet"]==true) =>
        {
            let kind = match name {
                "preview_vertex_bevel" => "vertex",
                "preview_face_offset" => "face",
                _ => "edge",
            };
            let items = if args.get("scope").is_some_and(|s|s=="all"){
                obs.report["block_edges"].as_array().into_iter().flatten().collect()
            }else{selected(&obs.report,kind)};
            if items.is_empty() {
                return Ok(Invocation::Preview(
                    json!({"answer":format!("Pick a {kind} to edit.")}),
                ));
            }
            let round = name == "preview_edge_round";
            let field = if round { "radius_mm" } else { "distance_mm" };
            let size = match args.get(field) {
                Some(v) => v.as_f64().ok_or("Supply a numeric distance in mm")?,
                None => 1.,
            };
            if !size.is_finite() || size == 0. || (kind != "face" && size < 0.) {
                return Err("Invalid edit distance".into());
            }
            let mut targets = Vec::new();
            for item in &items {
                let body = &item["body"];
                targets.push(match kind{
                    "edge" if item["curve"]=="Line"=>json!({"body":body,"endpoints_mm":item["endpoints_mm"]}),
                    "edge"=>return Ok(Invocation::Preview(json!({"answer":format!("{} is curved. Select straight block edges.",item["label"])}))),
                    "vertex"=>json!({"body":body,"vertex_mm":xyz(&item["point_mm"])?}),
                    _=>plane_target(item)?,
                });
            }
            let (op, params, verb) = match kind {
                "face" => (
                    "direct.push_pull",
                    json!({"faces":targets,"d":size}),
                    "Offset",
                ),
                "edge" if round => (
                    "part.edge_fillet",
                    json!({"edges":targets,"r":size,"propagation":"minimal"}),
                    "Fillet",
                ),
                _ => (
                    "part.chamfer",
                    json!({"edges":targets,"mode":"equal_distance","a":size}),
                    if kind == "vertex" { "Bevel" } else { "Chamfer" },
                ),
            };
            json!({"program":{"expected_revision":obs.revision,"requirements_rev":obs.requirements_rev,"program":[
                {"id":format!("edit_{}",obs.revision),"op":op,"args":params}
            ]},"answer":format!("{verb} preview for {} at {size} mm.",if items.len()==1 {items[0]["label"].as_str().unwrap_or(kind).to_string()} else {format!("{} selected {}",items.len(),match kind{"vertex"=>"vertices","face"=>"faces",_=>"edges"})})})
        }
        "preview_edge_round" => {
            let vertical = obs.report["vertical_corner_fillet"] == true;
            match args["scope"].as_str() {
                Some("all") => {
                    json!({"answer":if vertical {"Whole-body fillets are unavailable. I can preview a fillet on the vertical corners of this part."} else {"Whole-body fillets are unavailable for this geometry. The native fillet supports vertical corners on layered prismatic parts."}})
                }
                Some("vertical") if !vertical => {
                    json!({"answer":"This body does not support the native vertical-corner fillet."})
                }
                Some("vertical") => {
                    let radius = args.get("radius_mm").and_then(Value::as_f64).unwrap_or(1.0);
                    if !radius.is_finite() || radius <= 0. {
                        return Err("Supply a positive fillet radius".into());
                    }
                    json!({"program":{"expected_revision":obs.revision,"requirements_rev":obs.requirements_rev,"program":[
                        {"id":format!("round_{}",obs.revision),"op":"part.edge_fillet","args":{"edges":["all_vertical"],"r":radius,"propagation":"tangent"}}
                    ]},"answer":format!("Vertical corner fillet preview with radius {radius} mm.")})
                }
                _ => return Err("Choose selected, all or vertical scope".into()),
            }
        }
        "preview_block" | "preview_cylinder" => {
            let shape = name.trim_start_matches("preview_");
            args.insert("shape".into(), json!(shape));
            let answer = match shape {
                "block" => format!(
                    "Block preview: {} × {} × {} mm.",
                    args["width"], args["depth"], args["height"]
                ),
                _ => format!(
                    "Cylinder preview: {} mm diameter, {} mm high.",
                    args["diameter"], args["height"]
                ),
            };
            json!({"primitive":args,"answer":answer})
        }
        "preview_finish" => {
            let scope = args.remove("scope").unwrap();
            if !matches!(scope.as_str(), Some("face" | "body")) {
                return Err("Choose face or body scope".into());
            }
            if args.is_empty() || args.values().any(Value::is_null) {
                return Err("Supply a finish property to change".into());
            }
            json!({"scope":scope,"finish":args,"answer":format!("Finish preview for {} selected face(s).",selected(&obs.report,"face").len())})
        }
        "preview_geometry" => {
            let statements = args.remove("statements").unwrap();
            json!({"program":{"expected_revision":obs.revision,"requirements_rev":obs.requirements_rev,"program":statements},"answer":"Geometry preview ready."})
        }
        _ => return Err(format!("Unknown function {name}")),
    };
    Ok(Invocation::Preview(result))
}
