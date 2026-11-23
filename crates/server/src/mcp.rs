use super::{Server, tools};
use serde_json::{Value, json};
use std::sync::OnceLock;

pub(super) fn output_schema(name: &str) -> Option<Value> {
    let (properties, required) = match name {
        "query" => (
            json!({"revision":{"type":"integer"},"faces":{"type":"array"},"elements":{"type":"array"}}),
            vec!["revision", "faces", "elements"],
        ),
        "specification_tree" => (
            json!({"revision":{"type":"integer"},"features":{"type":"array"},"parameters":{"type":"object"}}),
            vec!["revision", "features", "parameters"],
        ),
        "run_program" => (json!({"report":{"type":"object"}}), vec!["report"]),
        "check_program" => (
            json!({"ok":{"type":"boolean"},"diagnostics":{"type":"array"}}),
            vec!["ok", "diagnostics"],
        ),
        "describe" => {
            return Some(
                json!({"type":"object","anyOf":[{"required":["bounding_box"]},{"required":["current","diff"]}]}),
            );
        }
        _ => return None,
    };
    Some(json!({"type":"object","properties":properties,"required":required}))
}

// The discovery table declares top-level types, required properties and enums.
// Nested domain values such as AgIR programs are checked by their CAD decoder.
fn validate_input(schema: &Value, arguments: &Value) -> Result<(), String> {
    let fields = arguments
        .as_object()
        .ok_or("Tool arguments must be an object")?;
    if let Some(required) = schema["required"].as_array() {
        for key in required.iter().filter_map(Value::as_str) {
            if !fields.contains_key(key) {
                return Err(format!("Missing required argument {key}"));
            }
        }
    }
    for (key, value) in fields {
        let property = &schema["properties"][key];
        let Some(kind) = property["type"].as_str() else {
            continue;
        };
        let valid = match kind {
            "array" => value.is_array(),
            "object" => value.is_object(),
            "integer" => value.is_u64() || value.is_i64(),
            "number" => value.is_number(),
            "boolean" => value.is_boolean(),
            "string" => value.is_string(),
            _ => true,
        };
        if !valid {
            return Err(format!("{key} must have type {kind}"));
        }
        if let Some(options) = property["enum"].as_array() {
            if !options.contains(value) {
                return Err(format!("{key} must be one of {}", property["enum"]));
            }
        }
    }
    Ok(())
}

pub(super) fn call(server: &mut Server, params: &Value) -> Result<Value, (i64, String)> {
    static TABLE: OnceLock<Vec<Value>> = OnceLock::new();
    let name = params["name"]
        .as_str()
        .ok_or_else(|| (-32602, "A tool name is required".into()))?;
    let tool = TABLE
        .get_or_init(tools)
        .iter()
        .find(|t| t["name"] == name)
        .ok_or_else(|| (-32602, format!("Unknown tool {name}")))?;
    let empty = json!({});
    let args = params.get("arguments").unwrap_or(&empty);
    validate_input(&tool["inputSchema"], args).map_err(|e| (-32602, e))?;
    match server.call_tool(name, args) {
        Ok(value) => {
            let failed = name == "run_program" && value["report"]["status"] != "ACCEPTED";
            let mut result = json!({"content":[{"type":"text","text":value.to_string()}],"isError":failed});
            if tool.get("outputSchema").is_some() {
                result["structuredContent"] = value;
            }
            Ok(result)
        }
        Err(reason) => Ok(json!({"content":[{"type":"text","text":reason}],"isError":true})),
    }
}
