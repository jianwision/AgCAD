use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
#[serde(tag = "shape", rename_all = "lowercase", deny_unknown_fields)]
pub enum Primitive {
    Block {
        width: f64,
        depth: f64,
        height: f64,
    },
    Cylinder {
        diameter: f64,
        height: f64,
    },
    Sphere {
        diameter: f64,
        #[serde(default)]
        center: [f64; 3],
    },
}

fn dimension(v: f64) -> Result<f64, String> {
    if v.is_finite() && v > 0.0 && v <= 1_000_000.0 {
        Ok(v)
    } else {
        Err("Enter positive dimensions up to 1,000,000 mm".into())
    }
}

impl Primitive {
    pub fn program(self) -> Result<Value, String> {
        let (profile, height) = match self {
            Self::Sphere { diameter, center } => {
                if center.iter().any(|v| !v.is_finite()) {
                    return Err("Enter a finite sphere center".into());
                }
                return Ok(
                    json!({"requirements_rev":1,"expected_revision":0,"program":[
                        {"id":"sphere_center","op":"datum.point","args":{"mode":"coordinates","refs":[],"value":center}},
                        {"id":"sphere","op":"part.primitive","args":{"kind":"sphere","axis":"%sphere_center","p":[dimension(diameter)?/2.0]}}
                    ]}),
                );
            }
            Self::Block {
                width,
                depth,
                height,
            } => {
                let (x, y) = (dimension(width)? / 2., dimension(depth)? / 2.);
                (
                    json!({"id":"outline","op":"sketch.rectangle","args":{"sketch":"%sketch","mode":"corners","pts":[[-x,-y],[x,y]]}}),
                    height,
                )
            }
            Self::Cylinder { diameter, height } => (
                json!({"id":"outline","op":"sketch.circle","args":{"sketch":"%sketch","mode":"center_radius","pts":[[0,0]],"r":dimension(diameter)?/2.}}),
                height,
            ),
        };
        Ok(
            json!({"requirements_rev":1,"expected_revision":0,"program":[
                {"id":"sketch","op":"sketch.create","args":{"support":"xy"}},profile,
                {"id":"region","op":"sketch.regions","args":{"sketch":"%outline"}},
                {"id":"solid","op":"part.pad","args":{"profile":"%region","limits":dimension(height)?}}
            ]}),
        )
    }
}
