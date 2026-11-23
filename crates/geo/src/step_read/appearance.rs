use super::{Arg, Entity};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Default, Serialize)]
pub struct StepMaterial {
    pub name: Option<String>,
    pub density: Option<f64>,
    /// Unit as declared in the STEP file. Preserve unusual combinations verbatim.
    pub density_unit: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct StepBodyStyle {
    pub entity: u64,
    pub name: String,
    pub color: Option<[f64; 3]>,
    pub material: StepMaterial,
    pub faces: Vec<(u64, Option<[f64; 3]>)>,
}

pub(super) struct Styles {
    colors: HashMap<u64, [f64; 3]>,
    materials: HashMap<u64, StepMaterial>,
}

type Index = HashMap<u64, Entity>;
fn link<'a>(m: &'a Index, args: &[Arg], i: usize) -> Option<&'a Entity> {
    m.get(&args.get(i)?.r()?)
}

fn style_color(m: &Index, id: u64, visited: &mut HashSet<u64>, depth: usize) -> Option<[f64; 3]> {
    if depth > 24 || !visited.insert(id) {
        return None;
    }
    let e = m.get(&id)?;
    let args = e.args();
    if e.ty() == "COLOUR_RGB" {
        let rgb = [
            args.get(1)?.num()?,
            args.get(2)?.num()?,
            args.get(3)?.num()?,
        ];
        return rgb.iter().all(|x| (0.0..=1.0).contains(x)).then_some(rgb);
    }
    if !matches!(
        e.ty(),
        "PRESENTATION_STYLE_ASSIGNMENT"
            | "SURFACE_STYLE_USAGE"
            | "SURFACE_SIDE_STYLE"
            | "SURFACE_STYLE_FILL_AREA"
            | "FILL_AREA_STYLE"
            | "FILL_AREA_STYLE_COLOUR"
            | "SURFACE_STYLE_RENDERING"
    ) {
        return None;
    }
    args.iter()
        .flat_map(|a| match a {
            Arg::List(v) => v.as_slice(),
            _ => std::slice::from_ref(a),
        })
        .filter_map(Arg::r)
        .find_map(|r| style_color(m, r, visited, depth + 1))
}

fn unit(m: &Index, id: u64, depth: usize) -> Option<String> {
    if depth > 4 {
        return None;
    }
    let e = m.get(&id)?;
    if let Some(a) = e.part("SI_UNIT") {
        let prefix = a.first()?.s().to_ascii_lowercase();
        return Some(format!("{prefix}{}", a.get(1)?.s().to_ascii_lowercase()));
    }
    if let Some(a) = e.part("DERIVED_UNIT") {
        let mut terms = Vec::new();
        for item in a.first()?.list() {
            let element = m.get(&item.r()?)?.part("DERIVED_UNIT_ELEMENT")?;
            terms.push(format!(
                "{}^{}",
                unit(m, element.first()?.r()?, depth + 1)?,
                element.get(1)?.num()?
            ));
        }
        return Some(terms.join(" · "));
    }
    e.part("CONVERSION_BASED_UNIT")?
        .first()
        .map(|a| a.s().to_owned())
}

impl Styles {
    pub(super) fn read(m: &Index) -> Self {
        let mut colors = HashMap::new();
        let mut properties = HashMap::<u64, StepMaterial>::new();
        let mut ids: Vec<_> = m.keys().copied().collect();
        ids.sort_unstable_by_key(|id| (m[id].ty() == "OVER_RIDING_STYLED_ITEM", *id));
        for id in ids {
            let e = &m[&id];
            let a = e.args();
            if matches!(e.ty(), "STYLED_ITEM" | "OVER_RIDING_STYLED_ITEM") {
                if let (Some(target), Some(styles)) = (a.get(2).and_then(Arg::r), a.get(1)) {
                    if let Some(rgb) = styles
                        .list()
                        .iter()
                        .filter_map(Arg::r)
                        .find_map(|id| style_color(m, id, &mut HashSet::new(), 0))
                    {
                        colors.insert(target, rgb);
                    }
                }
            } else if e.ty() == "MATERIAL_DESIGNATION" {
                if let (Some(name), Some(target)) = (a.first(), a.get(1).and_then(Arg::r)) {
                    properties.entry(target).or_default().name = Some(name.s().into());
                }
            } else if e.ty() == "PROPERTY_DEFINITION_REPRESENTATION" {
                let (Some(prop), Some(rep)) = (link(m, a, 0), link(m, a, 1)) else {
                    continue;
                };
                let Some(product) = prop.args().get(2).and_then(Arg::r) else {
                    continue;
                };
                let label = rep
                    .args()
                    .first()
                    .map(Arg::s)
                    .unwrap_or("")
                    .to_ascii_lowercase();
                for item in rep
                    .args()
                    .get(1)
                    .into_iter()
                    .flat_map(Arg::list)
                    .filter_map(Arg::r)
                    .filter_map(|r| m.get(&r))
                {
                    let a = item.args();
                    if label == "material name" && item.ty() == "DESCRIPTIVE_REPRESENTATION_ITEM" {
                        properties.entry(product).or_default().name =
                            a.first().map(|v| v.s().into());
                    } else if label == "density" && item.ty() == "MEASURE_REPRESENTATION_ITEM" {
                        let mat = properties.entry(product).or_default();
                        mat.density = a.get(1).and_then(Arg::num);
                        mat.density_unit = a.get(2).and_then(Arg::r).and_then(|r| unit(m, r, 0));
                    }
                }
            }
        }
        let mut materials = properties.clone();
        for e in m
            .values()
            .filter(|e| e.ty() == "SHAPE_DEFINITION_REPRESENTATION")
        {
            let (Some(def), Some(rep)) = (link(m, e.args(), 0), link(m, e.args(), 1)) else {
                continue;
            };
            let Some(mat) = def
                .args()
                .get(2)
                .and_then(Arg::r)
                .and_then(|id| properties.get(&id))
            else {
                continue;
            };
            for item in rep
                .args()
                .get(1)
                .into_iter()
                .flat_map(Arg::list)
                .filter_map(Arg::r)
            {
                materials.insert(item, mat.clone());
            }
        }
        Self { colors, materials }
    }
    pub(super) fn color(&self, id: u64) -> Option<[f64; 3]> {
        self.colors.get(&id).copied()
    }
    pub(super) fn body(&self, id: u64, name: &str) -> StepBodyStyle {
        StepBodyStyle {
            entity: id,
            name: name.into(),
            color: self.color(id),
            material: self.materials.get(&id).cloned().unwrap_or_default(),
            faces: vec![],
        }
    }
}
