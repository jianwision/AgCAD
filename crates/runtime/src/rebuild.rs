//! Specification-tree update (`document.update`, spec 02/05): features are pure functions of
//! inputs + parameters + configuration. `agcad-std` registers [`FeatureImpl`]s; unknown ops
//! report `UNSUPPORTED_OP` with their manifest (AG-07-002).

use crate::document::{FeatureId, FeatureNode, PartState};
use agcad_driver::AgStatus;
use agcad_geo::curve::Curve;
use agcad_geo::kernel::{ExactKernel, Solid};
use agcad_geo::math::{Aabb, Frame};
use agcad_geo::surface::Surface;
use agcad_geo::tessellate::Mesh;
use agcad_geo::builders::Profile;
use agcad_ir::program::Value;
use agcad_rtc::ExprGraph;
use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// Implicit body (spec 02 § Body types): AgLang distance function + parameters.
#[derive(Clone, Debug)]
pub struct ImplicitBody {
    pub graph: Arc<ExprGraph>,
    pub params: Vec<f64>,
    pub bounds: Aabb,
    pub lipschitz: f64,
}

impl ImplicitBody {
    pub fn eval(&self, p: agcad_geo::V3) -> f64 { self.graph.eval(&[p.x, p.y, p.z], &self.params)[0] }
}

/// Values produced by features (by feature id).
#[derive(Clone)]
pub enum Product {
    Solid(Solid),
    Sheet(agcad_geo::brep::Brep),
    Mesh(Mesh),
    Implicit(ImplicitBody),
    Curve(Curve),
    Surface(Surface),
    Frame(Frame),
    Regions(Vec<Profile>),
    Scalar(f64),
    Json(serde_json::Value),
    /// Extension products (sketches, networks, reports) owned by higher crates.
    Other(Arc<dyn Any + Send + Sync>),
}

impl Product {
    pub fn exact_body(&self) -> Option<std::borrow::Cow<'_, agcad_geo::brep::Brep>> {
        use std::borrow::Cow;
        match self {
            Product::Solid(Solid::Brep(b)) | Product::Sheet(b) => Some(Cow::Borrowed(b)),
            Product::Solid(s) => Some(Cow::Owned(s.brep())),
            _ => None,
        }
    }
}

impl std::fmt::Debug for Product {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let k = match self { Product::Solid(_) => "Solid", Product::Sheet(_) => "Sheet", Product::Mesh(_) => "Mesh", Product::Implicit(_) => "Implicit", Product::Curve(_) => "Curve", Product::Surface(_) => "Surface", Product::Frame(_) => "Frame", Product::Regions(_) => "Regions", Product::Scalar(_) => "Scalar", Product::Json(_) => "Json", Product::Other(_) => "Other" };
        write!(f, "Product::{k}")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FeatureError {
    pub status: AgStatus,
    pub message: String,
    /// Elements involved (stable names), for causal failure reports (spec 07).
    pub elements: Vec<String>,
    pub suggestion: Option<String>,
}

impl FeatureError {
    pub fn new(status: AgStatus, message: impl Into<String>) -> Self { FeatureError { status, message: message.into(), elements: vec![], suggestion: None } }
    pub fn with_elements(mut self, e: Vec<String>) -> Self { self.elements = e; self }
    pub fn suggest(mut self, s: impl Into<String>) -> Self { self.suggestion = Some(s.into()); self }
}

pub struct BuildContext<'a> {
    pub part: &'a PartState,
    pub kernel: &'a dyn ExactKernel,
    pub dispatches: BTreeMap<FeatureId, (&'static str, usize)>,
    pub products: BTreeMap<FeatureId, Product>,
    /// Current solid of each Body container.
    pub bodies: BTreeMap<usize, Solid>,
    /// Evaluated parameter values for this configuration (Knowledgeware + design table row).
    pub params: BTreeMap<String, f64>,
    /// Per-UDF scalar expression graphs, evaluated against `params`.
    pub exprs: Vec<ExprGraph>,
    /// Boolean tool solid + operation recorded per feature (patterns/mirrors replicate them).
    pub tools: BTreeMap<FeatureId, (Solid, agcad_geo::region2d::BoolOp2)>,
    /// Stable element names produced so far (for `query.named` / causal reports).
    pub names: BTreeMap<FeatureId, Vec<String>>,
}

impl<'a> BuildContext<'a> {
    /// Resolve a feature argument to a number: literal, quantity, `$param` ref, `$expr`.
    pub fn number(&self, v: &Value) -> Result<f64, FeatureError> {
        match v {
            Value::Int(i) => Ok(*i as f64),
            Value::Float(f) => Ok(*f),
            Value::Quantity(q) => Ok(q.value),
            Value::Bool(b) => Ok(*b as i32 as f64),
            Value::Ref { id, field: Some(name) } if id == "$param" => self.params.get(name).copied().ok_or_else(|| FeatureError::new(AgStatus::StaleReference, format!("unknown parameter '{name}'"))),
            Value::Formula(src) => {
                let e = agcad_ir::formula::parse(src).map_err(|m| FeatureError::new(AgStatus::InvalidParameter, m))?;
                let r = e.eval(&|n| self.params.get(n).copied()).map_err(|m| FeatureError::new(AgStatus::StaleReference, format!("{m} (known parameters: {})", self.params.keys().cloned().collect::<Vec<_>>().join(", "))))?;
                if r.is_finite() { Ok(r) } else { Err(FeatureError::new(AgStatus::NonfiniteParameter, format!("formula '={src}' evaluated to {r}"))) }
            }
            Value::Map(m) if m.contains_key("$expr") => {
                let idx = match m["$expr"] { Value::Int(i) => i as usize, _ => usize::MAX };
                let g = self.exprs.get(idx).ok_or_else(|| FeatureError::new(AgStatus::InvalidParameter, "bad $expr index"))?;
                let pv: Vec<f64> = g.param_names.iter().map(|n| self.params.get(n).copied().unwrap_or(f64::NAN)).collect();
                let r = g.eval(&[], &pv)[0];
                if r.is_finite() { Ok(r) } else { Err(FeatureError::new(AgStatus::NonfiniteParameter, "expression evaluated to a non-finite value")) }
            }
            Value::String(s) => self.params.get(s).copied().ok_or_else(|| FeatureError::new(AgStatus::InvalidParameter, format!("expected a number, got '{s}'"))),
            other => Err(FeatureError::new(AgStatus::InvalidParameter, format!("expected a number, got {other:?}"))),
        }
    }

    /// Resolve a `%ref` to an earlier product (statement ids map to feature ids at insert).
    pub fn product(&self, v: &Value) -> Result<&Product, FeatureError> {
        match v {
            Value::Ref { id, .. } => {
                let fid: FeatureId = id.trim_start_matches('f').parse().map_err(|_| FeatureError::new(AgStatus::StaleReference, format!("unknown reference '%{id}'")))?;
                self.products.get(&fid).ok_or_else(|| FeatureError::new(AgStatus::StaleReference, format!("reference '%{id}' has no product (failed or deactivated)")))
            }
            _ => Err(FeatureError::new(AgStatus::InvalidParameter, "expected a reference")),
        }
    }
}

pub trait FeatureImpl: Send + Sync {
    fn op(&self) -> &'static str;
    fn execute(&self, ctx: &mut BuildContext<'_>, node: &FeatureNode) -> Result<Product, FeatureError>;
}

#[derive(Default)]
pub struct Registry {
    map: HashMap<String, Arc<dyn FeatureImpl>>,
}

impl Registry {
    pub fn register(&mut self, f: Arc<dyn FeatureImpl>) { self.map.insert(f.op().to_string(), f); }
    pub fn get(&self, op: &str) -> Option<&Arc<dyn FeatureImpl>> { self.map.get(op) }
    pub fn implemented(&self) -> Vec<String> { let mut v: Vec<_> = self.map.keys().cloned().collect(); v.sort(); v }
}

#[derive(Clone, Debug)]
pub struct FeatureOutcome {
    pub feature: FeatureId,
    pub feature_ref: String,
    pub status: AgStatus,
    pub error: Option<FeatureError>,
}

pub struct UpdateResult {
    pub dispatches: BTreeMap<FeatureId, (&'static str, usize)>,
    pub outcomes: Vec<FeatureOutcome>,
    pub products: BTreeMap<FeatureId, Product>,
    pub bodies: BTreeMap<usize, Solid>,
}

impl UpdateResult {
    pub fn ok(&self) -> bool { self.outcomes.iter().all(|o| o.status.is_ok()) }
}

/// Update the specification tree for one configuration.
pub fn update(part: &PartState, registry: &Registry, kernel: &dyn ExactKernel, params: BTreeMap<String, f64>, exprs: Vec<ExprGraph>) -> UpdateResult {
    let mut ctx = BuildContext { part, kernel, dispatches: BTreeMap::new(), products: BTreeMap::new(), bodies: BTreeMap::new(), params, exprs, tools: BTreeMap::new(), names: BTreeMap::new() };
    let mut outcomes = Vec::new();
    for id in part.rebuild_order() {
        let node = &part.features[&id];
        let (status, error) = match registry.get(&node.op) {
            None => (AgStatus::UnsupportedOp, Some(FeatureError::new(AgStatus::UnsupportedOp, format!("feature '{}' is not implemented yet in this build", node.op)))),
            Some(f) => match f.execute(&mut ctx, node) {
                Ok(p) => {
                    if let Product::Solid(s) = &p { ctx.bodies.insert(node.container, s.clone()); }
                    ctx.products.insert(id, p);
                    (AgStatus::Accepted, None)
                }
                Err(e) => (e.status, Some(e)),
            },
        };
        outcomes.push(FeatureOutcome { feature: id, feature_ref: node.feature_ref(), status, error });
    }
    UpdateResult { dispatches: ctx.dispatches, outcomes, products: ctx.products, bodies: ctx.bodies }
}
