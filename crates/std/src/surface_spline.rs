//! Spline interpolation shared by section lofts and editable guide curves.
use crate::loft_device::{rows, Weighted};
use crate::util::{bad, frame, list, req, text, vec3};
use agcad_geo::curve::Curve;
use agcad_geo::nurbs::{ders_basis, find_span, NurbsCurve};
use agcad_ir::Value;
use agcad_runtime::document::FeatureNode;
use agcad_runtime::rebuild::{BuildContext, FeatureError, FeatureImpl, Product};

pub struct Interpolation {
    pub degree: usize,
    pub knots: Vec<f64>,
    pub inverse: Vec<Vec<f64>>,
}

pub fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1., |value, j| value * (n - j) as f64 / (j + 1) as f64)
}

impl Interpolation {
    pub fn new(params: &[f64], ruled: bool) -> Result<Self, FeatureError> {
        let n = params.len();
        if n < 2
            || params
                .windows(2)
                .any(|t| !t[0].is_finite() || t[1] - t[0] <= 1e-12)
        {
            return Err(bad("section or guide parameters must increase strictly"));
        }
        let degree = if ruled { 1 } else { 3.min(n - 1) };
        let mut knots = vec![0.; degree + 1];
        for i in 1..n - degree {
            knots.push(params[i..i + degree].iter().sum::<f64>() / degree as f64);
        }
        knots.extend(vec![1.; degree + 1]);
        Self::from_knots(params, degree, knots)
    }
    pub fn from_knots(
        params: &[f64],
        degree: usize,
        knots: Vec<f64>,
    ) -> Result<Self, FeatureError> {
        let n = params.len();
        let mut matrix = vec![vec![0.; 2 * n]; n];
        for (i, &t) in params.iter().enumerate() {
            let span = find_span(n - 1, degree, t, &knots);
            matrix[i][span - degree..=span]
                .copy_from_slice(&ders_basis(span, t, degree, 0, &knots)[0]);
            matrix[i][n + i] = 1.;
        }
        for col in 0..n {
            let pivot = (col..n)
                .max_by(|&i, &j| matrix[i][col].abs().total_cmp(&matrix[j][col].abs()))
                .unwrap();
            matrix.swap(pivot, col);
            let div = matrix[col][col];
            if div.abs() < 1e-13 {
                return Err(bad("ill-conditioned spline interpolation"));
            }
            for x in &mut matrix[col] {
                *x /= div;
            }
            let pivot = matrix[col].clone();
            for (i, row) in matrix.iter_mut().enumerate() {
                if i != col {
                    let c = row[col];
                    for (x, p) in row.iter_mut().zip(&pivot) {
                        *x -= c * p;
                    }
                }
            }
        }
        Ok(Self {
            degree,
            knots,
            inverse: matrix.into_iter().map(|r| r[n..].to_vec()).collect(),
        })
    }
}

pub struct Spline;
impl FeatureImpl for Spline {
    fn op(&self) -> &'static str {
        "wireframe.spline"
    }
    fn execute(
        &self,
        ctx: &mut BuildContext<'_>,
        n: &FeatureNode,
    ) -> Result<Product, FeatureError> {
        if n.args.contains_key("tangents") {
            return Err(bad("explicit guide tangent constraints are unsupported"));
        }
        let pts: Vec<_> = list(req(n, "pts")?)
            .iter()
            .map(|p| match p {
                Value::Ref { .. } => frame(ctx, p).map(|f| f.origin),
                _ => vec3(ctx, p),
            })
            .collect::<Result<_, _>>()?;
        if pts.len() < 2 {
            return Err(bad("guide needs at least two points"));
        }
        let mut t = vec![0.];
        for pair in pts.windows(2) {
            let delta = pair[1] - pair[0];
            let length = match text(n, "param").unwrap_or("chord") {
                "chord" => delta.norm(),
                "uniform" => 1.,
                "x" => delta.x.abs(),
                "y" => delta.y.abs(),
                "z" => delta.z.abs(),
                other => return Err(bad(format!("unsupported guide parameterization {other}"))),
            };
            if delta.norm() < 1e-9 || length <= 0. {
                return Err(bad("repeated guide station"));
            }
            t.push(t.last().unwrap() + length);
        }
        let last = *t.last().unwrap();
        for x in &mut t {
            *x /= last;
        }
        let fit = Interpolation::new(&t, false)?;
        let terms = fit
            .inverse
            .iter()
            .map(|r| r.iter().zip(&pts).map(|(&c, &p)| (c, (p, 1.))).collect())
            .collect::<Vec<_>>();
        let (p, device) = rows(&terms).map_err(|e| bad(format!("guide GPU interpolation: {e}")))?;
        ctx.dispatches.insert(n.id, (device, 1));
        Ok(Product::Curve(Curve::Nurbs(curve(
            fit.degree, fit.knots, p,
        ))))
    }
}

pub fn curve(degree: usize, knots: Vec<f64>, points: Vec<Weighted>) -> NurbsCurve {
    let (poles, weights) = points.into_iter().unzip();
    NurbsCurve {
        degree,
        knots,
        poles,
        weights,
    }
}
