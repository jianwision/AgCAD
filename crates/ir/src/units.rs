//! Units: internal mm, rad, kg, s. JSON accepts unit strings (spec 02 § Units; CONVENTIONS § 5).

use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dimension { Length, Angle, Mass, Time, Area, Volume, Density, Force, Pressure, Dimensionless }

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Quantity {
    /// Value in internal units.
    pub value: f64,
    pub dim: Dimension,
}

/// (suffix, dimension, factor to internal unit)
pub(crate) const UNITS: &[(&str, Dimension, f64)] = &[
    ("mm", Dimension::Length, 1.0),
    ("cm", Dimension::Length, 10.0),
    ("m", Dimension::Length, 1000.0),
    ("um", Dimension::Length, 1e-3),
    ("µm", Dimension::Length, 1e-3),
    ("in", Dimension::Length, 25.4),
    ("ft", Dimension::Length, 304.8),
    ("rad", Dimension::Angle, 1.0),
    ("deg", Dimension::Angle, core::f64::consts::PI / 180.0),
    ("°", Dimension::Angle, core::f64::consts::PI / 180.0),
    ("kg", Dimension::Mass, 1.0),
    ("g", Dimension::Mass, 1e-3),
    ("s", Dimension::Time, 1.0),
    ("ms", Dimension::Time, 1e-3),
    ("mm2", Dimension::Area, 1.0),
    ("mm3", Dimension::Volume, 1.0),
    ("kg/mm3", Dimension::Density, 1.0),
    ("g/cm3", Dimension::Density, 1e-6),
    ("N", Dimension::Force, 1.0),
    ("MPa", Dimension::Pressure, 1.0),
];

impl Quantity {
    pub fn new(value: f64, dim: Dimension) -> Self { Quantity { value, dim } }
    pub fn mm(v: f64) -> Self { Quantity::new(v, Dimension::Length) }
    pub fn rad(v: f64) -> Self { Quantity::new(v, Dimension::Angle) }

    /// Parse `"12 mm"`, `"0.5 in"`, `"30 deg"`, `"7"` (dimensionless).
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        let split = s.find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E'))).unwrap_or(s.len());
        // Guard "e" as exponent vs unit: only treat as number part if parse succeeds.
        let (num, unit) = s.split_at(split);
        let v: f64 = num.trim().parse().map_err(|_| format!("bad number in quantity '{s}'"))?;
        if !v.is_finite() {
            return Err(format!("non-finite quantity '{s}'"));
        }
        let unit = unit.trim();
        if unit.is_empty() {
            return Ok(Quantity::new(v, Dimension::Dimensionless));
        }
        UNITS.iter()
            .find(|(u, _, _)| *u == unit)
            .map(|(_, d, f)| Quantity::new(v * f, *d))
            .ok_or_else(|| format!("unknown unit '{unit}' in '{s}'"))
    }
}

/// Dimension implied by a catalog type name (`Length`, `Angle`, ...), if any.
pub fn dimension_of_type(ty: &str) -> Option<Dimension> {
    match ty {
        "Length" => Some(Dimension::Length),
        "Angle" => Some(Dimension::Angle),
        "Mass" => Some(Dimension::Mass),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_units() {
        assert_eq!(Quantity::parse("12 mm").unwrap(), Quantity::mm(12.0));
        assert!((Quantity::parse("0.5 in").unwrap().value - 12.7).abs() < 1e-12);
        assert!((Quantity::parse("180 deg").unwrap().value - core::f64::consts::PI).abs() < 1e-12);
        assert!(Quantity::parse("3 furlong").is_err());
        assert_eq!(Quantity::parse("7").unwrap().dim, Dimension::Dimensionless);
    }
}
