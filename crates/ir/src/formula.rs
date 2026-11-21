//! Formulas in AgIR JSON (spec 04): `"=od/2 - 5 mm"`. Deterministic, unit-aware literals
//! (`5 mm`, `30 deg` → internal mm / rad), parameters by name, `+ - * / ^`, parentheses and
//! `sqrt abs min max sin cos tan atan2 floor ceil round`. Parsed at decode/verify time so syntax
//! errors and unknown functions are reported before anything runs; evaluated per
//! configuration at rebuild (parameter values change across a study).

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Num(f64),
    Param(String),
    Neg(Box<Expr>),
    Bin(char, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
}

const FUNCS: &[(&str, usize)] = &[("sqrt", 1), ("abs", 1), ("min", 2), ("max", 2), ("sin", 1), ("cos", 1), ("tan", 1), ("atan2", 2), ("floor", 1), ("ceil", 1), ("round", 1)];

struct P<'a> { s: &'a [u8], i: usize, src: &'a str }

impl<'a> P<'a> {
    fn ws(&mut self) { while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() { self.i += 1; } }
    fn peek(&mut self) -> Option<u8> { self.ws(); self.s.get(self.i).copied() }
    fn err<T>(&self, m: impl Into<String>) -> Result<T, String> { Err(format!("{} at column {} in '={}'", m.into(), self.i + 1, self.src)) }
    fn ident(&mut self) -> Option<String> {
        self.ws();
        let st = self.i;
        while self.i < self.s.len() && (self.s[self.i].is_ascii_alphanumeric() || self.s[self.i] == b'_' || (self.i > st && self.s[self.i] == b'.')) { self.i += 1; }
        if self.i > st && !self.s[st].is_ascii_digit() { Some(self.src[st..self.i].to_string()) } else { self.i = st; None }
    }
    fn expr(&mut self) -> Result<Expr, String> {
        let mut l = self.term()?;
        while let Some(c @ (b'+' | b'-')) = self.peek() { self.i += 1; l = Expr::Bin(c as char, Box::new(l), Box::new(self.term()?)); }
        Ok(l)
    }
    fn term(&mut self) -> Result<Expr, String> {
        let mut l = self.unary()?;
        while let Some(c @ (b'*' | b'/')) = self.peek() { self.i += 1; l = Expr::Bin(c as char, Box::new(l), Box::new(self.unary()?)); }
        Ok(l)
    }
    fn unary(&mut self) -> Result<Expr, String> {
        if self.peek() == Some(b'-') { self.i += 1; return Ok(Expr::Neg(Box::new(self.unary()?))); }
        if self.peek() == Some(b'+') { self.i += 1; return self.unary(); }
        let base = self.atom()?;
        if self.peek() == Some(b'^') { self.i += 1; return Ok(Expr::Bin('^', Box::new(base), Box::new(self.unary()?))); }
        Ok(base)
    }
    fn atom(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Some(b'(') => {
                self.i += 1;
                let e = self.expr()?;
                if self.peek() != Some(b')') { return self.err("expected ')'"); }
                self.i += 1;
                Ok(e)
            }
            Some(c) if c.is_ascii_digit() || c == b'.' => {
                let st = self.i;
                while self.i < self.s.len() && (self.s[self.i].is_ascii_digit() || self.s[self.i] == b'.') { self.i += 1; }
                // exponent only when followed by a digit/sign+digit (so `2 e` is not eaten)
                if self.i < self.s.len() && matches!(self.s[self.i], b'e' | b'E') {
                    let j = self.i + 1 + matches!(self.s.get(self.i + 1), Some(b'+' | b'-')) as usize;
                    if self.s.get(j).is_some_and(u8::is_ascii_digit) { self.i = j; while self.i < self.s.len() && self.s[self.i].is_ascii_digit() { self.i += 1; } }
                }
                let v: f64 = match self.src[st..self.i].parse() { Ok(v) => v, Err(_) => return self.err("bad number") };
                // optional unit directly after a literal: `5 mm`, `30deg`, `45 °`
                let save = self.i;
                self.ws();
                let ust = self.i;
                while self.i < self.s.len() && (self.s[self.i].is_ascii_alphabetic() || self.s[self.i] >= 0x80) { self.i += 1; }
                let unit = &self.src[ust..self.i];
                if !unit.is_empty() {
                    if let Some((_, _, f)) = crate::units::UNITS.iter().find(|(u, _, _)| *u == unit) { return Ok(Expr::Num(v * f)); }
                }
                self.i = save;
                Ok(Expr::Num(v))
            }
            _ => {
                let Some(name) = self.ident() else { return self.err("expected a number, parameter or '('"); };
                if self.peek() == Some(b'(') {
                    self.i += 1;
                    let mut args = Vec::new();
                    if self.peek() != Some(b')') {
                        loop {
                            args.push(self.expr()?);
                            match self.peek() { Some(b',') => self.i += 1, Some(b')') => break, _ => return self.err("expected ',' or ')'") }
                        }
                    }
                    self.i += 1;
                    match FUNCS.iter().find(|(f, _)| *f == name) {
                        Some((_, n)) if *n == args.len() => Ok(Expr::Call(name, args)),
                        Some((_, n)) => self.err(format!("{name}() takes {n} argument(s), got {}", args.len())),
                        None => self.err(format!("unknown function '{name}' (available: {})", FUNCS.iter().map(|f| f.0).collect::<Vec<_>>().join(", "))),
                    }
                } else {
                    Ok(match name.as_str() { "pi" | "PI" => Expr::Num(core::f64::consts::PI), _ => Expr::Param(name) })
                }
            }
        }
    }
}

/// Parse formula source (without the leading `=`).
pub fn parse(src: &str) -> Result<Expr, String> {
    let mut p = P { s: src.as_bytes(), i: 0, src };
    let e = p.expr()?;
    if p.peek().is_some() { return p.err("unexpected trailing input"); }
    Ok(e)
}

impl Expr {
    /// Parameter names referenced (sorted, unique).
    pub fn params(&self) -> Vec<String> {
        fn go(e: &Expr, out: &mut Vec<String>) {
            match e {
                Expr::Param(n) => out.push(n.clone()),
                Expr::Neg(a) => go(a, out),
                Expr::Bin(_, a, b) => { go(a, out); go(b, out); }
                Expr::Call(_, a) => a.iter().for_each(|x| go(x, out)),
                Expr::Num(_) => {}
            }
        }
        let mut v = Vec::new();
        go(self, &mut v);
        v.sort();
        v.dedup();
        v
    }

    pub fn eval(&self, param: &dyn Fn(&str) -> Option<f64>) -> Result<f64, String> {
        Ok(match self {
            Expr::Num(v) => *v,
            Expr::Param(n) => param(n).ok_or_else(|| format!("unknown parameter '{n}' in formula"))?,
            Expr::Neg(a) => -a.eval(param)?,
            Expr::Bin(op, a, b) => {
                let (x, y) = (a.eval(param)?, b.eval(param)?);
                match op { '+' => x + y, '-' => x - y, '*' => x * y, '/' => x / y, _ => x.powf(y) }
            }
            Expr::Call(f, a) => {
                let v: Vec<f64> = a.iter().map(|x| x.eval(param)).collect::<Result<_, _>>()?;
                match f.as_str() {
                    "sqrt" => v[0].sqrt(), "abs" => v[0].abs(), "min" => v[0].min(v[1]), "max" => v[0].max(v[1]),
                    "sin" => v[0].sin(), "cos" => v[0].cos(), "tan" => v[0].tan(), "atan2" => v[0].atan2(v[1]),
                    "floor" => v[0].floor(), "ceil" => v[0].ceil(), _ => v[0].round(),
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn units_params_precedence_and_errors() {
        let e = parse("od/2 - 5 mm + 2^3 * max(t, 1in)").unwrap();
        assert_eq!(e.params(), vec!["od".to_string(), "t".to_string()]);
        let v = e.eval(&|n| match n { "od" => Some(100.0), "t" => Some(6.0), _ => None }).unwrap();
        assert!((v - (50.0 - 5.0 + 8.0 * 25.4)).abs() < 1e-12, "{v}");
        assert!((parse("-30 deg").unwrap().eval(&|_| None).unwrap() + core::f64::consts::PI / 6.0).abs() < 1e-15);
        assert!(parse("1e3").unwrap().eval(&|_| None).unwrap() == 1000.0);
        assert!(parse("od/").unwrap_err().contains("column"));
        assert!(parse("foo(1)").unwrap_err().contains("unknown function"));
        assert!(parse("x").unwrap().eval(&|_| None).unwrap_err().contains("unknown parameter 'x'"));
    }
}
