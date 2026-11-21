//! rustc-style diagnostics with spans and "did you mean" help (spec 04, AG-07 KPIs).

use agcad_driver::AgStatus;
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Diagnostic {
    pub severity: Severity,
    pub status: u16,
    pub message: String,
    pub line: usize,
    pub column: usize,
    pub help: Vec<String>,
}

impl Diagnostic {
    fn at(span: proc_macro2::Span, status: AgStatus, message: impl Into<String>) -> Self {
        let lc = span.start();
        Diagnostic { severity: Severity::Error, status: status.value(), message: message.into(), line: lc.line, column: lc.column + 1, help: Vec::new() }
    }
    pub fn sandbox(span: proc_macro2::Span, message: impl Into<String>) -> Self {
        Self::at(span, AgStatus::SandboxViolation, message)
    }
    pub fn compile(span: proc_macro2::Span, message: impl Into<String>) -> Self {
        Self::at(span, AgStatus::CompileFailed, message)
    }
    pub fn with_help(mut self, h: impl Into<String>) -> Self {
        self.help.push(h.into());
        self
    }

    /// rustc-like rendering against the source text.
    pub fn render(&self, file: &str, source: &str) -> String {
        let code = AgStatus::from_value(self.status).map(|s| s.name()).unwrap_or("?");
        let mut out = format!("error[{code}]: {}\n  --> {file}:{}:{}\n", self.message, self.line, self.column);
        if let Some(text) = source.lines().nth(self.line.saturating_sub(1)) {
            out += &format!("   |\n{:>3}| {text}\n   | {}^\n", self.line, " ".repeat(self.column.saturating_sub(1)));
        }
        for h in &self.help {
            out += &format!("   = help: {h}\n");
        }
        out
    }
}
