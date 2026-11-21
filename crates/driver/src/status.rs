//! `AgStatus`: fixed, append-only `u16` status codes (spec 06 § Error codes, AG-06-005, AG-CV-003).
//! GENERATED from docs/spec/06-harness.md by `cargo xtask gen-status`. Do not edit by hand.

/// Category = high byte of the status value.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum StatusCategory {
    Info = 0x00,
    Lifecycle = 0x01,
    Revision = 0x02,
    Parameter = 0x03,
    Check = 0x04,
    Reference = 0x05,
    Resource = 0x06,
    Code = 0x07,
    Solver = 0x08,
}

/// Command/feature outcome. `0x0000` is success. Values never change (append-only).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u16)]
pub enum AgStatus {
    /// Applied
    Accepted = 0x0000,
    /// Reused an already-checked result
    CacheHit = 0x0001,
    /// Configuration finished
    Stopped = 0x0002,
    /// Configuration masked
    Inactive = 0x0003,
    /// Exact-phase result not yet available
    PendingExact = 0x0004,
    /// Info: the capacity planner split the study into chunks ([05](05-execution.md#memory-capacity-planning)); reported, not an error
    Chunked = 0x0005,
    /// Feature or kernel version changed; recompile
    NeedRecompile = 0x0100,
    /// Starting state invalid for this operation
    InvalidBaseline = 0x0101,
    /// Branch merge touched the same feature parameters or topology
    MergeConflict = 0x0102,
    /// A saved document could not be migrated to the current schema/op versions (details per feature)
    MigrationFailed = 0x0103,
    /// expected_revision is not current
    StaleRevision = 0x0200,
    /// Requirements revision mismatch, or an attempt to change requirements
    StaleRequirements = 0x0201,
    /// Opcode not in the allow-list or unknown
    InvalidOpcode = 0x0300,
    /// Out of range, wrong type or wrong unit
    InvalidParameter = 0x0301,
    /// NaN or Inf
    NonfiniteParameter = 0x0302,
    /// A hard requirement was violated at this phase
    RequirementFailed = 0x0400,
    /// A geometric check failed (self-intersection, fold, invalid surface)
    GeometryCheckFailed = 0x0401,
    /// Exact topology construction failed (boolean, fillet, sew, split)
    TopologyFailed = 0x0402,
    /// A numerical-consistency check failed
    NumericalCheckFailed = 0x0403,
    /// Only when requirements demand monotone improvement
    NoImprovement = 0x0404,
    /// An implicit body's declared bound failed spot checks; evidence downgraded
    LipschitzViolation = 0x0405,
    /// A name or handle no longer resolves (lineage attached)
    StaleReference = 0x0500,
    /// A name or query matched several elements where one was required
    AmbiguousReference = 0x0501,
    /// A resource budget is exhausted
    BudgetExhausted = 0x0600,
    /// Declared capacity overflow (never silently truncated)
    CapacityExceeded = 0x0601,
    /// Feature or conversion not available for these inputs or on this backend
    UnsupportedOp = 0x0602,
    /// AgLang/AgIR did not compile (diagnostics attached)
    CompileFailed = 0x0700,
    /// Code violated purity, bounds or memory rules
    SandboxViolation = 0x0701,
    /// UDF publication or implementation promotion failed conformance
    ConformanceFailed = 0x0702,
    /// Sketch, assembly or optimizer did not converge within budget
    SolverDiverged = 0x0800,
    /// Solver: remaining DOF where full definition was required
    UnderConstrained = 0x0801,
    /// Solver: conflicting constraints (conflict set attached)
    OverConstrained = 0x0802,
}

impl AgStatus {
    /// Every defined status, in value order.
    pub const ALL: &[AgStatus] = &[
        AgStatus::Accepted,
        AgStatus::CacheHit,
        AgStatus::Stopped,
        AgStatus::Inactive,
        AgStatus::PendingExact,
        AgStatus::Chunked,
        AgStatus::NeedRecompile,
        AgStatus::InvalidBaseline,
        AgStatus::MergeConflict,
        AgStatus::MigrationFailed,
        AgStatus::StaleRevision,
        AgStatus::StaleRequirements,
        AgStatus::InvalidOpcode,
        AgStatus::InvalidParameter,
        AgStatus::NonfiniteParameter,
        AgStatus::RequirementFailed,
        AgStatus::GeometryCheckFailed,
        AgStatus::TopologyFailed,
        AgStatus::NumericalCheckFailed,
        AgStatus::NoImprovement,
        AgStatus::LipschitzViolation,
        AgStatus::StaleReference,
        AgStatus::AmbiguousReference,
        AgStatus::BudgetExhausted,
        AgStatus::CapacityExceeded,
        AgStatus::UnsupportedOp,
        AgStatus::CompileFailed,
        AgStatus::SandboxViolation,
        AgStatus::ConformanceFailed,
        AgStatus::SolverDiverged,
        AgStatus::UnderConstrained,
        AgStatus::OverConstrained,
    ];

    /// Raw `u16` value as written in status records.
    pub const fn value(self) -> u16 { self as u16 }

    /// Decode a raw value; unknown values (from a newer driver) return `None`.
    pub fn from_value(v: u16) -> Option<Self> {
        match v {
            0x0000 => Some(AgStatus::Accepted),
            0x0001 => Some(AgStatus::CacheHit),
            0x0002 => Some(AgStatus::Stopped),
            0x0003 => Some(AgStatus::Inactive),
            0x0004 => Some(AgStatus::PendingExact),
            0x0005 => Some(AgStatus::Chunked),
            0x0100 => Some(AgStatus::NeedRecompile),
            0x0101 => Some(AgStatus::InvalidBaseline),
            0x0102 => Some(AgStatus::MergeConflict),
            0x0103 => Some(AgStatus::MigrationFailed),
            0x0200 => Some(AgStatus::StaleRevision),
            0x0201 => Some(AgStatus::StaleRequirements),
            0x0300 => Some(AgStatus::InvalidOpcode),
            0x0301 => Some(AgStatus::InvalidParameter),
            0x0302 => Some(AgStatus::NonfiniteParameter),
            0x0400 => Some(AgStatus::RequirementFailed),
            0x0401 => Some(AgStatus::GeometryCheckFailed),
            0x0402 => Some(AgStatus::TopologyFailed),
            0x0403 => Some(AgStatus::NumericalCheckFailed),
            0x0404 => Some(AgStatus::NoImprovement),
            0x0405 => Some(AgStatus::LipschitzViolation),
            0x0500 => Some(AgStatus::StaleReference),
            0x0501 => Some(AgStatus::AmbiguousReference),
            0x0600 => Some(AgStatus::BudgetExhausted),
            0x0601 => Some(AgStatus::CapacityExceeded),
            0x0602 => Some(AgStatus::UnsupportedOp),
            0x0700 => Some(AgStatus::CompileFailed),
            0x0701 => Some(AgStatus::SandboxViolation),
            0x0702 => Some(AgStatus::ConformanceFailed),
            0x0800 => Some(AgStatus::SolverDiverged),
            0x0801 => Some(AgStatus::UnderConstrained),
            0x0802 => Some(AgStatus::OverConstrained),
            _ => None,
        }
    }

    /// Stable UPPER_SNAKE name (cf. `cudaGetErrorName`).
    pub const fn name(self) -> &'static str {
        match self {
            AgStatus::Accepted => "ACCEPTED",
            AgStatus::CacheHit => "CACHE_HIT",
            AgStatus::Stopped => "STOPPED",
            AgStatus::Inactive => "INACTIVE",
            AgStatus::PendingExact => "PENDING_EXACT",
            AgStatus::Chunked => "CHUNKED",
            AgStatus::NeedRecompile => "NEED_RECOMPILE",
            AgStatus::InvalidBaseline => "INVALID_BASELINE",
            AgStatus::MergeConflict => "MERGE_CONFLICT",
            AgStatus::MigrationFailed => "MIGRATION_FAILED",
            AgStatus::StaleRevision => "STALE_REVISION",
            AgStatus::StaleRequirements => "STALE_REQUIREMENTS",
            AgStatus::InvalidOpcode => "INVALID_OPCODE",
            AgStatus::InvalidParameter => "INVALID_PARAMETER",
            AgStatus::NonfiniteParameter => "NONFINITE_PARAMETER",
            AgStatus::RequirementFailed => "REQUIREMENT_FAILED",
            AgStatus::GeometryCheckFailed => "GEOMETRY_CHECK_FAILED",
            AgStatus::TopologyFailed => "TOPOLOGY_FAILED",
            AgStatus::NumericalCheckFailed => "NUMERICAL_CHECK_FAILED",
            AgStatus::NoImprovement => "NO_IMPROVEMENT",
            AgStatus::LipschitzViolation => "LIPSCHITZ_VIOLATION",
            AgStatus::StaleReference => "STALE_REFERENCE",
            AgStatus::AmbiguousReference => "AMBIGUOUS_REFERENCE",
            AgStatus::BudgetExhausted => "BUDGET_EXHAUSTED",
            AgStatus::CapacityExceeded => "CAPACITY_EXCEEDED",
            AgStatus::UnsupportedOp => "UNSUPPORTED_OP",
            AgStatus::CompileFailed => "COMPILE_FAILED",
            AgStatus::SandboxViolation => "SANDBOX_VIOLATION",
            AgStatus::ConformanceFailed => "CONFORMANCE_FAILED",
            AgStatus::SolverDiverged => "SOLVER_DIVERGED",
            AgStatus::UnderConstrained => "UNDER_CONSTRAINED",
            AgStatus::OverConstrained => "OVER_CONSTRAINED",
        }
    }

    /// Human-readable description (cf. `cudaGetErrorString`).
    pub const fn description(self) -> &'static str {
        match self {
            AgStatus::Accepted => "Applied",
            AgStatus::CacheHit => "Reused an already-checked result",
            AgStatus::Stopped => "Configuration finished",
            AgStatus::Inactive => "Configuration masked",
            AgStatus::PendingExact => "Exact-phase result not yet available",
            AgStatus::Chunked => "Info: the capacity planner split the study into chunks ([05](05-execution.md#memory-capacity-planning)); reported, not an error",
            AgStatus::NeedRecompile => "Feature or kernel version changed; recompile",
            AgStatus::InvalidBaseline => "Starting state invalid for this operation",
            AgStatus::MergeConflict => "Branch merge touched the same feature parameters or topology",
            AgStatus::MigrationFailed => "A saved document could not be migrated to the current schema/op versions (details per feature)",
            AgStatus::StaleRevision => "expected_revision is not current",
            AgStatus::StaleRequirements => "Requirements revision mismatch, or an attempt to change requirements",
            AgStatus::InvalidOpcode => "Opcode not in the allow-list or unknown",
            AgStatus::InvalidParameter => "Out of range, wrong type or wrong unit",
            AgStatus::NonfiniteParameter => "NaN or Inf",
            AgStatus::RequirementFailed => "A hard requirement was violated at this phase",
            AgStatus::GeometryCheckFailed => "A geometric check failed (self-intersection, fold, invalid surface)",
            AgStatus::TopologyFailed => "Exact topology construction failed (boolean, fillet, sew, split)",
            AgStatus::NumericalCheckFailed => "A numerical-consistency check failed",
            AgStatus::NoImprovement => "Only when requirements demand monotone improvement",
            AgStatus::LipschitzViolation => "An implicit body's declared bound failed spot checks; evidence downgraded",
            AgStatus::StaleReference => "A name or handle no longer resolves (lineage attached)",
            AgStatus::AmbiguousReference => "A name or query matched several elements where one was required",
            AgStatus::BudgetExhausted => "A resource budget is exhausted",
            AgStatus::CapacityExceeded => "Declared capacity overflow (never silently truncated)",
            AgStatus::UnsupportedOp => "Feature or conversion not available for these inputs or on this backend",
            AgStatus::CompileFailed => "AgLang/AgIR did not compile (diagnostics attached)",
            AgStatus::SandboxViolation => "Code violated purity, bounds or memory rules",
            AgStatus::ConformanceFailed => "UDF publication or implementation promotion failed conformance",
            AgStatus::SolverDiverged => "Sketch, assembly or optimizer did not converge within budget",
            AgStatus::UnderConstrained => "Solver: remaining DOF where full definition was required",
            AgStatus::OverConstrained => "Solver: conflicting constraints (conflict set attached)",
        }
    }

    pub fn category(self) -> StatusCategory {
        match (self.value() >> 8) as u8 {
            0x00 => StatusCategory::Info,
            0x01 => StatusCategory::Lifecycle,
            0x02 => StatusCategory::Revision,
            0x03 => StatusCategory::Parameter,
            0x04 => StatusCategory::Check,
            0x05 => StatusCategory::Reference,
            0x06 => StatusCategory::Resource,
            0x07 => StatusCategory::Code,
            0x08 => StatusCategory::Solver,
            _ => unreachable!("AgStatus values are defined only in known categories"),
        }
    }

    /// Success and informational codes (category 0x00) do not reject a command.
    pub fn is_ok(self) -> bool { matches!(self.category(), StatusCategory::Info) }
}

impl core::fmt::Display for AgStatus {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} (0x{:04X}): {}", self.name(), self.value(), self.description())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_unique_and_roundtrip() {
        let mut seen = std::collections::BTreeSet::new();
        for s in AgStatus::ALL {
            assert!(seen.insert(s.value()), "duplicate {}", s.name());
            assert_eq!(AgStatus::from_value(s.value()), Some(*s));
        }
        assert_eq!(AgStatus::ALL.len(), 32);
        assert_eq!(AgStatus::Accepted.value(), 0);
    }
}
