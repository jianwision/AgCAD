//! AgIR ⇄ bytecode (spec 04 § Bytecode; spec 07 § Cross-path equivalence).
//!
//! Lowering assigns each statement an SSA value slot (its index). Arguments map to the
//! catalog input positions: refs become value slots, up to two scalar immediates go in
//! `imm`, everything else is a typed record in the **arg arena** referenced by offset.

use crate::catalog::Catalog;
use crate::program::{Program, Stmt, Value};
use crate::units::{Dimension, Quantity};
use agcad_driver::abi::{flags, BlockHeader, CommandRecord};
use agcad_driver::AgStatus;
use std::collections::{BTreeMap, HashMap};

/// `configuration` value meaning "broadcast over all configurations".
pub const CONFIG_ALL: u32 = u32::MAX;
/// Arg slot encodings (top 4 bits = kind, low 28 bits = payload).
pub const ARG_NONE: u32 = 0x0000_0000;
pub const ARG_SLOT: u32 = 0x1000_0000;
pub const ARG_ARENA: u32 = 0x2000_0000;
pub const ARG_IMM0: u32 = 0x3000_0000;
pub const ARG_IMM1: u32 = 0x3000_0001;
const KIND_MASK: u32 = 0xF000_0000;

/// Typed record tags in the arg arena.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ArenaTag {
    F64Array = 1,
    I64Array = 2,
    Utf8 = 3,
    /// JSON-encoded structured value (maps, nested lists, named enums).
    Json = 4,
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct ArgArena {
    pub bytes: Vec<u8>,
}

impl ArgArena {
    fn push(&mut self, tag: ArenaTag, payload: &[u8]) -> u32 {
        // 16-byte aligned records: [tag u32][len u32][reserved u64][payload ...]
        while self.bytes.len() % 16 != 0 { self.bytes.push(0); }
        let off = self.bytes.len() as u32;
        self.bytes.extend_from_slice(&(tag as u32).to_le_bytes());
        self.bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        self.bytes.extend_from_slice(&0u64.to_le_bytes());
        self.bytes.extend_from_slice(payload);
        off
    }

    pub fn read(&self, off: u32) -> Option<(ArenaTag, &[u8])> {
        let o = off as usize;
        let tag = u32::from_le_bytes(self.bytes.get(o..o + 4)?.try_into().ok()?);
        let len = u32::from_le_bytes(self.bytes.get(o + 4..o + 8)?.try_into().ok()?) as usize;
        let payload = self.bytes.get(o + 16..o + 16 + len)?;
        let tag = match tag { 1 => ArenaTag::F64Array, 2 => ArenaTag::I64Array, 3 => ArenaTag::Utf8, 4 => ArenaTag::Json, _ => return None };
        Some((tag, payload))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bytecode {
    pub header: BlockHeader,
    pub records: Vec<CommandRecord>,
    pub arena: ArgArena,
    /// Slot → statement id (host-side, for diagnostics and decoding).
    pub slot_ids: Vec<String>,
}

fn numeric_list(v: &[Value]) -> Option<Vec<f64>> {
    v.iter().map(|x| x.as_f64()).collect()
}

pub fn lower(p: &Program, catalog: &Catalog, round: u32, producer_id: u32) -> Result<Bytecode, (String, AgStatus)> {
    let mut slots: HashMap<&str, u32> = HashMap::new();
    let mut records = Vec::with_capacity(p.statements.len());
    let mut arena = ArgArena::default();
    for (i, s) in p.statements.iter().enumerate() {
        let op = catalog.get(&s.op).ok_or_else(|| (s.id.clone(), AgStatus::InvalidOpcode))?;
        let params = op.params();
        if params.len() > 6 && s.args.len() > 6 {
            return Err((s.id.clone(), AgStatus::CapacityExceeded));
        }
        let mut rec: CommandRecord = bytemuck::Zeroable::zeroed();
        rec.opcode = op.opcode;
        rec.configuration = CONFIG_ALL;
        rec.dst = i as u32;
        rec.expected_revision = p.expected_revision;
        let mut imm_used = 0usize;
        // Arguments in catalog order; unknown extra args are packed as one JSON record in args[5].
        let mut extra: BTreeMap<String, Value> = BTreeMap::new();
        for (k, v) in &s.args {
            match params.iter().position(|pp| &pp.name == k) {
                Some(pos) if pos < 6 => {
                    rec.args[pos] = match v {
                        Value::Ref { id, field: None } => ARG_SLOT | *slots.get(id.as_str()).ok_or_else(|| (s.id.clone(), AgStatus::StaleReference))?,
                        Value::Int(_) | Value::Float(_) | Value::Quantity(_) | Value::Bool(_) if imm_used < 2 => {
                            let x = match v { Value::Bool(b) => *b as i32 as f64, _ => v.as_f64().unwrap() };
                            if !x.is_finite() { return Err((s.id.clone(), AgStatus::NonfiniteParameter)); }
                            rec.imm[imm_used] = x;
                            imm_used += 1;
                            if imm_used == 1 { ARG_IMM0 } else { ARG_IMM1 }
                        }
                        Value::List(l) if numeric_list(l).is_some() => {
                            let xs = numeric_list(l).unwrap();
                            let bytes: Vec<u8> = xs.iter().flat_map(|x| x.to_le_bytes()).collect();
                            ARG_ARENA | arena.push(ArenaTag::F64Array, &bytes)
                        }
                        Value::String(t) => ARG_ARENA | arena.push(ArenaTag::Utf8, t.as_bytes()),
                        other => {
                            let j = crate::json::encode_value(other).to_string();
                            ARG_ARENA | arena.push(ArenaTag::Json, j.as_bytes())
                        }
                    };
                }
                _ => { extra.insert(k.clone(), v.clone()); }
            }
        }
        if !extra.is_empty() {
            let j = crate::json::encode_value(&Value::Map(extra)).to_string();
            rec.args[5] = ARG_ARENA | arena.push(ArenaTag::Json, j.as_bytes());
        }
        if let Some(over) = &s.over {
            rec.flags |= flags::OVER_SUBSET;
            rec.configuration_mask_slot = *slots.get(over.as_str()).ok_or_else(|| (s.id.clone(), AgStatus::StaleReference))?;
        }
        if s.op == "document.checkpoint" { rec.flags |= flags::CHECKPOINT; }
        if op.phase == crate::catalog::Phase::Exact { rec.flags |= flags::EXPECTS_EXACT; }
        slots.insert(s.id.as_str(), i as u32);
        records.push(rec);
    }
    let header = BlockHeader::new(records.len() as u32, p.requirements_rev, catalog.hash(), round, producer_id);
    Ok(Bytecode { header, records, arena, slot_ids: p.statements.iter().map(|s| s.id.clone()).collect() })
}

/// Raise bytecode back to AgIR (replay across paths; spec 07). Immediates come back as
/// plain floats in internal units; dimensions are restored from catalog parameter types.
pub fn raise(bc: &Bytecode, catalog: &Catalog) -> Result<Program, AgStatus> {
    let mut p = Program::new(bc.header.requirements_rev, bc.records.first().map(|r| r.expected_revision).unwrap_or(0));
    for (i, r) in bc.records.iter().enumerate() {
        let op = catalog.by_opcode(r.opcode).ok_or(AgStatus::InvalidOpcode)?;
        let params = op.params();
        let mut args = BTreeMap::new();
        for (pos, a) in r.args.iter().enumerate() {
            if *a == ARG_NONE { continue; }
            let name = params.get(pos).map(|pp| pp.name.clone()).unwrap_or_else(|| format!("arg{pos}"));
            let dim = params.get(pos).and_then(|pp| pp.ty.names().first().and_then(|n| crate::units::dimension_of_type(n)));
            let v = match a & KIND_MASK {
                ARG_SLOT => Value::Ref { id: bc.slot_ids.get((a & !KIND_MASK) as usize).cloned().unwrap_or_else(|| format!("v{}", a & !KIND_MASK)), field: None },
                0x3000_0000 => {
                    let x = r.imm[(a & 1) as usize];
                    match dim { Some(d) if d != Dimension::Dimensionless => Value::Quantity(Quantity::new(x, d)), _ => Value::Float(x) }
                }
                ARG_ARENA => {
                    let (tag, payload) = bc.arena.read(a & !KIND_MASK).ok_or(AgStatus::InvalidParameter)?;
                    match tag {
                        ArenaTag::F64Array => Value::List(payload.chunks_exact(8).map(|c| Value::Float(f64::from_le_bytes(c.try_into().unwrap()))).collect()),
                        ArenaTag::I64Array => Value::List(payload.chunks_exact(8).map(|c| Value::Int(i64::from_le_bytes(c.try_into().unwrap()))).collect()),
                        ArenaTag::Utf8 => Value::String(String::from_utf8_lossy(payload).into_owned()),
                        ArenaTag::Json => {
                            let j: serde_json::Value = serde_json::from_slice(payload).map_err(|_| AgStatus::InvalidParameter)?;
                            let v = crate::json::decode_value(&j, "$").map_err(|_| AgStatus::InvalidParameter)?;
                            if pos == 5 && !params.get(5).is_some() {
                                if let Value::Map(m) = v { args.extend(m); continue; }
                            }
                            v
                        }
                    }
                }
                _ => return Err(AgStatus::InvalidParameter),
            };
            args.insert(name, v);
        }
        let over = if r.flags & flags::OVER_SUBSET != 0 { bc.slot_ids.get(r.configuration_mask_slot as usize).cloned() } else { None };
        p.statements.push(Stmt { id: bc.slot_ids.get(i).cloned().unwrap_or_else(|| format!("v{i}")), op: op.name.clone(), args, over });
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lower_raise_roundtrip_ops_and_refs() {
        let c = Catalog::global();
        let src = r#"{"requirements_rev":3,"expected_revision":41,"program":[
          {"id":"s","op":"sketch.create","args":{"support":"xy"}},
          {"id":"pd","op":"part.pad","args":{"profile":"%s","limits":"12 mm"}}]}"#;
        let p = crate::json::decode_program(src, c).unwrap();
        let bc = lower(&p, c, 0, 1).unwrap();
        assert_eq!(bc.records[1].args[0], ARG_SLOT | 0);
        let back = raise(&bc, c).unwrap();
        assert_eq!(back.statements[1].op, "part.pad");
        assert_eq!(back.statements[1].args["profile"], Value::Ref { id: "s".into(), field: None });
    }
}
