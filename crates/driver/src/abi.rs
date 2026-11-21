//! Bytecode ABI owned by the driver (spec 04 § Bytecode, spec 13 A3-1, AG-04-004).
//!
//! Little-endian, `repr(C)`, 16-byte aligned. Backward compatible within a MAJOR spec
//! version (AG-01-004): readers skip unknown record tails using `record_size`.

use crate::status::AgStatus;
use bytemuck::{Pod, Zeroable};

/// Magic at the start of every block: ASCII `AGB1`.
pub const BLOCK_MAGIC: [u8; 4] = *b"AGB1";
/// Current ABI version. Bump only with a backward-compatible reader path.
pub const ABI_VERSION: u16 = 1;
/// Size of one command record in this ABI version.
pub const RECORD_SIZE: u16 = 64;

/// Opcode ranges (spec 04).
pub mod opcode_range {
    /// Feature-table opcodes `(family << 8) | index`.
    pub const CATALOG: core::ops::RangeInclusive<u16> = 0x0000..=0x7FFF;
    /// Session-published UDF opcodes.
    pub const SESSION_UDF: core::ops::RangeInclusive<u16> = 0x8000..=0xFEFF;
    /// Expression opcodes interpreted by the device VM.
    pub const EXPRESSION: core::ops::RangeInclusive<u16> = 0xFF00..=0xFFFF;
}

/// Command flag bits.
pub mod flags {
    pub const OVER_SUBSET: u16 = 1 << 0;
    pub const CHECKPOINT: u16 = 1 << 1;
    pub const EXPECTS_EXACT: u16 = 1 << 2;
    pub const DIFFERENTIABLE_REQUEST: u16 = 1 << 3;
}

/// 64-byte block header preceding `record_count` command records.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
#[repr(C, align(16))]
pub struct BlockHeader {
    pub magic: [u8; 4],
    pub abi_version: u16,
    pub record_size: u16,
    pub record_count: u32,
    pub requirements_rev: u32,
    /// Feature-table version hash; mismatch → `NEED_RECOMPILE`.
    pub catalog_hash: u64,
    pub round: u32,
    pub producer_id: u32,
    pub reserved: [u32; 8],
}

impl BlockHeader {
    pub fn new(record_count: u32, requirements_rev: u32, catalog_hash: u64, round: u32, producer_id: u32) -> Self {
        Self {
            magic: BLOCK_MAGIC,
            abi_version: ABI_VERSION,
            record_size: RECORD_SIZE,
            record_count,
            requirements_rev,
            catalog_hash,
            round,
            producer_id,
            reserved: [0; 8],
        }
    }

    /// Host-side validation of framing. Device-side validation repeats the
    /// opcode/revision/budget checks per record (spec 06 wavefront step 1).
    pub fn validate(&self, expected_catalog_hash: u64) -> Result<(), AgStatus> {
        if self.magic != BLOCK_MAGIC {
            return Err(AgStatus::InvalidParameter);
        }
        if self.abi_version == 0 || self.abi_version > ABI_VERSION {
            return Err(AgStatus::NeedRecompile);
        }
        if self.record_size < RECORD_SIZE {
            return Err(AgStatus::InvalidParameter);
        }
        if self.catalog_hash != expected_catalog_hash {
            return Err(AgStatus::NeedRecompile);
        }
        Ok(())
    }
}

/// 64-byte command record.
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
#[repr(C, align(16))]
pub struct CommandRecord {
    pub opcode: u16,
    pub flags: u16,
    pub configuration: u32,
    /// SSA destination value slot.
    pub dst: u32,
    /// Value slots or arg-arena offsets.
    pub args: [u32; 6],
    pub expected_revision: u32,
    pub configuration_mask_slot: u32,
    pub reserved: u32,
    /// Immediates in internal units (mm, rad, kg, s).
    pub imm: [f64; 2],
}

/// 16-byte status record written by the GPU harness for every command.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct StatusRecord {
    pub status: u16,
    pub flags: u16,
    pub dst: u32,
    pub budget_used: u32,
    pub aux: u32,
}

impl StatusRecord {
    pub fn status(&self) -> Option<AgStatus> {
        AgStatus::from_value(self.status)
    }
}

/// Split a byte stream of blocks into (header, records), skipping unknown record tails.
pub fn parse_blocks(bytes: &[u8], expected_catalog_hash: u64) -> Result<Vec<(BlockHeader, Vec<CommandRecord>)>, AgStatus> {
    let mut out = Vec::new();
    let mut off = 0usize;
    let hsz = core::mem::size_of::<BlockHeader>();
    while off < bytes.len() {
        if bytes.len() - off < hsz {
            return Err(AgStatus::InvalidParameter);
        }
        let header: BlockHeader = bytemuck::pod_read_unaligned(&bytes[off..off + hsz]);
        header.validate(expected_catalog_hash)?;
        off += hsz;
        let rs = header.record_size as usize;
        let n = header.record_count as usize;
        let need = rs.checked_mul(n).ok_or(AgStatus::CapacityExceeded)?;
        if bytes.len() - off < need {
            return Err(AgStatus::InvalidParameter);
        }
        let mut recs = Vec::with_capacity(n);
        for i in 0..n {
            let start = off + i * rs;
            recs.push(bytemuck::pod_read_unaligned::<CommandRecord>(&bytes[start..start + RECORD_SIZE as usize]));
        }
        off += need;
        out.push((header, recs));
    }
    Ok(out)
}

/// Encode one block (current ABI).
pub fn encode_block(header: &BlockHeader, records: &[CommandRecord]) -> Vec<u8> {
    debug_assert_eq!(header.record_count as usize, records.len());
    let mut v = Vec::with_capacity(64 + records.len() * 64);
    v.extend_from_slice(bytemuck::bytes_of(header));
    for r in records {
        v.extend_from_slice(bytemuck::bytes_of(r));
    }
    v
}

const _: () = assert!(core::mem::size_of::<BlockHeader>() == 64);
const _: () = assert!(core::mem::size_of::<CommandRecord>() == 64);
const _: () = assert!(core::mem::size_of::<StatusRecord>() == 16);
const _: () = assert!(core::mem::offset_of!(CommandRecord, imm) == 48);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_roundtrip() {
        let mut r = CommandRecord::zeroed();
        r.opcode = 0x0700;
        r.imm = [12.0, 0.5];
        let h = BlockHeader::new(1, 3, 0xABCD, 0, 7);
        let bytes = encode_block(&h, &[r]);
        let blocks = parse_blocks(&bytes, 0xABCD).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].1[0], r);
        assert_eq!(parse_blocks(&bytes, 0x1).unwrap_err(), AgStatus::NeedRecompile);
    }
}
