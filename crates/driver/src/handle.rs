//! Handles: `(session, slot, generation, revision)` (spec 02 § Handles; Demo MODEL_SESSIONS.md).
//!
//! `generation` guards slot reuse after `document.delete`; `revision` bumps on every
//! workspace change and every version. Model code never sees raw device pointers (A1-3).

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Pod, Zeroable, Serialize, Deserialize)]
#[repr(C)]
pub struct Handle {
    pub session: u32,
    pub slot: u32,
    pub generation: u32,
    pub revision: u32,
}

impl Handle {
    pub const NULL: Handle = Handle { session: u32::MAX, slot: u32::MAX, generation: 0, revision: 0 };

    pub fn is_null(&self) -> bool {
        self.slot == u32::MAX
    }

    /// Packed `(slot, generation)` as used in device-side element references.
    pub fn device_key(&self) -> u64 {
        ((self.generation as u64) << 32) | self.slot as u64
    }

    pub fn with_revision(self, revision: u32) -> Self {
        Handle { revision, ..self }
    }
}

impl core::fmt::Display for Handle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "h{}:{}g{}r{}", self.session, self.slot, self.generation, self.revision)
    }
}
