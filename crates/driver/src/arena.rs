//! Resident typed slab arenas with generational handles (spec 05 § Memory; spec 02 § Handles).
//!
//! The host keeps slot bookkeeping only (free list, generations, revisions); element data
//! lives on the device. Overflow is `CAPACITY_EXCEEDED`, never truncation (AG-04-003).

use crate::device::{DeviceSlice, RawBuffer};
use crate::handle::Handle;
use crate::status::AgStatus;
use bytemuck::Pod;

pub struct Arena<T: Pod, B: RawBuffer> {
    storage: DeviceSlice<T, B>,
    session: u32,
    generations: Vec<u32>,
    revisions: Vec<u32>,
    live: Vec<bool>,
    free: Vec<u32>,
    next: u32,
}

impl<T: Pod, B: RawBuffer> Arena<T, B> {
    pub fn new(session: u32, storage: DeviceSlice<T, B>) -> Self {
        let cap = storage.len();
        Self {
            storage,
            session,
            generations: vec![0; cap],
            revisions: vec![0; cap],
            live: vec![false; cap],
            free: Vec::new(),
            next: 0,
        }
    }

    pub fn capacity(&self) -> usize { self.storage.len() }
    pub fn live_count(&self) -> usize { self.live.iter().filter(|l| **l).count() }
    pub fn storage(&self) -> &DeviceSlice<T, B> { &self.storage }

    /// Reserve a slot. Device data for the slot is written by kernels.
    pub fn allocate(&mut self) -> Result<Handle, AgStatus> {
        let slot = if let Some(s) = self.free.pop() {
            s
        } else if (self.next as usize) < self.capacity() {
            self.next += 1;
            self.next - 1
        } else {
            return Err(AgStatus::CapacityExceeded);
        };
        let i = slot as usize;
        self.live[i] = true;
        self.revisions[i] = 0;
        Ok(Handle { session: self.session, slot, generation: self.generations[i], revision: 0 })
    }

    /// Release a slot; bumps the generation so stale handles cannot address a new object.
    pub fn release(&mut self, h: Handle) -> Result<(), AgStatus> {
        self.check(h)?;
        let i = h.slot as usize;
        self.live[i] = false;
        self.generations[i] = self.generations[i].wrapping_add(1);
        self.free.push(h.slot);
        Ok(())
    }

    /// Validate session, slot, generation and revision (`STALE_REFERENCE` / `STALE_REVISION`).
    pub fn check(&self, h: Handle) -> Result<(), AgStatus> {
        let i = h.slot as usize;
        if h.session != self.session || i >= self.capacity() || !self.live[i] || self.generations[i] != h.generation {
            return Err(AgStatus::StaleReference);
        }
        if self.revisions[i] != h.revision {
            return Err(AgStatus::StaleRevision);
        }
        Ok(())
    }

    /// Record a mutation of the slot (workspace edit or version); returns the new handle.
    pub fn bump_revision(&mut self, h: Handle) -> Result<Handle, AgStatus> {
        self.check(h)?;
        let i = h.slot as usize;
        self.revisions[i] = self.revisions[i].wrapping_add(1);
        Ok(h.with_revision(self.revisions[i]))
    }

    /// Byte offset of a slot's element in the device storage.
    pub fn offset_of(&self, h: Handle) -> Result<usize, AgStatus> {
        self.check(h)?;
        Ok(h.slot as usize * core::mem::size_of::<T>())
    }
}
