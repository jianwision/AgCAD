//! CUDA virtual memory management: growable arenas, page-level copy-on-write for branches
//! (spec 05 § Memory) and read-only IPC to sandbox workers (spec 07; review A2-10).

use crate::{fault, CudaDevice};
use agcad_driver::*;
use cudarc::driver::sys;

pub struct CuVaRange {
    pub base: sys::CUdeviceptr,
    pub bytes: usize,
    /// Mapped (address, size) pieces; unmapped before the range is freed.
    mapped: std::sync::Mutex<Vec<(sys::CUdeviceptr, usize)>>,
}
unsafe impl Send for CuVaRange {}

impl Drop for CuVaRange {
    fn drop(&mut self) {
        unsafe {
            for (p, n) in self.mapped.get_mut().map(std::mem::take).unwrap_or_default() {
                let _ = sys::cuMemUnmap(p, n);
            }
            let _ = sys::cuMemAddressFree(self.base, self.bytes);
        }
    }
}

pub struct CuPhysPage {
    pub handle: sys::CUmemGenericAllocationHandle,
    pub bytes: usize,
}
unsafe impl Send for CuPhysPage {}

impl Drop for CuPhysPage {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuMemRelease(self.handle);
        }
    }
}

fn prop(device: i32) -> sys::CUmemAllocationProp {
    let mut p: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    p.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = device;
    // FD export for sandbox IPC exists only on POSIX; Windows pages are process-local.
    #[cfg(unix)]
    {
        p.requestedHandleTypes = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    }
    p
}

impl VmmExt for CudaDevice {
    type VaRange = CuVaRange;
    type PhysPage = CuPhysPage;

    fn granularity(&self) -> usize {
        let p = prop(self.context().ordinal() as i32);
        let mut g: usize = 0;
        unsafe {
            let _ = sys::cuMemGetAllocationGranularity(&mut g, &p, sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        }
        g.max(2 << 20)
    }

    fn reserve(&self, va_bytes: usize) -> DriverResult<CuVaRange> {
        self.context().bind_to_thread().map_err(fault::driver)?;
        let bytes = align_up(va_bytes, self.granularity());
        let mut base: sys::CUdeviceptr = 0;
        unsafe { fault::check(sys::cuMemAddressReserve(&mut base, bytes, 0, 0, 0)).map_err(fault::driver)? };
        Ok(CuVaRange { base, bytes, mapped: std::sync::Mutex::new(Vec::new()) })
    }

    fn create_page(&self, bytes: usize) -> DriverResult<CuPhysPage> {
        let p = prop(self.context().ordinal() as i32);
        let bytes = align_up(bytes, self.granularity());
        let mut h: sys::CUmemGenericAllocationHandle = 0;
        unsafe { fault::check(sys::cuMemCreate(&mut h, bytes, &p, 0)).map_err(fault::driver)? };
        Ok(CuPhysPage { handle: h, bytes })
    }

    fn map(&self, va: &CuVaRange, off: usize, page: &CuPhysPage, access: Access) -> DriverResult<()> {
        if off + page.bytes > va.bytes {
            return Err(EnqueueError::InvalidConfig("map out of range".into()).into());
        }
        let ptr = va.base + off as u64;
        unsafe { fault::check(sys::cuMemMap(ptr, page.bytes, 0, page.handle, 0)).map_err(fault::driver)? };
        va.mapped.lock().expect("vmm map list poisoned").push((ptr, page.bytes));
        let mut desc: sys::CUmemAccessDesc = unsafe { std::mem::zeroed() };
        desc.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        desc.location.id = self.context().ordinal() as i32;
        desc.flags = match access {
            Access::ReadOnly => sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ,
            Access::ReadWrite => sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        unsafe { fault::check(sys::cuMemSetAccess(ptr, page.bytes, &desc, 1)).map_err(fault::driver) }
    }

    #[cfg(unix)]
    fn export_fd(&self, page: &CuPhysPage) -> DriverResult<std::os::fd::OwnedFd> {
        use std::os::fd::FromRawFd;
        let mut fd: i32 = -1;
        unsafe {
            fault::check(sys::cuMemExportToShareableHandle(
                &mut fd as *mut i32 as *mut std::ffi::c_void,
                page.handle,
                sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0,
            ))
            .map_err(fault::driver)?;
            Ok(std::os::fd::OwnedFd::from_raw_fd(fd))
        }
    }

    #[cfg(unix)]
    fn import_fd(&self, fd: std::os::fd::OwnedFd, bytes: usize) -> DriverResult<CuPhysPage> {
        use std::os::fd::IntoRawFd;
        let raw = fd.into_raw_fd();
        let mut h: sys::CUmemGenericAllocationHandle = 0;
        unsafe {
            fault::check(sys::cuMemImportFromShareableHandle(
                &mut h,
                raw as usize as *mut std::ffi::c_void,
                sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            ))
            .map_err(fault::driver)?;
        }
        Ok(CuPhysPage { handle: h, bytes })
    }
}
