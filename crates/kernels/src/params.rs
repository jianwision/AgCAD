//! Parameter-block upload helper.

use agcad_driver::{AgDevice, DriverResult};
use bytemuck::Pod;

/// A parameter block resident on the device. Device buffers free stream-ordered, so the
/// block may be dropped right after the launch is enqueued (AG-08-004).
pub struct ParamUpload<D: AgDevice> {
    pub buffer: D::Buffer,
}

pub fn upload_params<D: AgDevice, P: Pod>(dev: &D, s: &D::Stream, p: &P) -> DriverResult<ParamUpload<D>> {
    let bytes = bytemuck::bytes_of(p);
    let buffer = dev.alloc_bytes(s, bytes.len())?;
    dev.upload_from_host_async(s, bytes, &buffer, 0)?;
    Ok(ParamUpload { buffer })
}
