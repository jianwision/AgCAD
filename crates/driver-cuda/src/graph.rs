//! CUDA Graphs: capture → instantiate → update → launch, plus conditional nodes
//! (IF/WHILE since CUDA 12.3, IF-ELSE/SWITCH since 12.8) used by the GPU-harness WHILE loop
//! and per-opcode SWITCH bins (spec 06 § GPU-resident harness; review F2). Device-side graph
//! launch is deliberately not used (support with conditionals unconfirmed).

use crate::{fault, CuStream, CudaDevice};
use agcad_driver::*;
use cudarc::driver::sys;

pub struct CuGraph {
    pub(crate) raw: sys::CUgraph,
    owned: bool,
}
unsafe impl Send for CuGraph {}

impl Drop for CuGraph {
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                let _ = sys::cuGraphDestroy(self.raw);
            }
        }
    }
}

pub struct CuGraphExec {
    raw: sys::CUgraphExec,
}
unsafe impl Send for CuGraphExec {}

impl Drop for CuGraphExec {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuGraphExecDestroy(self.raw);
        }
    }
}

impl GraphExt for CudaDevice {
    type Graph = CuGraph;
    type GraphExec = CuGraphExec;

    fn begin_capture(&self, s: &CuStream, mode: CaptureMode) -> DriverResult<()> {
        self.context().bind_to_thread().map_err(fault::driver)?;
        let m = match mode {
            CaptureMode::ThreadLocal => sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            CaptureMode::Global => sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
            CaptureMode::Relaxed => sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
        };
        unsafe { fault::check(sys::cuStreamBeginCapture_v2(s.raw(), m)).map_err(fault::driver) }
    }

    fn end_capture(&self, s: &CuStream) -> DriverResult<CuGraph> {
        let mut g: sys::CUgraph = std::ptr::null_mut();
        unsafe { fault::check(sys::cuStreamEndCapture(s.raw(), &mut g)).map_err(fault::driver)? };
        Ok(CuGraph { raw: g, owned: true })
    }

    fn instantiate(&self, g: &CuGraph) -> DriverResult<CuGraphExec> {
        let mut e: sys::CUgraphExec = std::ptr::null_mut();
        unsafe { fault::check(sys::cuGraphInstantiateWithFlags(&mut e, g.raw, 0)).map_err(fault::driver)? };
        Ok(CuGraphExec { raw: e })
    }

    fn update(&self, exec: &mut CuGraphExec, g: &CuGraph) -> DriverResult<UpdateResult> {
        let mut info: sys::CUgraphExecUpdateResultInfo = unsafe { std::mem::zeroed() };
        let r = unsafe { sys::cuGraphExecUpdate_v2(exec.raw, g.raw, &mut info) };
        if r == sys::CUresult::CUDA_SUCCESS {
            Ok(UpdateResult::Updated)
        } else {
            // Topology changed: re-instantiate.
            *exec = self.instantiate(g)?;
            Ok(UpdateResult::TopologyChanged)
        }
    }

    fn launch_graph(&self, s: &CuStream, exec: &CuGraphExec) -> Result<(), EnqueueError> {
        self.context().bind_to_thread().map_err(fault::enqueue)?;
        unsafe { fault::check(sys::cuGraphLaunch(exec.raw, s.raw())).map_err(fault::enqueue) }
    }

    fn new_graph(&self) -> DriverResult<CuGraph> {
        self.context().bind_to_thread().map_err(fault::driver)?;
        let mut g: sys::CUgraph = std::ptr::null_mut();
        unsafe { fault::check(sys::cuGraphCreate(&mut g, 0)).map_err(fault::driver)? };
        Ok(CuGraph { raw: g, owned: true })
    }

    fn capture_into(&self, s: &CuStream, g: &CuGraph, f: &mut dyn FnMut() -> DriverResult<()>) -> DriverResult<()> {
        self.context().bind_to_thread().map_err(fault::driver)?;
        let mode = sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
        unsafe { fault::check(sys::cuStreamBeginCaptureToGraph(s.raw(), g.raw, std::ptr::null(), std::ptr::null(), 0, mode)).map_err(fault::driver)? };
        let r = f();
        // Always end the capture (even on error) so the stream leaves capture mode.
        let mut out: sys::CUgraph = std::ptr::null_mut();
        let e = unsafe { fault::check(sys::cuStreamEndCapture(s.raw(), &mut out)).map_err(fault::driver) };
        r?;
        e
    }
}

impl ConditionalExt for CudaDevice {
    fn conditional(&self, g: &mut CuGraph, kind: CondKind, default: u32) -> DriverResult<(CondHandle, Vec<CuGraph>)> {
        let need = match kind { CondKind::If | CondKind::While => CondSupport::IfWhile, _ => CondSupport::Full };
        let have = self.caps().conditional_nodes;
        let ok = matches!((need, have), (CondSupport::IfWhile, CondSupport::IfWhile | CondSupport::Full) | (CondSupport::Full, CondSupport::Full));
        if !ok {
            return Err(EnqueueError::Unsupported("conditional graph nodes require CUDA 12.3 (IF/WHILE) or 12.8 (IF-ELSE/SWITCH)").into());
        }
        self.context().bind_to_thread().map_err(fault::driver)?;
        let mut handle: sys::CUgraphConditionalHandle = 0;
        unsafe {
            fault::check(sys::cuGraphConditionalHandleCreate(
                &mut handle,
                g.raw,
                self.context().cu_ctx(),
                default,
                1, // CU_GRAPH_COND_ASSIGN_DEFAULT
            ))
            .map_err(fault::driver)?;
        }
        let (ty, size) = match kind {
            CondKind::If => (sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_IF, 1u32),
            CondKind::IfElse => (sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_IF, 2u32),
            CondKind::While => (sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_WHILE, 1u32),
            CondKind::Switch(n) => (sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_SWITCH, n),
        };
       
        let mut params: sys::CUgraphNodeParams = unsafe { std::mem::zeroed() };
        params.type_ = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL;
        // Writing Copy fields of a union is safe.
        params.__bindgen_anon_1.conditional.handle = handle;
        params.__bindgen_anon_1.conditional.type_ = ty;
        params.__bindgen_anon_1.conditional.size = size;
        params.__bindgen_anon_1.conditional.ctx = self.context().cu_ctx();
        let mut node: sys::CUgraphNode = std::ptr::null_mut();
        unsafe {
            fault::check(sys::cuGraphAddNode(&mut node, g.raw, std::ptr::null(), 0, &mut params)).map_err(fault::driver)?;
        }
        let bodies = unsafe {
            let p = params.__bindgen_anon_1.conditional.phGraph_out;
            (0..size as usize).map(|i| CuGraph { raw: *p.add(i), owned: false }).collect()
        };
        Ok((CondHandle(handle), bodies))
    }
}
