//! `CuModel`: device-resident policy (spec 07 § GPU model path). Weights are safetensors
//! (written by the Candle trainer, `--features train`); inference runs in AgCAD kernels
//! (`cu/mlp.cu`, Tier N; CPU reference here) so it can live inside the rollout graph.
//! Candle-vs-kernel parity is checked per checkpoint (AG-07-005).

use crate::adaptor::*;
use agcad_driver::abi::{flags, CommandRecord};
use agcad_driver::*;
use agcad_kernels::upload_params;
use bytemuck::{Pod, Zeroable};
use std::collections::BTreeMap;

/// Minimal safetensors (f32 tensors only): 8-byte LE header length, JSON header, data.
pub fn read_safetensors(bytes: &[u8]) -> Result<BTreeMap<String, (Vec<usize>, Vec<f32>)>, String> {
    let n = u64::from_le_bytes(bytes.get(..8).ok_or("truncated")?.try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(bytes.get(8..8 + n).ok_or("truncated header")?).map_err(|e| e.to_string())?;
    let data = &bytes[8 + n..];
    let mut out = BTreeMap::new();
    for (k, v) in header.as_object().ok_or("header must be an object")? {
        if k == "__metadata__" { continue; }
        if v["dtype"] != "F32" { return Err(format!("{k}: only F32 supported")); }
        let shape: Vec<usize> = v["shape"].as_array().ok_or("shape")?.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
        let off = v["data_offsets"].as_array().ok_or("offsets")?;
        let (a, b) = (off[0].as_u64().unwrap_or(0) as usize, off[1].as_u64().unwrap_or(0) as usize);
        let raw = data.get(a..b).ok_or("tensor out of range")?;
        out.insert(k.clone(), (shape, bytemuck::pod_collect_to_vec(raw)));
    }
    Ok(out)
}

pub fn write_safetensors(tensors: &BTreeMap<String, (Vec<usize>, Vec<f32>)>) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (k, (shape, v)) in tensors {
        let a = data.len();
        data.extend_from_slice(bytemuck::cast_slice(v));
        header.insert(k.clone(), serde_json::json!({ "dtype": "F32", "shape": shape, "data_offsets": [a, data.len()] }));
    }
    let mut h = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    while (8 + h.len()) % 8 != 0 { h.push(b' '); }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend_from_slice(&data);
    out
}

/// Dense layers `layers[i] = (w [n_out × n_in], b [n_out])`.
#[derive(Clone, Debug, PartialEq)]
pub struct Mlp {
    pub layers: Vec<(usize, usize, Vec<f32>, Vec<f32>)>,
}

impl Mlp {
    /// Load `layer{i}.weight` / `layer{i}.bias` tensors.
    pub fn from_safetensors(bytes: &[u8]) -> Result<Mlp, String> {
        let t = read_safetensors(bytes)?;
        let mut layers = Vec::new();
        for i in 0.. {
            let (Some((ws, w)), Some((_, b))) = (t.get(&format!("layer{i}.weight")), t.get(&format!("layer{i}.bias"))) else { break };
            if ws.len() != 2 { return Err(format!("layer{i}.weight must be 2-D")); }
            layers.push((ws[1], ws[0], w.clone(), b.clone()));
        }
        if layers.is_empty() { return Err("no layers".into()); }
        Ok(Mlp { layers })
    }

    /// CPU reference forward over a batch `[configs × n_in]` (same order as the kernel).
    pub fn forward(&self, x: &[f32], configs: usize) -> Vec<f32> {
        let mut cur = x.to_vec();
        for (li, (n_in, n_out, w, b)) in self.layers.iter().enumerate() {
            let relu = li + 1 < self.layers.len();
            let mut y = vec![0f32; configs * n_out];
            for c in 0..configs { for o in 0..*n_out {
                let mut acc = b[o];
                for i in 0..*n_in { acc += w[o * n_in + i] * cur[c * n_in + i]; }
                y[c * n_out + o] = if relu && acc < 0.0 { 0.0 } else { acc };
            }}
            cur = y;
        }
        cur
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MlpLayerParams { pub x: u64, pub w: u64, pub b: u64, pub y: u64, pub configs: u32, pub n_in: u32, pub n_out: u32, pub relu: u32 }

/// # Safety: CPU reference device addresses.
pub unsafe fn ag_mlp_layer_cpu(params: *const u8, _: &LaunchConfig) {
    let p: MlpLayerParams = unsafe { core::ptr::read_unaligned(params as *const MlpLayerParams) };
    let (c, ni, no) = (p.configs as usize, p.n_in as usize, p.n_out as usize);
    let x = unsafe { core::slice::from_raw_parts(p.x as *const f32, c * ni) };
    let w = unsafe { core::slice::from_raw_parts(p.w as *const f32, no * ni) };
    let b = unsafe { core::slice::from_raw_parts(p.b as *const f32, no) };
    let y = unsafe { core::slice::from_raw_parts_mut(p.y as *mut f32, c * no) };
    for ci in 0..c { for o in 0..no {
        let mut acc = b[o];
        for i in 0..ni { acc += w[o * ni + i] * x[ci * ni + i]; }
        y[ci * no + o] = if p.relu != 0 && acc < 0.0 { 0.0 } else { acc };
    }}
}

pub const MODEL_CPU: &[(&str, CpuKernelFn)] = &[("ag_mlp_layer", ag_mlp_layer_cpu)];

pub fn model_library() -> LibrarySource {
    LibrarySource { name: "ag_model", cuda_source: Some(agcad_kernels_cuda::bundle(include_str!("../cu/mlp.cu"), &[])), cpu_kernels: MODEL_CPU, options: vec![] }
}

/// Persistent, capture-safe device-resident MLP (`Spec 08 § Models`): weights, biases,
/// intermediate activation buffers and parameter blocks are uploaded **once** at creation time
/// so every [`ResidentMlp::forward_into`] call performs zero allocations and zero host
/// transfers and can execute directly inside a captured CUDA Graph `WHILE` body.
pub struct ResidentMlp<D: KernelExt> {
    lib: LibraryId,
    configs: u32,
    n_in: usize,
    n_out: usize,
    layers: Vec<ResidentLayer<D>>,
}

struct ResidentLayer<D: KernelExt> {
    n_in: u32,
    n_out: u32,
    relu: u32,
    w: D::Buffer,
    b: D::Buffer,
    y: D::Buffer,
    /// Pre-uploaded parameter block when input pointer is fixed (layers >= 1, or cached obs).
    params: Option<(u64, agcad_kernels::params::ParamUpload<D>)>,
}

impl<D: KernelExt> ResidentMlp<D> {
    pub fn load(dev: &D, s: &D::Stream, lib: LibraryId, mlp: &Mlp, configs: u32) -> DriverResult<Self> {
        let up = |v: &[f32]| -> DriverResult<D::Buffer> {
            let b = dev.alloc_bytes(s, v.len().max(1) * 4)?;
            dev.upload_tagged_async(s, TransferKind::Control, bytemuck::cast_slice(v), &b, 0)?;
            Ok(b)
        };
        let mut layers = Vec::with_capacity(mlp.layers.len());
        let mut prev_ptr: Option<u64> = None;
        for (li, (n_in, n_out, w, b)) in mlp.layers.iter().enumerate() {
            let wb = up(w)?;
            let bb = up(b)?;
            let y = dev.alloc_bytes(s, (configs as usize * *n_out).max(1) * 4)?;
            let relu = (li + 1 < mlp.layers.len()) as u32;
            let params = if let Some(xptr) = prev_ptr {
                let prm = MlpLayerParams { x: xptr, w: wb.device_ptr(), b: bb.device_ptr(), y: y.device_ptr(), configs, n_in: *n_in as u32, n_out: *n_out as u32, relu };
                Some((xptr, upload_params(dev, s, &prm)?))
            } else {
                None
            };
            prev_ptr = Some(y.device_ptr());
            layers.push(ResidentLayer { n_in: *n_in as u32, n_out: *n_out as u32, relu, w: wb, b: bb, y, params });
        }
        let n_in = mlp.layers.first().map(|l| l.0).unwrap_or(0);
        let n_out = mlp.layers.last().map(|l| l.1).unwrap_or(0);
        Ok(ResidentMlp { lib, configs, n_in, n_out, layers })
    }

    /// Bind layer 0 to a fixed device observation buffer (makes subsequent [`Self::launch`]
    /// calls 100% allocation-free and host-transfer-free for CUDA graph capture).
    pub fn bind_input(&mut self, dev: &D, s: &D::Stream, obs: &D::Buffer) -> DriverResult<()> {
        let xptr = obs.device_ptr();
        if let Some(l0) = self.layers.first_mut() {
            if l0.params.as_ref().is_some_and(|(p, _)| *p == xptr) { return Ok(()); }
            let prm = MlpLayerParams { x: xptr, w: l0.w.device_ptr(), b: l0.b.device_ptr(), y: l0.y.device_ptr(), configs: self.configs, n_in: l0.n_in, n_out: l0.n_out, relu: l0.relu };
            l0.params = Some((xptr, upload_params(dev, s, &prm)?));
        }
        Ok(())
    }

    /// Enqueue the full forward pass using pre-bound input/layer buffers (zero alloc, zero host IO).
    pub fn launch(&self, dev: &D, s: &D::Stream) -> DriverResult<()> {
        for l in &self.layers {
            let Some((_, ref p)) = l.params else {
                return Err(EnqueueError::InvalidConfig("ResidentMlp: call bind_input before launch".into()).into());
            };
            dev.launch(s, self.lib, "ag_mlp_layer", &LaunchConfig::linear(self.configs as u64 * l.n_out as u64, 128), &p.buffer)?;
        }
        Ok(())
    }

    /// Bind (if needed) and launch; returns the device output buffer `[configs × n_out]` f32.
    pub fn forward_into(&mut self, dev: &D, s: &D::Stream, obs: &D::Buffer) -> DriverResult<&D::Buffer> {
        self.bind_input(dev, s, obs)?;
        self.launch(dev, s)?;
        Ok(self.output())
    }

    pub fn output(&self) -> &D::Buffer { &self.layers.last().expect("at least one layer").y }
    pub fn n_in(&self) -> usize { self.n_in }
    pub fn n_out(&self) -> usize { self.n_out }
    pub fn configs(&self) -> u32 { self.configs }
}

/// Device forward over `configs` observations; returns the output buffer.
pub fn forward_device<D: KernelExt>(dev: &D, s: &D::Stream, lib: LibraryId, mlp: &Mlp, obs: &D::Buffer, configs: u32) -> DriverResult<D::Buffer> {
    let mut rm = ResidentMlp::load(dev, s, lib, mlp, configs)?;
    rm.forward_into(dev, s, obs)?;
    Ok(rm.layers.pop().expect("at least one layer").y)
}

/// Parameter-delta policy head: output `k` adjusts parameter `params[k]` by `scale[k] * y`,
/// emitted as `knowledge.parameter` command records (the action space is the catalog
/// opcode allow-list; no fixed slot count — spec 07).
pub struct CuModel {
    manifest: CapabilityManifest,
    pub mlp: Mlp,
    pub params: Vec<String>,
    pub scale: Vec<f32>,
}

impl CuModel {
    pub fn new(mlp: Mlp, params: Vec<String>, scale: Vec<f32>) -> Self {
        let op = agcad_ir::Catalog::global().get("knowledge.parameter").map(|o| o.opcode).unwrap_or(0x0100);
        CuModel { manifest: CapabilityManifest { name: "cumodel-mlp".into(), opcodes: vec![op], precision: "f32".into(), max_context_tokens: None, device_resident: true }, mlp, params, scale }
    }

    /// Turn network outputs for one configuration into command records.
    pub fn commands(&self, y: &[f32], current: &[f64], configuration: u32, revision: u32) -> Vec<CommandRecord> {
        let op = self.manifest.opcodes[0];
        y.iter().enumerate().take(self.params.len()).map(|(k, v)| {
            let mut c: CommandRecord = Zeroable::zeroed();
            c.opcode = op;
            c.flags = flags::DIFFERENTIABLE_REQUEST;
            c.configuration = configuration;
            c.dst = k as u32;
            c.expected_revision = revision;
            c.imm = [current.get(k).copied().unwrap_or(0.0) + (*v as f64) * self.scale[k] as f64, 0.0];
            c
        }).collect()
    }
}

impl ModelAdaptor for CuModel {
    fn capabilities(&self) -> &CapabilityManifest { &self.manifest }
    fn decide(&mut self, _obs: &Observation) -> Result<Decision, (AgStatus, String)> {
        Ok(Decision::DeviceRollout { steps: 16 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn safetensors_roundtrip_and_kernel_parity() {
        let mut t = BTreeMap::new();
        t.insert("layer0.weight".into(), (vec![3, 2], vec![0.5, -1.0, 2.0, 0.25, -0.5, 1.5]));
        t.insert("layer0.bias".into(), (vec![3], vec![0.1, 0.0, -0.2]));
        t.insert("layer1.weight".into(), (vec![1, 3], vec![1.0, -1.0, 0.5]));
        t.insert("layer1.bias".into(), (vec![1], vec![0.0]));
        let mlp = Mlp::from_safetensors(&write_safetensors(&t)).unwrap();
        let x = [1.0f32, 2.0, -1.0, 0.5];
        let host = mlp.forward(&x, 2);
        let dev = agcad_driver_cpu::CpuDevice::new();
        let s = dev.stream(Priority::Normal).unwrap();
        let lib = dev.load_library(&LibrarySource { name: "ag_model", cuda_source: None, cpu_kernels: MODEL_CPU, options: vec![] }).unwrap();
        let ob = dev.alloc_bytes(&s, 16).unwrap();
        dev.upload_from_host_async(&s, bytemuck::cast_slice(&x), &ob, 0).unwrap();
        let y = forward_device(&*dev, &s, lib, &mlp, &ob, 2).unwrap();
        let got: Vec<f32> = bytemuck::pod_collect_to_vec(&y.as_slice()[..8]);
        assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), host.iter().map(|v| v.to_bits()).collect::<Vec<_>>());

        // ResidentMlp: bind once, then launch repeatedly with zero host transfers
        let mut rm = ResidentMlp::load(&*dev, &s, lib, &mlp, 2).unwrap();
        rm.bind_input(&*dev, &s, &ob).unwrap();
        dev.reset_transfer_stats();
        rm.launch(&*dev, &s).unwrap();
        assert_eq!(dev.transfer_stats().total(), 0, "capture-safe launch must do zero host transfers");
        let got2: Vec<f32> = bytemuck::pod_collect_to_vec(&rm.output().as_slice()[..8]);
        assert_eq!(got2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), host.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }
}
