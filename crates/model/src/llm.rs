//! `Llm` adaptor: POSTs the observation to a localhost model bridge (HTTP/1.1 over TcpStream,
//! no external client crates) and expects an AgIR JSON program back. Remote providers are
//! reached through a local bridge process, keeping credentials out of AgCAD (spec 17).

use crate::adaptor::*;
use agcad_driver::AgStatus;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

pub struct Llm {
    manifest: CapabilityManifest,
    pub addr: String,
    pub path: String,
}

impl Llm {
    /// Send a structured observation to the configured workstation bridge.
    pub fn observe(&self, obs: &Observation) -> Result<serde_json::Value, String> {
        let payload = serde_json::to_string(obs).map_err(|e| e.to_string())?;
        match serde_json::from_str(&self.post(&payload)?) {
            Err(e) if e.is_eof() => serde_json::from_str(&self.post(&payload)?).map_err(|e| format!("Incomplete or invalid bridge response: {e}")),
            result => result.map_err(|e| e.to_string()),
        }
    }
    pub fn new(addr: &str, path: &str, name: &str) -> Self {
        Llm { manifest: CapabilityManifest { name: name.into(), precision: "f64".into(), ..Default::default() }, addr: addr.into(), path: path.into() }
    }

    fn post(&self, body: &str) -> Result<String, String> {
        if !(self.addr.starts_with("127.0.0.1:") || self.addr.starts_with("localhost:") || self.addr.starts_with("[::1]:")) {
            return Err("Llm bridge must be on localhost (spec 17)".into());
        }
        let mut s = TcpStream::connect(&self.addr).map_err(|e| e.to_string())?;
        s.set_read_timeout(Some(Duration::from_secs(600))).ok();
        write!(s, "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", self.path, self.addr, body.len(), body).map_err(|e| e.to_string())?;
        let mut resp = String::new();
        s.read_to_string(&mut resp).map_err(|e| e.to_string())?;
        let (head, body) = resp.split_once("\r\n\r\n").ok_or("malformed HTTP response")?;
        if !head.starts_with("HTTP/1.1 200") && !head.starts_with("HTTP/1.0 200") { return Err(format!("bridge returned: {}", head.lines().next().unwrap_or(""))); }
        Ok(body.to_string())
    }
}

impl ModelAdaptor for Llm {
    fn capabilities(&self) -> &CapabilityManifest { &self.manifest }
    fn decide(&mut self, obs: &Observation) -> Result<Decision, (AgStatus, String)> {
        let body = serde_json::to_string(obs).map_err(|e| (AgStatus::InvalidParameter, e.to_string()))?;
        let text = self.post(&body).map_err(|e| (AgStatus::UnsupportedOp, format!("LLM bridge: {e}")))?;
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| (AgStatus::InvalidParameter, e.to_string()))?;
        if v.get("finish").and_then(|x| x.as_bool()) == Some(true) { return Ok(Decision::Finish); }
        let prog = v.get("program").unwrap_or(&v);
        let p = agcad_ir::json::decode_program(&prog.to_string(), agcad_ir::Catalog::global()).map_err(|e| (AgStatus::CompileFailed, e.to_string()))?;
        Ok(Decision::Program(p))
    }
}
