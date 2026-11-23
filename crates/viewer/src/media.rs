use std::{
    ffi::{CStr, CString, c_char, c_void},
    ptr::NonNull,
};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub xyz: [f32; 3],
    pub normal: [f32; 3],
    pub face: u32,
}
// Marker vertices share the mesh layout. The shader expands them in render pixels.
const MARKER: u32 = 1 << 31;
#[derive(Clone, Default)]
pub struct Highlight {
    face: Option<u32>,
    id: u64,
    marker: std::sync::Arc<[Vertex]>,
}
impl Highlight {
    pub fn combine(parts: impl Iterator<Item = Self>) -> Self {
        let mut markers = Vec::new();
        for p in parts {
            markers.extend_from_slice(&p.marker);
        }
        if markers.is_empty() {
            Self::default()
        } else {
            Self::marker(markers)
        }
    }
    pub fn face(index: u32) -> Self {
        Self {
            face: Some(index),
            ..Self::default()
        }
    }
    fn marker(vertices: Vec<Vertex>) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self {
            face: None,
            id: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            marker: vertices.into(),
        }
    }
    pub fn vertex(xyz: [f32; 3]) -> Self {
        let vertices = [
            [-9., 0.],
            [0., 9.],
            [9., 0.],
            [-9., 0.],
            [9., 0.],
            [0., -9.],
        ]
        .map(|[x, y]| Vertex {
            xyz,
            normal: [x, y, 0.],
            face: MARKER,
        });
        Self::marker(vertices.to_vec())
    }
    pub fn edge(points: impl Iterator<Item = [f32; 3]>) -> Self {
        let mut vertices = Vec::new();
        let mut prev = None;
        for next in points {
            if let Some(p) = prev {
                let v = |xyz, other, side| Vertex {
                    xyz,
                    normal: other,
                    face: MARKER + side,
                };
                let a = v(p, next, 1u32);
                let b = v(p, next, 2u32);
                let c = v(next, p, 2u32);
                let d = v(next, p, 1u32);
                vertices.extend([a, b, c, c, b, d]);
            }
            prev = Some(next);
        }
        Self::marker(vertices)
    }
}
#[repr(C)]
#[derive(Default, Clone)]
pub struct Frame {
    pub timestamp: u32,
    pub sequence: u32,
    pub scene: u64,
    pub rows: [[f32; 4]; 3],
    selected: u32,
    skybox: u32,
    padding: [u32; 2],
}
unsafe extern "C" {
    fn view_open_images(error: *mut *mut c_char) -> *mut c_void;
    fn view_image(
        p: *mut c_void,
        pose: *const f32,
        frame: *mut Frame,
        length: *mut usize,
    ) -> *mut c_void;
    fn view_open(
        offer: *const c_char,
        answer: *mut *mut c_char,
        error: *mut *mut c_char,
    ) -> *mut c_void;
    fn view_free_string(p: *mut c_char);
    fn view_scene(p: *mut c_void, id: u64, vertices: *const Vertex, count: usize);
    fn view_select(p: *mut c_void, face: u32, marker_id: u64, marker: *const Vertex, count: usize);
    fn view_palette(p: *mut c_void, values: *const [f32; 8], count: usize);
    fn view_skybox(p: *mut c_void, preset: u32);
    fn view_frame(p: *mut c_void, timestamp: u32, out: *mut Frame) -> bool;
    fn view_error(p: *mut c_void) -> *mut c_char;
    fn view_close(p: *mut c_void);
}
pub struct ImageRenderer {
    peer: Peer,
    previous: Option<std::sync::Arc<crate::document::Scene>>,
    generation: u64,
}
impl ImageRenderer {
    pub fn new() -> Result<Self, String> {
        let mut error = std::ptr::null_mut();
        let raw = unsafe { view_open_images(&mut error) };
        let p = NonNull::new(raw).ok_or_else(|| {
            take(error).unwrap_or_else(|| "GPU image initialization failed".into())
        })?;
        Ok(Self {
            peer: Peer(p),
            previous: None,
            generation: 0,
        })
    }
    pub fn render(
        &mut self,
        scene: std::sync::Arc<crate::document::Scene>,
        palette: &[[f32; 8]],
        highlight: &Highlight,
        skybox: u32,
        pose: [f32; 5],
    ) -> Result<(Vec<u8>, Frame), String> {
        if self
            .previous
            .as_ref()
            .is_none_or(|old| !std::sync::Arc::ptr_eq(old, &scene))
        {
            self.generation += 1;
            self.peer.scene(self.generation, &scene.vertices);
            self.previous = Some(scene.clone());
        }
        self.peer.palette(palette);
        self.peer.select(highlight);
        self.peer.skybox(skybox);
        let mut frame = Frame::default();
        let mut len = 0;
        let raw = unsafe { view_image(self.peer.0.as_ptr(), pose.as_ptr(), &mut frame, &mut len) };
        if raw.is_null() {
            return Err(self
                .peer
                .error()
                .unwrap_or_else(|| "GPU did not produce a preview image".into()));
        }
        let bytes = unsafe {
            let b = std::slice::from_raw_parts(raw.cast::<u8>(), len).to_vec();
            view_free_string(raw.cast());
            b
        };
        frame.scene = scene.id;
        Ok((bytes, frame))
    }
}
fn take(p: *mut c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    unsafe {
        let s = CStr::from_ptr(p).to_string_lossy().into_owned();
        view_free_string(p);
        Some(s)
    }
}
pub struct Peer(NonNull<c_void>);
// The owner serializes control calls; streaming callbacks use the native peer's mutex.
unsafe impl Send for Peer {}
impl Peer {
    pub fn open(sdp: &str) -> Result<(Self, String), String> {
        let sdp = CString::new(sdp).map_err(|_| "Invalid SDP")?;
        let (mut answer, mut error) = (std::ptr::null_mut(), std::ptr::null_mut());
        let p = unsafe { view_open(sdp.as_ptr(), &mut answer, &mut error) };
        let handle = NonNull::new(p)
            .ok_or_else(|| take(error).unwrap_or_else(|| "Media startup failed".into()))?;
        Ok((Self(handle), take(answer).ok_or("Missing SDP answer")?))
    }
    pub fn scene(&mut self, id: u64, mesh: &[Vertex]) {
        unsafe { view_scene(self.0.as_ptr(), id, mesh.as_ptr(), mesh.len()) }
    }
    pub fn palette(&mut self, colors: &[[f32; 8]]) {
        unsafe { view_palette(self.0.as_ptr(), colors.as_ptr(), colors.len()) }
    }
    pub fn skybox(&mut self, preset: u32) {
        unsafe { view_skybox(self.0.as_ptr(), preset) }
    }
    pub fn select(&mut self, h: &Highlight) {
        unsafe {
            view_select(
                self.0.as_ptr(),
                h.face.unwrap_or(u32::MAX),
                h.id,
                h.marker.as_ptr(),
                h.marker.len(),
            )
        }
    }
    pub fn frame(&self, timestamp: u32) -> Result<Frame, String> {
        let mut f = Frame::default();
        if unsafe { view_frame(self.0.as_ptr(), timestamp, &mut f) } {
            Ok(f)
        } else {
            Err("Displayed frame expired. Click again.".into())
        }
    }
    pub fn error(&self) -> Option<String> {
        take(unsafe { view_error(self.0.as_ptr()) })
    }
}
impl Drop for Peer {
    fn drop(&mut self) {
        unsafe { view_close(self.0.as_ptr()) }
    }
}
