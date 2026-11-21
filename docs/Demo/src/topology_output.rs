use crate::{Result, mesh};
use std::{fs, path::Path};
const MV: usize = 4096;
const MT: usize = 4096;
const MESH_WORDS: usize = 8 + 6 * MV + 3 * MT;
const AUX: usize = 5 * MESH_WORDS;
pub const WORDS: usize = AUX + 128;
pub fn write(data: &[f64], out: &Path) -> Result<()> {
    if data.len() != WORDS
        || data.iter().any(|v| !v.is_finite())
        || data[AUX] != 1.
        || data[AUX + 1] != 1.
    {
        return Err("Invalid topology arena".into());
    }
    let names = [
        "delete_faces",
        "fill_hole",
        "revolve_profile",
        "smooth_laplacian",
        "smooth_taubin",
    ];
    let mut records = Vec::new();
    let mut gallery = Vec::new();
    for (id, name) in names.iter().enumerate() {
        let m = &data[id * MESH_WORDS..(id + 1) * MESH_WORDS];
        let (nv, nt) = (m[0] as usize, m[1] as usize);
        if m[2] != 0.
            || nv == 0
            || nt == 0
            || nv > MV
            || nt > MT
            || m[0] != nv as f64
            || m[1] != nt as f64
        {
            return Err(format!("Topology operator {name} failed").into());
        }
        let p = &m[8..8 + 3 * nv];
        let n = &m[8 + 3 * MV..8 + 3 * MV + 3 * nv];
        let ids = &m[8 + 6 * MV..8 + 6 * MV + 3 * nt];
        let mut soup = Vec::new();
        for index in ids {
            let k = *index as usize;
            if k >= nv || *index != k as f64 {
                return Err("Invalid topology mesh index".into());
            }
            soup.extend(p[3 * k..3 * k + 3].iter().map(|v| *v as f32));
        }
        for normal in n.chunks_exact(3) {
            if (normal.iter().map(|v| v * v).sum::<f64>() - 1.).abs() > 1e-10 {
                return Err("Invalid topology normal".into());
            }
        }
        let st = mesh::analyze(&soup)?;
        let (boundary, euler) = match id {
            0 => (4, 1),
            2 => (0, 0),
            _ => (0, 2),
        };
        if st.vertices != nv
            || st.triangles != nt
            || st.boundary_edges != boundary
            || st.euler != euler
            || st.nonmanifold_edges + st.orientation_errors + st.degenerate_triangles > 0
            || id != 0 && st.volume <= 0.
        {
            return Err(format!("Topology {name} failed: {st:?}").into());
        }
        mesh::write_stl(&out.join(format!("{name}.stl")), &soup)?;
        for p in soup.chunks_exact(3) {
            gallery.extend([
                p[0] + 4. * (id % 3) as f32 - 4.,
                p[1] + 4. * (id / 3) as f32 - 2.,
                p[2],
            ]);
        }
        records.push(format!("{{\"name\":\"{name}\",\"vertices\":{nv},\"triangles\":{nt},\"positions\":{p:?},\"normals\":{n:?},\"indices\":{ids:?},\"boundary_edges\":{},\"euler\":{},\"volume_mesh_f32\":{}}}",st.boundary_edges,st.euler,st.volume));
    }
    fs::write(
        out.join("topology_operations.json"),
        format!(
            "{{\"meshes\":[{}],\"aux\":{:?}}}",
            records.join(","),
            &data[AUX..]
        ),
    )?;
    mesh::write_stl(&out.join("topology_model.stl"), &gallery)?;
    mesh::write_viewer(&out.join("topology_viewer.html"), &gallery, 7.)?;
    let html=fs::read_to_string(out.join("topology_viewer.html"))?.replace("让代码成为形状","拓扑与平滑操作").replace("沿不同角度检查表面和贯通孔","检查删面、填洞、旋转和网格平滑").replace("href=\"model.stl\"","href=\"topology_model.stl\"").replace("</aside>","<details class=\"small\"><summary>排列顺序（逐行）</summary>delete · fill hole · revolve<br>Laplacian · Taubin</details></aside>");
    fs::write(out.join("topology_viewer.html"), html)?;
    Ok(())
}
