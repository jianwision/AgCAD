use crate::{Result, mesh};
use std::{fs, path::Path};
const SURF_WORDS: usize = 4192;
const MV: usize = 4096;
const MT: usize = 4096;
const MESH_WORDS: usize = 8 + 6 * MV + 3 * MT;
const MESH_OFFSET: usize = 2 * SURF_WORDS;
const SAMPLE_OFFSET: usize = MESH_OFFSET + 7 * MESH_WORDS;
const AUX_OFFSET: usize = SAMPLE_OFFSET + 2 * 81 * 3;
pub const WORDS: usize = AUX_OFFSET + 256;

pub fn write(data: &[f64], out: &Path) -> Result<()> {
    if data.len() != WORDS || data.iter().any(|v| !v.is_finite()) {
        return Err("Invalid operation arena".into());
    }
    if data[AUX_OFFSET..AUX_OFFSET + 7].iter().any(|v| *v != 1.) || data[AUX_OFFSET + 7] != 0. {
        return Err("Operation rejection/resample checks failed".into());
    }
    let mut surfaces = Vec::new();
    for id in 0..2 {
        let s = &data[id * SURF_WORDS..(id + 1) * SURF_WORDS];
        let (nu, nv, pu, pv) = (s[0] as usize, s[1] as usize, s[2] as usize, s[3] as usize);
        if s[11] != 0. || nu > 32 || nv > 32 || nu < 2 || nv < 2 || pu > 5 || pv > 5 {
            return Err("Operation surface builder failed".into());
        }
        surfaces.push(format!("{{\"id\":{id},\"size\":[{nu},{nv}],\"degree\":[{pu},{pv}],\"u_knots\":{:?},\"v_knots\":{:?},\"poles_xyzw\":{:?}}}",&s[16..16+nu+pu+1],&s[56..56+nv+pv+1],&s[96..96+4*nu*nv]));
    }
    let names = [
        "bend",
        "twist",
        "taper",
        "sweep_path",
        "catmull_clark",
        "loop_subdivide",
        "resample_grid",
    ];
    let mut records = Vec::new();
    let mut gallery = Vec::new();
    for (id, name) in names.iter().enumerate() {
        let m = &data[MESH_OFFSET + id * MESH_WORDS..MESH_OFFSET + (id + 1) * MESH_WORDS];
        let (nv, nt) = (m[0] as usize, m[1] as usize);
        if m[2] != 0.
            || nv == 0
            || nt == 0
            || nv > MV
            || nt > MT
            || m[0] != nv as f64
            || m[1] != nt as f64
        {
            return Err(format!("Operation mesh {name} failed").into());
        }
        let positions = &m[8..8 + 3 * nv];
        let normals = &m[8 + 3 * MV..8 + 3 * MV + 3 * nv];
        let indices = &m[8 + 6 * MV..8 + 6 * MV + 3 * nt];
        let mut soup = Vec::new();
        for index in indices {
            let k = *index as usize;
            if k >= nv || *index != k as f64 {
                return Err("Operation index out of range".into());
            }
            soup.extend(positions[3 * k..3 * k + 3].iter().map(|v| *v as f32));
        }
        for n in normals.chunks_exact(3) {
            if (n.iter().map(|v| v * v).sum::<f64>() - 1.).abs() > 1e-10 {
                return Err("Invalid operation normal".into());
            }
        }
        let st = mesh::analyze(&soup)?;
        let (boundary, euler) = if id == 6 { (56, 1) } else { (0, 2) };
        if st.vertices != nv
            || st.triangles != nt
            || st.boundary_edges != boundary
            || st.euler != euler
            || st.nonmanifold_edges + st.orientation_errors + st.degenerate_triangles > 0
            || id != 6 && st.volume <= 0.
        {
            return Err(format!("Operation {name} topology failed: {st:?}").into());
        }
        mesh::write_stl(&out.join(format!("operation_{name}.stl")), &soup)?;
        for p in soup.chunks_exact(3) {
            gallery.extend([
                p[0] + 4. * (id % 4) as f32 - 6.,
                p[1] + 4. * (id / 4) as f32 - 2.,
                p[2],
            ]);
        }
        records.push(format!("{{\"name\":\"{name}\",\"vertices\":{nv},\"triangles\":{nt},\"positions\":{positions:?},\"normals\":{normals:?},\"indices\":{indices:?},\"boundary_edges\":{},\"euler\":{},\"volume_mesh_f32\":{}}}",st.boundary_edges,st.euler,st.volume));
    }
    let samples = &data[SAMPLE_OFFSET..AUX_OFFSET];
    for i in 0..8 {
        for j in 0..8 {
            for k in [
                9 * i + j,
                9 * (i + 1) + j,
                9 * (i + 1) + j + 1,
                9 * i + j,
                9 * (i + 1) + j + 1,
                9 * i + j + 1,
            ] {
                let p = &samples[3 * (81 + k)..3 * (81 + k) + 3];
                gallery.extend([p[0] as f32 + 6., p[1] as f32 + 2., p[2] as f32]);
            }
        }
    }
    fs::write(
        out.join("operations.json"),
        format!(
            "{{\"surfaces\":[{}],\"meshes\":[{}],\"surface_samples\":{samples:?},\"aux\":{:?}}}",
            surfaces.join(","),
            records.join(","),
            &data[AUX_OFFSET..]
        ),
    )?;
    mesh::write_stl(&out.join("operations_model.stl"), &gallery)?;
    mesh::write_viewer(&out.join("operations_viewer.html"), &gallery, 8.)?;
    let html=fs::read_to_string(out.join("operations_viewer.html"))?
        .replace("让代码成为形状","建模操作原语")
        .replace("沿不同角度检查表面和贯通孔","检查变形、路径扫掠、光顺细分和重采样")
        .replace("href=\"model.stl\"","href=\"operations_model.stl\"")
        .replace("</aside>","<details class=\"small\"><summary>排列顺序（逐行）</summary>bend · twist · taper · path sweep<br>Catmull-Clark · Loop · resample grid · chordal loft</details></aside>");
    fs::write(out.join("operations_viewer.html"), html)?;
    Ok(())
}
