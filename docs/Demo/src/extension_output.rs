use crate::{Result, mesh};
use std::{fs, path::Path};

const CURVES: usize = 9;
const CURVE_WORDS: usize = 192;
const SAMPLES: usize = 513;
const OP_WORDS: usize = 256;
const SOLIDS: usize = 10;
const SOLID_WORDS: usize = 64;
const MV: usize = 4096;
const MT: usize = 4096;
const MESH_WORDS: usize = 8 + 6 * MV + 3 * MT;
const SAMPLE_OFFSET: usize = CURVES * CURVE_WORDS;
const OP_OFFSET: usize = SAMPLE_OFFSET + CURVES * SAMPLES * 9;
const DEF_OFFSET: usize = OP_OFFSET + OP_WORDS;
const MESH_OFFSET: usize = DEF_OFFSET + SOLIDS * SOLID_WORDS;
pub const WORDS: usize = MESH_OFFSET + SOLIDS * MESH_WORDS;

pub fn write(data: &[f64], out: &Path) -> Result<()> {
    if data.len() != WORDS || data.iter().any(|x| !x.is_finite()) {
        return Err("Invalid extension arena".into());
    }
    let records: Vec<_> = data[..SAMPLE_OFFSET]
        .chunks_exact(CURVE_WORDS)
        .map(|s| format!("{s:?}"))
        .collect();
    if data[..SAMPLE_OFFSET]
        .chunks_exact(CURVE_WORDS)
        .any(|s| s[20] != 0.)
        || data[OP_OFFSET + 13..OP_OFFSET + 21]
            .iter()
            .any(|v| *v != 1.)
    {
        return Err("Extension curve builder/status checks failed".into());
    }
    fs::write(
        out.join("extension_curves.json"),
        format!(
            "{{\"records\":[{}],\"samples\":{:?},\"ops\":{:?}}}",
            records.join(","),
            &data[SAMPLE_OFFSET..OP_OFFSET],
            &data[OP_OFFSET..DEF_OFFSET]
        ),
    )?;
    let names = [
        "solid_sphere",
        "solid_ellipsoid",
        "solid_cylinder",
        "solid_cone",
        "solid_frustum",
        "solid_torus",
        "solid_capsule",
        "solid_prism",
        "solid_pyramid",
        "solid_wedge",
    ];
    let mut solids = Vec::new();
    let mut gallery = Vec::new();
    for (id, name) in names.iter().enumerate() {
        let spec = &data[DEF_OFFSET + id * SOLID_WORDS..DEF_OFFSET + (id + 1) * SOLID_WORDS];
        let m = &data[MESH_OFFSET + id * MESH_WORDS..MESH_OFFSET + (id + 1) * MESH_WORDS];
        if spec[20] != 0. || m[2] != 0. {
            return Err(format!("Solid builder {name} rejected input").into());
        }
        let (nv, nt) = (m[0] as usize, m[1] as usize);
        if nv == 0 || nt == 0 || nv > MV || nt > MT || m[0] != nv as f64 || m[1] != nt as f64 {
            return Err("Invalid solid mesh counts".into());
        }
        let p = &m[8..8 + 3 * nv];
        let n = &m[8 + 3 * MV..8 + 3 * MV + 3 * nv];
        let ids = &m[8 + 6 * MV..8 + 6 * MV + 3 * nt];
        let mut soup = Vec::with_capacity(nt * 9);
        for index in ids {
            let k = *index as usize;
            if k >= nv || *index != k as f64 {
                return Err("Solid mesh index out of range".into());
            }
            soup.extend(p[3 * k..3 * k + 3].iter().map(|v| *v as f32));
        }
        for normal in n.chunks_exact(3) {
            if (normal.iter().map(|v| v * v).sum::<f64>() - 1.).abs() > 1e-10 {
                return Err("Invalid solid mesh normal".into());
            }
        }
        let stats = mesh::analyze(&soup)?;
        let expected_euler = if id == 5 { 0 } else { 2 };
        if stats.vertices != nv
            || stats.triangles != nt
            || stats.euler != expected_euler
            || stats.boundary_edges
                + stats.nonmanifold_edges
                + stats.orientation_errors
                + stats.degenerate_triangles
                > 0
            || stats.volume <= 0.
        {
            return Err(format!("Solid mesh {name} topology failed: {stats:?}").into());
        }
        mesh::write_stl(&out.join(format!("{name}.stl")), &soup)?;
        for p in soup.chunks_exact(3) {
            gallery.extend([
                p[0] + 4.5 * (id % 5) as f32 - 9.,
                p[1] + 4.5 * (id / 5) as f32 - 2.25,
                p[2] - 1.,
            ]);
        }
        solids.push(format!("{{\"id\":{id},\"name\":\"{name}\",\"spec\":{spec:?},\"vertices\":{nv},\"triangles\":{nt},\"positions\":{p:?},\"normals\":{n:?},\"indices\":{ids:?},\"euler\":{},\"boundary_edges\":{},\"volume_mesh_f32\":{}}}",stats.euler,stats.boundary_edges,stats.volume));
    }
    fs::write(
        out.join("extension_solids.json"),
        format!("[{}]", solids.join(",")),
    )?;
    mesh::write_stl(&out.join("solid_model.stl"), &gallery)?;
    mesh::write_viewer(&out.join("solid_viewer.html"), &gallery, 9.)?;
    let html = fs::read_to_string(out.join("solid_viewer.html"))?
        .replace("让代码成为形状", "基本体原语测试集")
        .replace("沿不同角度检查表面和贯通孔", "检查解析基本体、胶囊、棱柱、棱锥和楔体")
        .replace("href=\"model.stl\"", "href=\"solid_model.stl\"")
        .replace("</aside>", "<details class=\"small\"><summary>排列顺序（逐行）</summary>sphere · ellipsoid · cylinder · cone · frustum<br>torus · capsule · prism · pyramid · wedge</details></aside>");
    fs::write(out.join("solid_viewer.html"), html)?;

    let mut lines = Vec::new();
    for id in 0..CURVES {
        for i in 0..SAMPLES - 1 {
            for k in [i, i + 1] {
                let p = &data[SAMPLE_OFFSET + 9 * (id * SAMPLES + k)..][..3];
                lines.extend([
                    p[0] as f32 + 5. * (id % 3) as f32 - 5.,
                    p[1] as f32 + 5. * (id / 3) as f32 - 5.,
                    p[2] as f32 - 0.5,
                ]);
            }
        }
    }
    mesh::write_viewer(&out.join("curve_viewer.html"), &lines, 8.)?;
    let html = fs::read_to_string(out.join("curve_viewer.html"))?
        .replace("让代码成为形状", "曲线原语扩展集")
        .replace("三角形", "线段")
        .replace("三维网格", "参数曲线")
        .replace("沿不同角度检查表面和贯通孔", "检查插值、圆锥曲线、螺旋线、反向与偏移")
        .replace("gl.TRIANGLES", "gl.LINES")
        .replace("gl_FragColor=vec4(sqrt(color),1.);", "gl_FragColor=vec4(.3,.85,1.,1.);")
        .replace(&format!("triangleCount={}", lines.len() / 9), &format!("triangleCount={}", lines.len() / 6))
        .replace("href=\"model.stl\" download>下载 STL", "href=\"extended_curves.step\" download>下载 STEP")
        .replace("</aside>", "<details class=\"small\"><summary>排列顺序（逐行）</summary>Hermite · parabola · hyperbola<br>helix · spiral · 3-point arc<br>interpolate · reverse · offset</details></aside>");
    fs::write(out.join("curve_viewer.html"), html)?;
    Ok(())
}
