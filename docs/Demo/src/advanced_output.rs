use crate::{Result, mesh};
use std::{fs, path::Path};
const N: usize = 9;
const S: usize = 4192;
const SAMPLE: usize = N * S;
const AUX: usize = SAMPLE + N * 81 * 9;
pub const WORDS: usize = AUX + 128;
pub fn write(data: &[f64], out: &Path) -> Result<()> {
    if data.len() != WORDS || data.iter().any(|v| !v.is_finite()) || data[AUX..AUX + 2] != [1., 1.]
    {
        return Err("Advanced surface data/check failure".into());
    }
    let mut records = Vec::new();
    let mut soup = Vec::new();
    for id in 0..N {
        let s = &data[id * S..(id + 1) * S];
        let (nu, nv, pu, pv) = (s[0] as usize, s[1] as usize, s[2] as usize, s[3] as usize);
        if s[11] != 0. || !(2..=32).contains(&nu) || !(2..=32).contains(&nv) || pu > 5 || pv > 5 {
            return Err("Invalid advanced control net".into());
        }
        records.push(format!("{{\"id\":{id},\"size\":[{nu},{nv}],\"degree\":[{pu},{pv}],\"u_knots\":{:?},\"v_knots\":{:?},\"poles_xyzw\":{:?}}}",&s[16..16+nu+pu+1],&s[56..56+nv+pv+1],&s[96..96+4*nu*nv]));
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
                    let p = &data[SAMPLE + 9 * (id * 81 + k)..SAMPLE + 9 * (id * 81 + k) + 3];
                    soup.extend([
                        p[0] as f32 + 3. * (id % 3) as f32 - 3.,
                        p[1] as f32 + 3. * (id / 3) as f32 - 3.,
                        p[2] as f32,
                    ]);
                }
            }
        }
    }
    fs::write(
        out.join("advanced_surfaces.json"),
        format!(
            "{{\"surfaces\":[{}],\"jets\":{:?},\"aux\":{:?}}}",
            records.join(","),
            &data[SAMPLE..AUX],
            &data[AUX..]
        ),
    )?;
    mesh::write_stl(&out.join("advanced_surfaces.stl"), &soup)?;
    mesh::write_viewer(&out.join("advanced_viewer.html"), &soup, 6.)?;
    let html = fs::read_to_string(out.join("advanced_viewer.html"))?
        .replace("让代码成为形状", "高级曲面操作")
        .replace("href=\"model.stl\"", "href=\"advanced_surfaces.stl\"")
        .replace(
            "沿不同角度检查表面和贯通孔",
            "导向 / 缩放 / 变截面 / 闭环扫掠，G1/G2 匹配及联合平顺",
        );
    fs::write(out.join("advanced_viewer.html"), html)?;
    Ok(())
}
