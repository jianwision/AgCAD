//! Deterministic store-only ZIP writer/reader (used by `.agcad` and 3MF; spec 16 § 3,
//! AG-16-002). Entries are written in the given order with a fixed DOS timestamp and no
//! compression, so identical inputs give identical bytes.

const DOS_TIME: u16 = 0; // 00:00:00
const DOS_DATE: u16 = (46 << 9) | (1 << 5) | 1; // 2026-01-01

pub fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let t = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for i in 0..256u32 {
            let mut c = i;
            for _ in 0..8 { c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 }; }
            t[i as usize] = c;
        }
        t
    });
    let mut c = 0xFFFF_FFFFu32;
    for &b in data { c = t[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8); }
    c ^ 0xFFFF_FFFF
}

pub fn write_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    let le16 = |v: &mut Vec<u8>, x: u16| v.extend_from_slice(&x.to_le_bytes());
    let le32 = |v: &mut Vec<u8>, x: u32| v.extend_from_slice(&x.to_le_bytes());
    for (name, data) in entries {
        let off = out.len() as u32;
        let crc = crc32(data);
        le32(&mut out, 0x0403_4b50);
        le16(&mut out, 20); le16(&mut out, 0); le16(&mut out, 0); le16(&mut out, DOS_TIME); le16(&mut out, DOS_DATE);
        le32(&mut out, crc); le32(&mut out, data.len() as u32); le32(&mut out, data.len() as u32);
        le16(&mut out, name.len() as u16); le16(&mut out, 0);
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);
        le32(&mut central, 0x0201_4b50);
        le16(&mut central, 20); le16(&mut central, 20); le16(&mut central, 0); le16(&mut central, 0); le16(&mut central, DOS_TIME); le16(&mut central, DOS_DATE);
        le32(&mut central, crc); le32(&mut central, data.len() as u32); le32(&mut central, data.len() as u32);
        le16(&mut central, name.len() as u16); le16(&mut central, 0); le16(&mut central, 0); le16(&mut central, 0); le16(&mut central, 0);
        le32(&mut central, 0); le32(&mut central, off);
        central.extend_from_slice(name.as_bytes());
    }
    let cd_off = out.len() as u32;
    out.extend_from_slice(&central);
    le32(&mut out, 0x0605_4b50);
    le16(&mut out, 0); le16(&mut out, 0); le16(&mut out, entries.len() as u16); le16(&mut out, entries.len() as u16);
    le32(&mut out, central.len() as u32); le32(&mut out, cd_off); le16(&mut out, 0);
    out
}

/// Read a store-only ZIP (rejects compressed entries and CRC mismatches).
pub fn read_zip(data: &[u8]) -> Result<Vec<(String, Vec<u8>)>, String> {
    let rd16 = |o: usize| -> Result<u16, String> { data.get(o..o + 2).map(|b| u16::from_le_bytes([b[0], b[1]])).ok_or_else(|| "truncated".to_string()) };
    let rd32 = |o: usize| -> Result<u32, String> { data.get(o..o + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).ok_or_else(|| "truncated".to_string()) };
    let mut out = Vec::new();
    let mut o = 0usize;
    while o + 4 <= data.len() && rd32(o)? == 0x0403_4b50 {
        let method = rd16(o + 8)?;
        if method != 0 { return Err("compressed entries are not supported".into()); }
        let crc = rd32(o + 14)?;
        let size = rd32(o + 18)? as usize;
        let nlen = rd16(o + 26)? as usize;
        let xlen = rd16(o + 28)? as usize;
        let name = String::from_utf8(data.get(o + 30..o + 30 + nlen).ok_or("truncated name")?.to_vec()).map_err(|_| "bad name")?;
        let start = o + 30 + nlen + xlen;
        let body = data.get(start..start + size).ok_or("truncated body")?.to_vec();
        if crc32(&body) != crc { return Err(format!("CRC mismatch in {name}")); }
        out.push((name, body));
        o = start + size;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_and_known_crc() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        let z = write_zip(&[("a.txt", b"hello"), ("dir/b.bin", &[1, 2, 3])]);
        let r = read_zip(&z).unwrap();
        assert_eq!(r[0].0, "a.txt");
        assert_eq!(r[1].1, vec![1, 2, 3]);
        assert_eq!(z, write_zip(&[("a.txt", b"hello"), ("dir/b.bin", &[1, 2, 3])]));
    }
}
