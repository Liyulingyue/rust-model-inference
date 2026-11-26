use super::{Band, TrellisWeight};
use crate::core::tensor::{typed_tensor_bytes, GGMLType, MetaValue, MetaValueType, TensorSource};
use crate::ops::kernel::Weight;

fn scales(
    source: &dyn TensorSource,
    name: &str,
    n: usize,
    ty: GGMLType,
) -> Result<Vec<f32>, String> {
    let bytes = typed_tensor_bytes(source, name, &[n as u64], ty)?;
    let values: Vec<f32> = if ty == GGMLType::F16 {
        bytes
            .chunks_exact(2)
            .map(|b| crate::ops::f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect()
    } else {
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    if values.iter().any(|s| !s.is_finite()) {
        return Err(format!("{name}: nonfinite scale"));
    }
    Ok(values)
}

fn permutation(source: &dyn TensorSource, name: &str, n: usize) -> Result<Vec<u32>, String> {
    let bytes = typed_tensor_bytes(source, name, &[n as u64], GGMLType::I32)?;
    let mut seen = vec![false; n];
    bytes
        .chunks_exact(4)
        .map(|b| {
            let v = u32::from_le_bytes(b.try_into().unwrap());
            match seen.get_mut(v as usize) {
                Some(flag) if !*flag => {
                    *flag = true;
                    Ok(v)
                }
                _ => Err(format!("{name}: not a permutation of 0..{n}")),
            }
        })
        .collect()
}

fn block(name: &str, n: usize, spec: &[i32]) -> Result<usize, String> {
    match spec {
        [0, _, _] => Ok(1),
        [1, b, _]
            if *b >= 2 && (*b as usize).is_power_of_two() && n.is_multiple_of(*b as usize) =>
        {
            Ok(*b as usize)
        }
        _ => Err(format!(
            "{name}: unsupported NariQ transform {spec:?} for width {n}"
        )),
    }
}

fn load_band<'a>(
    source: &'a dyn TensorSource,
    base: &str,
    k: usize,
    embedding: bool,
) -> Result<Band<'a>, String> {
    let name = format!("{base}.weight");
    let info = source
        .tensor_info(&name)
        .ok_or_else(|| format!("Missing {name}"))?;
    info.ggml_type
        .trellis_bits()
        .ok_or_else(|| format!("{name}: expected TRL1..TRL6"))?;
    let &[cols, rows] = info.dims.as_slice() else {
        return Err(format!("{name}: expected a matrix"));
    };
    if cols != k as u64
        || k == 0
        || rows == 0
        || rows > u32::MAX as u64
        || k % 16 != 0
        || rows % 16 != 0
    {
        return Err(format!("{name}: invalid tile dimensions {:?}", info.dims));
    }
    let rows = rows as usize;
    let codes = typed_tensor_bytes(source, &name, &[cols, rows as u64], info.ggml_type)?;
    let desc_name = format!("trellis.desc.{base}");
    let Some(MetaValue::Array(MetaValueType::Int32, entries)) = source.metadata(&desc_name) else {
        return Err(format!("Missing INT32 descriptor {desc_name}"));
    };
    let desc: Vec<_> = entries
        .iter()
        .map(|v| match v {
            MetaValue::Int32(i) => Ok(*i),
            _ => Err(format!("{desc_name}: expected INT32")),
        })
        .collect::<Result<_, _>>()?;
    if desc.len() < 7 || !(0..=8).contains(&desc[6]) || desc.len() != 7 + desc[6] as usize {
        return Err(format!("{desc_name}: invalid part count"));
    }
    let out_block = block(&desc_name, rows, &desc[..3])?;
    let in_block = block(&desc_name, k, &desc[3..6])?;
    let optional_perm = |suffix, size| {
        let name = format!("{base}.trellis_{suffix}");
        source
            .tensor_info(&name)
            .map(|_| permutation(source, &name, size))
            .transpose()
    };
    let pin = optional_perm("pin", k)?;
    let pout = optional_perm("pout", rows)?;
    if embedding && (out_block != 1 || pin.is_some() || pout.is_some()) {
        return Err(format!(
            "{base}: embedding requires unrotated rows and no matrix gathers"
        ));
    }
    let scatter = match pout {
        Some(gather) => inverse(&gather),
        None => (0..rows as u32).collect(),
    };
    Ok(Band {
        codes,
        storage_type: info.ggml_type,
        rows,
        in_block,
        out_block,
        s16: scales(
            source,
            &format!("{base}.trellis_s16"),
            k / 16,
            GGMLType::F16,
        )?,
        pre: scales(source, &format!("{base}.trellis_pre"), k, GGMLType::F32)?,
        post: scales(source, &format!("{base}.trellis_post"), rows, GGMLType::F32)?,
        pin,
        scatter,
    })
}

fn inverse(gather: &[u32]) -> Vec<u32> {
    let mut inverse = vec![0; gather.len()];
    for (out, &stored) in gather.iter().enumerate() {
        inverse[stored as usize] = out as u32;
    }
    inverse
}

fn band_count_key(base: &str) -> Option<&'static str> {
    match base {
        "output" => Some("trellis.lm_head_bands"),
        "token_embd" => Some("trellis.embed_bands"),
        _ => None,
    }
}

pub(crate) fn is_coded(source: &dyn TensorSource, name: &str) -> bool {
    let Some(base) = name.strip_suffix(".weight") else {
        return false;
    };
    band_count_key(base).is_some_and(|key| source.metadata(key).is_some())
        || source
            .tensor_info(name)
            .is_some_and(|info| info.ggml_type.trellis_bits().is_some())
        || source.metadata(&format!("trellis.desc.{base}")).is_some()
}

/// The logical shape is checked against the trunk even when the physical tensor
/// is split across vocabulary bands of different bit widths.
pub(crate) fn load_weight<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    k: usize,
    n: usize,
) -> Result<Weight<'a>, String> {
    if !matches!(
        source.metadata("trellis.version"),
        Some(MetaValue::Uint32(1))
    ) {
        return Err("NariQ requires trellis.version = UINT32 1".into());
    }
    let base = name
        .strip_suffix(".weight")
        .ok_or_else(|| format!("{name}: expected a matrix weight"))?;
    let mut bands = Vec::new();
    let mut vocab_gather = None;
    if let Some(value) = band_count_key(base).and_then(|key| source.metadata(key)) {
        let MetaValue::Uint32(count) = value else {
            return Err(format!("{base}: expected UINT32 band count"));
        };
        if *count == 0 || *count as usize > n / 16 || source.tensor_info(name).is_some() {
            return Err(format!("{base}: invalid or ambiguous band layout"));
        }
        let gather = permutation(source, &format!("{base}.vocab_gather"), n)?;
        let vocab_scatter = inverse(&gather);
        let mut start = 0;
        for i in 0..*count {
            let mut band = load_band(
                source,
                &format!("{base}.band{i:02}"),
                k,
                base == "token_embd",
            )?;
            if band.rows > n - start {
                return Err(format!("{base}: too many vocabulary rows"));
            }
            for r in &mut band.scatter {
                *r = vocab_scatter[start + *r as usize];
            }
            start += band.rows;
            bands.push(band);
        }
        if start != n {
            return Err(format!("{base}: expected {n} vocabulary rows, got {start}"));
        }
        vocab_gather = Some(gather);
    } else {
        let band = load_band(source, base, k, base == "token_embd")?;
        if band.rows != n {
            return Err(format!("{name}: expected {n} rows, got {}", band.rows));
        }
        bands.push(band);
    }
    // Weight exposes one dispatch type. The kernel retains every band's actual width.
    let dtype = bands[0].storage_type;
    Ok(Weight {
        kernel: Box::new(TrellisWeight {
            bands,
            vocab_gather,
        }),
        n_in: k,
        n_out: n,
        ggml_type: dtype,
    })
}
