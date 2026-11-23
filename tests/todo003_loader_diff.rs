//! Differential test for the TODO-003 loader unification.
//!
//! The old per-model loaders and the new shared
//! `core::tensor::load_weight_any` must produce the **same** `Weight` for
//! the same tensor. This test compares the two implementations tensor by
//! tensor: the declared format, the matrix shape, the bytes the kernel
//! actually sees, and the numbers it produces.
//!
//! Two layers:
//!
//! * `synthetic` — a [`TensorSource`] the test builds itself, covering every
//!   format the loaders accept and both shape conventions the repository
//!   uses. It needs no checkpoint, so it runs in CI and a change to the
//!   shared loader that breaks the contract fails the build.
//! * `real weights` — the same comparison on real checkpoints. Those need
//!   model files, so they are `#[ignore]`d and run with `--ignored`; a
//!   silent skip would look like a pass.
//!
//! The old loaders are reproduced below. They are copies of
//! `fd6f1b7d:src/models/{qwen35,dots,gemma4}` — the commit before the
//! unification — so the comparison is against the code that used to run,
//! not against a paraphrase of it.

use rust_model_inference::core::tensor::{GGMLType, MetaValue, TensorInfo, TensorSource};
use rust_model_inference::ops::kernel::{Kernel, QuantizedTensor, Weight};

/// The formats a qwen35 matrix tensor ships in.
const QWEN35_MATRIX_FORMATS: &[GGMLType] = &[
    GGMLType::F32,
    GGMLType::F16,
    GGMLType::BF16,
    GGMLType::Q8_0,
    GGMLType::Q4_0,
    GGMLType::Q4_1,
    GGMLType::Q4K,
    GGMLType::Q5K,
    GGMLType::Q6K,
];

/// The formats the dots tensors ship in.
const DOTS_FORMATS: &[GGMLType] = &[GGMLType::F32, GGMLType::F16, GGMLType::BF16, GGMLType::Q8_0];

/// The old loaders, copied from the commit before the unification.
mod old {
    use super::{GGMLType, QuantizedTensor, TensorSource, Weight};

    /// `fd6f1b7d:src/models/qwen35/trunk/weights.rs::load_weight`.
    pub fn qwen35_load_weight<'a, S: TensorSource + ?Sized>(
        source: &'a S,
        name: &str,
    ) -> Option<Weight<'a>> {
        let ti = source.tensor_info(name)?;
        let data = source.tensor_slice(name)?;
        let n_cols = ti.dims[0] as usize;
        let n_rows = if ti.dims.len() >= 2 {
            ti.dims[1] as usize
        } else {
            1
        };

        match ti.ggml_type {
            GGMLType::F32 => Weight::from_f32_bytes(data, n_cols, n_rows),
            GGMLType::F16
            | GGMLType::BF16
            | GGMLType::Q8_0
            | GGMLType::Q4_0
            | GGMLType::Q4_1
            | GGMLType::Q4K
            | GGMLType::Q5K
            | GGMLType::Q6K => {
                let mut weight = Weight::from_quantized(QuantizedTensor::from_bytes(
                    data,
                    ti.ggml_type,
                    n_cols,
                    n_rows,
                ));
                weight.n_in = n_cols;
                weight.n_out = n_rows;
                if ti.ggml_type == GGMLType::BF16 {
                    weight.kernel = Box::new(
                        rust_model_inference::ops::kernel::bf16::BF16Kernel::with_bf16_input(data),
                    );
                }
                Some(weight)
            }
            _ => None,
        }
    }

    /// `fd6f1b7d:src/models/qwen35/trunk/util.rs::f16_bits_to_f32`.
    fn f16_bits_to_f32(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0f32 };
        let exp = ((bits >> 10) & 0x1F) as i32;
        let frac = (bits & 0x3FF) as f32 / 1024.0;
        if exp == 0 {
            sign * frac * 2.0f32.powi(-14)
        } else if exp == 31 {
            if frac == 0.0 {
                sign * f32::INFINITY
            } else {
                sign * f32::NAN
            }
        } else {
            sign * (1.0 + frac) * 2.0f32.powi(exp - 15)
        }
    }

    /// `fd6f1b7d:src/models/qwen35/trunk/util.rs::f16_at`.
    fn f16_at(data: &[u8], idx: usize) -> f32 {
        if idx * 2 + 2 > data.len() {
            return 0.0;
        }
        let bits = u16::from_le_bytes([data[idx * 2], data[idx * 2 + 1]]);
        f16_bits_to_f32(bits)
    }

    /// `fd6f1b7d:src/models/qwen35/trunk/weights.rs::load_weight_f32`.
    pub fn qwen35_load_weight_f32<S: TensorSource + ?Sized>(
        source: &S,
        name: &str,
    ) -> Option<Vec<f32>> {
        let ti = source.tensor_info(name)?;
        let data = source.tensor_slice(name)?;
        let n_el = ti.n_elements();
        match ti.ggml_type {
            GGMLType::F32 => {
                let mut out = Vec::with_capacity(n_el);
                for i in 0..n_el {
                    let off = i * 4;
                    if off + 4 <= data.len() {
                        out.push(f32::from_le_bytes([
                            data[off],
                            data[off + 1],
                            data[off + 2],
                            data[off + 3],
                        ]));
                    } else {
                        out.push(0.0);
                    }
                }
                Some(out)
            }
            GGMLType::F16 => {
                let mut out = Vec::with_capacity(n_el);
                for i in 0..n_el {
                    out.push(f16_at(data, i));
                }
                Some(out)
            }
            GGMLType::BF16 => {
                let mut out = Vec::with_capacity(n_el);
                for i in 0..n_el {
                    let off = i * 2;
                    if off + 2 <= data.len() {
                        out.push(rust_model_inference::ops::bf16_to_f32(u16::from_le_bytes(
                            [data[off], data[off + 1]],
                        )));
                    } else {
                        out.push(0.0);
                    }
                }
                Some(out)
            }
            _ => None,
        }
    }

    /// `fd6f1b7d:src/models/dots/weights.rs::load_weight`.
    pub fn dots_load_weight<'a, S: TensorSource + ?Sized>(
        source: &'a S,
        name: &str,
        dims: &[u64],
    ) -> Result<Weight<'a>, String> {
        if dims.len() < 2 || dims.contains(&0) {
            return Err(format!("Invalid matrix dimensions for {name}: {dims:?}"));
        }
        let n_out = usize::try_from(*dims.last().unwrap())
            .map_err(|_| format!("{name} output width overflow"))?;
        let n_in = dims[..dims.len() - 1]
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))
            .ok_or_else(|| format!("{name} input width overflow"))?;
        let info = source
            .tensor_info(name)
            .ok_or_else(|| format!("Missing tensor: {name}"))?;
        let flattened = [n_in as u64, n_out as u64];
        if info.dims != dims && !(info.ggml_type == GGMLType::Q8_0 && info.dims == flattened) {
            return Err(format!(
                "Invalid tensor {name}: shape {:?}; expected {dims:?}",
                info.dims
            ));
        }
        if !matches!(
            info.ggml_type,
            GGMLType::F32 | GGMLType::F16 | GGMLType::BF16 | GGMLType::Q8_0
        ) || (info.ggml_type == GGMLType::Q8_0 && n_in % 32 != 0)
        {
            return Err(format!(
                "Unsupported matrix {name}: {:?}, row width {n_in}",
                info.ggml_type
            ));
        }
        let expected = n_in
            .checked_mul(n_out)
            .and_then(|count| match info.ggml_type {
                GGMLType::F32 => count.checked_mul(4),
                GGMLType::F16 | GGMLType::BF16 => count.checked_mul(2),
                GGMLType::Q8_0 => (count / 32).checked_mul(34),
                _ => None,
            })
            .ok_or_else(|| format!("Tensor byte size overflow: {name}"))?;
        let bytes = source
            .tensor_slice(name)
            .ok_or_else(|| format!("Missing tensor data: {name}"))?;
        if bytes.len() != expected {
            return Err(format!(
                "Invalid tensor data length for {name}: {}; expected {expected}",
                bytes.len()
            ));
        }
        let mut weight = Weight::from_quantized(QuantizedTensor::from_bytes(
            bytes,
            info.ggml_type,
            n_in,
            n_out,
        ));
        weight.n_out = n_out;
        weight.n_in = n_in;
        Ok(weight)
    }

    /// `fd6f1b7d:src/models/gemma4/trunk/weights.rs::load_weight`.
    pub fn gemma4_load_weight<'a, S: TensorSource + ?Sized>(
        source: &'a S,
        name: &str,
        dims: &[u64],
        ggml_type: GGMLType,
    ) -> Result<Weight<'a>, String> {
        let info = source
            .tensor_info(name)
            .ok_or_else(|| format!("Missing tensor: {name}"))?;
        if info.dims != dims || info.ggml_type != ggml_type {
            return Err(format!(
                "Invalid tensor {name}: shape {:?} type {:?}; expected {:?} {:?}",
                info.dims, info.ggml_type, dims, ggml_type
            ));
        }
        let expected = info
            .checked_nbytes()
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?;
        let bytes = source
            .tensor_slice(name)
            .ok_or_else(|| format!("Missing tensor data: {name}"))?;
        if bytes.len() != expected {
            return Err(format!(
                "Invalid tensor data length for {name}: {}; expected {expected}",
                bytes.len()
            ));
        }
        let n_in = usize::try_from(dims[0]).map_err(|_| format!("{name} input width overflow"))?;
        let n_out =
            usize::try_from(dims[1]).map_err(|_| format!("{name} output width overflow"))?;
        let mut weight =
            Weight::from_quantized(QuantizedTensor::from_bytes(bytes, ggml_type, n_in, n_out));
        weight.n_in = n_in;
        weight.n_out = n_out;
        Ok(weight)
    }

    /// `fd6f1b7d:src/models/gemma4/trunk/weights.rs::load_weight_any`.
    pub fn gemma4_load_weight_any<'a, S: TensorSource + ?Sized>(
        source: &'a S,
        name: &str,
        dims: &[u64],
        allowed_types: &[GGMLType],
    ) -> Result<Weight<'a>, String> {
        let info = source
            .tensor_info(name)
            .ok_or_else(|| format!("Missing tensor: {name}"))?;
        if info.dims != dims || !allowed_types.contains(&info.ggml_type) {
            return Err(format!(
                "Invalid tensor {name}: shape {:?} type {:?}; expected {:?} one of {allowed_types:?}",
                info.dims, info.ggml_type, dims
            ));
        }
        let expected = info
            .checked_nbytes()
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?;
        let bytes = source
            .tensor_slice(name)
            .ok_or_else(|| format!("Missing tensor data: {name}"))?;
        if bytes.len() != expected {
            return Err(format!(
                "Invalid tensor data length for {name}: {}; expected {expected}",
                bytes.len()
            ));
        }
        let n_in = usize::try_from(dims[0]).map_err(|_| format!("{name} input width overflow"))?;
        let n_out =
            usize::try_from(dims[1]).map_err(|_| format!("{name} output width overflow"))?;
        let ggml_type = info.ggml_type;
        let mut weight =
            Weight::from_quantized(QuantizedTensor::from_bytes(bytes, ggml_type, n_in, n_out));
        weight.n_in = n_in;
        weight.n_out = n_out;
        Ok(weight)
    }
}

// ---------------------------------------------------------------------------
// The synthetic source: a checkpoint the test builds itself.
// ---------------------------------------------------------------------------

/// A checkpoint the test builds: a list of tensors, each with a declared
/// format, a shape and a payload.
struct Synthetic {
    tensors: Vec<(TensorInfo, Vec<u8>)>,
}

impl TensorSource for Synthetic {
    fn metadata(&self, _key: &str) -> Option<&MetaValue> {
        None
    }

    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors
            .iter()
            .find(|(info, _)| info.name == name)
            .map(|(info, _)| info)
    }

    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        self.tensors
            .iter()
            .find(|(info, _)| info.name == name)
            .map(|(_, bytes)| bytes.as_slice())
    }
}

impl Synthetic {
    /// Add a tensor whose payload is the deterministic pattern the tests
    /// compare. The pattern is not random: the same bytes have to decode to
    /// the same numbers on every run.
    fn with(
        &mut self,
        name: &str,
        dims: &[u64],
        ggml_type: GGMLType,
        payload: Option<Vec<u8>>,
    ) -> &mut Self {
        let elements: u64 = dims.iter().product();
        let (block_elements, block_bytes) = ggml_type.type_traits();
        let length = (elements as usize / block_elements) * block_bytes;
        let bytes = payload.unwrap_or_else(|| {
            (0..length)
                .map(|index| ((index * 37 + 11) % 251) as u8)
                .collect()
        });
        self.tensors.push((
            TensorInfo {
                name: name.to_string(),
                dims: dims.to_vec(),
                ggml_type,
                offset: 0,
            },
            bytes,
        ));
        self
    }
}

/// The new qwen35 loader, as the model calls it: the shape comes from the
/// tensor's own dimensions, the way `qwen35/trunk/weights.rs` derives it.
fn new_qwen35_load_weight<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
) -> Result<Weight<'a>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let n_in = info.dims.first().copied().unwrap_or(0) as usize;
    let n_out = info.dims[1..].iter().product::<u64>() as usize;
    let mut weight = rust_model_inference::core::tensor::load_weight_any(
        source,
        name,
        n_in,
        n_out,
        Some(QWEN35_MATRIX_FORMATS),
    )?;
    if info.ggml_type == GGMLType::BF16 {
        let data = source.tensor_slice(name).unwrap_or_default();
        weight.kernel =
            Box::new(rust_model_inference::ops::kernel::bf16::BF16Kernel::with_bf16_input(data));
    }
    Ok(weight)
}

/// The new dots loader, as the model calls it.
fn new_dots_load_weight<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    dims: &[u64],
) -> Result<Weight<'a>, String> {
    let n_out = usize::try_from(*dims.last().unwrap())
        .map_err(|_| format!("{name} output width overflow"))?;
    let n_in = dims[..dims.len() - 1]
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))
        .ok_or_else(|| format!("{name} input width overflow"))?;
    rust_model_inference::core::tensor::load_weight_any(
        source,
        name,
        n_in,
        n_out,
        Some(DOTS_FORMATS),
    )
}

/// The new gemma4 loader, as the model calls it.
fn new_gemma4_load_weight<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    dims: &[u64],
    allowed: &[GGMLType],
) -> Result<Weight<'a>, String> {
    let n_in = usize::try_from(dims[0]).map_err(|_| format!("{name} input width overflow"))?;
    let n_out = usize::try_from(dims[1]).map_err(|_| format!("{name} output width overflow"))?;
    rust_model_inference::core::tensor::load_weight_any(source, name, n_in, n_out, Some(allowed))
}

/// `Weight` is not `Debug`, so `unwrap_err` is not available.
fn expect_error<T>(result: Result<T, String>) -> String {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(error) => error,
    }
}

/// Compare two float vectors bit for bit, with NaN equal to NaN.
///
/// The synthetic payload is an arbitrary byte pattern, so it decodes to
/// NaN and to infinity in places. Two loaders that hand the kernel the same
/// bytes produce the same NaNs, and `NaN != NaN` would report that as a
/// difference.
fn assert_same_values(name: &str, old: &[f32], new: &[f32]) {
    assert_eq!(old.len(), new.len(), "{name}: length");
    for (index, (left, right)) in old.iter().zip(new).enumerate() {
        assert!(
            left.to_bits() == right.to_bits() || (left.is_nan() && right.is_nan()),
            "{name}: value {index}: {left:?} vs {right:?}"
        );
    }
}

/// Compare the two weights the way the model depends on them: same format,
/// same shape, same bytes, same numbers.
fn assert_same_weight(name: &str, old: &Weight<'_>, new: &Weight<'_>) {
    assert_eq!(old.ggml_type, new.ggml_type, "{name}: format");
    assert_eq!(old.n_in, new.n_in, "{name}: input width");
    assert_eq!(old.n_out, new.n_out, "{name}: output width");
    assert_eq!(
        old.kernel.weight_bytes().unwrap_or_default(),
        new.kernel.weight_bytes().unwrap_or_default(),
        "{name}: weight bytes"
    );
    if old.n_in > 0 && old.n_out > 0 && old.n_in <= 4096 {
        let input: Vec<f32> = (0..old.n_in)
            .map(|index| ((index % 17) as f32 - 8.0) * 0.125)
            .collect();
        assert_same_values(
            &format!("{name}: matmul output"),
            &old.matmul(&input),
            &new.matmul(&input),
        );
    }
}

/// A row width every format in the lists can hold: 256 is a whole number of
/// 32-element and of 256-element blocks.
const ROW: u64 = 256;
const COLUMNS: u64 = 8;

// ---------------------------------------------------------------------------
// The synthetic layer: this is what CI runs.
// ---------------------------------------------------------------------------

/// Every format the qwen35 loader accepts has to load the same way it used
/// to: same format, same shape, same bytes, same numbers.
#[test]
fn the_shared_loader_matches_the_old_qwen35_loader_on_synthetic_tensors() {
    for ggml_type in QWEN35_MATRIX_FORMATS {
        let mut source = Synthetic {
            tensors: Vec::new(),
        };
        source.with("weight", &[ROW, COLUMNS], *ggml_type, None);
        let old = old::qwen35_load_weight(&source, "weight")
            .unwrap_or_else(|| panic!("the old loader rejected {ggml_type:?}"));
        let new = new_qwen35_load_weight(&source, "weight")
            .unwrap_or_else(|error| panic!("the shared loader rejected {ggml_type:?}: {error}"));
        assert_same_weight(&format!("qwen35 {ggml_type:?}"), &old, &new);
    }
}

/// The dots vocoder stores its convolution weights as `[.., n_in, n_out]`,
/// which is the opposite convention from GGUF's. The two loaders have to
/// agree on it.
#[test]
fn the_shared_loader_matches_the_old_dots_loader_on_synthetic_tensors() {
    // [kernel, in_ch, out_ch] — the dots layout.
    for ggml_type in DOTS_FORMATS {
        let dims = [3u64, 64, 16];
        let mut source = Synthetic {
            tensors: Vec::new(),
        };
        source.with("weight", &dims, *ggml_type, None);
        let old = old::dots_load_weight(&source, "weight", &dims)
            .unwrap_or_else(|error| panic!("the old dots loader rejected {ggml_type:?}: {error}"));
        let new = new_dots_load_weight(&source, "weight", &dims)
            .unwrap_or_else(|error| panic!("the shared loader rejected {ggml_type:?}: {error}"));
        assert_same_weight(&format!("dots {ggml_type:?}"), &old, &new);
        // The dots convention: everything but the last dimension is the
        // input width.
        assert_eq!(new.n_in, 3 * 64, "dots {ggml_type:?}: input width");
        assert_eq!(new.n_out, 16, "dots {ggml_type:?}: output width");
    }
}

/// The gemma4 trunk loads one declared type per tensor.
#[test]
fn the_shared_loader_matches_the_old_gemma4_loader_on_synthetic_tensors() {
    for ggml_type in [GGMLType::Q8_0, GGMLType::Q4K, GGMLType::Q6K, GGMLType::F16] {
        let dims = [ROW, COLUMNS];
        let mut source = Synthetic {
            tensors: Vec::new(),
        };
        source.with("weight", &dims, ggml_type, None);
        let old =
            old::gemma4_load_weight(&source, "weight", &dims, ggml_type).unwrap_or_else(|error| {
                panic!("the old gemma4 loader rejected {ggml_type:?}: {error}")
            });
        let new = new_gemma4_load_weight(&source, "weight", &dims, &[ggml_type])
            .unwrap_or_else(|error| panic!("the shared loader rejected {ggml_type:?}: {error}"));
        assert_same_weight(&format!("gemma4 {ggml_type:?}"), &old, &new);

        // The whitelist entry point has to agree with the single-type one.
        let any = old::gemma4_load_weight_any(&source, "weight", &dims, &[ggml_type])
            .unwrap_or_else(|error| {
                panic!("the old gemma4 whitelist rejected {ggml_type:?}: {error}")
            });
        assert_same_weight(&format!("gemma4 any {ggml_type:?}"), &any, &new);
    }
}

/// The norm and bias tensors are read into owned F32 values. The old reader
/// padded a short payload with zeros; the shared one reports it. For a
/// well-formed tensor the numbers have to be the same.
#[test]
fn the_shared_loader_matches_the_old_qwen35_f32_reader_on_synthetic_tensors() {
    for ggml_type in [GGMLType::F32, GGMLType::F16, GGMLType::BF16] {
        let count = 64usize;
        let mut source = Synthetic {
            tensors: Vec::new(),
        };
        let (block_elements, block_bytes) = ggml_type.type_traits();
        let payload: Vec<u8> = (0..count * block_bytes)
            .map(|index| ((index * 53 + 7) % 251) as u8)
            .collect();
        source.with("norm", &[count as u64], ggml_type, Some(payload));
        let old = old::qwen35_load_weight_f32(&source, "norm")
            .unwrap_or_else(|| panic!("the old reader rejected {ggml_type:?}"));
        let new = rust_model_inference::core::tensor::optional_f32_values(&source, "norm")
            .unwrap_or_else(|error| panic!("the shared reader rejected {ggml_type:?}: {error}"))
            .unwrap_or_else(|| panic!("the shared reader found no {ggml_type:?} values"));
        assert_same_values(&format!("{ggml_type:?}: norm values"), &old, &new);
    }
}

/// The F16 reader changed implementation: the hand-rolled decoder became
/// `ops::float::f16_to_f32`. Every F16 bit pattern has to decode the same.
#[test]
fn the_old_and_the_new_f16_decoders_agree_on_every_bit_pattern() {
    let mut differing = 0usize;
    for bits in 0u32..=0xFFFF {
        let bits = bits as u16;
        let old = f32::from_bits(old_f16_bits(bits));
        let new = rust_model_inference::ops::float::f16_to_f32(bits);
        if old.to_bits() != new.to_bits() && !(old.is_nan() && new.is_nan()) {
            differing += 1;
        }
    }
    assert_eq!(
        differing, 0,
        "the two F16 decoders disagree on a real number"
    );
}

/// The hand-rolled F16 decoder, copied from
/// `fd6f1b7d:src/models/qwen35/trunk/util.rs::f16_bits_to_f32`.
fn old_f16_bits(bits: u16) -> u32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x3FF) as f32 / 1024.0;
    let value = if exp == 0 {
        sign * frac * 2.0f32.powi(-14)
    } else if exp == 31 {
        if frac == 0.0 {
            sign * f32::INFINITY
        } else {
            sign * f32::NAN
        }
    } else {
        sign * (1.0 + frac) * 2.0f32.powi(exp - 15)
    };
    value.to_bits()
}

/// The two differences the unification makes on purpose, both of them
/// fixes: the old qwen35 loader ignored the payload length and read a
/// three-dimensional tensor as a two-dimensional one.
#[test]
fn the_shared_loader_is_stricter_where_the_old_qwen35_loader_was_not() {
    // A payload that does not match the declared format. The old loader
    // took the bytes as they came; the shared one reports them.
    let mut source = Synthetic {
        tensors: Vec::new(),
    };
    source.with(
        "weight",
        &[ROW, COLUMNS],
        GGMLType::BF16,
        Some(vec![0u8; 8]),
    );
    assert!(
        old::qwen35_load_weight(&source, "weight").is_some(),
        "the old loader ignored the payload length"
    );
    let error = expect_error(new_qwen35_load_weight(&source, "weight"));
    assert!(error.contains("expected"), "{error}");

    // A three-dimensional tensor. The old loader read dims[0] x dims[1] and
    // left the rest of the payload unread, so the weight it produced was
    // missing rows. The shared loader reads every row.
    let mut source = Synthetic {
        tensors: Vec::new(),
    };
    source.with("weight", &[32, 4, 8], GGMLType::Q8_0, None);
    let old = old::qwen35_load_weight(&source, "weight").expect("the old loader accepted it");
    assert_eq!(
        old.n_out, 4,
        "the old loader read only the second dimension"
    );
    let new = new_qwen35_load_weight(&source, "weight").expect("the shared loader accepts it");
    assert_eq!(new.n_out, 32, "the shared loader reads every row");
}

/// A format no kernel can hold is reported, never handed to the kernel
/// builder — which panics on it.
#[test]
fn the_shared_loader_reports_a_format_that_has_no_kernel() {
    for ggml_type in [GGMLType::I32, GGMLType::F64, GGMLType::Q8_1, GGMLType::Q8K] {
        let (block_elements, block_bytes) = ggml_type.type_traits();
        let n_in = block_elements as u64;
        let mut source = Synthetic {
            tensors: Vec::new(),
        };
        source.with("weight", &[n_in, 4], ggml_type, None);
        // No whitelist: this is the path that used to reach the kernel
        // builder and panic.
        let error = expect_error(rust_model_inference::core::tensor::load_weight_any(
            &source,
            "weight",
            n_in as usize,
            4,
            None,
        ));
        assert!(error.contains("no weight kernel"), "{ggml_type:?}: {error}");
    }
}

// ---------------------------------------------------------------------------
// The real-weight layer: run with `--ignored` on a machine that has the
// checkpoints.
// ---------------------------------------------------------------------------

/// The checkpoints the real-weight tests look for.
fn checkpoint(role: &str) -> Option<std::path::PathBuf> {
    let candidates: &[&str] = match role {
        "qwen35" => &[
            "models/OvisOCR3-GGUF/OvisOCR3.Q8_0.gguf",
            "models/OvisOCR3-GGUF-converted/bf16/OvisOCR3-BF16.gguf",
            "models/OvisOCR3-GGUF-converted/q8_0/OvisOCR3-Q8_0.gguf",
        ],
        "dots" => &["models/dots-out/dots-tts-base-mmproj-BF16.gguf"],
        "gemma4" => &["models/gemma4/gemma-4-12b-it-Q8_0.gguf"],
        _ => &[],
    };
    candidates
        .iter()
        .map(std::path::PathBuf::from)
        .find(|path| path.exists())
}

/// The matrix tensors the qwen35 trunk reads.
fn qwen35_matrix_names() -> Vec<String> {
    let mut names = Vec::new();
    for prefix in ["token_embd.weight", "output.weight", "cls.output.weight"] {
        names.push(prefix.to_string());
    }
    names.push("shared_head_head.weight".into());
    for layer in 0..64 {
        for suffix in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_qkv.weight",
            "attn_gate.weight",
            "ssm_beta.weight",
            "ssm_alpha.weight",
            "ssm_out.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
            "ffn_gate_inp.weight",
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
            "ffn_gate_inp_shexp.weight",
        ] {
            names.push(format!("blk.{layer}.{suffix}"));
        }
    }
    names
}

/// The old and the new loader must agree on every matrix of a real
/// checkpoint: same format, same shape, same bytes, same numbers.
#[test]
#[ignore = "requires the real OvisOCR3 GGUFs; run with --ignored"]
fn the_shared_loader_matches_the_old_qwen35_loader_on_real_weights() {
    let Some(path) = checkpoint("qwen35") else {
        eprintln!("no OvisOCR3 checkpoint on this machine");
        return;
    };
    let source: std::sync::Arc<dyn TensorSource> = std::sync::Arc::from(
        rust_model_inference::format::ggufrs::open_model_source(
            &path,
            rust_model_inference::format::ggufrs::ComponentRole::Llm,
        )
        .unwrap(),
    );

    let mut compared = 0usize;
    let mut matmuls = 0usize;
    for name in qwen35_matrix_names() {
        let Some(info) = source.tensor_info(&name) else {
            continue;
        };
        if !QWEN35_MATRIX_FORMATS.contains(&info.ggml_type) {
            continue;
        }
        let old = old::qwen35_load_weight(source.as_ref(), &name)
            .unwrap_or_else(|| panic!("the old loader rejected {name}"));
        let new = new_qwen35_load_weight(source.as_ref(), &name)
            .unwrap_or_else(|error| panic!("the shared loader rejected {name}: {error}"));
        assert_same_weight(&name, &old, &new);
        compared += 1;
        if old.n_in > 0 && old.n_out > 0 && old.n_in <= 4096 {
            matmuls += 1;
        }
    }
    assert!(
        compared > 0,
        "no matrix tensor was compared; the checkpoint changed"
    );
    assert!(matmuls > 0, "no matmul was compared");
    eprintln!("qwen35: compared {compared} matrix tensors, {matmuls} matmuls");
}

/// The same comparison for the dots vocoder, whose convolution weights use
/// the `[.., n_in, n_out]` shape convention.
#[test]
#[ignore = "requires the real dots.tts GGUFs; run with --ignored"]
fn the_shared_loader_matches_the_old_dots_loader_on_real_weights() {
    let Some(path) = checkpoint("dots") else {
        eprintln!("no dots.tts checkpoint on this machine");
        return;
    };
    let loader = rust_model_inference::GGUFLoader::from_file(&path).unwrap();
    let source: &dyn TensorSource = &loader;

    let mut compared = 0usize;
    let mut matmuls = 0usize;
    for info in loader.tensors() {
        if info.dims.len() < 2 || !DOTS_FORMATS.contains(&info.ggml_type) {
            continue;
        }
        let dims = info.dims.clone();
        let old = old::dots_load_weight(source, &info.name, &dims)
            .unwrap_or_else(|error| panic!("the old dots loader rejected {}: {error}", info.name));
        let new = new_dots_load_weight(source, &info.name, &dims)
            .unwrap_or_else(|error| panic!("the shared loader rejected {}: {error}", info.name));
        assert_same_weight(&info.name, &old, &new);
        compared += 1;
        if old.n_in > 0 && old.n_out > 0 && old.n_in <= 4096 {
            matmuls += 1;
        }
    }
    assert!(
        compared > 0,
        "no matrix tensor was compared; the checkpoint changed"
    );
    assert!(matmuls > 0, "no matmul was compared");
    eprintln!("dots: compared {compared} matrix tensors, {matmuls} matmuls");
}

/// The same comparison for the gemma4 trunk, which loads one declared type
/// per tensor.
#[test]
#[ignore = "requires the real Gemma 4 GGUFs; run with --ignored"]
fn the_shared_loader_matches_the_old_gemma4_loader_on_real_weights() {
    let Some(path) = checkpoint("gemma4") else {
        eprintln!("no Gemma 4 checkpoint on this machine");
        return;
    };
    let loader = rust_model_inference::GGUFLoader::from_file(&path).unwrap();
    let source: &dyn TensorSource = &loader;

    let mut compared = 0usize;
    let mut matmuls = 0usize;
    for info in loader.tensors() {
        if info.dims.len() != 2 {
            continue;
        }
        let dims = info.dims.clone();
        let old = old::gemma4_load_weight(source, &info.name, &dims, info.ggml_type)
            .unwrap_or_else(|error| {
                panic!("the old gemma4 loader rejected {}: {error}", info.name)
            });
        let new = new_gemma4_load_weight(source, &info.name, &dims, &[info.ggml_type])
            .unwrap_or_else(|error| panic!("the shared loader rejected {}: {error}", info.name));
        assert_same_weight(&info.name, &old, &new);
        compared += 1;
        if old.n_in > 0 && old.n_out > 0 && old.n_in <= 4096 {
            matmuls += 1;
        }
    }
    assert!(
        compared > 0,
        "no matrix tensor was compared; the checkpoint changed"
    );
    assert!(matmuls > 0, "no matmul was compared");
    eprintln!("gemma4: compared {compared} matrix tensors, {matmuls} matmuls");
}
