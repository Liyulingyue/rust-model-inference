use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::{QuantizedTensor, Weight};
use crate::ops::quant::{BlockQ8K, QK_K};
use crate::ops::{dot_f32, sum_f32, sum_sq_centered_f32, vec_add_into};

pub use crate::ops::{
    gelu_approx_inplace, gelu_inplace, rms_norm, rms_norm_inplace, rope_neox_partial,
    silu_approx_inplace, silu_inplace, silu_mul_approx_inplace, silu_mul_inplace,
};

pub fn checked_len(name: &str, dimensions: &[usize]) -> Result<usize, String> {
    if dimensions.is_empty() || dimensions.contains(&0) {
        return Err(format!("{name} dimensions must be non-zero"));
    }
    dimensions.iter().try_fold(1usize, |length, &dimension| {
        length
            .checked_mul(dimension)
            .ok_or_else(|| format!("{name} shape overflows usize"))
    })
}

pub struct Linear<'a> {
    weight: Weight<'a>,
    bias: Option<Vec<f32>>,
}

impl<'a> Linear<'a> {
    pub fn from_weight(weight: Weight<'a>, bias: Option<Vec<f32>>) -> Result<Self, String> {
        if weight.n_in == 0 || weight.n_out == 0 {
            return Err("DreamX linear dimensions must be non-zero".into());
        }
        if bias.as_ref().is_some_and(|bias| bias.len() != weight.n_out) {
            return Err("DreamX linear bias width mismatch".into());
        }
        Ok(Self { weight, bias })
    }

    pub fn from_source(
        source: &'a dyn TensorSource,
        weight_name: &str,
        bias_name: Option<&str>,
        n_in: usize,
        n_out: usize,
    ) -> Result<Self, String> {
        let info = source
            .tensor_info(weight_name)
            .ok_or_else(|| format!("Missing tensor: {weight_name}"))?;
        let expected_dims = [n_in as u64, n_out as u64];
        if info.dims != expected_dims {
            return Err(format!(
                "Invalid tensor {weight_name}: shape {:?}; expected {:?}",
                info.dims, expected_dims
            ));
        }
        match info.ggml_type {
            GGMLType::F32
            | GGMLType::F16
            | GGMLType::BF16
            | GGMLType::Q8_0
            | GGMLType::Q6K
            | GGMLType::Q2K
            | GGMLType::Q3K
            | GGMLType::IQ4_NL
            | GGMLType::IQ2_XXS
            | GGMLType::IQ2_S
            | GGMLType::IQ2_XS
            | GGMLType::IQ3_XXS
            | GGMLType::IQ3_S
            | GGMLType::IQ4_XS
            | GGMLType::IQ1_M
            | GGMLType::IQ1_S
            | GGMLType::Q4_0
            | GGMLType::Q4_1
            | GGMLType::Q4K
            | GGMLType::Q5K => {}
            other => return Err(format!("Unsupported DreamX linear type {other:?}")),
        }
        let bytes = source
            .tensor_slice(weight_name)
            .ok_or_else(|| format!("Missing tensor data: {weight_name}"))?;
        let expected_bytes = info
            .checked_nbytes()
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| format!("Invalid tensor byte size: {weight_name}"))?;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "Invalid tensor data length for {weight_name}: {}; expected {expected_bytes}",
                bytes.len()
            ));
        }
        let mut weight = Weight::from_quantized(QuantizedTensor::from_bytes(
            bytes,
            info.ggml_type,
            n_in,
            n_out,
        ));
        weight.n_in = n_in;
        weight.n_out = n_out;
        let bias = bias_name
            .map(|name| load_float_tensor(source, name, n_out))
            .transpose()?;
        Self::from_weight(weight, bias)
    }

    pub fn n_in(&self) -> usize {
        self.weight.n_in
    }

    pub fn n_out(&self) -> usize {
        self.weight.n_out
    }

    pub fn embedding_lookup(&self, token_id: u32, output: &mut [f32]) -> Result<(), String> {
        if token_id as usize >= self.n_out() || output.len() != self.n_in() {
            return Err("Invalid DreamX embedding lookup".into());
        }
        self.weight.embedding_lookup(token_id, output);
        Ok(())
    }

    pub fn forward(
        &self,
        pool: &ComputePool,
        input: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>, String> {
        self.forward_impl(pool, input, rows, false)
    }

    /// Forward a projection and repair a device result that the CPU kernel
    /// does not agree with.
    ///
    /// The device intermittently returns a projection that is wrong in whole
    /// rows (TODO-044). Rechecking a few elements of the result against the
    /// CPU kernel costs a fraction of the projection itself, and a result
    /// that disagrees is recomputed on the CPU, so the caller always gets
    /// the numbers the CPU would have produced.
    pub fn forward_verified(
        &self,
        pool: &ComputePool,
        input: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>, String> {
        self.forward_impl(pool, input, rows, true)
    }

    fn forward_impl(
        &self,
        pool: &ComputePool,
        input: &[f32],
        rows: usize,
        verify: bool,
    ) -> Result<Vec<f32>, String> {
        if rows == 0 || input.len() != checked_len("DreamX linear input", &[rows, self.n_in()])? {
            return Err("Invalid DreamX linear input shape".into());
        }
        let mut output = vec![0.0; checked_len("DreamX linear output", &[rows, self.n_out()])?];
        #[cfg(feature = "vulkan")]
        if self.weight.try_vulkan_rows(input, &mut output, rows) {
            // `RMI_GPU_CPU_CHECK=1` compares every GPU projection with the CPU
            // kernel on the same weight and input. The two summation orders
            // differ, so a healthy projection shows a relative difference
            // around 1e-6; a device that returned the previous contents of the
            // arena shows 1.0, and one that ran only part of the dispatch
            // shows up between the two.
            #[cfg(feature = "vulkan")]
            if std::env::var_os("RMI_GPU_CPU_CHECK").is_some() {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static COUNTER: AtomicUsize = AtomicUsize::new(0);
                let index = COUNTER.fetch_add(1, Ordering::Relaxed);
                if let Some(weight) = self.weight.kernel.bf16_bytes() {
                    let mut cpu = vec![0.0f32; output.len()];
                    linear_bf16(
                        pool,
                        weight,
                        input,
                        rows,
                        self.n_in(),
                        self.n_out(),
                        None,
                        &mut cpu,
                    );
                    let mut max_abs = 0.0f32;
                    let mut scale = 0.0f32;
                    for (gpu, cpu) in output.iter().zip(cpu.iter()) {
                        max_abs = max_abs.max((gpu - cpu).abs());
                        scale = scale.max(cpu.abs());
                    }
                    let relative = if scale > 0.0 {
                        max_abs / scale
                    } else {
                        max_abs
                    };
                    // The relative difference is always reported: a healthy
                    // projection sits around 1e-6 because the two summation
                    // orders differ, and the first projection whose number
                    // moves between two runs is where the drift starts.
                    eprintln!(
                        "[GPU-CHECK] {index} {rows} {} {} {relative:.12e}",
                        self.n_in(),
                        self.n_out(),
                    );
                }
            }
            if verify {
                if let Some(weight) = self.weight.kernel.bf16_bytes() {
                    // The device has been seen to return the numbers of an
                    // earlier call, so a projection is checked against the CPU
                    // kernel before it is used. A retry of the same call
                    // usually returns the right numbers, and a retry keeps the
                    // bits of a healthy run, which leaves the caller
                    // reproducible. Only a projection that stays wrong is
                    // recomputed on the CPU.
                    // How far a healthy projection sits from the CPU kernel.
                    // The check has to sit above this and below the damage.
                    if std::env::var_os("RMI_GPU_VERIFY_TRACE").is_some() {
                        let worst = projection_worst_difference(
                            weight,
                            input,
                            &output,
                            rows,
                            self.n_in(),
                            self.n_out(),
                        );
                        HEALTHY_WORST.fetch_max(worst.to_bits(), Ordering::Relaxed);
                    }
                    let mut attempts = 0;
                    if std::env::var_os("RMI_GPU_VERIFY_TRACE").is_some() {
                        // When the check fails, the whole output is compared
                        // against the CPU kernel, so the shape of the damage
                        // is visible: how many elements differ, and how they
                        // are spread over the output.
                        if !projection_matches_cpu(
                            pool,
                            weight,
                            input,
                            &output,
                            rows,
                            self.n_in(),
                            self.n_out(),
                        ) {
                            let mut cpu = vec![0.0f32; output.len()];
                            linear_bf16(
                                pool,
                                weight,
                                input,
                                rows,
                                self.n_in(),
                                self.n_out(),
                                None,
                                &mut cpu,
                            );
                            let mut differing = 0usize;
                            let mut first = usize::MAX;
                            let mut last = 0usize;
                            let mut per_row: Vec<usize> = vec![0; rows];
                            for index in 0..output.len() {
                                if output[index] != cpu[index] {
                                    differing += 1;
                                    first = first.min(index);
                                    last = last.max(index);
                                    per_row[index / self.n_out()] += 1;
                                }
                            }
                            let rows_hit = per_row.iter().filter(|c| **c > 0).count();
                            let worst_row = per_row.iter().copied().max().unwrap_or(0);
                            eprintln!(
                                "[GPU-VERIFY-SHAPE] rows={rows} n_out={} 不同的元素={differing}/{} 受影响的行={rows_hit} 最差行的错列数={worst_row} 跨度={}",
                                self.n_out(),
                                output.len(),
                                last.saturating_sub(first)
                            );
                        }
                    }
                    if std::env::var_os("RMI_GPU_VERIFY_TRACE").is_some() {
                        eprintln!(
                            "[GPU-VERIFY] linear rows={rows} n_in={} n_out={} worst={:.3e}",
                            self.n_in(),
                            self.n_out(),
                            projection_worst_difference(
                                weight,
                                input,
                                &output,
                                rows,
                                self.n_in(),
                                self.n_out(),
                            )
                        );
                    }
                    while attempts < GPU_RETRY_LIMIT
                        && !projection_matches_cpu(
                            pool,
                            weight,
                            input,
                            &output,
                            rows,
                            self.n_in(),
                            self.n_out(),
                        )
                    {
                        attempts += 1;
                        if !self.weight.try_vulkan_rows(input, &mut output, rows) {
                            break;
                        }
                    }
                    if attempts > 0 {
                        eprintln!(
                            "[GPU-VERIFY] a projection was wrong and {attempts} retries fixed it"
                        );
                    }
                    if !projection_matches_cpu(
                        pool,
                        weight,
                        input,
                        &output,
                        rows,
                        self.n_in(),
                        self.n_out(),
                    ) {
                        eprintln!(
                            "[GPU] a projection did not match the CPU kernel after {attempts} retries; recomputing it on the CPU"
                        );
                        let _cpu_scope = ComputePool::disable_gpu_matmul_for_scope();
                        linear_bf16(
                            pool,
                            weight,
                            input,
                            rows,
                            self.n_in(),
                            self.n_out(),
                            None,
                            &mut output,
                        );
                    }
                }
            }
            if let Some(bias) = &self.bias {
                for row in output.chunks_exact_mut(self.n_out()) {
                    vec_add_into(bias, row);
                }
            }
            return Ok(output);
        }
        #[cfg(feature = "vulkan")]
        let _cpu_scope = ComputePool::disable_gpu_matmul_for_scope();
        if let Some(weight) = self.weight.kernel.bf16_bytes() {
            linear_bf16(
                pool,
                weight,
                input,
                rows,
                self.n_in(),
                self.n_out(),
                self.bias.as_deref(),
                &mut output,
            );
            return Ok(output);
        }
        let mut q8 = vec![0; self.n_in()];
        let mut scales = vec![0.0; self.n_in().div_ceil(32)];
        let mut q8k = vec![
            BlockQ8K {
                d: 0.0,
                qs: [0; QK_K],
                bsums: [0; QK_K / 16],
            };
            self.n_in().div_ceil(QK_K)
        ];
        for row in 0..rows {
            let input_row = &input[row * self.n_in()..(row + 1) * self.n_in()];
            let output_row = &mut output[row * self.n_out()..(row + 1) * self.n_out()];
            self.weight.quantize_and_matmul_with_scratch(
                input_row,
                &mut q8k,
                &mut q8,
                &mut scales,
                output_row,
                pool,
            );
            if let Some(bias) = &self.bias {
                vec_add_into(bias, output_row);
            }
        }
        Ok(output)
    }
}

fn load_float_tensor(
    source: &dyn TensorSource,
    name: &str,
    expected_len: usize,
) -> Result<Vec<f32>, String> {
    load_float_values(source, name, &[expected_len as u64])
}

pub fn load_float_values(
    source: &dyn TensorSource,
    name: &str,
    expected_dims: &[u64],
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != expected_dims {
        return Err(format!(
            "Invalid tensor {name}: shape {:?}; expected {expected_dims:?}",
            info.dims,
        ));
    }
    let expected_len = expected_dims
        .iter()
        .try_fold(1usize, |length, &dimension| {
            usize::try_from(dimension)
                .ok()
                .and_then(|dimension| length.checked_mul(dimension))
                .ok_or_else(|| format!("Invalid tensor shape: {name}"))
        })?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    let expected_bytes = info
        .checked_nbytes()
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected_bytes}",
            bytes.len()
        ));
    }
    let values: Vec<f32> = match info.ggml_type {
        GGMLType::F32 => bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect(),
        GGMLType::F16 => bytes
            .chunks_exact(2)
            .map(|bytes| crate::ops::f16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())))
            .collect(),
        GGMLType::BF16 => bytes
            .chunks_exact(2)
            .map(|bytes| crate::ops::bf16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())))
            .collect(),
        other => {
            return Err(format!(
                "Unsupported DreamX float tensor type {other:?}: {name}"
            ))
        }
    };
    if values.len() != expected_len {
        return Err(format!("Invalid tensor data length for {name}"));
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn linear_bf16(
    pool: &ComputePool,
    weight: &[u8],
    input: &[f32],
    rows: usize,
    n_in: usize,
    n_out: usize,
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    let output_address = output.as_mut_ptr() as usize;
    let output_len = rows * n_out;
    pool.compute(|thread, threads| {
        for index in (thread..output_len).step_by(threads) {
            let row = index / n_out;
            let column = index % n_out;
            let input = &input[row * n_in..(row + 1) * n_in];
            let weight = &weight[column * n_in * 2..(column + 1) * n_in * 2];
            let value = dot_bf16_f32(weight, input) + bias.map_or(0.0, |bias| bias[column]);
            unsafe { (output_address as *mut f32).add(index).write(value) };
        }
    });
}

/// How often a projection is asked of the device again after the CPU kernel
/// disagreed with it.
use std::sync::atomic::Ordering;

const GPU_RETRY_LIMIT: usize = 3;

/// The largest relative difference a healthy projection showed against the
/// CPU kernel.
static HEALTHY_WORST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The largest relative difference a healthy projection showed.
pub fn healthy_worst_difference() -> f32 {
    f32::from_bits(HEALTHY_WORST.load(Ordering::Relaxed))
}

/// How many elements of a projection are checked against the CPU kernel.
const GPU_CHECK_SAMPLES: usize = 65_536;

/// How many consecutive columns of a projection are checked at once. The
/// weight of a projection is read in order this way, and a small block
/// covers more of the output with the same number of samples.
const GPU_CHECK_BLOCK: usize = 8;

/// How many rows of a projection are checked in full. The damage is
/// sometimes a run of wrong columns, which a sample of single elements
/// steps over.
const GPU_CHECK_ROWS: usize = 8;

/// The largest relative difference between a device projection and the CPU
/// kernel over the sampled elements.
fn projection_worst_difference(
    weight: &[u8],
    input: &[f32],
    output: &[f32],
    rows: usize,
    n_in: usize,
    n_out: usize,
) -> f32 {
    let scale = output
        .iter()
        .fold(0.0f32, |peak, value| peak.max(value.abs()))
        .max(1.0e-6);
    let mut worst = 0.0f32;
    let cells = rows * n_out;
    let samples = GPU_CHECK_SAMPLES.min(cells);
    for sample in 0..samples {
        let cell = (sample as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) % cells as u64;
        let row = (cell as usize) / n_out;
        let column = (cell as usize) % n_out;
        let reference = dot_bf16_f32(
            &weight[column * n_in * 2..(column + 1) * n_in * 2],
            &input[row * n_in..(row + 1) * n_in],
        );
        worst = worst.max((output[cell as usize] - reference).abs() / scale);
    }
    worst
}

/// Whether a device projection agrees with the CPU kernel.
///
/// The two summation orders differ, so a healthy projection differs from the
/// CPU by a relative amount around 1e-6. A device that returned the numbers
/// of an earlier call differs in whole rows, which is far above that.
///
/// The check looks at a few elements of a few rows instead of the whole
/// projection, so it costs a small fraction of the projection itself. A
/// device that returned the wrong numbers gets those wrong in every row, so
/// a handful of elements is enough to catch it.
pub(crate) fn projection_matches_cpu(
    pool: &ComputePool,
    weight: &[u8],
    input: &[f32],
    output: &[f32],
    rows: usize,
    n_in: usize,
    n_out: usize,
) -> bool {
    if rows == 0 || n_in == 0 || n_out == 0 {
        return true;
    }
    // The scale of the whole result is the yardstick: the healthy
    // difference between the two summation orders is a small fraction of it.
    let scale = output
        .iter()
        .fold(0.0f32, |peak, value| peak.max(value.abs()))
        .max(1.0e-6);
    // The healthy difference between the two summation orders is around 1e-6
    // of the scale, and a broken projection is off by 1e-2 of it or more, so
    // a limit of 1e-4 tells the two apart with room on both sides.
    let limit = 1.0e-4 * scale;
    // The device damages the output in three ways: a whole row, a run of
    // consecutive columns, and a few scattered values. The check therefore
    // looks at
    //
    // - a spread of single elements over the whole output, which is what
    //   scattered damage needs;
    // - a few rows in full, which is what a run of columns needs.
    //
    // The spread is over the output and not over the columns of every row:
    // a projection of the decoder has sixty-five thousand rows of thirty-two
    // columns, and a sample of every row would spend all its time on the
    // same thirty-two columns.
    // The samples come in blocks of consecutive columns, so the weight is
    // read in order instead of one six kilobyte row at a time.
    let block = GPU_CHECK_BLOCK.min(n_out);
    let blocks_per_row = n_out.div_ceil(block);
    let total_blocks = rows * blocks_per_row;
    let blocks = (GPU_CHECK_SAMPLES / block).clamp(1, total_blocks);
    // The check is the same work for every block, so the pool runs it.
    let mismatch = std::sync::atomic::AtomicBool::new(false);
    pool.compute(|thread, threads| {
        for sample in (thread..blocks).step_by(threads) {
            if mismatch.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }
            // The golden ratio spreads the blocks over the output whatever
            // its size is, and keeps them apart for a power of two.
            let index = (sample as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) % total_blocks as u64;
            let row = (index as usize) / blocks_per_row;
            let first = (index as usize) % blocks_per_row * block;
            for offset in 0..block {
                let column = first + offset;
                if column >= n_out {
                    break;
                }
                let reference = dot_bf16_f32(
                    &weight[column * n_in * 2..(column + 1) * n_in * 2],
                    &input[row * n_in..(row + 1) * n_in],
                );
                if (output[row * n_out + column] - reference).abs() > limit {
                    mismatch.store(true, std::sync::atomic::Ordering::Relaxed);
                    return;
                }
            }
        }
    });
    if mismatch.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    // Every column of a few rows, so a run of wrong columns cannot hide.
    pool.compute(|thread, threads| {
        for sample in (thread..GPU_CHECK_ROWS).step_by(threads) {
            if mismatch.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }
            let row = (sample as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) % rows as u64;
            for column in 0..n_out {
                let reference = dot_bf16_f32(
                    &weight[column * n_in * 2..(column + 1) * n_in * 2],
                    &input[row as usize * n_in..(row as usize + 1) * n_in],
                );
                if (output[row as usize * n_out + column] - reference).abs() > limit {
                    mismatch.store(true, std::sync::atomic::Ordering::Relaxed);
                    return;
                }
            }
        }
    });
    !mismatch.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn dot_bf16_f32(weight: &[u8], input: &[f32]) -> f32 {
    crate::ops::dot_bf16_f32(input, weight, input.len())
}

pub fn rms_norm_rows(
    input: &[f32],
    rows: usize,
    weight: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    let width = weight.len();
    if rows == 0
        || width == 0
        || input.len() != checked_len("DreamX RMSNorm input", &[rows, width])?
        || !epsilon.is_finite()
        || epsilon < 0.0
    {
        return Err("Invalid DreamX RMSNorm tensors".into());
    }
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let range = row * width..(row + 1) * width;
        if crate::ops::scalar_mode() {
            let values = &input[range.clone()];
            let mut sum_sq = 0.0f32;
            for &value in values {
                sum_sq += value * value;
            }
            let scale = 1.0f32 / (sum_sq / width as f32 + epsilon).sqrt();
            for index in 0..width {
                output[range.start + index] = values[index] * scale * weight[index];
            }
        } else {
            rms_norm(&input[range.clone()], weight, &mut output[range], epsilon);
        }
    }
    Ok(output)
}

pub fn layer_norm_rows(
    input: &[f32],
    rows: usize,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    let width = weight
        .map(<[f32]>::len)
        .or_else(|| bias.map(<[f32]>::len))
        .ok_or("DreamX LayerNorm requires a weight or bias")?;
    if rows == 0
        || width == 0
        || weight.is_some_and(|values| values.len() != width)
        || bias.is_some_and(|values| values.len() != width)
        || input.len() != checked_len("DreamX LayerNorm input", &[rows, width])?
        || !epsilon.is_finite()
        || epsilon < 0.0
    {
        return Err("Invalid DreamX LayerNorm tensors".into());
    }
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let range = row * width..(row + 1) * width;
        let values = &input[range.clone()];
        if crate::ops::scalar_mode() {
            let mut sum = 0.0f32;
            for &value in values {
                sum += value;
            }
            let mean = sum / width as f32;
            let mut variance = 0.0f32;
            for &value in values {
                let centered = value - mean;
                variance += centered * centered;
            }
            let inverse = 1.0 / (variance / width as f32 + epsilon).sqrt();
            for index in 0..width {
                output[range.start + index] =
                    (values[index] - mean) * inverse * weight.map_or(1.0, |weight| weight[index])
                        + bias.map_or(0.0, |bias| bias[index]);
            }
            continue;
        }
        let mean = (sum_f32(values) / width as f64) as f32;
        let variance = (sum_sq_centered_f32(values, mean) / width as f64) as f32;
        let inverse = 1.0 / (variance + epsilon).sqrt();
        for index in 0..width {
            output[range.start + index] =
                (values[index] - mean) * inverse * weight.map_or(1.0, |weight| weight[index])
                    + bias.map_or(0.0, |bias| bias[index]);
        }
    }
    Ok(output)
}

pub fn group_norm_ncthw(
    input: &[f32],
    shape: [usize; 4],
    groups: usize,
    weight: &[f32],
    bias: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    let [channels, depth, height, width] = shape;
    if groups == 0
        || channels % groups != 0
        || weight.len() != channels
        || bias.len() != channels
        || input.len() != checked_len("DreamX GroupNorm input", &shape)?
        || !epsilon.is_finite()
        || epsilon < 0.0
    {
        return Err("Invalid DreamX GroupNorm tensors".into());
    }
    let plane = checked_len("DreamX GroupNorm plane", &[depth, height, width])?;
    let channels_per_group = channels / groups;
    let values_per_group = checked_len("DreamX GroupNorm group", &[channels_per_group, plane])?;
    let mut output = vec![0.0; input.len()];
    for group in 0..groups {
        let start = group * values_per_group;
        let values = &input[start..start + values_per_group];
        let (mean, variance) = if crate::ops::scalar_mode() {
            let mut sum = 0.0f32;
            for &value in values {
                sum += value;
            }
            let mean = sum / values_per_group as f32;
            let mut variance = 0.0f32;
            for &value in values {
                let centered = value - mean;
                variance += centered * centered;
            }
            (mean, variance / values_per_group as f32)
        } else {
            let mean = (sum_f32(values) / values_per_group as f64) as f32;
            (
                mean,
                (sum_sq_centered_f32(values, mean) / values_per_group as f64) as f32,
            )
        };
        let inverse = 1.0 / (variance + epsilon).sqrt();
        for channel_offset in 0..channels_per_group {
            let channel = group * channels_per_group + channel_offset;
            let channel_start = channel * plane;
            for index in 0..plane {
                output[channel_start + index] =
                    (input[channel_start + index] - mean) * inverse * weight[channel]
                        + bias[channel];
            }
        }
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionSpec {
    pub query_tokens: usize,
    pub key_tokens: usize,
    pub query_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub causal: bool,
    pub scale: f32,
}

impl AttentionSpec {
    fn validate(self, query: &[f32], key: &[f32], value: &[f32]) -> Result<(), String> {
        if self.query_tokens == 0
            || self.key_tokens == 0
            || self.query_heads == 0
            || self.key_value_heads == 0
            || self.head_dim == 0
            || self.query_heads % self.key_value_heads != 0
            || !self.scale.is_finite()
            || query.len()
                != checked_len(
                    "DreamX attention query",
                    &[self.query_tokens, self.query_heads, self.head_dim],
                )?
            || key.len()
                != checked_len(
                    "DreamX attention key",
                    &[self.key_tokens, self.key_value_heads, self.head_dim],
                )?
            || value.len() != key.len()
            || (self.causal && self.query_tokens > self.key_tokens)
        {
            return Err("Invalid DreamX attention tensors".into());
        }
        Ok(())
    }

    fn key_visible(self, query: usize, key: usize) -> bool {
        !self.causal || key <= query + (self.key_tokens - self.query_tokens)
    }
}

pub fn attention_scalar(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    spec: AttentionSpec,
) -> Result<Vec<f32>, String> {
    attention_scalar_with_dot(query, key, value, spec, dot_f32)
}

thread_local! {
    /// Set while the CPU attention runs, so the check that compares a device
    /// result against it does not ask the device again.
    static ATTENTION_ON_CPU: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The attention the CPU kernel produces, without the device path.
///
/// The device has been seen to return the numbers of an earlier call
/// (TODO-044), so the device result is compared against this one.
pub fn attention_on_cpu(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    spec: AttentionSpec,
) -> Result<Vec<f32>, String> {
    ATTENTION_ON_CPU.with(|flag| flag.set(true));
    let result = attention_scalar_with_dot(query, key, value, spec, dot_f32);
    ATTENTION_ON_CPU.with(|flag| flag.set(false));
    result
}

/// Whether the CPU attention is the one being computed right now.
pub(crate) fn attention_is_on_cpu() -> bool {
    ATTENTION_ON_CPU.with(|flag| flag.get())
}

/// Whether a device attention agrees with the CPU kernel.
///
/// The two summation orders differ, so a healthy attention differs from the
/// CPU by a small relative amount. A device that returned the numbers of an
/// earlier call differs in every row, which is far above that.
pub(crate) fn attention_matches_cpu(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    output: &[f32],
    spec: AttentionSpec,
) -> bool {
    if spec.query_heads == 0 || spec.head_dim == 0 || spec.query_tokens == 0 {
        return true;
    }
    // The scale of the whole result is the yardstick.
    let scale = output
        .iter()
        .fold(0.0f32, |peak, value| peak.max(value.abs()))
        .max(1.0e-6);
    // The healthy difference between the two summation orders is around 1e-6
    // of the scale, and a broken result is off by 1e-2 of it or more, so a
    // limit of 1e-4 tells the two apart with room on both sides.
    let limit = 1.0e-4 * scale;
    let Ok(reference) = attention_on_cpu(query, key, value, spec) else {
        return true;
    };
    if reference.len() != output.len() {
        return true;
    }
    // The device gets whole tokens or whole heads wrong, so every token is
    // looked at, at one element of every head.
    let width = spec.head_dim * spec.query_heads;
    if width == 0 || output.len() % width != 0 {
        return true;
    }
    // Every token and every head is looked at, at several elements of the
    // head. One element per head is not enough: the device has been seen to
    // damage a part of a head only, and that part would slip through.
    for token in 0..output.len() / width {
        for head in 0..spec.query_heads {
            let base = token * width + head * spec.head_dim;
            for step in 0..GPU_ATTENTION_CHECK_ELEMENTS {
                let index = base + (step * 1_030_003 + head * 8_177) % spec.head_dim;
                if (output[index] - reference[index]).abs() > limit {
                    return false;
                }
            }
            // A run of wrong dimensions is missed by the spread sampling
            // above, so a consecutive block is looked at as well.
            for step in 0..GPU_ATTENTION_CHECK_BLOCK {
                let index = base + (step + head * 31) % spec.head_dim;
                if (output[index] - reference[index]).abs() > limit {
                    return false;
                }
            }
        }
    }
    true
}

/// Sequential F32 QK reduction in scalar mode, as used by MageVAE's torch.bmm.
pub fn attention_scalar_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    spec: AttentionSpec,
) -> Result<Vec<f32>, String> {
    attention_scalar_with_dot(query, key, value, spec, crate::ops::dot_f32_exact)
}

fn attention_scalar_with_dot(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    spec: AttentionSpec,
    dot: fn(&[f32], &[f32], usize) -> f32,
) -> Result<Vec<f32>, String> {
    attention_scalar_with_bias(query, key, value, None, spec, dot)
}

fn attention_scalar_with_bias(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    bias: Option<&[f32]>,
    spec: AttentionSpec,
    dot: fn(&[f32], &[f32], usize) -> f32,
) -> Result<Vec<f32>, String> {
    spec.validate(query, key, value)?;
    #[cfg(feature = "vulkan")]
    if bias.is_none() && !attention_is_on_cpu() {
        if let Some(output) = crate::ops::kernel::vulkan::try_attention(query, key, value, spec) {
            return Ok(output);
        }
    }
    if let Some(bias) = bias {
        let expected = checked_len(
            "DreamX attention bias",
            &[spec.query_heads, spec.query_tokens, spec.key_tokens],
        )?;
        if bias.len() != expected {
            return Err("Invalid DreamX attention bias shape".into());
        }
    }
    let mut output = vec![0.0; query.len()];
    let mut scores = vec![f32::NEG_INFINITY; spec.key_tokens];
    let group_size = spec.query_heads / spec.key_value_heads;
    for query_token in 0..spec.query_tokens {
        for query_head in 0..spec.query_heads {
            let query_start = (query_token * spec.query_heads + query_head) * spec.head_dim;
            let key_head = query_head / group_size;
            let mut maximum = f32::NEG_INFINITY;
            for key_token in 0..spec.key_tokens {
                if !spec.key_visible(query_token, key_token) {
                    scores[key_token] = f32::NEG_INFINITY;
                    continue;
                }
                let key_start = (key_token * spec.key_value_heads + key_head) * spec.head_dim;
                let score = dot(
                    &query[query_start..query_start + spec.head_dim],
                    &key[key_start..key_start + spec.head_dim],
                    spec.head_dim,
                ) * spec.scale;
                let score = score
                    + bias.map_or(0.0, |bias| {
                        bias[(query_head * spec.query_tokens + query_token) * spec.key_tokens
                            + key_token]
                    });
                scores[key_token] = score;
                maximum = maximum.max(score);
            }
            let mut normalizer = 0.0;
            for &score in &scores {
                if score.is_finite() {
                    normalizer += (score - maximum).exp();
                }
            }
            let output_row = &mut output[query_start..query_start + spec.head_dim];
            for (key_token, &score) in scores.iter().enumerate() {
                if !score.is_finite() {
                    continue;
                }
                let probability = (score - maximum).exp() / normalizer;
                let value_start = (key_token * spec.key_value_heads + key_head) * spec.head_dim;
                for dimension in 0..spec.head_dim {
                    output_row[dimension] += probability * value[value_start + dimension];
                }
            }
        }
    }
    Ok(output)
}

pub fn attention_online(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    spec: AttentionSpec,
) -> Result<Vec<f32>, String> {
    if crate::ops::scalar_mode() {
        return attention_scalar_f32(query, key, value, spec);
    }
    spec.validate(query, key, value)?;
    #[cfg(feature = "vulkan")]
    if let Some(output) = crate::ops::kernel::vulkan::try_attention(query, key, value, spec) {
        return Ok(output);
    }
    let mut output = vec![0.0; query.len()];
    let group_size = spec.query_heads / spec.key_value_heads;
    for query_token in 0..spec.query_tokens {
        for query_head in 0..spec.query_heads {
            let query_start = (query_token * spec.query_heads + query_head) * spec.head_dim;
            let output_row = &mut output[query_start..query_start + spec.head_dim];
            let key_head = query_head / group_size;
            let mut running_max = f32::NEG_INFINITY;
            let mut normalizer = 0.0;
            for key_token in 0..spec.key_tokens {
                if !spec.key_visible(query_token, key_token) {
                    continue;
                }
                let key_start = (key_token * spec.key_value_heads + key_head) * spec.head_dim;
                let score = dot_f32(
                    &query[query_start..query_start + spec.head_dim],
                    &key[key_start..key_start + spec.head_dim],
                    spec.head_dim,
                ) * spec.scale;
                let new_max = running_max.max(score);
                let old_scale = (running_max - new_max).exp();
                let new_scale = (score - new_max).exp();
                normalizer = normalizer * old_scale + new_scale;
                let value_start = (key_token * spec.key_value_heads + key_head) * spec.head_dim;
                for dimension in 0..spec.head_dim {
                    output_row[dimension] = output_row[dimension] * old_scale
                        + new_scale * value[value_start + dimension];
                }
                running_max = new_max;
            }
            for value in output_row {
                *value /= normalizer;
            }
        }
    }
    Ok(output)
}

pub fn attention_online_with_bias(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    bias: &[f32],
    spec: AttentionSpec,
) -> Result<Vec<f32>, String> {
    if crate::ops::scalar_mode() {
        return attention_scalar_with_bias(
            query,
            key,
            value,
            Some(bias),
            spec,
            crate::ops::dot_f32_exact,
        );
    }
    spec.validate(query, key, value)?;
    if bias.len()
        != checked_len(
            "DreamX attention bias",
            &[spec.query_heads, spec.query_tokens, spec.key_tokens],
        )?
    {
        return Err("Invalid DreamX attention bias shape".into());
    }
    let mut output = vec![0.0; query.len()];
    let group_size = spec.query_heads / spec.key_value_heads;
    for query_token in 0..spec.query_tokens {
        for query_head in 0..spec.query_heads {
            let query_start = (query_token * spec.query_heads + query_head) * spec.head_dim;
            let output_row = &mut output[query_start..query_start + spec.head_dim];
            let key_head = query_head / group_size;
            let mut running_max = f32::NEG_INFINITY;
            let mut normalizer = 0.0;
            for key_token in 0..spec.key_tokens {
                if !spec.key_visible(query_token, key_token) {
                    continue;
                }
                let key_start = (key_token * spec.key_value_heads + key_head) * spec.head_dim;
                let bias_index =
                    (query_head * spec.query_tokens + query_token) * spec.key_tokens + key_token;
                let score = dot_f32(
                    &query[query_start..query_start + spec.head_dim],
                    &key[key_start..key_start + spec.head_dim],
                    spec.head_dim,
                ) * spec.scale
                    + bias[bias_index];
                let new_max = running_max.max(score);
                let old_scale = (running_max - new_max).exp();
                let new_scale = (score - new_max).exp();
                normalizer = normalizer * old_scale + new_scale;
                let value_start = (key_token * spec.key_value_heads + key_head) * spec.head_dim;
                for dimension in 0..spec.head_dim {
                    output_row[dimension] = output_row[dimension] * old_scale
                        + new_scale * value[value_start + dimension];
                }
                running_max = new_max;
            }
            for value in output_row {
                *value /= normalizer;
            }
        }
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug)]
pub struct Conv3dSpec {
    pub stride: [usize; 3],
    pub padding: [usize; 3],
    pub dilation: [usize; 3],
    pub groups: usize,
}

impl Default for Conv3dSpec {
    fn default() -> Self {
        Self {
            stride: [1; 3],
            padding: [0; 3],
            dilation: [1; 3],
            groups: 1,
        }
    }
}

pub fn conv3d(
    pool: &ComputePool,
    input: &[f32],
    input_shape: [usize; 4],
    weight: &[f32],
    weight_shape: [usize; 5],
    stride: [usize; 3],
) -> Result<Vec<f32>, String> {
    conv3d_with_options(
        pool,
        input,
        input_shape,
        weight,
        weight_shape,
        None,
        Conv3dSpec {
            stride,
            ..Conv3dSpec::default()
        },
    )
    .map(|(output, _)| output)
}

pub fn conv3d_with_options(
    pool: &ComputePool,
    input: &[f32],
    input_shape: [usize; 4],
    weight: &[f32],
    weight_shape: [usize; 5],
    bias: Option<&[f32]>,
    spec: Conv3dSpec,
) -> Result<(Vec<f32>, [usize; 4]), String> {
    let [input_channels, depth, height, width] = input_shape;
    let [output_channels, weight_channels, kernel_depth, kernel_height, kernel_width] =
        weight_shape;
    if spec.groups == 0
        || input_channels % spec.groups != 0
        || output_channels % spec.groups != 0
        || weight_channels != input_channels / spec.groups
        || spec.stride.contains(&0)
        || spec.dilation.contains(&0)
        || bias.is_some_and(|bias| bias.len() != output_channels)
        || input.len() != checked_len("DreamX Conv3D input", &input_shape)?
        || weight.len() != checked_len("DreamX Conv3D weight", &weight_shape)?
    {
        return Err("Invalid DreamX Conv3D tensors".into());
    }
    let output_depth = conv_output(
        depth,
        kernel_depth,
        spec.stride[0],
        spec.padding[0],
        spec.dilation[0],
    )?;
    let output_height = conv_output(
        height,
        kernel_height,
        spec.stride[1],
        spec.padding[1],
        spec.dilation[1],
    )?;
    let output_width = conv_output(
        width,
        kernel_width,
        spec.stride[2],
        spec.padding[2],
        spec.dilation[2],
    )?;
    let output_shape = [output_channels, output_depth, output_height, output_width];
    let output_len = checked_len("DreamX Conv3D output", &output_shape)?;
    let mut output = vec![0.0; output_len];
    #[cfg(feature = "vulkan")]
    if crate::ops::kernel::vulkan::try_conv3d(
        input,
        weight,
        bias,
        &mut output,
        input_shape,
        weight_shape,
        output_shape,
        spec,
    ) {
        // The device has been seen to return a convolution that disagrees
        // with the CPU kernel, so the result is checked before it is used.
        // A retry of the same call usually returns the right numbers.
        let mut attempts = 0;
        while attempts < GPU_RETRY_LIMIT
            && !conv3d_matches_cpu(
                input,
                weight,
                bias,
                &output,
                input_shape,
                weight_shape,
                output_shape,
                spec,
            )
        {
            attempts += 1;
            if !crate::ops::kernel::vulkan::try_conv3d(
                input,
                weight,
                bias,
                &mut output,
                input_shape,
                weight_shape,
                output_shape,
                spec,
            ) {
                break;
            }
        }
        if attempts > 0 {
            eprintln!("[GPU-VERIFY] a convolution was wrong and {attempts} retries fixed it");
        }
        if conv3d_matches_cpu(
            input,
            weight,
            bias,
            &output,
            input_shape,
            weight_shape,
            output_shape,
            spec,
        ) {
            return Ok((output, output_shape));
        }
        // The device keeps returning a wrong convolution, so the CPU kernel
        // computes the whole thing.
        let _scope = crate::core::thread_pool::ComputePool::disable_gpu_matmul_for_scope();
        conv3d_cpu(
            pool,
            input,
            weight,
            bias,
            &mut output,
            input_shape,
            weight_shape,
            output_shape,
            spec,
        );
        return Ok((output, output_shape));
    }
    conv3d_cpu(
        pool,
        input,
        weight,
        bias,
        &mut output,
        input_shape,
        weight_shape,
        output_shape,
        spec,
    );
    Ok((output, output_shape))
}

/// How many elements of a convolution output are checked against the CPU
/// kernel.
const GPU_CONV_CHECK_SAMPLES: usize = 65_536;

/// How many elements of every head of an attention result are checked
/// against the CPU kernel.
const GPU_ATTENTION_CHECK_ELEMENTS: usize = 8;

/// How many consecutive dimensions of every head are checked as well.
const GPU_ATTENTION_CHECK_BLOCK: usize = 16;

/// The largest relative difference between a device convolution and the CPU
/// kernel over the sampled elements.
#[allow(clippy::too_many_arguments)]
fn conv3d_matches_cpu(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &[f32],
    input_shape: [usize; 4],
    weight_shape: [usize; 5],
    output_shape: [usize; 4],
    spec: Conv3dSpec,
) -> bool {
    let output_len = match checked_len("DreamX Conv3D output", &output_shape) {
        Ok(len) => len,
        Err(_) => return false,
    };
    if output.len() != output_len || output_len == 0 {
        return false;
    }
    // The scale of the whole output sets the tolerance, because the two
    // implementations sum in a different order.
    let scale = output
        .iter()
        .fold(0.0f32, |peak, value| peak.max(value.abs()))
        .max(1.0);
    let tolerance = 1.0e-4 * scale;
    if std::env::var_os("RMI_GPU_VERIFY_TRACE").is_some() {
        eprintln!(
            "[GPU-VERIFY] conv check 输出长度={output_len} 形状={output_shape:?} 采样={GPU_CONV_CHECK_SAMPLES} 容差={tolerance:.3e}"
        );
    }
    let reference = vec![0.0f32; output_len];
    for sample in 0..GPU_CONV_CHECK_SAMPLES {
        // The golden ratio spreads the samples over the whole output
        // whatever its size is.
        let index =
            ((sample as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) % output_len as u64) as usize;
        let value = conv3d_element(
            input,
            weight,
            bias,
            &reference,
            index,
            input_shape,
            weight_shape,
            output_shape,
            spec,
        );
        if (value - output[index]).abs() > tolerance {
            if std::env::var_os("RMI_GPU_VERIFY_TRACE").is_some() {
                eprintln!(
                    "[GPU-VERIFY-SHAPE] conv 采样命中 index={index} gpu={:.6e} cpu={:.6e} 容差={tolerance:.3e} 输出长度={output_len}",
                    output[index], value
                );
            }
            return false;
        }
    }
    true
}

/// One element of the convolution, computed the way the CPU kernel does.
#[allow(clippy::too_many_arguments)]
fn conv3d_element(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    scratch: &[f32],
    output_index: usize,
    input_shape: [usize; 4],
    weight_shape: [usize; 5],
    output_shape: [usize; 4],
    spec: Conv3dSpec,
) -> f32 {
    let [_, depth, height, width] = input_shape;
    let [_, weight_channels, kernel_depth, kernel_height, kernel_width] = weight_shape;
    let [_, output_depth, output_height, output_width] = output_shape;
    let output_channels_per_group = output_shape[0] / spec.groups;
    let ow = output_index % output_width;
    let rest = output_index / output_width;
    let oh = rest % output_height;
    let rest = rest / output_height;
    let od = rest % output_depth;
    let oc = rest / output_depth;
    let group = oc / output_channels_per_group;
    let mut sum = bias.map_or(0.0, |bias| bias[oc]);
    for local_channel in 0..weight_channels {
        let input_channel = group * weight_channels + local_channel;
        for kd in 0..kernel_depth {
            let padded_depth = od * spec.stride[0] + kd * spec.dilation[0];
            if padded_depth < spec.padding[0] {
                continue;
            }
            let input_depth = padded_depth - spec.padding[0];
            if input_depth >= depth {
                continue;
            }
            for kh in 0..kernel_height {
                let padded_height = oh * spec.stride[1] + kh * spec.dilation[1];
                if padded_height < spec.padding[1] {
                    continue;
                }
                let input_height = padded_height - spec.padding[1];
                if input_height >= height {
                    continue;
                }
                let input_row =
                    ((input_channel * depth + input_depth) * height + input_height) * width;
                let weight_row = (((oc * weight_channels + local_channel) * kernel_depth + kd)
                    * kernel_height
                    + kh)
                    * kernel_width;
                for kw in 0..kernel_width {
                    let padded_width = ow * spec.stride[2] + kw * spec.dilation[2];
                    if padded_width < spec.padding[2] {
                        continue;
                    }
                    let input_width = padded_width - spec.padding[2];
                    if input_width < width {
                        sum += input[input_row + input_width] * weight[weight_row + kw];
                    }
                }
            }
        }
    }
    let _ = scratch;
    sum
}

/// The CPU convolution. The device version is checked against a sample of
/// this kernel, so both have to agree on the same arithmetic.
#[allow(clippy::too_many_arguments)]
fn conv3d_cpu(
    pool: &ComputePool,
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    input_shape: [usize; 4],
    weight_shape: [usize; 5],
    output_shape: [usize; 4],
    spec: Conv3dSpec,
) {
    let [_, depth, height, width] = input_shape;
    let [output_channels, weight_channels, kernel_depth, kernel_height, kernel_width] =
        weight_shape;
    let [_, output_depth, output_height, output_width] = output_shape;
    let output_len = checked_len("DreamX Conv3D output", &output_shape).unwrap_or(0);
    let output_address = output.as_mut_ptr() as usize;
    let output_channels_per_group = output_channels / spec.groups;
    let strict = crate::ops::scalar_mode();
    pool.compute(|thread, threads| {
        for output_index in (thread..output_len).step_by(threads) {
            let ow = output_index % output_width;
            let rest = output_index / output_width;
            let oh = rest % output_height;
            let rest = rest / output_height;
            let od = rest % output_depth;
            let oc = rest / output_depth;
            let group = oc / output_channels_per_group;
            let mut sum = if strict {
                0.0
            } else {
                bias.map_or(0.0, |bias| bias[oc])
            };
            for local_channel in 0..weight_channels {
                let input_channel = group * weight_channels + local_channel;
                for kd in 0..kernel_depth {
                    let padded_depth = od * spec.stride[0] + kd * spec.dilation[0];
                    if padded_depth < spec.padding[0] {
                        continue;
                    }
                    let input_depth = padded_depth - spec.padding[0];
                    if input_depth >= depth {
                        continue;
                    }
                    for kh in 0..kernel_height {
                        let padded_height = oh * spec.stride[1] + kh * spec.dilation[1];
                        if padded_height < spec.padding[1] {
                            continue;
                        }
                        let input_height = padded_height - spec.padding[1];
                        if input_height >= height {
                            continue;
                        }
                        let input_row =
                            ((input_channel * depth + input_depth) * height + input_height) * width;
                        let weight_row = (((oc * weight_channels + local_channel) * kernel_depth
                            + kd)
                            * kernel_height
                            + kh)
                            * kernel_width;
                        if strict {
                            for kw in 0..kernel_width {
                                let padded_width = ow * spec.stride[2] + kw * spec.dilation[2];
                                if padded_width >= spec.padding[2] {
                                    let input_width = padded_width - spec.padding[2];
                                    if input_width < width {
                                        sum += input[input_row + input_width]
                                            * weight[weight_row + kw];
                                    }
                                }
                            }
                        } else if spec.dilation[2] == 1 {
                            let base = ow * spec.stride[2];
                            let first = spec.padding[2].saturating_sub(base).min(kernel_width);
                            let last = kernel_width
                                .min(width.saturating_add(spec.padding[2]).saturating_sub(base));
                            if first < last {
                                let input_width = base + first - spec.padding[2];
                                sum += dot_f32(
                                    &input[input_row + input_width
                                        ..input_row + input_width + last - first],
                                    &weight[weight_row + first..weight_row + last],
                                    last - first,
                                );
                            }
                        } else {
                            for kw in 0..kernel_width {
                                let padded_width = ow * spec.stride[2] + kw * spec.dilation[2];
                                if padded_width < spec.padding[2] {
                                    continue;
                                }
                                let input_width = padded_width - spec.padding[2];
                                if input_width < width {
                                    sum += input[input_row + input_width] * weight[weight_row + kw];
                                }
                            }
                        }
                    }
                }
            }
            if strict {
                if let Some(bias) = bias {
                    sum += bias[oc];
                }
            }
            unsafe { (output_address as *mut f32).add(output_index).write(sum) };
        }
    });
}

pub fn conv2d(
    pool: &ComputePool,
    input: &[f32],
    input_shape: [usize; 3],
    weight: &[f32],
    weight_shape: [usize; 4],
    bias: Option<&[f32]>,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Result<(Vec<f32>, [usize; 3]), String> {
    let [channels, height, width] = input_shape;
    let [output_channels, weight_channels, kernel_height, kernel_width] = weight_shape;
    let (output, shape) = conv3d_with_options(
        pool,
        input,
        [channels, 1, height, width],
        weight,
        [
            output_channels,
            weight_channels,
            1,
            kernel_height,
            kernel_width,
        ],
        bias,
        Conv3dSpec {
            stride: [1, stride[0], stride[1]],
            padding: [0, padding[0], padding[1]],
            dilation: [1, dilation[0], dilation[1]],
            groups,
        },
    )?;
    Ok((output, [shape[0], shape[2], shape[3]]))
}

pub fn conv1d(
    pool: &ComputePool,
    input: &[f32],
    input_shape: [usize; 2],
    weight: &[f32],
    weight_shape: [usize; 3],
    bias: Option<&[f32]>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<(Vec<f32>, [usize; 2]), String> {
    let [channels, width] = input_shape;
    let [output_channels, weight_channels, kernel_width] = weight_shape;
    let (output, shape) = conv3d_with_options(
        pool,
        input,
        [channels, 1, 1, width],
        weight,
        [output_channels, weight_channels, 1, 1, kernel_width],
        bias,
        Conv3dSpec {
            stride: [1, 1, stride],
            padding: [0, 0, padding],
            dilation: [1, 1, dilation],
            groups,
        },
    )?;
    Ok((output, [shape[0], shape[3]]))
}

pub fn depthwise_conv2d(
    pool: &ComputePool,
    input: &[f32],
    input_shape: [usize; 3],
    weight: &[f32],
    weight_shape: [usize; 4],
    bias: Option<&[f32]>,
    stride: [usize; 2],
    padding: [usize; 2],
) -> Result<(Vec<f32>, [usize; 3]), String> {
    conv2d(
        pool,
        input,
        input_shape,
        weight,
        weight_shape,
        bias,
        stride,
        padding,
        [1; 2],
        input_shape[0],
    )
}

#[derive(Clone, Copy, Debug)]
pub struct ConvTranspose3dSpec {
    pub stride: [usize; 3],
    pub padding: [usize; 3],
    pub output_padding: [usize; 3],
    pub dilation: [usize; 3],
    pub groups: usize,
}

pub fn conv_transpose3d(
    pool: &ComputePool,
    input: &[f32],
    input_shape: [usize; 4],
    weight: &[f32],
    weight_shape: [usize; 5],
    bias: Option<&[f32]>,
    spec: ConvTranspose3dSpec,
) -> Result<(Vec<f32>, [usize; 4]), String> {
    let [input_channels, depth, height, width] = input_shape;
    let [weight_input_channels, output_channels_per_group, kernel_depth, kernel_height, kernel_width] =
        weight_shape;
    let output_channels = output_channels_per_group
        .checked_mul(spec.groups)
        .ok_or("DreamX transposed-convolution channel overflow")?;
    if spec.groups == 0
        || input_channels != weight_input_channels
        || input_channels % spec.groups != 0
        || spec.stride.contains(&0)
        || spec.dilation.contains(&0)
        || spec
            .output_padding
            .iter()
            .zip(spec.stride)
            .any(|(&output_padding, stride)| output_padding >= stride)
        || bias.is_some_and(|bias| bias.len() != output_channels)
        || input.len() != checked_len("DreamX ConvTranspose3D input", &input_shape)?
        || weight.len() != checked_len("DreamX ConvTranspose3D weight", &weight_shape)?
    {
        return Err("Invalid DreamX ConvTranspose3D tensors".into());
    }
    let output_depth = conv_transpose_output(
        depth,
        kernel_depth,
        spec.stride[0],
        spec.padding[0],
        spec.output_padding[0],
        spec.dilation[0],
    )?;
    let output_height = conv_transpose_output(
        height,
        kernel_height,
        spec.stride[1],
        spec.padding[1],
        spec.output_padding[1],
        spec.dilation[1],
    )?;
    let output_width = conv_transpose_output(
        width,
        kernel_width,
        spec.stride[2],
        spec.padding[2],
        spec.output_padding[2],
        spec.dilation[2],
    )?;
    let output_shape = [output_channels, output_depth, output_height, output_width];
    let output_len = checked_len("DreamX ConvTranspose3D output", &output_shape)?;
    let mut output = vec![0.0; output_len];
    let output_address = output.as_mut_ptr() as usize;
    let input_channels_per_group = input_channels / spec.groups;
    pool.compute(|thread, threads| {
        for output_index in (thread..output_len).step_by(threads) {
            let ow = output_index % output_width;
            let rest = output_index / output_width;
            let oh = rest % output_height;
            let rest = rest / output_height;
            let od = rest % output_depth;
            let oc = rest / output_depth;
            let group = oc / output_channels_per_group;
            let output_channel = oc % output_channels_per_group;
            let mut sum = bias.map_or(0.0, |bias| bias[oc]);
            for local_input_channel in 0..input_channels_per_group {
                let input_channel = group * input_channels_per_group + local_input_channel;
                for kd in 0..kernel_depth {
                    let Some(input_depth) = transpose_input_index(
                        od,
                        kd,
                        depth,
                        spec.stride[0],
                        spec.padding[0],
                        spec.dilation[0],
                    ) else {
                        continue;
                    };
                    for kh in 0..kernel_height {
                        let Some(input_height) = transpose_input_index(
                            oh,
                            kh,
                            height,
                            spec.stride[1],
                            spec.padding[1],
                            spec.dilation[1],
                        ) else {
                            continue;
                        };
                        for kw in 0..kernel_width {
                            let Some(input_width) = transpose_input_index(
                                ow,
                                kw,
                                width,
                                spec.stride[2],
                                spec.padding[2],
                                spec.dilation[2],
                            ) else {
                                continue;
                            };
                            let input_index = (((input_channel * depth + input_depth) * height
                                + input_height)
                                * width)
                                + input_width;
                            let weight_index = ((((input_channel * output_channels_per_group
                                + output_channel)
                                * kernel_depth
                                + kd)
                                * kernel_height
                                + kh)
                                * kernel_width)
                                + kw;
                            sum += input[input_index] * weight[weight_index];
                        }
                    }
                }
            }
            unsafe { (output_address as *mut f32).add(output_index).write(sum) };
        }
    });
    Ok((output, output_shape))
}

fn conv_output(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize, String> {
    let effective_kernel = dilation
        .checked_mul(kernel.saturating_sub(1))
        .and_then(|value| value.checked_add(1))
        .ok_or("DreamX convolution kernel overflow")?;
    let padded = padding
        .checked_mul(2)
        .and_then(|padding| input.checked_add(padding))
        .ok_or("DreamX convolution padding overflow")?;
    if kernel == 0 || stride == 0 || dilation == 0 || padded < effective_kernel {
        return Err("Invalid DreamX convolution geometry".into());
    }
    Ok((padded - effective_kernel) / stride + 1)
}

fn conv_transpose_output(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
) -> Result<usize, String> {
    if input == 0 || kernel == 0 || stride == 0 || dilation == 0 || output_padding >= stride {
        return Err("Invalid DreamX transposed-convolution geometry".into());
    }
    let length = (input - 1)
        .checked_mul(stride)
        .and_then(|value| value.checked_add(dilation.checked_mul(kernel - 1)?))
        .and_then(|value| value.checked_add(output_padding))
        .and_then(|value| value.checked_add(1))
        .ok_or("DreamX transposed-convolution shape overflow")?;
    length
        .checked_sub(padding.checked_mul(2).ok_or("DreamX padding overflow")?)
        .filter(|&length| length > 0)
        .ok_or_else(|| "Invalid DreamX transposed-convolution output shape".into())
}

fn transpose_input_index(
    output: usize,
    kernel: usize,
    input_length: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Option<usize> {
    let padded_output = output.checked_add(padding)?;
    let kernel_offset = kernel.checked_mul(dilation)?;
    let numerator = padded_output.checked_sub(kernel_offset)?;
    if numerator % stride != 0 {
        return None;
    }
    let input = numerator / stride;
    (input < input_length).then_some(input)
}

#[cfg(test)]
mod tests {
    use super::projection_matches_cpu;

    fn bf16(value: f32) -> [u8; 2] {
        ((value.to_bits() >> 16) as u16).to_le_bytes()
    }

    /// A device projection that the CPU kernel agrees with has to pass the
    /// check, and one that came back with a stale row has to fail it.
    #[test]
    fn a_broken_convolution_fails_the_convolution_check() {
        let input_shape = [2usize, 1, 4, 4];
        let weight_shape = [3usize, 2, 3, 3, 3];
        let output_shape = [3usize, 1, 4, 4];
        let spec = Conv3dSpec {
            stride: [1, 1, 1],
            padding: [1, 1, 1],
            dilation: [1, 1, 1],
            groups: 1,
        };
        let input: Vec<f32> = (0..input_shape.iter().product())
            .map(|i| (i % 7) as f32 / 3.0 - 1.0)
            .collect();
        let weight: Vec<f32> = (0..weight_shape.iter().product())
            .map(|i| (i % 5) as f32 / 4.0 - 0.5)
            .collect();
        let bias: Vec<f32> = vec![0.25, -0.5, 0.75];
        let mut output = vec![0.0f32; output_shape.iter().product()];
        let pool = ComputePool::new(1);
        super::conv3d_cpu(
            &pool,
            &input,
            &weight,
            Some(&bias),
            &mut output,
            input_shape,
            weight_shape,
            output_shape,
            spec,
        );
        assert!(super::conv3d_matches_cpu(
            &input,
            &weight,
            Some(&bias),
            &output,
            input_shape,
            weight_shape,
            output_shape,
            spec,
        ));
        // A slice of the output that came back wrong is what a broken
        // dispatch looks like.
        let mut broken = output.clone();
        broken[0] += 10.0;
        assert!(!super::conv3d_matches_cpu(
            &input,
            &weight,
            Some(&bias),
            &broken,
            input_shape,
            weight_shape,
            output_shape,
            spec,
        ));
    }

    fn a_stale_row_fails_the_projection_check() {
        let n_in = 64usize;
        let n_out = 32usize;
        let rows = 8usize;
        let mut weight = vec![0u8; n_out * n_in * 2];
        for column in 0..n_out {
            for k in 0..n_in {
                let value = ((column * 7 + k * 13) % 11) as f32 / 8.0 - 0.5;
                weight[(column * n_in + k) * 2..(column * n_in + k) * 2 + 2]
                    .copy_from_slice(&bf16(value));
            }
        }
        let input: Vec<f32> = (0..rows * n_in)
            .map(|i| (i % 17) as f32 / 4.0 - 2.0)
            .collect();
        let mut output = vec![0.0f32; rows * n_out];
        let pool = ComputePool::new(4);
        for row in 0..rows {
            for column in 0..n_out {
                output[row * n_out + column] = super::dot_bf16_f32(
                    &weight[column * n_in * 2..(column + 1) * n_in * 2],
                    &input[row * n_in..(row + 1) * n_in],
                );
            }
        }
        assert!(projection_matches_cpu(
            &pool, &weight, &input, &output, rows, n_in, n_out
        ));
        // A row that came back zero is what a stale activation looks like.
        let mut stale = output.clone();
        for column in 0..n_out {
            stale[3 * n_out + column] = 0.0;
        }
        assert!(!projection_matches_cpu(
            &pool, &weight, &input, &stale, rows, n_in, n_out
        ));
        // A row that came back with different numbers is caught as well.
        let mut wrong = output.clone();
        for column in 0..n_out {
            wrong[5 * n_out + column] += 3.0;
        }
        assert!(!projection_matches_cpu(
            &pool, &weight, &input, &wrong, rows, n_in, n_out
        ));
        // The numbers of an earlier call, which is what the device was seen
        // to return, are caught as well.
        let mut earlier = output.clone();
        for row in 0..rows {
            for column in 0..n_out {
                earlier[row * n_out + column] *= 0.5;
            }
        }
        assert!(!projection_matches_cpu(
            &pool, &weight, &input, &earlier, rows, n_in, n_out
        ));
    }

    use super::*;
    use crate::core::thread_pool::ComputePool;

    fn values(len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| ((index * 17 % 29) as f32 - 14.0) / 11.0)
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value {index}: {actual} != {expected}"
            );
        }
    }

    fn conv3d_reference(
        input: &[f32],
        input_shape: [usize; 4],
        weight: &[f32],
        weight_shape: [usize; 5],
        stride: [usize; 3],
    ) -> Vec<f32> {
        let [channels, depth, height, width] = input_shape;
        let [output_channels, weight_channels, kernel_depth, kernel_height, kernel_width] =
            weight_shape;
        assert_eq!(channels, weight_channels);
        let output_depth = (depth - kernel_depth) / stride[0] + 1;
        let output_height = (height - kernel_height) / stride[1] + 1;
        let output_width = (width - kernel_width) / stride[2] + 1;
        let mut output = vec![0.0; output_channels * output_depth * output_height * output_width];
        for oc in 0..output_channels {
            for od in 0..output_depth {
                for oh in 0..output_height {
                    for ow in 0..output_width {
                        let mut sum = 0.0;
                        for ic in 0..channels {
                            for kd in 0..kernel_depth {
                                for kh in 0..kernel_height {
                                    for kw in 0..kernel_width {
                                        let input_index = (((ic * depth + od * stride[0] + kd)
                                            * height
                                            + oh * stride[1]
                                            + kh)
                                            * width)
                                            + ow * stride[2]
                                            + kw;
                                        let weight_index =
                                            ((((oc * channels + ic) * kernel_depth + kd)
                                                * kernel_height
                                                + kh)
                                                * kernel_width)
                                                + kw;
                                        sum += input[input_index] * weight[weight_index];
                                    }
                                }
                            }
                        }
                        let output_index =
                            (((oc * output_depth + od) * output_height + oh) * output_width) + ow;
                        output[output_index] = sum;
                    }
                }
            }
        }
        output
    }

    #[test]
    fn conv3d_matches_scalar_reference() {
        let input = values(2 * 3 * 4 * 5);
        let weight = values(4 * 2 * 3 * 3 * 3);
        let expected = conv3d_reference(&input, [2, 3, 4, 5], &weight, [4, 2, 3, 3, 3], [1, 1, 1]);
        let actual = conv3d(
            &ComputePool::new(2),
            &input,
            [2, 3, 4, 5],
            &weight,
            [4, 2, 3, 3, 3],
            [1, 1, 1],
        )
        .unwrap();
        assert_close(&actual, &expected, 1e-5);
    }

    #[test]
    fn online_attention_matches_materialized_softmax() {
        let q = values(2 * 2 * 4);
        let k = values(3 * 2 * 4);
        let v = values(3 * 2 * 4);
        let spec = AttentionSpec {
            query_tokens: 2,
            key_tokens: 3,
            query_heads: 2,
            key_value_heads: 2,
            head_dim: 4,
            causal: false,
            scale: 0.5,
        };
        let expected = attention_scalar(&q, &k, &v, spec).unwrap();
        let actual = attention_online(&q, &k, &v, spec).unwrap();
        assert_close(&actual, &expected, 1e-5);
    }

    #[test]
    fn scalar_attention_bias_is_materialized_in_key_order() {
        if !crate::ops::scalar_mode() {
            return;
        }
        let q = values(2 * 1 * 3);
        let k = values(3 * 1 * 3);
        let v = values(3 * 1 * 3);
        let bias = [0.25, -0.5, 0.75, 0.0, 0.125, -0.25];
        let spec = AttentionSpec {
            query_tokens: 2,
            key_tokens: 3,
            query_heads: 1,
            key_value_heads: 1,
            head_dim: 3,
            causal: false,
            scale: 0.5,
        };
        fn scalar_dot(a: &[f32], b: &[f32], n: usize) -> f32 {
            let mut sum = 0.0f32;
            for index in 0..n {
                sum += a[index] * b[index];
            }
            sum
        }
        let actual = attention_online_with_bias(&q, &k, &v, &bias, spec).unwrap();
        let expected =
            attention_scalar_with_bias(&q, &k, &v, Some(&bias), spec, scalar_dot).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn scalar_rms_norm_uses_f32_sum_of_squares() {
        if !crate::ops::scalar_mode() {
            return;
        }
        let input = [3.0f32, -2.0, 1.5, 0.25];
        let weight = [1.0f32, 0.5, 2.0, -1.0];
        let actual = rms_norm_rows(&input, 1, &weight, 1e-6).unwrap();
        let mut sum = 0.0f32;
        for value in input {
            sum += value * value;
        }
        let scale = 1.0f32 / (sum / input.len() as f32 + 1e-6).sqrt();
        let expected: Vec<f32> = input
            .iter()
            .zip(weight)
            .map(|(&value, weight)| value * scale * weight)
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn scalar_conv3d_adds_bias_after_sequential_kernel_sum() {
        if !crate::ops::scalar_mode() {
            return;
        }
        let input = [1.25f32, -2.0, 3.5];
        let weight = [0.5f32, 1.25, -0.75];
        let bias = [0.125f32];
        let actual = conv3d_with_options(
            &ComputePool::new(1),
            &input,
            [1, 1, 1, 3],
            &weight,
            [1, 1, 1, 1, 3],
            Some(&bias),
            Conv3dSpec::default(),
        )
        .unwrap()
        .0;
        let mut expected = 0.0f32;
        for (&value, coefficient) in input.iter().zip(weight) {
            expected += value * coefficient;
        }
        expected += bias[0];
        assert_eq!(actual, [expected]);
    }

    #[test]
    fn scalar_bf16_dot_rounds_storage_then_accumulates_in_order() {
        if !crate::ops::scalar_mode() {
            return;
        }
        let input = values(19);
        let bytes: Vec<u8> = values(19)
            .iter()
            .flat_map(|&value| crate::ops::f32_to_bf16(value).to_le_bytes())
            .collect();
        let mut expected = 0.0f32;
        for (index, chunk) in bytes.chunks_exact(2).enumerate() {
            expected += crate::ops::bf16_to_f32(u16::from_le_bytes(chunk.try_into().unwrap()))
                * input[index];
        }
        assert_eq!(dot_bf16_f32(&bytes, &input), expected);
    }

    #[test]
    fn linear_applies_rows_and_bias() {
        let raw: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let mut weight =
            Weight::from_quantized(QuantizedTensor::from_bytes(&raw, GGMLType::F32, 3, 2));
        weight.n_in = 3;
        weight.n_out = 2;
        let linear = Linear::from_weight(weight, Some(vec![0.5, -0.5])).unwrap();
        let output = linear
            .forward(&ComputePool::new(2), &[1.0, 0.0, -1.0, 2.0, 1.0, 0.0], 2)
            .unwrap();
        assert_close(&output, &[-1.5, -2.5, 4.5, 12.5], 1e-6);
    }

    #[test]
    fn bf16_dot_dispatch_matches_scalar_values() {
        let input = values(19);
        let weight = values(19);
        let bytes: Vec<u8> = weight
            .iter()
            .flat_map(|&value| crate::ops::f32_to_bf16(value).to_le_bytes())
            .collect();
        let rounded: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|bytes| crate::ops::bf16_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())))
            .collect();
        let expected = rounded.iter().zip(&input).map(|(a, b)| a * b).sum::<f32>();
        let actual = dot_bf16_f32(&bytes, &input);
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }

    #[test]
    fn transposed_convolution_uses_pytorch_weight_layout() {
        let (output, shape) = conv_transpose3d(
            &ComputePool::new(2),
            &[1.0, 2.0],
            [1, 1, 1, 2],
            &[1.0, 10.0, 100.0],
            [1, 1, 1, 1, 3],
            None,
            ConvTranspose3dSpec {
                stride: [1, 1, 2],
                padding: [0, 0, 1],
                output_padding: [0, 0, 1],
                dilation: [1; 3],
                groups: 1,
            },
        )
        .unwrap();
        assert_eq!(shape, [1, 1, 1, 4]);
        assert_eq!(output, [10.0, 102.0, 20.0, 200.0]);
    }
}
