//! Phi-3 / Phi-4 source wrapper.
//!
//! Phi-4 stores attention and FFN as *fused* tensors:
//! - `blk.{l}.attn_qkv.weight` of shape `[n_embd, n_embd_q + 2 * n_embd_gqa]`
//! - `blk.{l}.ffn_up.weight` of shape `[n_embd, 2 * n_ff]`
//!
//! The llama trunk expects the standard llama layout (separate `attn_q`,
//! `attn_k`, `attn_v`, `ffn_gate`, `ffn_up` tensors per layer). [`Phi3Source`]
//! physically splits the fused tensors once at construction time and serves
//! the resulting per-projection byte slices under their canonical names, so
//! the trunk code can stay llama-shaped.
//!
//! Pre-computed YaRN rope tables (`rope_factors_long.weight`,
//! `rope_factors_short.weight`) are passed through unchanged. For
//! positions 0..`rope.scaling.original_context_length` (= 4096 for Phi-4)
//! YaRN reduces to the plain RoPE recurrence with the published `freq_base`,
//! so the trunk's plain `apply_rope` is bit-exact for short prompts. Long
//! prompts (> 4096 tokens) would need YaRN scaling folded into the RoPE
//! recurrence — left as a follow-up.
//!
//! Memory cost: one full split per layer. For Phi-4-mini (32 layers, Q4_K_M)
//! that's ~290 MB of `Vec<u8>` allocations held for the source's lifetime.
//! For a 3.8B model loaded once per inference this is acceptable.

use std::collections::HashMap;
use std::sync::Arc;

use crate::core::tensor::{MetaValue, TensorInfo, TensorSource};

/// Wrap `inner` so that `arch="phi3"` fused QKV / fused gate-up tensors
/// become per-projection tensors the llama trunk can read directly.
///
/// `n_embd_q` and `n_embd_gqa` and `n_ff` are taken from the metadata the
/// caller has already validated. The split happens once on construction;
/// subsequent `tensor_info` / `tensor_slice` calls are HashMap lookups.
pub struct Phi3Source<S: TensorSource + ?Sized> {
    inner: Arc<S>,
    /// Pre-split tensors: `(TensorInfo, raw bytes)` keyed by the canonical
    /// llama-style name. Holds the synthetic `attn_q` / `attn_k` / etc.
    /// views; non-fused tensors are passed through to `inner`.
    splits: HashMap<String, (TensorInfo, Vec<u8>)>,
}

impl<S: TensorSource + ?Sized> Phi3Source<S> {
    pub fn new(inner: Arc<S>, n_embd_q: usize, n_embd_gqa: usize, n_ff: usize) -> Self {
        let mut splits = HashMap::new();
        // Iterate over the inner source's tensors and split the fused ones.
        // We assume the caller has already checked that `arch = "phi3"`.
        //
        // We discover tensor names via `tensor_info`; pass-through tensors
        // (token_embd, output, norms, rope_factors_*) we copy byte-for-byte.
        // Fused tensors we split.
        //
        // We don't have a list of tensor names on the trait, so we use the
        // standard phi3 naming and probe each layer.
        let n_layer: usize = inner
            .metadata("phi3.block_count")
            .and_then(|v| v.to_u64())
            .map(|v| v as usize)
            .unwrap_or(0);
        let embd: usize = inner
            .metadata("phi3.embedding_length")
            .and_then(|v| v.to_u64())
            .map(|v| v as usize)
            .unwrap_or(0);

        for l in 0..n_layer {
            // --- Split fused QKV ---
            //
            // GGUF reports the fused `attn_qkv.weight` tensor as
            //   dims = [n_embd_in, n_embd_q + 2*n_embd_gqa]
            // but the BYTES are stored in matmul-friendly order:
            //   [n_embd_q + 2*n_embd_gqa] rows × [n_embd_in] elements/row
            // i.e. each storage row is a full `n_embd_in`-element column
            // of the math matrix. The matmul kernel reads `n_out * row_bytes`
            // bytes total, where `row_bytes = n_in * bytes_per_block`. So the
            // split must carve out contiguous *rows* of the storage, not
            // contiguous *columns* of the math matrix.
            //
            // (Earlier versions of this split by column prefix/middle/suffix,
            // producing `[n_in, n_out]` byte order. The matmul kernel reads
            // `[n_out, n_in]` storage, so the resulting weights computed the
            // transpose of the intended matmul — for square Q this is silent
            // because the matrix is its own transpose up to block shuffling,
            // but K/V/ffn_gate/ffn_up produced completely wrong activations.)
            let qkv_name = format!("blk.{l}.attn_qkv.weight");
            if let Some(info) = inner.tensor_info(&qkv_name) {
                if let Some(bytes) = inner.tensor_slice(&qkv_name) {
                    let ggml_type = info.ggml_type;
                    let n_in = info.dims[0] as usize;
                    let total_cols = info.dims[1] as usize;
                    if total_cols == n_embd_q + 2 * n_embd_gqa {
                        let (block_size, type_size) = ggml_type.type_traits();
                        // Storage row length in bytes (= n_in * bytes_per_block).
                        let row_bytes = (n_in / block_size) * type_size;
                        // QKV storage rows = n_embd_q + 2*n_embd_gqa (each
                        // one represents one output row of the math matrix
                        // with `n_in` input elements).
                        let q_bytes = bytes[..n_embd_q * row_bytes].to_vec();
                        let k_bytes = bytes
                            [n_embd_q * row_bytes..(n_embd_q + n_embd_gqa) * row_bytes]
                            .to_vec();
                        let v_bytes = bytes
                            [(n_embd_q + n_embd_gqa) * row_bytes..total_cols * row_bytes]
                            .to_vec();
                        splits.insert(
                            format!("blk.{l}.attn_q.weight"),
                            (
                                TensorInfo {
                                    name: format!("blk.{l}.attn_q.weight"),
                                    dims: vec![n_in as u64, n_embd_q as u64],
                                    ggml_type,
                                    offset: 0,
                                },
                                q_bytes,
                            ),
                        );
                        splits.insert(
                            format!("blk.{l}.attn_k.weight"),
                            (
                                TensorInfo {
                                    name: format!("blk.{l}.attn_k.weight"),
                                    dims: vec![n_in as u64, n_embd_gqa as u64],
                                    ggml_type,
                                    offset: 0,
                                },
                                k_bytes,
                            ),
                        );
                        splits.insert(
                            format!("blk.{l}.attn_v.weight"),
                            (
                                TensorInfo {
                                    name: format!("blk.{l}.attn_v.weight"),
                                    dims: vec![n_in as u64, n_embd_gqa as u64],
                                    ggml_type,
                                    offset: 0,
                                },
                                v_bytes,
                            ),
                        );
                    }
                }
            }

            // --- Split fused gate+up FFN ---
            //
            // Same convention as QKV: storage is `[2*n_ff] rows × [n_in] elems`,
            // not `[n_in] rows × [2*n_ff] elems`. Split by rows: first `n_ff`
            // rows are the gate weights, next `n_ff` rows are the up weights.
            let ffn_up_name = format!("blk.{l}.ffn_up.weight");
            if let Some(info) = inner.tensor_info(&ffn_up_name) {
                if let Some(bytes) = inner.tensor_slice(&ffn_up_name) {
                    let ggml_type = info.ggml_type;
                    let n_in = info.dims[0] as usize;
                    let total_cols = info.dims[1] as usize;
                    if total_cols == 2 * n_ff {
                        let (block_size, type_size) = ggml_type.type_traits();
                        let row_bytes = (n_in / block_size) * type_size;
                        let gate_bytes = bytes[..n_ff * row_bytes].to_vec();
                        let up_bytes = bytes[n_ff * row_bytes..total_cols * row_bytes].to_vec();
                        splits.insert(
                            format!("blk.{l}.ffn_gate.weight"),
                            (
                                TensorInfo {
                                    name: format!("blk.{l}.ffn_gate.weight"),
                                    dims: vec![n_in as u64, n_ff as u64],
                                    ggml_type,
                                    offset: 0,
                                },
                                gate_bytes,
                            ),
                        );
                        splits.insert(
                            format!("blk.{l}.ffn_up.weight"),
                            (
                                TensorInfo {
                                    name: format!("blk.{l}.ffn_up.weight"),
                                    dims: vec![n_in as u64, n_ff as u64],
                                    ggml_type,
                                    offset: 0,
                                },
                                up_bytes,
                            ),
                        );
                    }
                }
            }
        }
        Self { inner, splits }
    }
}

impl<S: TensorSource + ?Sized> TensorSource for Phi3Source<S> {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.inner.metadata(key)
    }

    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        if let Some((info, _)) = self.splits.get(name) {
            return Some(info);
        }
        self.inner.tensor_info(name)
    }

    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        if let Some((_, bytes)) = self.splits.get(name) {
            return Some(bytes.as_slice());
        }
        self.inner.tensor_slice(name)
    }
}

/// Bytes per row for a quantized tensor, computed from the per-row block
/// count. Assumes the standard GGUF convention where rows are stored as a
/// sequence of `dims[0]` blocks (one block per `BLOCK_SIZE` elements).
fn block_row_bytes(ggml_type: crate::core::tensor::GGMLType, n_in: u64) -> usize {
    let (block_size, type_size) = ggml_type.type_traits();
    let n_blocks = (n_in as usize + block_size - 1) / block_size;
    n_blocks * type_size
}

/// Per-block encoded size for a quantized type (the `type_size` half of
/// `type_traits()`).
fn ggml_type_bytes_per_block(ggml_type: crate::core::tensor::GGMLType) -> usize {
    ggml_type.type_traits().1
}

/// Extract the leading `prefix_bytes` of every row of `bytes`, where each
/// row is `row_bytes` long, and concatenate them into a fresh `Vec<u8>`.
fn extract_block_prefix_per_row(
    bytes: &[u8],
    n_rows: usize,
    row_bytes: usize,
    prefix_bytes: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_rows * prefix_bytes);
    for r in 0..n_rows {
        let start = r * row_bytes;
        out.extend_from_slice(&bytes[start..start + prefix_bytes]);
    }
    out
}

/// Extract the trailing `suffix_bytes` of every row (i.e. the bytes at
/// `row_bytes - suffix_bytes..row_bytes`).
fn extract_block_suffix_per_row(
    bytes: &[u8],
    n_rows: usize,
    row_bytes: usize,
    suffix_bytes: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_rows * suffix_bytes);
    for r in 0..n_rows {
        let end = (r + 1) * row_bytes;
        out.extend_from_slice(&bytes[end - suffix_bytes..end]);
    }
    out
}

/// Extract a middle slice of every row: skip the first `prefix_bytes`,
/// take the next `middle_bytes`.
fn extract_block_middle_per_row(
    bytes: &[u8],
    n_rows: usize,
    row_bytes: usize,
    prefix_bytes: usize,
    middle_bytes: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_rows * middle_bytes);
    for r in 0..n_rows {
        let start = r * row_bytes + prefix_bytes;
        out.extend_from_slice(&bytes[start..start + middle_bytes]);
    }
    out
}
