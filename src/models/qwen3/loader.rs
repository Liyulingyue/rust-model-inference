//! `Qwen3Model::from_source` and stateless accessors.
//!
//! `from_source` is the canonical constructor: it reads `Qwen3Config`,
//! loads the per-layer weight stack via `load_layers_static`, and
//! materialises the embedding + output projection tables.
//!
//! Accessors (`config`, `tokenizer`, `pool`, `layers`, `output_norm`)
//! are kept here because they are part of the model-build contract and
//! have no runtime side effects.
//!
//! `embed_tokens` and `text_encode` live here too because they only
//! depend on `self.token_embedding` / `self.config` and never touch the
//! scratchpad or KV cache.

use super::base::{text_encode, Qwen3Config, Qwen3Model};
use super::skeleton::{load_layers_static, Qwen3LayerWeights};
use super::util::{
    check_allocation, checked_product, load_f32_tensor, usize_to_u64, validate_token_ids,
};
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::core::tokenizer::BPETokenizer;
use crate::ops::kernel::{Kernel, QuantizedTensor, Weight};
use std::sync::Arc;

impl Qwen3Model {
    pub fn from_source(
        source: Arc<dyn TensorSource>,
        tokenizer: Arc<BPETokenizer>,
        pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        let config = Qwen3Config::from_source(source.as_ref())?;
        if config.vocab != tokenizer.vocab_size() {
            return Err(format!(
                "{} vocabulary size {} does not match tokenizer vocab {}",
                config.architecture,
                config.vocab,
                tokenizer.vocab_size()
            ));
        }
        let n_embd_q = checked_product("query width", config.n_head, config.n_embd_head_k)?;
        let n_embd_k = checked_product("key width", config.n_head_kv, config.n_embd_head_k)?;
        let n_embd_v = checked_product("value width", config.n_head_kv, config.n_embd_head_v)?;

        let output_norm = load_f32_tensor(
            source.as_ref(),
            "output_norm.weight",
            &[usize_to_u64(config.n_embd, "embedding width")?],
        )?;
        let token_embedding_info = source.tensor_info("token_embd.weight").expect("no token_embd.weight");
        let token_embedding_bytes = source.tensor_slice("token_embd.weight").expect("no embd");
        let token_embedding_bytes_static: &'static [u8] = unsafe { std::mem::transmute(token_embedding_bytes) };
        let token_embedding = Weight::from_quantized(QuantizedTensor::from_bytes(
            token_embedding_bytes_static,
            token_embedding_info.ggml_type,
            config.n_embd,
            config.vocab,
        ));

        let output_info = source.tensor_info("output.weight").unwrap_or(token_embedding_info);
        let output_bytes = source.tensor_slice("output.weight").unwrap_or(token_embedding_bytes);
        let output_bytes_static: &'static [u8] = unsafe { std::mem::transmute(output_bytes) };
        let output = Weight::from_quantized(QuantizedTensor::from_bytes(
            output_bytes_static,
            output_info.ggml_type,
            config.n_embd,
            config.vocab,
        ));

        let layers: Vec<Qwen3LayerWeights<'static>> = load_layers_static(
            Arc::clone(&source),
            config.n_layer,
            config.n_embd,
            checked_product("query width", config.n_head, config.n_embd_head_k)?,
            checked_product("key width", config.n_head_kv, config.n_embd_head_k)?,
            config.n_ff,
            config.n_embd_head_k,
            config.has_qk_norm,
        );

        Ok(Self {
            source,
            tokenizer,
            pool,
            config,
            layers,
            output_norm,
            token_embedding,
            output,
        })
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    pub fn tokenizer(&self) -> &BPETokenizer {
        &self.tokenizer
    }

    pub fn pool(&self) -> Arc<ComputePool> {
        Arc::clone(&self.pool)
    }

    pub fn layers(&self) -> &Vec<Qwen3LayerWeights> {
        &self.layers
    }

    pub fn output_norm(&self) -> &Vec<f32> {
        &self.output_norm
    }

    pub fn embed_tokens(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        validate_token_ids(token_ids, self.config.vocab)?;
        let len = checked_product(
            "token embedding values",
            token_ids.len(),
            self.config.n_embd,
        )?;
        check_allocation("token embeddings", len, std::mem::size_of::<f32>())?;
        let mut embeddings = vec![0.0; len];
        for (row, &token_id) in embeddings
            .chunks_exact_mut(self.config.n_embd)
            .zip(token_ids)
        {
            self.token_embedding.embedding_lookup(token_id, row);
        }
        Ok(embeddings)
    }

    pub fn text_encode(&self, token_ids: &[u32], positions: &[[usize; 4]]) -> Result<Vec<f32>, String> {
        text_encode(self, token_ids, positions)
    }
}