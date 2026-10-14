use crate::core::tensor::TensorSource;
use crate::models::edge0::weights::load_affine;
use crate::models::qwen35::trunk::config::Qwen35Config;
use crate::ops::kernel::Weight;

/// MoE weights for one Occamy block.
///
/// The routed experts ship as a single fused `ffn_gate_up_exps`
/// `[n_embd, 2*n_ff_exp, n_expert]` tensor, where the first `n_ff_exp` rows are
/// the SwiGLU gate and the rest are the up projection, matching
/// `create_tensor_gate_up_exps` upstream. The shared expert has its own
/// gate/up/down plus a scalar gate that is passed through a sigmoid.
pub struct OccamyMoeWeights<'a> {
    pub router: Weight<'a>,
    pub gate: Vec<Weight<'a>>,
    pub up: Vec<Weight<'a>>,
    pub down: Vec<Weight<'a>>,
    /// `ffn_gate_inp_shexp` projects the token to a single value, which
    /// upstream feeds through a sigmoid and multiplies the shared expert by.
    pub shared_gate: Weight<'a>,
    pub shared_up: Weight<'a>,
    pub shared_down: Weight<'a>,
    pub used: usize,
    /// Router post-processing carried by the metadata; absent here, so the raw
    /// softmax probabilities are used without renormalising or scaling.
    pub hparams: MoeRoutingParams,
}

/// The subset of llama.cpp's  that affects expert routing.
#[derive(Debug, Clone, Copy)]
pub struct MoeRoutingParams {
    pub expert_weights_norm: Option<bool>,
    pub expert_weights_scale: Option<f32>,
}

pub fn load_occamy_moe<'a>(
    source: &'a dyn TensorSource,
    config: &Qwen35Config,
) -> Result<Vec<OccamyMoeWeights<'a>>, String> {
    let count = meta_usize(source, "qwen35moe.expert_count")?;
    let used = meta_usize(source, "qwen35moe.expert_used_count")?;
    let n_ff_exp = meta_usize(source, "qwen35moe.expert_feed_forward_length")?;
    // Not published in the metadata; upstream falls back to the block FFN
    // width, and occamy sets both to 512.
    let n_ff_shexp = source
        .metadata("qwen35moe.expert_shared_feed_forward_length")
        .and_then(|v| v.to_u64())
        .map(|v| v as usize)
        .unwrap_or(n_ff_exp);

    let routing = MoeRoutingParams {
        expert_weights_norm: source.metadata("qwen35moe.expert_weights_norm").and_then(
            |v| match v {
                crate::core::tensor::MetaValue::Bool(b) => Some(*b),
                _ => None,
            },
        ),
        expert_weights_scale: source
            .metadata("qwen35moe.expert_weights_scale")
            .and_then(|v| v.to_f64())
            .map(|v| v as f32),
    };
    if count == 0 || used == 0 || used > count {
        return Err(format!(
            "invalid Occamy MoE contract: {used} of {count} experts"
        ));
    }

    (0..config.n_layer_impl())
        .map(|layer| {
            let prefix = format!("blk.{layer}");
            let router = load_affine(source, &format!("{prefix}.ffn_gate_inp.weight"), None)?;
            if router.n_in != config.n_embd || router.n_out != count {
                return Err(format!(
                    "Occamy router shape mismatch at layer {layer}: {}x{}",
                    router.n_in, router.n_out
                ));
            }

            // This build ships the split `ffn_gate_exps` / `ffn_up_exps`
            // rather than the fused `ffn_gate_up_exps` that upstream also
            // accepts, so load them separately.
            let gate_experts = load_experts_3d(
                source,
                &format!("{prefix}.ffn_gate_exps.weight"),
                count,
                config.n_embd,
                n_ff_exp,
            )?;
            let up_experts = load_experts_3d(
                source,
                &format!("{prefix}.ffn_up_exps.weight"),
                count,
                config.n_embd,
                n_ff_exp,
            )?;

            let down = load_experts_3d(
                source,
                &format!("{prefix}.ffn_down_exps.weight"),
                count,
                n_ff_exp,
                config.n_embd,
            )?;

            let shared_gate =
                load_affine(source, &format!("{prefix}.ffn_gate_inp_shexp.weight"), None)?;
            let shared_up = load_affine(source, &format!("{prefix}.ffn_up_shexp.weight"), None)?;
            let shared_down =
                load_affine(source, &format!("{prefix}.ffn_down_shexp.weight"), None)?;
            if shared_gate.n_in != config.n_embd
                || shared_gate.n_out != 1
                || shared_up.n_in != config.n_embd
                || shared_up.n_out != n_ff_shexp
                || shared_down.n_in != n_ff_shexp
                || shared_down.n_out != config.n_embd
            {
                return Err(format!(
                    "Occamy shared expert shape mismatch at layer {layer}"
                ));
            }

            Ok(OccamyMoeWeights {
                router,
                gate: gate_experts,
                up: up_experts,
                down,
                shared_gate,
                shared_up,
                shared_down,
                used,
                hparams: routing,
            })
        })
        .collect()
}

fn meta_usize(source: &dyn TensorSource, key: &str) -> Result<usize, String> {
    source
        .metadata(key)
        .and_then(|v| v.to_u64())
        .map(|v| v as usize)
        .ok_or_else(|| format!("Missing clip metadata: {key}"))
}

/// Load a plain 3-D `[n_in, n_out, n_expert]` expert tensor.
fn load_experts_3d<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    count: usize,
    n_in: usize,
    n_out: usize,
) -> Result<Vec<Weight<'a>>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("missing {name}"))?;
    if info.dims.len() != 3 || info.dims[2] as usize != count {
        return Err(format!("Occamy expert count mismatch for {name}"));
    }
    (0..count)
        .map(|expert| {
            let w = load_affine(source, name, Some(expert))?;
            if w.n_in != n_in || w.n_out != n_out {
                return Err(format!("Occamy expert shape mismatch for {name}"));
            }
            Ok(w)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// occamy-1.0-Q8_0.gguf, read with the repository's own loader.
    const MODEL: &str = "models/occamy-1.0-GGUF/occamy-1.0-Q8_0.gguf";

    fn meta<'a>(source: &'a dyn TensorSource, key: &str) -> Option<&'a str> {
        source.metadata(key).and_then(|v| v.to_string_val())
    }

    #[test]
    fn occamy_declares_the_moe_contract_we_implement() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("skipping: {MODEL} not present");
            return;
        }
        let path = MODEL;
        let source = match crate::format::ggufrs::open_model_source(
            std::path::Path::new(path),
            crate::format::ggufrs::ComponentRole::Llm,
        ) {
            Ok(s) => s,
            Err(e) => panic!("open {MODEL}: {e}"),
        };
        assert_eq!(
            meta(source.as_ref(), "general.architecture").as_deref(),
            Some("qwen35moe"),
            "occamy must land on the qwen35moe arch, not the config.json name"
        );
        assert_eq!(
            meta_usize(source.as_ref(), "qwen35moe.expert_count").unwrap(),
            256
        );
        assert_eq!(
            meta_usize(source.as_ref(), "qwen35moe.expert_used_count").unwrap(),
            8
        );
        assert_eq!(
            meta_usize(source.as_ref(), "qwen35moe.expert_feed_forward_length").unwrap(),
            512
        );
        // The shared-expert gate is a scalar per token, not a router.
        let info = source
            .tensor_info("blk.0.ffn_gate_inp_shexp.weight")
            .unwrap();
        assert_eq!(info.dims, vec![2048, 1], "shared gate is n_embd -> 1");
    }
}

/// Occamy-1.0 as a hybrid trunk plus per-layer MoE FFN.
pub struct OccamyModel<'a> {
    pub(crate) trunk: crate::models::qwen35::trunk::HybridTrunk<'a>,
    pub(crate) moe: Vec<OccamyMoeWeights<'a>>,
}

impl<'a> OccamyModel<'a> {
    pub fn from_source(source: &'a dyn TensorSource) -> Result<Self, String> {
        let arch = source
            .metadata("general.architecture")
            .and_then(|value| value.to_string_val());
        if arch != Some("qwen35moe") {
            return Err(format!(
                "OccamyModel requires general.architecture=qwen35moe, got {arch:?}"
            ));
        }
        let config = Qwen35Config::from_source_for_arch(source, "qwen35moe")?;
        let trunk = crate::models::qwen35::trunk::HybridTrunk::load_trunk(source, config, false)?;
        let moe = load_occamy_moe(source, &trunk.config)?;
        Ok(Self { trunk, moe })
    }
}

impl<'m> crate::models::qwen35::trunk::session::HybridTrunkModel<'m> for OccamyModel<'m> {
    fn trunk(&self) -> &crate::models::qwen35::trunk::HybridTrunk<'m> {
        &self.trunk
    }

    fn forward_at(
        &mut self,
        n_tokens: usize,
        base_position: usize,
        kv_cache: &mut crate::core::scratchpad::KvCache,
        scratch: &mut crate::models::qwen35::Qwen35Scratchpad,
        pool: &crate::core::thread_pool::ComputePool,
        positions: &[[usize; 4]],
    ) -> Result<Vec<f32>, String> {
        self.trunk.forward_impl(
            n_tokens,
            Some(base_position),
            kv_cache,
            scratch,
            pool,
            positions,
            None,
            Some(&self.moe),
        )
    }

    #[cfg(feature = "vulkan")]
    fn supports_vulkan(&self) -> bool {
        false
    }
}
