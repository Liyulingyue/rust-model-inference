pub mod app;
pub use app::LayerWeights;
pub use app::open_or_exit;
pub use app::run_or_exit;
pub use app::get_f32_tensor;
pub mod core;
pub mod format;
pub mod models;
pub mod ops;
#[cfg(feature = "parity-trace")]
#[doc(hidden)]
pub mod parity_trace;
pub mod prompt;
#[cfg(feature = "vulkan")]
pub mod vulkan;
#[cfg(feature = "wgpu")]
pub mod wgpu;

pub use asr::*;
pub use clip_config::{ClipVisionConfig, Qwen35Config};
pub use format::ggufrs::{
    export_ggufrs, open_model_source, ComponentInfo, ComponentRole, ExportOptions, GgufrsError,
    GgufrsFile, LoadedComponent, SegmentKind, GGUFRS_SEGMENT_ALIGNMENT, GGUFRS_VERSION,
};
pub use format::load_plan::{
    build_load_plan, load_logical_cpu, LoadPlan, LogicalCpuDeviceLoad, LogicalCpuLoad,
    LogicalCpuPlacement, LogicalDevice, Placement, PlacementPolicy, PlacementSlice,
};
pub use core::loader::{model_config_from_source, GGUFLoader};
pub use core::memory::{BlockAllocator, KVCacheView, MemoryArena, PagedKVBlock};
pub use core::model::{ModelGraph, QuantizedLinear};
pub use core::scratchpad::{ExecutionScratchpad, KvCache, KvCacheF16, KvCacheF32};
pub use core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
pub use core::thread_pool::ComputePool;
pub use core::tokenizer::{BPETokenizer, EncodeOptions, StreamingDecoder};
pub use core::traits::{ExecContext, Layer, ModelConfig};

/// Backward-compatibility facade re-exporting everything that used to live
/// in `src/model.rs`. New code should prefer `crate::core::{tensor,loader,model}`.
#[deprecated(note = "use crate::core::{tensor,loader,model} instead")]
pub mod model {
    pub use crate::core::loader::*;
    pub use crate::core::model::*;
    pub use crate::core::tensor::*;
}

/// Backward-compatibility facade for `src/memory.rs` (Phase 5A).
#[deprecated(note = "use crate::core::memory instead")]
pub mod memory {
    pub use crate::core::memory::*;
}

/// Backward-compatibility facade for `src/thread_pool.rs` (Phase 5A).
#[deprecated(note = "use crate::core::thread_pool instead")]
pub mod thread_pool {
    pub use crate::core::thread_pool::*;
}

/// Backward-compatibility facade for `src/scratchpad.rs` (Phase 5A).
#[deprecated(note = "use crate::core::scratchpad instead")]
pub mod scratchpad {
    pub use crate::core::scratchpad::*;
}

/// Backward-compatibility facade for `src/traits.rs` (Phase 5A).
#[deprecated(note = "use crate::core::traits instead")]
pub mod traits {
    pub use crate::core::traits::*;
}

/// Backward-compatibility facade for `src/tokenizer.rs` (Phase 5A).
#[deprecated(note = "use crate::core::tokenizer instead")]
pub mod tokenizer {
    pub use crate::core::tokenizer::*;
}

/// Backward-compatibility facade for `src/ggufrs.rs` (Phase 5B).
#[deprecated(note = "use crate::format::ggufrs instead")]
pub mod ggufrs {
    pub use crate::format::ggufrs::*;
}

/// Backward-compatibility facade for `src/load_plan.rs` (Phase 5B).
#[deprecated(note = "use crate::format::load_plan instead")]
pub mod load_plan {
    pub use crate::format::load_plan::*;
}

/// Backward-compatibility facade for `src/asr.rs` (Phase 4 prep).
#[deprecated(note = "use crate::models::asr instead")]
pub mod asr {
    pub use crate::models::asr::*;
}

/// Backward-compatibility facade for `src/clip_config.rs` (Phase 4 prep).
#[deprecated(note = "use crate::models::clip_config instead")]
pub mod clip_config {
    pub use crate::models::clip_config::*;
}

/// Backward-compatibility facade for `src/qwen3.rs` (Phase 4 prep).
#[deprecated(note = "use crate::models::qwen3 instead")]
pub mod qwen3 {
    pub use crate::models::qwen3::*;
}

/// Backward-compatibility facade for `src/qwen35.rs` (Phase 4 prep).
#[deprecated(note = "use crate::models::qwen35 instead")]
pub mod qwen35 {
    pub use crate::models::qwen35::*;
}

/// Backward-compatibility facade for `src/qwen3a.rs` (Phase 4 prep).
#[deprecated(note = "use crate::models::qwen3a instead")]
pub mod qwen3a {
    pub use crate::models::qwen3a::*;
}

/// Backward-compatibility facade for `src/vision.rs` (Phase 4 prep).
#[deprecated(note = "use crate::models::vision instead")]
pub mod vision {
    pub use crate::models::vision::*;
}

/// Backward-compatibility facade for `src/pig.rs` (Phase 4 prep).
#[deprecated(note = "use crate::models::diffusion::pig instead")]
pub mod pig {
    pub use crate::models::diffusion::pig::*;
}
pub use ops::*;
pub use prompt::{
    append_qwen_assistant_prefix, append_qwen_message_tokens, build_hunyuan_chat_prompt,
    build_qwen_chat_prompt, build_simple_prompt, HunyuanMessage, QwenMessage,
};
pub use ops::quant::{dequant_weight_q4k, BlockQ8K, QK_K, dequantize_q4_k_weight};

#[deprecated(note = "use crate::ops::quant instead")]
pub mod quant {
    pub use crate::ops::quant::*;
}
pub use qwen3::*;
pub use qwen35::{build_qwen35_positions, Qwen35Model};
pub use pig::{PigConfig, PigModel, PigVAE};
pub use vision::{qwen_smart_resize, VisionEncoder, VisionGrid, VisionScratchpad};
