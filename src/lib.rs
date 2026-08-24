pub mod app;
pub use app::open_or_exit;
pub use app::run_or_exit;
pub use models::qwen3::{Qwen3LayerWeights, get_f32_tensor};
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

pub use core::loader::{model_config_from_source, GGUFLoader};
pub use core::memory::{BlockAllocator, KVCacheView, MemoryArena, PagedKVBlock};
pub use core::model::{ModelGraph, QuantizedLinear};
pub use core::scratchpad::{ExecutionScratchpad, KvCache, KvCacheF16, KvCacheF32};
pub use core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
pub use core::thread_pool::ComputePool;
pub use core::tokenizer::{BPETokenizer, EncodeOptions, StreamingDecoder};
pub use core::traits::{ExecContext, Layer, ModelConfig};
pub use format::ggufrs::{
    export_ggufrs, open_model_source, ComponentInfo, ComponentRole, ExportOptions, GgufrsError,
    GgufrsFile, LoadedComponent, SegmentKind, GGUFRS_SEGMENT_ALIGNMENT, GGUFRS_VERSION,
};
pub use format::load_plan::{
    build_load_plan, load_logical_cpu, LoadPlan, LogicalCpuDeviceLoad, LogicalCpuLoad,
    LogicalCpuPlacement, LogicalDevice, Placement, PlacementPolicy, PlacementSlice,
};
pub use models::asr::*;
pub use models::clip_config::{ClipVisionConfig, Qwen35Config};
pub use models::qwen3_multimodal::*;
pub use models::qwen35::{build_qwen35_positions, Qwen35Model};
pub use models::vision::{qwen_smart_resize, VisionEncoder, VisionGrid, VisionScratchpad};

pub use ops::*;
pub use ops::quant::{dequant_weight_q4k, BlockQ8K, QK_K, dequantize_q4_k_weight};
pub use prompt::{
    append_qwen_assistant_prefix, append_qwen_message_tokens, build_hunyuan_chat_prompt,
    build_qwen_chat_prompt, build_simple_prompt, HunyuanMessage, QwenMessage,
};
