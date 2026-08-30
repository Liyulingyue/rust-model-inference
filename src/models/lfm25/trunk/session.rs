//! LFM2 session state types
//!
//! LFM2 has no explicit `Session` struct — the forward loop runs as a single
//! function call. This file holds [`KvCacheFmt`], the cache-precision enum
//! that callers thread through `run_inference`.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheFmt {
    F16,
    F32,
}