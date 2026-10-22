//! Occamy-1.0 (`qwen35moe`) — a Qwen3.5 hybrid trunk whose FFN is replaced by
//! a 256-expert sparse MoE.
//!
//! Uses the shared `qwen35moe` loader, routed/shared expert kernels and session.
//! Keeping one implementation also keeps text and image generation on the
//! same numerical path.

pub mod run;
