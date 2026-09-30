//! JEV adapters for backends that cannot be expressed by the
//! [`crate::app::jev::single::JevScorer`] trait.
//!
//! `JevScorer` scores by reading the LM head's next-token logit for a label's
//! token id (`compute_jev_result`), which requires a causal decoder with a
//! chat template and single-token A..Z labels. A backend that violates any of
//! those — cosine in a projection space, a logit per marker position — plugs in
//! here instead, and only has to produce a `Vec<JevResult>` in the shared
//! shape so the CLI and HTTP surfaces do not fork.
//!
//! `src/app/jev/clm.rs` is the earlier instance of the same pattern and has not
//! been moved: it predates this layout and moving it would churn code that is
//! already merged and verified.

pub mod gliner2;
