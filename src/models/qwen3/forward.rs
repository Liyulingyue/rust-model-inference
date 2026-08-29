//! `Qwen3Model::generate` and the ASR-trace variants.
//!
//! `generate` and `generate_asr` are thin wrappers that pick whether the
//! `Qwen3Session` runs with ASR tracing on or off. The real work lives in
//! [`super::session::Qwen3Session::generate_with_asr_trace`].

use super::base::{Qwen3GenerateOptions, Qwen3Input, Qwen3Model};
use super::session::Qwen3Session;
use super::util::{checked_session_capacity, validate_generation};

impl Qwen3Model {
    pub fn generate(
        &self,
        input: Qwen3Input<'_>,
        options: Qwen3GenerateOptions,
    ) -> Result<super::base::Qwen3Generation, String> {
        self.generate_with_asr_trace(input, options, false)
    }

    pub(crate) fn generate_asr(
        &self,
        input: Qwen3Input<'_>,
        options: Qwen3GenerateOptions,
    ) -> Result<super::base::Qwen3Generation, String> {
        self.generate_with_asr_trace(input, options, true)
    }

    fn generate_with_asr_trace(
        &self,
        input: Qwen3Input<'_>,
        options: Qwen3GenerateOptions,
        asr_trace: bool,
    ) -> Result<super::base::Qwen3Generation, String> {
        validate_generation(self, &input, options)?;
        let capacity = checked_session_capacity(
            input.token_ids.len(),
            options.max_new_tokens,
            self.config.n_ctx,
        )?;
        Qwen3Session::new(self, capacity)?.generate_with_asr_trace(input, options, asr_trace)
    }
}