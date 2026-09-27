//! Generation runtime types shared by the CLI and the HTTP server.
//!
//! Phase 2-3 of the CLI/HTTP unification
//! (see `docs/develop/TEXT_RUNTIME_UNIFICATION.md`). The per-arch adapters
//! in `app::text::runtime` implement [`TextRuntime`]; the server holds
//! `Box<dyn TextRuntime>` and streams through a [`TokenSink`].
//!
//! Lives under `ops` (not `core`) because it is transport-agnostic glue
//! between model code and I/O, and `ops::*` is already re-exported at the
//! crate root for the server's convenience.

/// What a generation call needs.
#[derive(Clone, Debug)]
pub struct GenerationRequest {
    pub token_ids: Vec<u32>,
    pub max_new_tokens: usize,
    pub sampling: SamplingParams,
}

impl GenerationRequest {
    pub fn new(token_ids: Vec<u32>, max_new_tokens: usize) -> Self {
        Self {
            token_ids,
            max_new_tokens,
            sampling: SamplingParams::default(),
        }
    }
}

/// Sampling inputs. HTTP only fills `temperature` (top_k / top_p /
/// repetition_penalty / seed stay warn-ignored per the unification plan);
/// the llama-family CLI path additionally uses `repetition_penalty`.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: None,
        }
    }
}

/// Result of a generation.
#[derive(Clone, Debug)]
pub struct GeneratedText {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub finish: Finish,
}

/// Why generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// `max_new_tokens` reached.
    Limit,
    /// The runtime hit the model's eos / `im_end` id.
    Eos,
    /// A stop sequence matched.
    Sequence,
    /// The sink returned [`Flow::Stop`] (client disconnected / cancelled).
    Cancelled,
}

/// Continue-or-stop answer from a [`TokenSink`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Stop,
}

/// Receives decoded text as it is produced. Streaming callers (HTTP SSE)
/// write each chunk; batch callers (CLI) accumulate and print.
pub trait TokenSink {
    fn push_text(&mut self, chunk: &str) -> Flow;
}

/// Per-arch generation adapter. The server builds one at startup and calls
/// `generate` per request; the CLI keeps its arch `run_inference` functions.
pub trait TextRuntime: Send {
    /// Architecture string, matching the GGUF `general.architecture`.
    fn arch(&self) -> &str;
    /// Maximum context (prompt + generated) the runtime's KV cache holds.
    /// Used by the server to reject requests that would overflow.
    fn context_length(&self) -> usize;
    /// Generate from `request.token_ids`, streaming decoded chunks to `sink`.
    fn generate(
        &mut self,
        request: &GenerationRequest,
        sink: &mut dyn TokenSink,
    ) -> Result<GeneratedText, String>;
}

/// Shared handle type for the server: a `Mutex` around the per-arch adapter.
/// The server serializes on it (`generation_slot` already does too), and the
/// mutex is what lets [`Qwen35TextRuntime`] hand out `&mut model` per request.
pub type TextRuntimeHandle = std::sync::Mutex<Box<dyn TextRuntime>>;

/// Collects all chunks into one string. Used by tests and by the HTTP
/// non-streaming path.
#[derive(Default)]
pub struct CollectSink {
    pub text: String,
}

impl CollectSink {
    pub fn new() -> Self {
        Self::default()
    }
}

impl TokenSink for CollectSink {
    fn push_text(&mut self, chunk: &str) -> Flow {
        self.text.push_str(chunk);
        Flow::Continue
    }
}

/// Writes every chunk to stdout, flushing after each — the CLI's terminal
/// behaviour, factored out so adapters can reuse it.
pub struct StdoutSink;

impl TokenSink for StdoutSink {
    fn push_text(&mut self, chunk: &str) -> Flow {
        use std::io::Write;
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        let _ = out.write_all(chunk.as_bytes());
        let _ = out.flush();
        Flow::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_sink_accumulates() {
        let mut sink = CollectSink::new();
        assert_eq!(sink.push_text("a"), Flow::Continue);
        assert_eq!(sink.push_text("b"), Flow::Continue);
        assert_eq!(sink.text, "ab");
    }

    #[test]
    fn sampling_defaults_pin_server_conventions() {
        // The server's HTTP default temperature (protocol.rs) and the
        // no-op repetition penalty used by the llama family.
        let params = SamplingParams::default();
        assert!((params.temperature - 0.6).abs() < f32::EPSILON);
        assert_eq!(params.top_k, 1);
        assert!((params.top_p - 1.0).abs() < f32::EPSILON);
        assert!((params.repetition_penalty - 1.0).abs() < f32::EPSILON);
        assert!(params.seed.is_none());
    }

    #[test]
    fn generation_request_new_defaults_sampling() {
        let request = GenerationRequest::new(vec![1, 2, 3], 16);
        assert_eq!(request.token_ids, vec![1, 2, 3]);
        assert_eq!(request.max_new_tokens, 16);
        assert!((request.sampling.temperature - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn temperature_sampler_greedy_matches_argmax() {
        let logits = [0.1f32, 5.0, 0.2, -1.0];
        assert_eq!(
            crate::ops::sampling::sample_temperature_greedy_or_random(&logits, 0.0),
            1
        );
    }

    #[test]
    fn temperature_sampler_single_candidate_is_deterministic() {
        // With one candidate every temperature returns it.
        assert_eq!(
            crate::ops::sampling::sample_temperature_greedy_or_random(&[3.0], 0.7),
            0
        );
    }
}