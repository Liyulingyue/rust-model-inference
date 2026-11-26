//! Tokenizer for `SenseNova-U1.6-8B-MoT`.
//!
//! The GGUF carries **no** tokenizer metadata, so the vocabulary is loaded
//! from the HuggingFace files that ship next to it (`vocab.json`,
//! `merges.txt`). [`BPETokenizer::from_vocab_json_and_merges`] reproduces the
//! ids of `transformers`' `Qwen2Tokenizer`, including the family's
//! `added_tokens.json` entries.
//!
//! Everything here is a thin wrapper: the byte-level BPE lives in
//! `core::tokenizer`, and this module only adds the two things the diffusion
//! pipeline needs on top of it — the `neo1_0` chat template (`MPT` separators,
//! rendered the way `Conversation.get_prompt()` renders a single user turn
//! with an open assistant turn) and the T2I queries of contract §5.
//!
//! ```text
//! conditional:   <|im_start|>system\n{SYSTEM_MESSAGE_FOR_GEN}<|im_end|>\n
//!                <|im_start|>user\n{prompt}<|im_end|>\n
//!                <|im_start|>assistant\n
//!                <think>\n\n</think>\n\n<img>
//! unconditional: <|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<img>
//! ```

use std::path::Path;

use crate::core::tokenizer::{BPETokenizer, EncodeOptions};
use crate::models::diffusion::sensenova_u1::config;

/// `sep` of the `neo1_0` template (`SeparatorStyle::MPT`).
const SEP: &str = "<|im_end|>\n";
/// `roles[0]` of the `neo1_0` template.
const ROLE_USER: &str = "<|im_start|>user\n";
/// `roles[1]` of the `neo1_0` template.
const ROLE_ASSISTANT: &str = "<|im_start|>assistant\n";
/// `system_template` of the `neo1_0` template, up to `{system_message}`.
const SYSTEM_TEMPLATE: &str = "<|im_start|>system\n";
/// Appended to the conditional query by `t2i_generate` when `think_mode` is
/// off: an empty `<think>` block followed by the image start marker.
const CONDITIONAL_SUFFIX: &str = "<think>\n\n</think>\n\n<img>";
/// Appended to the unconditional query: just the image start marker.
const UNCONDITIONAL_SUFFIX: &str = "<img>";

/// `SYSTEM_MESSAGE_FOR_GEN` of
/// `references/src/sensenova_u1/models/neo_unify/utils.py`, character for
/// character. The Python source is a concatenation of adjacent string
/// literals, so the only newlines in the value are the explicit `\n\n`
/// paragraph breaks below.
pub(crate) const SYSTEM_MESSAGE_FOR_GEN: &str = concat!(
    "You are an image generation and editing assistant that accurately understands",
    " and executes user intent.\n\n",
    "You support two modes:\n\n",
    "1. Think Mode:\nIf the task requires reasoning, you MUST start with a",
    " <think></think> block. Put all reasoning inside the block using plain text.",
    " DO NOT include any image tags. Keep it reasonable and directly useful for",
    " producing the final image.\n\n",
    "2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final",
    " image.\n\n",
    "Task Types:\n\n",
    "A. Text-to-Image Generation:\n- Generate a high-quality image based on the",
    " user's description.\n- Ensure visual clarity, semantic consistency, and",
    " completeness.\n- DO NOT introduce elements that contradict or override the",
    " user's intent.\n\n",
    "B. Image Editing:\n- Use the provided image(s) as input or reference for",
    " modification or transformation.\n- The result can be an edited image or a new",
    " image based on the reference(s).\n- Preserve all unspecified attributes",
    " unless explicitly changed.\n\n",
    "General Rules:\n- For any visible text in the image, follow the language",
    " specified for the rendered text in the user's description, not the language",
    " of the prompt. If no language is specified, use the user's input language.",
);

pub(crate) struct Tokenizer {
    inner: BPETokenizer,
}

impl Tokenizer {
    /// `dir` holds `vocab.json` and `merges.txt`.
    pub(crate) fn load(dir: &Path) -> Result<Self, String> {
        let vocab_path = dir.join("vocab.json");
        let merges_path = dir.join("merges.txt");
        let vocab = std::fs::read_to_string(&vocab_path)
            .map_err(|error| format!("read {}: {error}", vocab_path.display()))?;
        let merges = std::fs::read_to_string(&merges_path)
            .map_err(|error| format!("read {}: {error}", merges_path.display()))?;
        let inner = BPETokenizer::from_vocab_json_and_merges(&vocab, &merges)?;
        // The ids have to index the embedding matrix, so a tokenizer whose
        // vocabulary disagrees with `llm.vocab_size` is the wrong file pair.
        if inner.vocab_size() != config::VOCAB {
            return Err(format!(
                "tokenizer vocabulary has {} ids but the model has {} embedding rows",
                inner.vocab_size(),
                config::VOCAB
            ));
        }
        Ok(Self { inner })
    }

    /// Token ids for an already-rendered prompt string.
    ///
    /// Matches `transformers`' `tokenizer(text, return_tensors="pt")`: the
    /// special markers of the prompt are matched whole (`parse_special`), and
    /// `add_special` is a no-op for this family because `tokenizer_config.json`
    /// sets `add_bos_token = false` / `add_eos_token = false`.
    pub(crate) fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        Ok(self.inner.encode(
            text,
            EncodeOptions {
                add_special: true,
                parse_special: true,
            },
        ))
    }

    /// The conditional T2I query (contract §5).
    pub(crate) fn conditional_prompt(&self, user_prompt: &str) -> String {
        let mut prompt = render_prompt(SYSTEM_MESSAGE_FOR_GEN, user_prompt);
        prompt.push_str(CONDITIONAL_SUFFIX);
        prompt
    }

    /// The unconditional (CFG negative) T2I query.
    pub(crate) fn unconditional_prompt(&self) -> String {
        let mut prompt = render_prompt("", "");
        prompt.push_str(UNCONDITIONAL_SUFFIX);
        prompt
    }
}

/// Port of `Conversation.get_prompt()` for `neo1_0` (`SeparatorStyle::MPT`)
/// over the fixed message list `[(user, user_message), (assistant, None)]`
/// that `_build_t2i_query` builds.
///
/// Two of the upstream branches are easy to get wrong and are reproduced
/// exactly here:
///
/// * an **empty** system message suppresses the system turn entirely
///   (`ret = '' if system_prompt == ''`);
/// * an **empty** user message still emits the role *and* the separator
///   (`if message: ... else: ret += role + self.sep`, because the message is
///   `''` and not `None`), while the `None` assistant message emits only the
///   role.
fn render_prompt(system_message: &str, user_message: &str) -> String {
    let mut prompt = String::new();
    if !system_message.is_empty() {
        prompt.push_str(SYSTEM_TEMPLATE);
        prompt.push_str(system_message);
        prompt.push_str(SEP);
    }
    prompt.push_str(ROLE_USER);
    if user_message.is_empty() {
        prompt.push_str(SEP);
    } else {
        prompt.push_str(user_message);
        prompt.push_str(SEP);
    }
    prompt.push_str(ROLE_ASSISTANT);
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `references/sn_tokenizer/` is gitignored, so a checkout without the
    /// companion files skips the token-id checks instead of failing.
    fn tokenizer() -> Option<Tokenizer> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("references/sn_tokenizer");
        if !dir.join("vocab.json").is_file() || !dir.join("merges.txt").is_file() {
            eprintln!("references/sn_tokenizer is absent, the token-id check is skipped");
            return None;
        }
        Some(Tokenizer::load(&dir).expect("the reference tokenizer files load"))
    }

    #[test]
    fn conditional_prompt_matches_the_neo1_0_template() {
        let prompt = tokenizer().map(|tokenizer| tokenizer.conditional_prompt("hi"));
        // The rendering itself needs no tokenizer files.
        let rendered = render_prompt(SYSTEM_MESSAGE_FOR_GEN, "hi");
        assert_eq!(
            rendered.clone() + CONDITIONAL_SUFFIX,
            concat!(
                "<|im_start|>system\nYou are an image generation and editing assistant that",
                " accurately understands and executes user intent.\n\n",
                "You support two modes:\n\n",
                "1. Think Mode:\nIf the task requires reasoning, you MUST start with a",
                " <think></think> block. Put all reasoning inside the block using plain text.",
                " DO NOT include any image tags. Keep it reasonable and directly useful for",
                " producing the final image.\n\n",
                "2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final",
                " image.\n\n",
                "Task Types:\n\n",
                "A. Text-to-Image Generation:\n- Generate a high-quality image based on the",
                " user's description.\n- Ensure visual clarity, semantic consistency, and",
                " completeness.\n- DO NOT introduce elements that contradict or override the",
                " user's intent.\n\n",
                "B. Image Editing:\n- Use the provided image(s) as input or reference for",
                " modification or transformation.\n- The result can be an edited image or a new",
                " image based on the reference(s).\n- Preserve all unspecified attributes",
                " unless explicitly changed.\n\n",
                "General Rules:\n- For any visible text in the image, follow the language",
                " specified for the rendered text in the user's description, not the language",
                " of the prompt. If no language is specified, use the user's input",
                " language.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
                "<think>\n\n",
                "</think>\n\n",
                "<img>",
            )
        );
        if let Some(prompt) = prompt {
            assert_eq!(
                prompt,
                render_prompt(SYSTEM_MESSAGE_FOR_GEN, "hi") + CONDITIONAL_SUFFIX
            );
        }
    }

    #[test]
    fn unconditional_prompt_is_the_empty_turn_plus_the_image_marker() {
        let rendered = render_prompt("", "");
        assert_eq!(
            rendered.clone() + UNCONDITIONAL_SUFFIX,
            "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<img>",
        );
        if let Some(tokenizer) = tokenizer() {
            assert_eq!(
                tokenizer.unconditional_prompt(),
                "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<img>",
            );
        }
    }

    #[test]
    fn empty_system_message_drops_the_system_turn() {
        // `ret = '' if system_prompt == ''` — no `<|im_start|>system` at all.
        assert!(!render_prompt("", "hi").contains("<|im_start|>system"));
        assert!(render_prompt("sys", "hi").starts_with("<|im_start|>system\nsys<|im_end|>\n"));
    }

    /// Regression pin for the T2I path: the conditional prompt of a typical
    /// caption, tokenized exactly like the oracle
    /// (`transformers`' `Qwen2Tokenizer`, `add_special_tokens=True`).
    #[test]
    fn conditional_prompt_token_ids_match_the_oracle() {
        let Some(tokenizer) = tokenizer() else {
            return;
        };
        let prompt = tokenizer.conditional_prompt(
            "A studio photograph of a red apple on a white table, soft daylight.",
        );
        let ids = tokenizer.encode(&prompt).expect("the prompt encodes");
        assert_eq!(ids.len(), 266, "{ids:?}");
        assert_eq!(
            &ids[..8],
            &[151_644u32, 8948, 198, 2610, 525, 458, 2168, 9471]
        );
        assert_eq!(
            &ids[ids.len() - 8..],
            &[151_644u32, 77091, 198, 151667, 271, 151668, 271, 151670]
        );
        // Whole vector, for diffing against the oracle.
        #[rustfmt::skip]
        let expected: &[u32] = &[
            151644, 8948, 198, 2610, 525, 458, 2168, 9471, 323, 15664, 17847, 429,
            29257, 30769, 323, 51435, 1196, 7385, 382, 2610, 1824, 1378, 19777, 1447,
            16, 13, 21149, 14562, 510, 2679, 279, 3383, 7460, 32711, 11, 498,
            27732, 1191, 448, 264, 220, 151667, 151668, 2504, 13, 10224, 678, 32711,
            4766, 279, 2504, 1667, 14396, 1467, 13, 9319, 4183, 2924, 894, 2168,
            9492, 13, 13655, 432, 13276, 323, 5961, 5390, 369, 17387, 279, 1590,
            2168, 382, 17, 13, 11581, 12, 38687, 14562, 510, 2679, 902, 32711,
            374, 4362, 11, 5961, 8193, 279, 1590, 2168, 382, 6262, 20768, 1447,
            32, 13, 2918, 4686, 12, 1906, 23470, 510, 12, 19813, 264, 1550,
            22092, 2168, 3118, 389, 279, 1196, 594, 4008, 624, 12, 29279, 9124,
            31273, 11, 41733, 28137, 11, 323, 79314, 624, 12, 9319, 4183, 19131,
            5424, 429, 42461, 476, 2812, 279, 1196, 594, 7385, 382, 33, 13,
            4654, 61053, 510, 12, 5443, 279, 3897, 2168, 1141, 8, 438, 1946,
            476, 5785, 369, 17030, 476, 17991, 624, 12, 576, 1102, 646, 387,
            458, 19138, 2168, 476, 264, 501, 2168, 3118, 389, 279, 5785, 1141,
            4292, 12, 81306, 678, 54311, 8201, 7241, 20975, 5497, 382, 15415, 22847,
            510, 12, 1752, 894, 9434, 1467, 304, 279, 2168, 11, 1795, 279,
            4128, 5189, 369, 279, 22383, 1467, 304, 279, 1196, 594, 4008, 11,
            537, 279, 4128, 315, 279, 9934, 13, 1416, 902, 4128, 374, 5189,
            11, 990, 279, 1196, 594, 1946, 4128, 13, 151645, 198, 151644, 872,
            198, 32, 14029, 10300, 315, 264, 2518, 23268, 389, 264, 4158, 1965,
            11, 8413, 52021, 13, 151645, 198, 151644, 77091, 198, 151667, 271, 151668,
            271, 151670,
        ];
        assert_eq!(ids, expected);
    }

    #[test]
    fn unconditional_prompt_token_ids_match_the_oracle() {
        let Some(tokenizer) = tokenizer() else {
            return;
        };
        let ids = tokenizer
            .encode(&tokenizer.unconditional_prompt())
            .expect("the prompt encodes");
        assert_eq!(
            ids,
            [151_644u32, 872, 198, 151_645, 198, 151_644, 77091, 198, 151670]
        );
    }
}
