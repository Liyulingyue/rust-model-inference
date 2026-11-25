use crate::core::tensor::{MetaValue, TensorSource};
use crate::prompt::jinja::{JinjaChatTemplate, Options};

pub(crate) fn load_template(
    source: &dyn TensorSource,
    jinja: &Options,
) -> Result<JinjaChatTemplate, String> {
    if let Some(path) = &jinja.file {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("chat template file {}: {e}", path.display()))?;
        return JinjaChatTemplate::compile_huggingface(&text, &path.display().to_string());
    }
    let text = source
        .metadata("tokenizer.chat_template")
        .and_then(MetaValue::to_string_val)
        .ok_or("Aion requires its GGUF chat template or --chat-template-file")?;
    JinjaChatTemplate::compile_huggingface(text, "Aion GGUF tokenizer.chat_template")
}
