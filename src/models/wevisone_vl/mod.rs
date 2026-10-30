//! Tencent WeVisOne-VL: Qwen3-VL backbone and the released FP32 pixel decoder.
mod decoder;
pub use decoder::{Decoder, Feature};

pub fn build_prompt(task: &str, query: &str) -> Result<String, String> {
    let query = query.trim();
    let phrase = if query.is_empty() || query.ends_with(['.', '?', '!']) {
        query.to_string()
    } else {
        format!("{query}.")
    };
    let labels = query
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(". ");
    Ok(match task.trim().to_lowercase().as_str() {
        "detection" => format!("Locate all the instances that match the following description: {labels}."),
        "referring" => format!("Locate the instances that match the following description: {phrase}"),
        "point" => format!("point to: {labels}"),
        "count_box" => format!("How many {} in the image? Please count them with boxes.", if query.is_empty() { "objects" } else { query }),
        "count_point" => format!("How many {} in the image?", if query.is_empty() { "objects" } else { query }),
        "segmentation" | "matting" => format!("Locate the instances that match the following description and provide the alpha mattes: {phrase}"),
        "ocr" => "detect all the word in the box format".into(),
        "ocr_grounding" => format!("Locate the following text in the image: {query}"),
        "freeform" => query.into(),
        _ => return Err(format!("unsupported WeVisOne task: {task}")),
    })
}

/// The released decoder consumes only boxes immediately followed by `<OBJ>`.
pub fn parse_seg_boxes(text: &str) -> Vec<[f32; 4]> {
    let boxes = regex::Regex::new(r"(?is)<box>(.*?)(?:</box>|$)").unwrap();
    let coords = regex::Regex::new(r"<(\d{1,4})>").unwrap();
    let obj = regex::Regex::new(r"(?i)<OBJ>").unwrap();
    let mut result = Vec::new();
    for body in boxes.captures_iter(text.split("<|im_end|>").next().unwrap_or_default()) {
        let parts = obj.split(&body[1]).collect::<Vec<_>>();
        for prefix in &parts[..parts.len().saturating_sub(1)] {
            let values = coords
                .captures_iter(prefix)
                .map(|c| c[1].parse::<u32>().unwrap())
                .collect::<Vec<_>>();
            if values.len() >= 4 {
                let v = &values[values.len() - 4..];
                if v.iter().all(|&x| x <= 1000) && v[2] > v[0] && v[3] > v[1] {
                    result.push(std::array::from_fn(|i| v[i] as f32 / 1000.0));
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_prompts_match_released_instructions() {
        assert_eq!(
            build_prompt("detection", " cat\n dog ").unwrap(),
            "Locate all the instances that match the following description: cat. dog."
        );
        assert_eq!(build_prompt("matting", "the person").unwrap(), "Locate the instances that match the following description and provide the alpha mattes: the person.");
        assert!(build_prompt("unknown", "cat").is_err());
    }

    #[test]
    fn segmentation_boxes_require_valid_obj_delimiters() {
        assert_eq!(
            parse_seg_boxes("<box><1><2><900><999><OBJ>,<10><20><30><40><OBJ></box>"),
            vec![[0.001, 0.002, 0.9, 0.999], [0.01, 0.02, 0.03, 0.04]]
        );
        for text in [
            "None",
            "<box><1><2><3><4></box>",
            "<box><9><2><3><4><OBJ></box>",
            "<box><0><0><1001><999><OBJ></box>",
        ] {
            assert!(parse_seg_boxes(text).is_empty());
        }
    }
}
