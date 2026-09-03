#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum XVectorMode {
    #[default]
    Auto,
    On,
    Off,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditRequest {
    pub source_text: String,
    pub target_text: String,
    pub use_xvector: bool,
}

use std::str::FromStr;

const CONTAINERS: &[&str] = &["del", "ins", "sub", "emo", "pitch", "rate", "enhance", "bg"];
const EMPTY: &[&str] = &["pause", "spk_transfer"];
const AUTO_OFF: &[&str] = &["emo", "bg", "enhance"];

#[derive(Debug)]
struct Node {
    tag: String,
    opening: String,
    children: Vec<Part>,
}

#[derive(Debug)]
enum Part {
    Text(String),
    Node(Node),
}

impl FromStr for XVectorMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "on" => Ok(Self::On),
            "off" => Ok(Self::Off),
            _ => Err(format!(
                "invalid xvector mode {value:?}; expected auto, on, or off"
            )),
        }
    }
}

pub fn resolve_edit_request(
    instruction: &str,
    source_text: Option<&str>,
    target_text: Option<&str>,
    mode: XVectorMode,
) -> Result<EditRequest, String> {
    let (parts, operations) = parse_instruction(instruction)?;
    let rendered_source = normalize_rendered(render(&parts, false)?);
    let rendered_target = normalize_rendered(render(&parts, true)?);
    let source_text = source_text.unwrap_or(&rendered_source).trim().to_owned();
    let target_text = target_text.unwrap_or(&rendered_target).trim().to_owned();
    if source_text.is_empty() || target_text.is_empty() {
        return Err("edit source and target text must both be non-empty".into());
    }
    let use_xvector = match mode {
        XVectorMode::On => true,
        XVectorMode::Off => false,
        XVectorMode::Auto => {
            operations.is_empty()
                || operations
                    .iter()
                    .any(|tag| !AUTO_OFF.contains(&tag.as_str()))
        }
    };
    Ok(EditRequest {
        source_text,
        target_text,
        use_xvector,
    })
}

fn parse_instruction(instruction: &str) -> Result<(Vec<Part>, Vec<String>), String> {
    let mut root = Vec::new();
    let mut stack: Vec<Node> = Vec::new();
    let mut operations = Vec::new();
    let mut cursor = 0;
    while cursor < instruction.len() {
        let rest = &instruction[cursor..];
        let next = rest.find(|ch| ch == '<' || ch == '>');
        let Some(offset) = next else {
            push_text(&mut root, &mut stack, decode_entities(rest)?);
            break;
        };
        let at = cursor + offset;
        if instruction.as_bytes()[at] == b'>' {
            return Err("stray > in edit instruction".into());
        }
        push_text(
            &mut root,
            &mut stack,
            decode_entities(&instruction[cursor..at])?,
        );
        let end = instruction[at + 1..]
            .find('>')
            .map(|end| at + 1 + end)
            .ok_or_else(|| "unclosed < in edit instruction".to_owned())?;
        let opening = &instruction[at + 1..end];
        if opening.ends_with('/') {
            return Err("self-closing edit tags are not supported".into());
        }
        if let Some(closing) = opening.strip_prefix('/') {
            let tag = tag_name(closing)?;
            if !closing[tag.len()..].trim().is_empty() {
                return Err("closing edit tags cannot have attributes".into());
            }
            let node = stack
                .pop()
                .ok_or_else(|| "unexpected closing edit tag".to_owned())?;
            if node.tag != tag {
                return Err(format!("mismatched edit tag: expected </{}>", node.tag));
            }
            if EMPTY.contains(&tag.as_str()) && !node.children.is_empty() {
                return Err(format!("<{tag}> must be empty"));
            }
            push_part(&mut root, &mut stack, Part::Node(node));
        } else {
            let tag = tag_name(opening)?;
            if !CONTAINERS.contains(&tag.as_str()) && !EMPTY.contains(&tag.as_str()) {
                return Err(format!("unknown edit tag <{tag}>"));
            }
            if tag == "sub" {
                sub_target(opening)?;
            }
            operations.push(tag.clone());
            stack.push(Node {
                tag,
                opening: opening.to_owned(),
                children: Vec::new(),
            });
        }
        cursor = end + 1;
    }
    if !stack.is_empty() {
        return Err("unclosed edit tag".into());
    }
    Ok((root, operations))
}

fn push_text(root: &mut Vec<Part>, stack: &mut [Node], text: String) {
    if !text.is_empty() {
        push_part(root, stack, Part::Text(text));
    }
}

fn push_part(root: &mut Vec<Part>, stack: &mut [Node], part: Part) {
    if let Some(parent) = stack.last_mut() {
        parent.children.push(part);
    } else {
        root.push(part);
    }
}

fn tag_name(opening: &str) -> Result<String, String> {
    let name = opening
        .split_whitespace()
        .next()
        .ok_or_else(|| "empty edit tag".to_owned())?;
    if !name
        .bytes()
        .all(|byte| byte.is_ascii_alphabetic() || byte == b'_')
    {
        return Err("invalid edit tag name".into());
    }
    Ok(name.to_ascii_lowercase())
}

fn sub_target(opening: &str) -> Result<String, String> {
    let mut rest = opening[3..].trim_start();
    let mut target = None;
    while !rest.is_empty() {
        let name_end = rest
            .find(|ch: char| ch.is_whitespace() || ch == '=')
            .ok_or_else(|| "sub attributes need =\"value\"".to_owned())?;
        let name = &rest[..name_end];
        rest = rest[name_end..].trim_start();
        rest = rest
            .strip_prefix('=')
            .ok_or_else(|| "sub attributes need =\"value\"".to_owned())?
            .trim_start();
        let quote = rest
            .chars()
            .next()
            .filter(|quote| *quote == '\'' || *quote == '\"')
            .ok_or_else(|| "sub attributes must be quoted".to_owned())?;
        rest = &rest[quote.len_utf8()..];
        let end = rest
            .find(quote)
            .ok_or_else(|| "unclosed sub attribute".to_owned())?;
        let value = decode_entities(&rest[..end])?;
        if name == "targ" {
            if target.replace(value).is_some() {
                return Err("sub requires exactly one targ attribute".into());
            }
        }
        rest = rest[end + quote.len_utf8()..].trim_start();
    }
    target.ok_or_else(|| "sub requires a quoted targ attribute".into())
}

fn decode_entities(text: &str) -> Result<String, String> {
    let mut decoded = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('&') {
        decoded.push_str(&rest[..start]);
        let entity = &rest[start + 1..];
        let end = entity
            .find(';')
            .ok_or_else(|| "unclosed XML entity".to_owned())?;
        decoded.push_str(match &entity[..end] {
            "amp" => "&",
            "lt" => "<",
            "gt" => ">",
            "quot" => "\"",
            "apos" => "'",
            _ => return Err("unknown XML entity".into()),
        });
        rest = &entity[end + 1..];
    }
    decoded.push_str(rest);
    Ok(decoded)
}

fn render(parts: &[Part], target: bool) -> Result<String, String> {
    let mut rendered = String::new();
    for part in parts {
        match part {
            Part::Text(text) => rendered.push_str(text),
            Part::Node(node) => match node.tag.as_str() {
                "del" if target => {}
                "ins" if !target => {}
                "pause" | "spk_transfer" => {}
                "sub" if target => rendered.push_str(&sub_target(&node.opening)?),
                _ => rendered.push_str(&render(&node.children, target)?),
            },
        }
    }
    Ok(rendered)
}

fn normalize_rendered(text: String) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::{resolve_edit_request, XVectorMode};

    #[test]
    fn renders_source_and_target_from_text_edit_tags() {
        let deletion =
            resolve_edit_request("<del>预</del>热", None, None, XVectorMode::Auto).unwrap();
        assert_eq!(
            (deletion.source_text.as_str(), deletion.target_text.as_str()),
            ("预热", "热")
        );

        let insertion = resolve_edit_request(
            "hello <ins>brave</ins> world",
            None,
            None,
            XVectorMode::Auto,
        )
        .unwrap();
        assert_eq!(insertion.source_text, "hello world");
        assert_eq!(insertion.target_text, "hello brave world");

        let substitution = resolve_edit_request(
            "<sub targ=\"new &amp; safe\">old</sub>",
            None,
            None,
            XVectorMode::Auto,
        )
        .unwrap();
        assert_eq!(substitution.source_text, "old");
        assert_eq!(substitution.target_text, "new & safe");

        let nested = resolve_edit_request(
            "<emo>a<rate> b </rate>c</emo>",
            None,
            None,
            XVectorMode::Auto,
        )
        .unwrap();
        assert_eq!(nested.source_text, "a b c");
        assert_eq!(nested.target_text, "a b c");

        let entities = resolve_edit_request(
            "<emo>&lt;&gt;&quot;&apos;&amp;</emo>",
            None,
            None,
            XVectorMode::Auto,
        )
        .unwrap();
        assert_eq!(entities.source_text, "<>\"'&");
        assert_eq!(entities.target_text, "<>\"'&");
    }

    #[test]
    fn explicit_transcripts_override_rendering_but_not_instruction_validation() {
        let request = resolve_edit_request(
            "<del>old</del><ins>new</ins>",
            Some("spoken old"),
            Some("spoken new"),
            XVectorMode::On,
        )
        .unwrap();
        assert_eq!(request.source_text, "spoken old");
        assert_eq!(request.target_text, "spoken new");
        assert!(request.use_xvector);
        assert!(
            resolve_edit_request("<del>broken", Some("old"), Some("new"), XVectorMode::On).is_err()
        );
    }

    #[test]
    fn auto_xvector_disables_only_style_only_operation_sets() {
        for instruction in [
            "<emo value=\"happy\">hello</emo>",
            "<bg type=\"rain\">hello</bg>",
            "<enhance>hello</enhance>",
        ] {
            assert!(
                !resolve_edit_request(instruction, Some("hello"), Some("hello"), XVectorMode::Auto)
                    .unwrap()
                    .use_xvector
            );
        }
        assert!(
            resolve_edit_request("hello", Some("hello"), Some("hello"), XVectorMode::Auto)
                .unwrap()
                .use_xvector
        );
        assert!(
            resolve_edit_request(
                "<del>old</del><emo>new</emo>",
                Some("old"),
                Some("new"),
                XVectorMode::Auto
            )
            .unwrap()
            .use_xvector
        );
        assert!(
            !resolve_edit_request("<del>old</del>", Some("old"), Some("new"), XVectorMode::Off)
                .unwrap()
                .use_xvector
        );
    }

    #[test]
    fn rejects_malformed_and_empty_edit_instructions() {
        for instruction in [
            "<wat>text</wat>",
            "<del/>",
            "<del>text</ins>",
            "<del>text",
            "text >",
            "<del>text<",
            "<del>text &bogus;</del>",
            "<del></del>",
        ] {
            assert!(resolve_edit_request(instruction, None, None, XVectorMode::Auto).is_err());
        }
    }

    #[test]
    fn parses_mode_and_supports_instruction_tags() {
        assert_eq!("auto".parse::<XVectorMode>().unwrap(), XVectorMode::Auto);
        assert_eq!("on".parse::<XVectorMode>().unwrap(), XVectorMode::On);
        assert_eq!("off".parse::<XVectorMode>().unwrap(), XVectorMode::Off);
        assert!("Auto".parse::<XVectorMode>().is_err());

        for instruction in [
            "<emo>text</emo>",
            "<pitch>text</pitch>",
            "<rate>text</rate>",
            "<enhance>text</enhance>",
            "<bg>text</bg>",
            "<pause></pause><spk_transfer></spk_transfer>text",
            "<sub targ='new'>old</sub>",
        ] {
            assert!(resolve_edit_request(
                instruction,
                Some("text"),
                Some("text"),
                XVectorMode::Auto
            )
            .is_ok());
        }
    }
}
