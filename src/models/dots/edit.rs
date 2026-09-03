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
    let rendered_source = render_source(&parts)?;
    let rendered_target = render_target(&parts)?;
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
        let end = tag_end(instruction, at + 1)?;
        let opening = &instruction[at + 1..end];
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
            push_part(&mut root, &mut stack, Part::Node(node));
        } else {
            let trimmed = opening.trim_end();
            let (opening, self_closing) = match trimmed.strip_suffix('/') {
                Some(opening) => (opening.trim_end(), true),
                None => (opening, false),
            };
            let (tag, attr_start) = opening_tag(opening)?;
            if !CONTAINERS.contains(&tag.as_str()) && !EMPTY.contains(&tag.as_str()) {
                return Err(format!("unknown edit tag <{tag}>"));
            }
            parse_attributes(opening, attr_start)?;
            if tag == "sub" {
                sub_target(opening)?;
            }
            operations.push(tag.clone());
            if self_closing {
                if !EMPTY.contains(&tag.as_str()) {
                    return Err(format!("self-closing <{tag}/> is not supported"));
                }
                push_part(
                    &mut root,
                    &mut stack,
                    Part::Node(Node {
                        tag,
                        opening: opening.to_owned(),
                        children: Vec::new(),
                    }),
                );
            } else {
                if EMPTY.contains(&tag.as_str()) {
                    return Err(format!("<{tag}> must be self-closing"));
                }
                stack.push(Node {
                    tag,
                    opening: opening.to_owned(),
                    children: Vec::new(),
                });
            }
        }
        cursor = end + 1;
    }
    if !stack.is_empty() {
        return Err("unclosed edit tag".into());
    }
    Ok((root, operations))
}

fn tag_end(instruction: &str, start: usize) -> Result<usize, String> {
    for (offset, ch) in instruction[start..].char_indices() {
        match ch {
            '<' => return Err("raw < inside edit tag".into()),
            '>' => return Ok(start + offset),
            _ => {}
        }
    }
    Err("unclosed < in edit instruction".into())
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
    let (tag, _) = opening_tag(opening)?;
    Ok(tag)
}

fn opening_tag(opening: &str) -> Result<(String, usize), String> {
    let leading = opening.len() - opening.trim_start().len();
    let rest = &opening[leading..];
    let end = rest
        .find(|ch: char| ch.is_whitespace() || ch == ',')
        .unwrap_or(rest.len());
    let name = &rest[..end];
    if name.is_empty()
        || !name.bytes().enumerate().all(|(index, byte)| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'_' => true,
            b'0'..=b'9' | b'-' => index > 0,
            _ => false,
        })
    {
        return Err("invalid edit tag name".into());
    }
    Ok((name.to_ascii_lowercase(), leading + end))
}

fn sub_target(opening: &str) -> Result<String, String> {
    let (_, attr_start) = opening_tag(opening)?;
    parse_attributes(opening, attr_start)?
        .ok_or_else(|| "sub requires a quoted targ attribute".into())
}

fn parse_attributes(opening: &str, attr_start: usize) -> Result<Option<String>, String> {
    let mut rest = &opening[attr_start..];
    let mut target = None;
    while !rest.is_empty() {
        let separator = rest
            .chars()
            .next()
            .ok_or_else(|| "missing attribute separator".to_owned())?;
        if separator == ',' {
            rest = rest[1..].trim_start();
        } else if separator.is_whitespace() {
            rest = rest.trim_start();
        } else {
            return Err("attributes require whitespace or a comma separator".into());
        }
        if rest.is_empty() {
            return Err("missing attribute after separator".into());
        }
        let name_end = rest
            .find(|ch: char| ch.is_whitespace() || ch == '=')
            .unwrap_or(rest.len());
        let name = &rest[..name_end];
        if name.is_empty()
            || !name.bytes().enumerate().all(|(index, byte)| match byte {
                b'A'..=b'Z' | b'a'..=b'z' | b'_' => true,
                b'0'..=b'9' | b'-' => index > 0,
                _ => false,
            })
        {
            return Err("invalid attribute name".into());
        }
        rest = rest[name_end..].trim_start();
        rest = rest
            .strip_prefix('=')
            .ok_or_else(|| "attributes need a value".to_owned())?
            .trim_start();
        if rest.is_empty() {
            return Err("attributes need a value".into());
        }
        let (value, quoted, following) = match rest.chars().next().unwrap() {
            quote @ ('\'' | '\"') => {
                let body = &rest[quote.len_utf8()..];
                let end = body
                    .find(quote)
                    .ok_or_else(|| "unclosed quoted attribute".to_owned())?;
                (&body[..end], true, &body[end + quote.len_utf8()..])
            }
            _ => {
                let end = rest
                    .find(|ch: char| ch.is_whitespace() || ch == ',')
                    .unwrap_or(rest.len());
                (&rest[..end], false, &rest[end..])
            }
        };
        if value.is_empty() {
            return Err("attributes need a value".into());
        }
        if !quoted && !is_decimal(value) {
            return Err("unquoted attribute values must be signed decimals".into());
        }
        let value = decode_entities(value)?;
        if name.eq_ignore_ascii_case("targ") {
            if !quoted {
                return Err("sub targ attribute must be quoted".into());
            }
            if target.replace(value).is_some() {
                return Err("sub requires exactly one targ attribute".into());
            }
        }
        rest = following;
    }
    Ok(target)
}

fn is_decimal(value: &str) -> bool {
    let value = value.strip_prefix(['+', '-']).unwrap_or(value);
    let mut parts = value.split('.');
    let whole = parts.next().unwrap_or_default();
    let fractional = parts.next();
    if parts.next().is_some() || !whole.bytes().all(|byte| byte.is_ascii_digit()) {
        return false;
    }
    match fractional {
        None => !whole.is_empty(),
        Some(part) => !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()),
    }
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

fn render_source(parts: &[Part]) -> Result<String, String> {
    let parts = source_parts(parts)?;
    let mut rendered = String::new();
    for (index, part) in parts.iter().enumerate() {
        if let Some(text) = part {
            rendered.push_str(text);
            continue;
        }
        let right = parts[index + 1..]
            .iter()
            .find_map(|part| part.as_deref().filter(|text| !text.is_empty()))
            .unwrap_or("");
        if source_insertion_needs_space(&rendered, right) {
            rendered.push(' ');
        }
    }
    Ok(rendered)
}

fn source_parts(parts: &[Part]) -> Result<Vec<Option<String>>, String> {
    let mut rendered = Vec::new();
    for part in parts {
        match part {
            Part::Text(text) => rendered.push(Some(text.clone())),
            Part::Node(node) => {
                let children = source_parts(&node.children)?;
                match node.tag.as_str() {
                    "ins" => rendered.push(None),
                    "sub" => {
                        sub_target(&node.opening)?;
                        rendered.extend(children);
                    }
                    "pause" | "spk_transfer" => {}
                    _ => rendered.extend(children),
                }
            }
        }
    }
    Ok(rendered)
}

fn source_insertion_needs_space(left: &str, right: &str) -> bool {
    let (Some(left), Some(right)) = (left.chars().last(), right.chars().next()) else {
        return false;
    };
    [left, right]
        .iter()
        .all(|ch| !ch.is_whitespace() && !is_punctuation(*ch) && !is_cjk(*ch))
}

fn render_target(parts: &[Part]) -> Result<String, String> {
    let (segments, has_text_edit) = target_segments(parts)?;
    if !has_text_edit {
        return Ok(segments.into_iter().map(|(text, _)| text).collect());
    }
    let mut parts = Vec::new();
    let mut unchanged = String::new();
    for (text, is_edit) in segments {
        if is_edit {
            if !unchanged.is_empty() {
                parts.push(std::mem::take(&mut unchanged));
            }
            parts.push(text);
        } else {
            unchanged.push_str(&text);
        }
    }
    if !unchanged.is_empty() {
        parts.push(unchanged);
    }
    Ok(normalize_target_parts(&parts))
}

fn target_segments(parts: &[Part]) -> Result<(Vec<(String, bool)>, bool), String> {
    let mut rendered = Vec::new();
    let mut has_text_edit = false;
    for part in parts {
        match part {
            Part::Text(text) => rendered.push((text.clone(), false)),
            Part::Node(node) => {
                let (children, child_has_text_edit) = target_segments(&node.children)?;
                has_text_edit |= child_has_text_edit;
                match node.tag.as_str() {
                    "del" => {
                        rendered.push((String::new(), true));
                        has_text_edit = true;
                    }
                    "sub" => {
                        rendered.push((sub_target(&node.opening)?, true));
                        has_text_edit = true;
                    }
                    "ins" => {
                        rendered.push((children.into_iter().map(|(text, _)| text).collect(), true));
                        has_text_edit = true;
                    }
                    "pause" | "spk_transfer" => {}
                    _ => rendered.extend(children),
                }
            }
        }
    }
    Ok((rendered, has_text_edit))
}

fn normalize_target_parts(parts: &[String]) -> String {
    let clean: Vec<&str> = parts.iter().map(|part| part.trim()).collect();
    let mut rendered = String::new();
    for (index, part) in clean.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        let next = clean[index + 1..]
            .iter()
            .copied()
            .find(|part| !part.is_empty())
            .unwrap_or("");
        if !rendered.is_empty() && attach_connector_to_left(part, &rendered, next) {
            rendered.push_str(part);
            continue;
        }
        if !rendered.is_empty() && needs_target_space(&rendered, part) {
            rendered.push(' ');
        }
        rendered.push_str(part);
    }
    let collapsed = rendered.split_whitespace().collect::<Vec<_>>().join(" ");
    trim_space_before_punctuation(&collapsed)
}

fn needs_target_space(left: &str, right: &str) -> bool {
    !word_internal_join(left, right)
        && (contains_ascii_word(left.chars().last()) || contains_ascii_word(right.chars().next()))
}

fn word_internal_join(left: &str, right: &str) -> bool {
    let Some(left_edge) = left.chars().last() else {
        return false;
    };
    let Some(right_edge) = right.chars().next() else {
        return false;
    };
    if matches!(right_edge, '\'' | '-' | '’') {
        return is_ascii_word(left_edge)
            && apostrophe_or_hyphen_suffix(&right[right_edge.len_utf8()..], right_edge);
    }
    if matches!(left_edge, '\'' | '-' | '’') {
        let before = left[..left.len() - left_edge.len_utf8()].chars().last();
        return before.is_some_and(is_ascii_word)
            && is_ascii_word(right_edge)
            && (left_edge == '-' || apostrophe_suffix(right));
    }
    false
}

fn attach_connector_to_left(part: &str, left: &str, right: &str) -> bool {
    let Some(left_edge) = left.chars().last() else {
        return false;
    };
    if !is_ascii_word(left_edge) {
        return false;
    }
    match part {
        "-" => right.chars().next().is_some_and(is_ascii_word),
        "'" | "’" => apostrophe_suffix(right) || matches!(left_edge, 's' | 'S'),
        _ => false,
    }
}

fn apostrophe_or_hyphen_suffix(right: &str, connector: char) -> bool {
    connector == '-' && right.chars().next().is_some_and(is_ascii_word)
        || connector != '-' && apostrophe_suffix(right)
}

fn apostrophe_suffix(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    ["s", "t", "re", "ve", "ll", "d", "m"].iter().any(|suffix| {
        lower
            .strip_prefix(suffix)
            .is_some_and(|rest| !rest.chars().next().is_some_and(is_ascii_word))
    })
}

fn contains_ascii_word(ch: Option<char>) -> bool {
    ch.is_some_and(is_ascii_word)
}

fn is_ascii_word(ch: char) -> bool {
    ch.is_ascii_alphanumeric()
}

fn is_punctuation(ch: char) -> bool {
    matches!(
        ch,
        ',' | '.' | ';' | ':' | '!' | '?' | '，' | '。' | '？' | '！' | '；' | '：'
    ) || ch.is_ascii_punctuation()
}

fn is_cjk(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0x2CEB0..=0x2EBEF
            | 0x30000..=0x3134F
    )
}

fn trim_space_before_punctuation(text: &str) -> String {
    let mut rendered = String::new();
    for ch in text.chars() {
        if matches!(
            ch,
            ',' | '.' | ';' | ':' | '!' | '?' | '，' | '。' | '？' | '！' | '；' | '：'
        ) {
            while rendered.ends_with(char::is_whitespace) {
                rendered.pop();
            }
        }
        rendered.push(ch);
    }
    rendered.trim().to_owned()
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
        assert_eq!(insertion.source_text, "hello  world");
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
            "<pause/><spk_transfer/>text",
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

    #[test]
    fn preserves_style_whitespace_and_normalizes_only_text_edit_boundaries() {
        let text_edit =
            resolve_edit_request("foo<ins>bar</ins>baz", None, None, XVectorMode::Auto).unwrap();
        assert_eq!(text_edit.source_text, "foo baz");
        assert_eq!(text_edit.target_text, "foo bar baz");

        let connector =
            resolve_edit_request("well<ins>-</ins>known", None, None, XVectorMode::Auto).unwrap();
        assert_eq!(connector.source_text, "well known");
        assert_eq!(connector.target_text, "well-known");

        let style_only = resolve_edit_request(
            "<emo>line one\tline two\nline three</emo>",
            None,
            None,
            XVectorMode::Auto,
        )
        .unwrap();
        assert_eq!(style_only.source_text, "line one\tline two\nline three");
        assert_eq!(style_only.target_text, "line one\tline two\nline three");
    }

    #[test]
    fn accepts_official_empty_tags_and_attribute_forms() {
        let empty = resolve_edit_request(
            "hello<pause duration=0.5/><spk_transfer/> world",
            None,
            None,
            XVectorMode::Auto,
        )
        .unwrap();
        assert_eq!(empty.source_text, "hello world");
        assert_eq!(empty.target_text, "hello world");

        for instruction in [
            "<pitch, semitones=-2>hello</pitch>",
            "<rate, factor=0.8>hello</rate>",
            "<sub note='keep' targ='new'>old</sub>",
        ] {
            assert!(resolve_edit_request(instruction, None, None, XVectorMode::Auto).is_ok());
        }
    }

    #[test]
    fn rejects_invalid_attribute_and_empty_tag_forms() {
        for instruction in [
            "<emo value=>x</emo>",
            "<sub targ='new'junk='x'>old</sub>",
            "<sub targ='new < safe'>old</sub>",
            "<pause></pause>",
            "<spk_transfer></spk_transfer>",
        ] {
            assert!(resolve_edit_request(instruction, None, None, XVectorMode::Auto).is_err());
        }
    }

    #[test]
    fn normalizes_internal_target_whitespace_only_for_text_edits() {
        let request =
            resolve_edit_request("foo\tbar<ins>x</ins>baz", None, None, XVectorMode::Auto).unwrap();
        assert_eq!(request.source_text, "foo\tbar baz");
        assert_eq!(request.target_text, "foo bar x baz");
    }

    #[test]
    fn rejects_merged_unquoted_numeric_attributes() {
        assert!(resolve_edit_request(
            "<pitch semitones=-2rate=1>hello</pitch>",
            None,
            None,
            XVectorMode::Auto,
        )
        .is_err());
    }

    #[test]
    fn renders_curly_apostrophe_insertions_without_panicking() {
        let request =
            resolve_edit_request("foo<ins>’s</ins>bar", None, None, XVectorMode::Auto).unwrap();
        assert_eq!(request.source_text, "foo bar");
        assert_eq!(request.target_text, "foo’s bar");
    }

    #[test]
    fn rejects_whitespace_only_rendered_surfaces() {
        assert!(resolve_edit_request(" \t\n ", None, None, XVectorMode::Auto).is_err());
    }
}
