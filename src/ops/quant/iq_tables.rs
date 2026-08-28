use std::sync::OnceLock;

const GGML_COMMON: &str = include_str!("../../../references/llama.cpp/ggml/src/ggml-common.h");

fn parse_table<T>(type_name: &str, table_name: &str, expected: usize) -> Vec<T>
where
    T: TryFrom<i128>,
{
    let marker = format!("GGML_TABLE_BEGIN({type_name}, {table_name},");
    let begin = GGML_COMMON
        .find(&marker)
        .unwrap_or_else(|| panic!("missing ggml table {type_name}, {table_name}"));
    let tail = &GGML_COMMON[begin..];
    let end = tail
        .find("GGML_TABLE_END")
        .unwrap_or_else(|| panic!("unterminated ggml table {type_name}, {table_name}"));
    let mut values = Vec::with_capacity(expected);
    for token in tail[..end]
        .split(|character: char| character == ',' || character == ';' || character.is_whitespace())
        .filter(|token| !token.is_empty())
    {
        let parsed = if let Some(hex) = token.strip_prefix("0x") {
            i128::from_str_radix(hex, 16)
        } else {
            token.parse::<i128>()
        };
        let value = match parsed {
            Ok(value) => match T::try_from(value) {
                Ok(value) => value,
                Err(_) => panic!("value out of range in ggml table {table_name}: {value}"),
            },
            Err(_) => continue,
        };
        values.push(value);
    }
    assert_eq!(values.len(), expected, "wrong length for ggml table {table_name}");
    values
}

pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

pub fn iq2_xs_grid() -> &'static [u64] {
    static GRID: OnceLock<Vec<u64>> = OnceLock::new();
    GRID.get_or_init(|| parse_table("uint64_t", "iq2xs_grid", 512))
}

pub fn iq3_s_grid() -> &'static [u32] {
    static GRID: OnceLock<Vec<u32>> = OnceLock::new();
    GRID.get_or_init(|| parse_table("uint32_t", "iq3s_grid", 512))
}

pub fn iq2_xs_signs() -> &'static [u8] {
    static SIGNS: OnceLock<Vec<u8>> = OnceLock::new();
    SIGNS.get_or_init(|| parse_table("uint8_t", "ksigns_iq2xs", 128))
}

pub fn iq2_xs_mask() -> &'static [u8] {
    static MASK: OnceLock<Vec<u8>> = OnceLock::new();
    MASK.get_or_init(|| parse_table("uint8_t", "kmask_iq2xs", 8))
}
