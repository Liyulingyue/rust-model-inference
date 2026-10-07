fn main() {
    use regex::Regex;
    let text = "The trip was great. Paris was lovely.";
    for pat in [
        r"(?i)(?<![\p{L}\p{N}_])paris(?![\p{L}\p{N}_])",
        r"(?i)(?<!\w)paris(?!\w)",
        r"(?i)\bparis\b",
    ] {
        match Regex::new(pat) {
            Ok(re) => {
                let hits: Vec<(usize, usize)> =
                    re.find_iter(text).map(|m| (m.start(), m.end())).collect();
                println!("{pat:48} -> {hits:?}");
            }
            Err(e) => println!("{pat:48} -> ERR {e}"),
        }
    }
    let escaped = regex::escape("paris");
    let pat = format!(r"(?i)(?<![\p{{L}}\p{{N}}_]){escaped}(?![\p{{L}}\p{{N}}_])");
    println!("built = {pat}");
    match Regex::new(&pat) {
        Ok(re) => println!(
            "  -> {:?}",
            re.find_iter(text)
                .map(|m| (m.start(), m.end()))
                .collect::<Vec<_>>()
        ),
        Err(e) => println!("  -> ERR {e}"),
    }
}
