/// Hold only a possible stop-sequence prefix, so a match never reaches the client.
pub struct StopFilter {
    pending: String,
    sequences: Vec<String>,
    pub hit: Option<String>,
}

impl StopFilter {
    pub fn new(sequences: Vec<String>) -> Self {
        Self {
            pending: String::new(),
            sequences,
            hit: None,
        }
    }

    pub fn push(&mut self, text: &str) -> String {
        if self.hit.is_some() {
            return String::new();
        }
        self.pending.push_str(text);
        if let Some((offset, sequence)) = self
            .sequences
            .iter()
            .filter_map(|s| self.pending.find(s).map(|offset| (offset, s)))
            .min_by_key(|(offset, _)| *offset)
        {
            self.hit = Some(sequence.clone());
            let output = self.pending[..offset].to_string();
            self.pending.clear();
            return output;
        }
        let keep = self
            .sequences
            .iter()
            .flat_map(|sequence| {
                sequence
                    .char_indices()
                    .skip(1)
                    .map(move |(end, _)| &sequence[..end])
            })
            .filter(|prefix| self.pending.ends_with(prefix))
            .map(str::len)
            .max()
            .unwrap_or(0);
        let remaining = self.pending.split_off(self.pending.len() - keep);
        std::mem::replace(&mut self.pending, remaining)
    }

    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stop_sequence_never_leaks_across_chunks() {
        let mut filter = StopFilter::new(vec!["終わり".into(), "END".into()]);
        assert_eq!(filter.push("hello E"), "hello ");
        assert_eq!(filter.push("N"), "");
        assert_eq!(filter.push("D trailing"), "");
        assert_eq!(filter.hit.as_deref(), Some("END"));
        assert_eq!(filter.finish(), "");
    }
    #[test]
    fn incomplete_prefix_is_flushed() {
        let mut filter = StopFilter::new(vec!["END".into()]);
        assert_eq!(filter.push("hello EN"), "hello ");
        assert_eq!(filter.finish(), "EN");
    }
    #[test]
    fn unicode_suffix_is_preserved() {
        let mut filter = StopFilter::new(vec!["終わり".into()]);
        assert_eq!(filter.push("你好終"), "你好");
        assert_eq!(filter.push("わり後"), "");
        assert_eq!(filter.hit.as_deref(), Some("終わり"));
    }
}
