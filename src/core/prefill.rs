use std::ops::Range;

pub const DEFAULT_PREFILL_BATCH_SIZE: usize = 64;

pub fn checked_prefill_batch_size(value: Option<usize>) -> Result<usize, String> {
    match value.unwrap_or(DEFAULT_PREFILL_BATCH_SIZE) {
        0 => Err("prefill batch size must be at least 1".into()),
        value => Ok(value),
    }
}

pub(crate) fn prefill_chunks(len: usize, batch_size: usize) -> impl Iterator<Item = Range<usize>> {
    (0..len)
        .step_by(batch_size)
        .map(move |start| start..(start + batch_size).min(len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_batch_size_defaults_to_64_and_rejects_zero() {
        assert_eq!(checked_prefill_batch_size(None).unwrap(), 64);
        assert_eq!(checked_prefill_batch_size(Some(1)).unwrap(), 1);
        assert_eq!(checked_prefill_batch_size(Some(128)).unwrap(), 128);
        assert!(checked_prefill_batch_size(Some(0))
            .unwrap_err()
            .contains("at least 1"));
    }

    #[test]
    fn prefill_chunks_cover_input_without_padding() {
        let ranges = prefill_chunks(130, 64).collect::<Vec<_>>();
        assert_eq!(ranges, [0..64, 64..128, 128..130]);
    }
}
