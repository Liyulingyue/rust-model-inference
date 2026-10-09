//! Commit position only after validated device state reaches the CPU shadow.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TokenCommitState {
    committed_len: usize,
    capacity: usize,
    pending: Option<(usize, usize)>,
}

impl TokenCommitState {
    pub(crate) fn new(committed_len: usize, capacity: usize) -> Self {
        Self {
            committed_len,
            capacity,
            pending: None,
        }
    }

    pub(crate) fn begin(&mut self, position: usize, rows: usize) -> Result<(), String> {
        if self.pending.is_some() {
            return Err("a Vulkan chunk is already pending".into());
        }
        if position != self.committed_len {
            return Err(format!(
                "Vulkan token position {position} does not match committed length {}",
                self.committed_len
            ));
        }
        if rows == 0
            || position
                .checked_add(rows)
                .is_none_or(|end| end > self.capacity)
        {
            return Err("Vulkan chunk exceeds capacity or has zero rows".into());
        }
        self.pending = Some((position, rows));
        Ok(())
    }

    pub(crate) fn commit(&mut self) {
        if let Some((base, rows)) = self.pending.take() {
            self.committed_len = base + rows;
        }
    }

    pub(crate) fn abort(&mut self) {
        self.pending = None;
    }

    pub(crate) fn committed_len(&self) -> usize {
        self.committed_len
    }

    pub(crate) fn reset(&mut self) {
        self.committed_len = 0;
        self.pending = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn token_commit_failed_chunk_does_not_advance_committed_kv() {
        let mut state = TokenCommitState::new(7, 16);
        state.begin(7, 3).unwrap();
        state.abort();
        assert_eq!(state.committed_len(), 7);
    }

    #[test]
    fn token_commit_chunk_advances_once_and_checks_capacity() {
        let mut state = TokenCommitState::new(7, 9);
        assert!(state.begin(6, 3).is_err());
        assert!(state.begin(7, 0).is_err());
        assert!(state.begin(7, 3).is_err());
        assert!(state.begin(7, usize::MAX).is_err());
        let mut state = TokenCommitState::new(7, 10);
        state.begin(7, 3).unwrap();
        assert!(state.begin(7, 3).is_err());
        state.commit();
        state.commit();
        assert_eq!(state.committed_len(), 10);
    }

    #[test]
    fn reset_rewinds_committed_length() {
        let mut state = TokenCommitState::new(7, 10);
        state.reset();
        assert_eq!(state.committed_len(), 0);
        state.begin(0, 3).unwrap();
    }
}
