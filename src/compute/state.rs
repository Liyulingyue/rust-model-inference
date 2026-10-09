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

/// Validate every layer before publishing any part of a device KV delta.
#[allow(clippy::too_many_arguments)]
pub(crate) fn commit_kv_cache(
    cache: &mut crate::core::scratchpad::KvCache,
    n_layer: usize,
    capacity: usize,
    stride: usize,
    committed_len: usize,
    position: usize,
    rows: usize,
    k_delta: &[f32],
    v_delta: &[f32],
) -> Result<(), String> {
    use crate::core::scratchpad::KvCache;
    let delta_len = n_layer
        .checked_mul(stride)
        .and_then(|len| len.checked_mul(rows))
        .ok_or_else(|| "KV delta length overflow".to_string())?;
    if rows == 0
        || position != committed_len
        || position.checked_add(rows).is_none_or(|end| end > capacity)
        || k_delta.len() != delta_len
        || v_delta.len() != delta_len
        || k_delta.iter().chain(v_delta).any(|value| {
            !value.is_finite()
                || (matches!(cache, KvCache::F16(_))
                    && crate::ops::f32_to_f16(*value) & 0x7c00 == 0x7c00)
        })
    {
        return Err(format!(
            "Invalid Vulkan KV delta: position={position}/{} k={} v={} expected={delta_len}",
            capacity,
            k_delta.len(),
            v_delta.len()
        ));
    }
    let cache_len = n_layer
        .checked_mul(capacity)
        .and_then(|len| len.checked_mul(stride))
        .ok_or_else(|| "KV cache length overflow".to_string())?;
    let cache_lengths_match = match &*cache {
        KvCache::F16(cache) => cache.k.len() == cache_len && cache.v.len() == cache_len,
        KvCache::F32(cache) => cache.k.len() == cache_len && cache.v.len() == cache_len,
    };
    if !cache_lengths_match {
        return Err("Invalid CPU shadow KV cache length".into());
    }

    match cache {
        KvCache::F16(cache) => {
            for layer in 0..n_layer {
                let source = layer * rows * stride;
                let target = (layer * capacity + position) * stride;
                for index in 0..rows * stride {
                    cache.k[target + index] = crate::ops::f32_to_f16(k_delta[source + index]);
                    cache.v[target + index] = crate::ops::f32_to_f16(v_delta[source + index]);
                }
            }
        }
        KvCache::F32(cache) => {
            for layer in 0..n_layer {
                let source = layer * rows * stride;
                let target = (layer * capacity + position) * stride;
                cache.k[target..target + rows * stride]
                    .copy_from_slice(&k_delta[source..source + rows * stride]);
                cache.v[target..target + rows * stride]
                    .copy_from_slice(&v_delta[source..source + rows * stride]);
            }
        }
    }
    Ok(())
}
