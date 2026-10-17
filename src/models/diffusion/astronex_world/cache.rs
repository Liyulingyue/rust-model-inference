//! Raw K/V, overwritten during denoising and committed by the final clean pass.

enum Storage {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
}

impl Storage {
    fn resize(&mut self, len: usize) {
        match self {
            Self::F32(v) => v.resize(len, 0.0),
            Self::Bf16(v) => v.resize(len, 0),
        }
    }
    fn roll(&mut self, source: std::ops::Range<usize>, destination: usize) {
        match self {
            Self::F32(v) => v.copy_within(source, destination),
            Self::Bf16(v) => v.copy_within(source, destination),
        }
    }
    fn write(&mut self, start: usize, values: &[f32]) {
        match self {
            Self::F32(v) => v[start..start + values.len()].copy_from_slice(values),
            Self::Bf16(v) => {
                for (dst, &src) in v[start..start + values.len()].iter_mut().zip(values) {
                    *dst = crate::ops::f32_to_bf16(src);
                }
            }
        }
    }
    fn read(&self, len: usize) -> Vec<f32> {
        match self {
            Self::F32(v) => v[..len].to_vec(),
            Self::Bf16(v) => v[..len]
                .iter()
                .map(|&v| f32::from_bits((v as u32) << 16))
                .collect(),
        }
    }
}

pub(super) struct KvCache {
    k: Storage,
    v: Storage,
    global_end: usize,
    local_frames: usize,
    frame_values: usize,
    capacity: usize,
    sink: usize,
}

impl KvCache {
    pub(super) fn new(frame_values: usize, bf16: bool) -> Self {
        let storage = || {
            if bf16 {
                Storage::Bf16(Vec::new())
            } else {
                Storage::F32(Vec::new())
            }
        };
        Self {
            k: storage(),
            v: storage(),
            global_end: 0,
            local_frames: 0,
            frame_values,
            capacity: 20,
            sink: 4,
        }
    }

    pub(super) fn update(
        &mut self,
        start: usize,
        frames: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, usize), String> {
        if frames == 0
            || frames > self.capacity - self.sink
            || k.len() != frames * self.frame_values
            || v.len() != k.len()
        {
            return Err("Invalid Astronex KV block".into());
        }
        let end = start
            .checked_add(frames)
            .ok_or("Astronex KV frame overflow")?;
        if end != self.global_end && start != self.global_end {
            return Err("Astronex KV blocks must be contiguous; repeated steps must overwrite the same block".into());
        }
        let new = end > self.global_end;
        if new {
            let evict = (self.local_frames + frames).saturating_sub(self.capacity);
            if evict != 0 {
                let source =
                    (self.sink + evict) * self.frame_values..self.local_frames * self.frame_values;
                self.k.roll(source.clone(), self.sink * self.frame_values);
                self.v.roll(source, self.sink * self.frame_values);
                self.local_frames -= evict;
            }
            self.local_frames += frames;
            self.global_end = end;
        }
        let len = self.local_frames * self.frame_values;
        self.k.resize(len);
        self.v.resize(len);
        let offset = len - k.len();
        self.k.write(offset, k);
        self.v.write(offset, v);
        Ok((self.k.read(len), self.v.read(len), self.local_frames))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_steps_clean_commit_and_sink_survive_eviction() {
        for bf16 in [false, true] {
            let mut cache = KvCache::new(1, bf16);
            for block in 0..4 {
                let data: Vec<f32> = (block * 8..block * 8 + 8).map(|i| i as f32).collect();
                let (k, _, frames) = cache.update(block * 8, 8, &data, &data).unwrap();
                assert_eq!(frames, ((block + 1) * 8).min(20));
                let noisy = vec![99.0; 8];
                let repeated = cache.update(block * 8, 8, &noisy, &noisy).unwrap().0;
                assert_eq!(repeated.len(), k.len());
                let clean = cache.update(block * 8, 8, &data, &data).unwrap().0;
                assert_eq!(clean, k);
                if block >= 2 {
                    assert_eq!(&clean[..4], &[0.0, 1.0, 2.0, 3.0]);
                    let tail: Vec<f32> = ((block + 1) * 8 - 16..(block + 1) * 8)
                        .map(|i| i as f32)
                        .collect();
                    assert_eq!(&clean[4..], tail);
                }
            }
            assert!(cache.update(40, 8, &[0.0; 8], &[0.0; 8]).is_err());
        }
    }
}
