use super::{IMAGE, PAGE};

pub(super) struct Pages {
    pub coords: Vec<(usize, usize)>,
    pub image_pages: Vec<Option<usize>>,
    pub query_pages: Vec<Option<usize>>,
    counts: Vec<usize>,
    markers: usize,
    prompt_pages: usize,
}

impl Pages {
    pub fn new(ids: &[u32]) -> Result<Self, String> {
        let last = ids
            .iter()
            .rposition(|&id| id == IMAGE)
            .ok_or("Unlimited-OCR prompt has no image tokens")?;
        let prompt_pages = ids[..=last].iter().filter(|&&id| id == PAGE).count();
        if prompt_pages == 0 {
            return Err("Unlimited-OCR requires page markers before image runs".into());
        }
        let mut result = Self {
            coords: Vec::new(),
            image_pages: Vec::new(),
            query_pages: Vec::new(),
            counts: vec![0; prompt_pages + 1],
            markers: 0,
            prompt_pages,
        };
        for (i, &id) in ids.iter().enumerate() {
            result.markers += usize::from(id == PAGE);
            let page = if i <= last {
                result.markers
            } else {
                result.markers.saturating_sub(prompt_pages)
            };
            if page > prompt_pages {
                return Err("Invalid Unlimited-OCR prompt page sequence".into());
            }
            result.coords.push((result.counts[page], page));
            result.counts[page] += 1;
            result
                .image_pages
                .push((id == IMAGE).then(|| result.markers - 1));
            result.query_pages.push(if i <= last {
                result.markers.checked_sub(1)
            } else {
                result.markers.checked_sub(prompt_pages + 1)
            });
        }
        Ok(result)
    }

    pub fn push(&mut self, id: u32) {
        let (tok, page) = *self.coords.last().unwrap();
        let coordinate = if id == PAGE {
            self.markers += 1;
            let p = (page + 1).min(self.prompt_pages);
            (self.counts[p], p)
        } else {
            (tok + 1, page)
        };
        self.coords.push(coordinate);
        self.image_pages.push(None);
        self.query_pages
            .push(self.markers.checked_sub(self.prompt_pages + 1));
    }

    pub fn candidate_blocks(&self, query: usize, nblocks: usize) -> Vec<bool> {
        let page = self.query_pages[query];
        let mut allowed = vec![false; nblocks];
        if let Some(page) = page {
            for (i, p) in self.image_pages.iter().enumerate() {
                if *p == Some(page) {
                    allowed[i / 32] = true;
                }
            }
        }
        if !allowed.iter().any(|&x| x) {
            allowed.fill(true);
        }
        allowed
    }
}

/// Neox layout, 63 resampled token frequency pairs and one alternating page pair.
pub(super) fn rope(x: &mut [f32], coords: &[(usize, usize)], heads: usize) {
    for (row, &(tok, page)) in x.chunks_exact_mut(heads * 128).zip(coords) {
        for head in row.chunks_exact_mut(128) {
            for i in 0..64 {
                let angle = if i == 63 {
                    (page % 2) as f32
                } else {
                    tok as f32 * (1.0 / 10000_f32.powf(i as f32 / 63.0))
                };
                let (sin, cos) = angle.sin_cos();
                let a = head[i];
                let b = head[i + 64];
                head[i] = a * cos - b * sin;
                head[i + 64] = b * cos + a * sin;
            }
        }
    }
}

pub(super) fn entmax15(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let x: Vec<f32> = scores.iter().map(|v| (v - max) / 2.0).collect();
    let mut sorted = x.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let (mut sum, mut sq, mut tau) = (0.0, 0.0, 0.0);
    for (i, &v) in sorted.iter().enumerate() {
        sum += v;
        sq += v * v;
        let n = (i + 1) as f32;
        let mean = sum / n;
        let t = mean - ((1.0 - n * (sq / n - mean * mean)) / n).max(0.0).sqrt();
        if t <= v {
            tau = t;
        }
    }
    x.iter().map(|v| (v - tau).max(0.0).powi(2)).collect()
}

/// Top-16 entmax prior, top-12 read set; fall back to all remote blocks if needed.
pub(super) fn route(scores: &[f32], allowed: &[bool]) -> Vec<(usize, f32)> {
    let mut choices: Vec<_> = scores
        .iter()
        .enumerate()
        .filter(|(i, s)| s.is_finite() && **s > -5.0e8 && allowed[*i])
        .map(|(i, &s)| (i, s))
        .collect();
    if choices.is_empty() {
        choices = scores
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_finite() && **s > -5.0e8)
            .map(|(i, &s)| (i, s))
            .collect();
    }
    choices.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    choices.truncate(16);
    let prior = entmax15(
        &choices
            .iter()
            .map(|x| x.1 * (128.0 / 128_f32.sqrt()))
            .collect::<Vec<_>>(),
    );
    choices.truncate(12);
    let logp: Vec<_> = prior
        .iter()
        .take(choices.len())
        .map(|p| p.max(0.0001).ln())
        .collect();
    let mean = logp.iter().sum::<f32>() / logp.len().max(1) as f32;
    choices
        .iter()
        .zip(logp)
        .map(|(x, p)| (x.0, p - mean))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn page_coordinates_rope_and_routes_match_pinned_official_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/unlimited_ocr_oracle.json"
        ))
        .unwrap();
        let mut ids = vec![0, PAGE];
        ids.extend(std::iter::repeat_n(IMAGE, 273));
        ids.push(PAGE);
        ids.extend(std::iter::repeat_n(IMAGE, 273));
        ids.extend([37460, 4366, 76466, 16]);
        let pages = Pages::new(&ids).unwrap();
        for (index, coord) in fixture["coordinate_samples"].as_object().unwrap() {
            assert_eq!(
                pages.coords[index.parse::<usize>().unwrap()],
                (
                    coord[0].as_u64().unwrap() as usize,
                    coord[1].as_u64().unwrap() as usize
                )
            );
        }
        let coords: Vec<_> = fixture["rope"]["coords"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                (
                    v[0].as_u64().unwrap() as usize,
                    v[1].as_u64().unwrap() as usize,
                )
            })
            .collect();
        let mut rotated = vec![1.0; coords.len() * 128];
        rope(&mut rotated, &coords, 1);
        for (actual, expected) in rotated
            .iter()
            .zip(fixture["rope"]["ones_rotated"].as_array().unwrap())
        {
            assert!((actual - expected.as_f64().unwrap() as f32).abs() < 0.002);
        }
        let fixture = &fixture["route"];
        let scores: Vec<_> = fixture["scores"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let allowed: Vec<_> = fixture["allowed"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect();
        let routed = route(&scores, &allowed);
        assert_eq!(routed.len(), fixture["indices"].as_array().unwrap().len());
        for (i, &(block, bias)) in routed.iter().enumerate() {
            assert_eq!(block, fixture["indices"][i].as_u64().unwrap() as usize);
            assert!((bias - fixture["bias"][i].as_f64().unwrap() as f32).abs() < 0.0001);
        }
    }
    #[test]
    fn page_rope_continues_each_prefill_bucket_on_decode() {
        let mut p = Pages::new(&[0, PAGE, IMAGE, IMAGE, PAGE, IMAGE, 99]).unwrap();
        assert_eq!(
            p.coords,
            [(0, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (1, 0)]
        );
        p.push(PAGE);
        p.push(22);
        p.push(PAGE);
        p.push(23);
        assert_eq!(&p.coords[7..], [(3, 1), (4, 1), (2, 2), (3, 2)]);
        assert_eq!(&p.query_pages[7..], [Some(0), Some(0), Some(1), Some(1)]);
    }
    #[test]
    fn page_candidates_and_no_candidate_fallback() {
        let mut ids = vec![0, PAGE];
        ids.extend(std::iter::repeat_n(IMAGE, 64));
        ids.push(PAGE);
        ids.extend(std::iter::repeat_n(IMAGE, 64));
        ids.push(42);
        let mut p = Pages::new(&ids).unwrap();
        assert_eq!(p.candidate_blocks(68, 5), [false, false, true, true, true]);
        assert_eq!(p.candidate_blocks(131, 5), [true; 5]);
        p.push(PAGE);
        assert_eq!(p.candidate_blocks(132, 5), [true, true, true, false, false]);
    }
    #[test]
    fn entmax_is_sparse_and_shift_invariant() {
        assert!(entmax15(&[0.0, 0.0]).iter().all(|x| (x - 0.5).abs() < 1e-7));
        assert_eq!(entmax15(&[5.0, 0.0, -10.0]), [1.0, 0.0, 0.0]);
        assert_eq!(entmax15(&[105.0, 100.0, 90.0]), [1.0, 0.0, 0.0]);
    }
    #[test]
    fn router_obeys_candidates_but_falls_back_if_all_are_local() {
        assert_eq!(route(&[0.2, 0.8, -1e9], &[true, false, true]), [(0, 0.0)]);
        assert_eq!(route(&[0.2, -1e9], &[false, true]), [(0, 0.0)]);
        assert!(route(&[-1e9; 3], &[true; 3]).is_empty());
    }
}
