//! Oracle companion; see tools/oracle/hy_simulmt/README.md.
use rust_model_inference::core::{
    loader::GGUFLoader,
    scratchpad::KvFormat,
    tensor::TensorSource,
    thread_pool::ComputePool,
    tokenizer::{BPETokenizer, EncodeOptions},
};
use rust_model_inference::models::hy_simulmt::{HyAudioEncoder, HySimulMt};
use std::path::Path;
use std::sync::Arc;

fn pcm(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        return Err("invalid f32 PCM/vector".into());
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    if values.iter().any(|x| !x.is_finite()) {
        return Err("non-finite f32 input".into());
    }
    Ok(values)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 8 {
        return Err("usage: MODE MODEL FIXTURE OUT THREADS MAXTOK INTERLEAVE".into());
    }
    let s: Arc<dyn TensorSource> = Arc::new(GGUFLoader::from_file(&a[2])?);
    std::fs::create_dir_all(&a[4])?;
    if a[1] == "tokens" {
        let t = BPETokenizer::from_gguf_metadata(|k| s.metadata(k).cloned())?;
        let mut cases: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(&a[3])?)?;
        for c in &mut cases {
            c["ids"] = serde_json::to_value(t.encode(
                c["text"].as_str().ok_or("text")?,
                EncodeOptions {
                    add_special: c["add_special"].as_bool().ok_or("add_special")?,
                    parse_special: c["parse_special"].as_bool().ok_or("parse_special")?,
                },
            ))?;
        }
        std::fs::write(
            format!("{}/tokens.json", a[4]),
            serde_json::to_vec_pretty(&cases)?,
        )?;
        return Ok(());
    }
    if a[1] == "projections" {
        #[cfg(feature = "parity-trace")]
        {
            use rust_model_inference::core::tensor::GGMLType;
            use rust_model_inference::ops::kernel::{QuantizedTensor, Weight};
            use rust_model_inference::parity_trace::{checkpoint, report};
            let cases: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(&a[3])?)?;
            for c in cases {
                let name = c["tensor"].as_str().ok_or("tensor")?;
                let expert = c["expert"].as_u64().ok_or("expert")? as usize;
                let info = s.tensor_info(name).ok_or("missing projection")?;
                if !(2..=3).contains(&info.dims.len())
                    || !matches!(
                        info.ggml_type,
                        GGMLType::Q4K | GGMLType::Q6K | GGMLType::F32
                    )
                {
                    return Err("unsupported projection probe".into());
                }
                let experts = info.dims.get(2).copied().unwrap_or(1) as usize;
                if expert >= experts {
                    return Err("expert out of range".into());
                }
                let (ni, no) = (info.dims[0] as usize, info.dims[1] as usize);
                let data = s.tensor_slice(name).ok_or("missing projection bytes")?;
                let size = usize::try_from(info.checked_nbytes().ok_or("projection size")?)?;
                if data.len() != size {
                    return Err("truncated projection".into());
                }
                let size = size / experts;
                let weight = Weight::from_quantized(QuantizedTensor::from_bytes(
                    &data[expert * size..(expert + 1) * size],
                    info.ggml_type,
                    ni,
                    no,
                ));
                let input = pcm(&Path::new(&a[3])
                    .parent()
                    .unwrap()
                    .join(c["input"].as_str().ok_or("input")?))?;
                if input.len() != ni {
                    return Err("projection input shape".into());
                }
                let mut output = vec![0.0; no];
                weight.kernel.forward(&input, &mut output, ni, no);
                report(checkpoint(
                    &format!("hy.projection.{name}.{expert}"),
                    None,
                    &[no],
                    &output,
                ));
            }
            return Ok(());
        }
        #[cfg(not(feature = "parity-trace"))]
        return Err("projections requires --features parity-trace".into());
    }
    let samples = pcm(Path::new(&a[3]))?;
    let pool = Arc::new(ComputePool::new(a[5].parse()?));
    if a[1] == "audio" {
        let enc = HyAudioEncoder::from_source(s, pool)?;
        let mut offset = 0;
        while offset < samples.len() {
            let n = (if offset == 0 { 25600 } else { 12800 }).min(samples.len() - offset);
            enc.encode(&samples[offset..offset + n])?;
            offset += n;
        }
    } else if a[1] == "translate" {
        let model = HySimulMt::from_source(s, pool)?;
        let result = model.translate(
            &samples,
            "zh",
            None,
            a[6].parse()?,
            1024,
            256,
            KvFormat::F32,
        )?;
        std::fs::write(
            format!("{}/turns.json", a[4]),
            serde_json::to_vec_pretty(&result.turns)?,
        )?;
        std::fs::write(format!("{}/translation.txt", a[4]), result.text)?;
    } else {
        return Err("invalid mode".into());
    }
    Ok(())
}
