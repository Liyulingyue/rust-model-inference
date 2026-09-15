use crate::core::tensor::{MetaValue, TensorSource};

const PLANNER: &str = "qwen_drive_planner";
const PERCEPTION: &str = "qwen_drive_perception";

pub fn architecture<S: TensorSource + ?Sized>(source: &S) -> Result<&str, String> {
    source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        .ok_or_else(|| "Missing string metadata: general.architecture".to_string())
}

fn require_architecture<S: TensorSource + ?Sized>(
    source: &S,
    expected: &str,
) -> Result<(), String> {
    let actual = architecture(source)?;
    if actual != expected {
        return Err(format!("Expected {expected} GGUF, got {actual}"));
    }
    Ok(())
}

fn fixed_usize<S: TensorSource + ?Sized>(
    source: &S,
    architecture: &str,
    key: &str,
    expected: usize,
) -> Result<usize, String> {
    let name = format!("{architecture}.{key}");
    let value = source
        .metadata(&name)
        .and_then(MetaValue::to_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("Missing integer metadata: {name}"))?;
    if value != expected {
        return Err(format!("Invalid {name}: {value}; expected {expected}"));
    }
    Ok(value)
}

fn fixed_f32<S: TensorSource + ?Sized>(
    source: &S,
    architecture: &str,
    key: &str,
    expected: f32,
) -> Result<f32, String> {
    let name = format!("{architecture}.{key}");
    let value = source
        .metadata(&name)
        .and_then(MetaValue::to_f64)
        .map(|value| value as f32)
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("Missing finite numeric metadata: {name}"))?;
    if value.to_bits() != expected.to_bits() {
        return Err(format!("Invalid {name}: {value}; expected {expected}"));
    }
    Ok(value)
}

fn fixed_usize_array<const N: usize, S: TensorSource + ?Sized>(
    source: &S,
    architecture: &str,
    key: &str,
    expected: [usize; N],
) -> Result<[usize; N], String> {
    let name = format!("{architecture}.{key}");
    let values = source
        .metadata(&name)
        .and_then(MetaValue::to_arr)
        .ok_or_else(|| format!("Missing integer-array metadata: {name}"))?;
    let actual: Vec<usize> = values
        .iter()
        .map(|value| {
            value
                .to_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| format!("Invalid integer-array metadata: {name}"))
        })
        .collect::<Result<_, _>>()?;
    let actual: [usize; N] = actual
        .try_into()
        .map_err(|_| format!("Invalid {name} length; expected {N}"))?;
    if actual != expected {
        return Err(format!("Invalid {name}: {actual:?}; expected {expected:?}"));
    }
    Ok(actual)
}

fn fixed_f32_array<const N: usize, S: TensorSource + ?Sized>(
    source: &S,
    architecture: &str,
    key: &str,
    expected: [f32; N],
) -> Result<[f32; N], String> {
    let name = format!("{architecture}.{key}");
    let values = source
        .metadata(&name)
        .and_then(MetaValue::to_arr)
        .ok_or_else(|| format!("Missing float-array metadata: {name}"))?;
    let actual: Vec<f32> = values
        .iter()
        .map(|value| {
            value
                .to_f64()
                .map(|value| value as f32)
                .filter(|value| value.is_finite())
                .ok_or_else(|| format!("Invalid float-array metadata: {name}"))
        })
        .collect::<Result<_, _>>()?;
    let actual: [f32; N] = actual
        .try_into()
        .map_err(|_| format!("Invalid {name} length; expected {N}"))?;
    if actual
        .iter()
        .zip(expected)
        .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
    {
        return Err(format!("Invalid {name}: {actual:?}; expected {expected:?}"));
    }
    Ok(actual)
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlannerConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub layers_per_kv: usize,
    pub rms_eps: f32,
    pub future_points: usize,
    pub history_points: usize,
    pub point_dim: usize,
    pub nav_classes: usize,
    pub ego_status_dim: usize,
    pub history_dynamics_dim: usize,
    pub time_embed_dim: usize,
    pub time_embed_scale: f32,
    pub fourier_features: usize,
    pub fourier_max_frequency: f32,
    pub mrope_sections: [usize; 3],
    pub rope_theta: f32,
    pub trajectory_scale: [f32; 3],
    pub steps: usize,
    pub min_one_minus_t: f32,
    pub noise_init_std: f32,
    pub noise_seed: usize,
    pub trajectory_hz: f32,
    pub max_reasoning_tokens: usize,
}

impl PlannerConfig {
    pub fn from_source<S: TensorSource + ?Sized>(source: &S) -> Result<Self, String> {
        require_architecture(source, PLANNER)?;
        Ok(Self {
            hidden: fixed_usize(source, PLANNER, "hidden_size", 1024)?,
            intermediate: fixed_usize(source, PLANNER, "intermediate_size", 3584)?,
            layers: fixed_usize(source, PLANNER, "num_hidden_layers", 32)?,
            heads: fixed_usize(source, PLANNER, "num_attention_heads", 16)?,
            kv_heads: fixed_usize(source, PLANNER, "num_key_value_heads", 4)?,
            head_dim: fixed_usize(source, PLANNER, "head_dim", 256)?,
            layers_per_kv: fixed_usize(source, PLANNER, "layers_per_kv", 4)?,
            rms_eps: fixed_f32(source, PLANNER, "rms_norm_eps", 1e-5)?,
            future_points: fixed_usize(source, PLANNER, "num_future_points", 50)?,
            history_points: fixed_usize(source, PLANNER, "num_history_points", 16)?,
            point_dim: fixed_usize(source, PLANNER, "trajectory_point_dim", 3)?,
            nav_classes: fixed_usize(source, PLANNER, "nav_command_classes", 3)?,
            ego_status_dim: fixed_usize(source, PLANNER, "ego_status_dim", 8)?,
            history_dynamics_dim: fixed_usize(source, PLANNER, "history_dynamics_dim", 2)?,
            time_embed_dim: fixed_usize(source, PLANNER, "time_embed_dim", 128)?,
            time_embed_scale: fixed_f32(source, PLANNER, "time_embed_scale", 1000.0)?,
            fourier_features: fixed_usize(source, PLANNER, "fourier_num_features", 16)?,
            fourier_max_frequency: fixed_f32(source, PLANNER, "fourier_max_frequency", 16.0)?,
            mrope_sections: fixed_usize_array(source, PLANNER, "mrope_section", [11, 11, 10])?,
            rope_theta: fixed_f32(source, PLANNER, "rope_theta", 10_000_000.0)?,
            trajectory_scale: fixed_f32_array(
                source,
                PLANNER,
                "trajectory_scale",
                [165.0, 25.0, 1.5703125],
            )?,
            steps: fixed_usize(source, PLANNER, "num_inference_steps", 10)?,
            min_one_minus_t: fixed_f32(source, PLANNER, "min_one_minus_t", 0.1)?,
            noise_init_std: fixed_f32(source, PLANNER, "noise_init_std", 1.0)?,
            noise_seed: fixed_usize(source, PLANNER, "noise_seed", 42)?,
            trajectory_hz: fixed_f32(source, PLANNER, "trajectory_hz", 10.0)?,
            max_reasoning_tokens: fixed_usize(source, PLANNER, "max_reasoning_tokens", 256)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PerceptionConfig {
    pub llm_dim: usize,
    pub vit_dim: usize,
    pub embed_dim: usize,
    pub bev: [usize; 2],
    pub occ_pillar_h: usize,
    pub occ_dim: usize,
    pub num_queries: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub det_classes: usize,
    pub occ_classes: usize,
    pub map_classes: usize,
    pub code_size: usize,
    pub image_size: [usize; 2],
    pub det_pc_range: [f32; 6],
    pub det_voxel_size: [f32; 3],
    pub nuscenes_occ_pc_range: [f32; 6],
    pub nuscenes_occ_voxel_size: [f32; 3],
    pub nuplan_occ_pc_range: [f32; 6],
    pub nuplan_occ_voxel_size: [f32; 3],
    pub map_xbound: [f32; 3],
    pub map_ybound: [f32; 3],
    pub frustum_range: [f32; 6],
    pub frustum_size: [f32; 3],
}

impl PerceptionConfig {
    pub fn from_source<S: TensorSource + ?Sized>(source: &S) -> Result<Self, String> {
        require_architecture(source, PERCEPTION)?;
        Ok(Self {
            llm_dim: fixed_usize(source, PERCEPTION, "llm_dim", 2560)?,
            vit_dim: fixed_usize(source, PERCEPTION, "vit_dim", 1024)?,
            embed_dim: fixed_usize(source, PERCEPTION, "embed_dim", 256)?,
            bev: [
                fixed_usize(source, PERCEPTION, "bev_h", 200)?,
                fixed_usize(source, PERCEPTION, "bev_w", 200)?,
            ],
            occ_pillar_h: fixed_usize(source, PERCEPTION, "occ_pillar_h", 16)?,
            occ_dim: fixed_usize(source, PERCEPTION, "occ_dim", 32)?,
            num_queries: fixed_usize(source, PERCEPTION, "num_query", 900)?,
            encoder_layers: fixed_usize(source, PERCEPTION, "num_encoder_layers", 6)?,
            decoder_layers: fixed_usize(source, PERCEPTION, "num_decoder_layers", 6)?,
            det_classes: fixed_usize(source, PERCEPTION, "det_num_classes", 7)?,
            occ_classes: fixed_usize(source, PERCEPTION, "occ_num_classes", 10)?,
            map_classes: fixed_usize(source, PERCEPTION, "map_num_classes", 6)?,
            code_size: fixed_usize(source, PERCEPTION, "code_size", 10)?,
            image_size: fixed_usize_array(source, PERCEPTION, "image_size", [896, 512])?,
            det_pc_range: fixed_f32_array(
                source,
                PERCEPTION,
                "det_pc_range",
                [-51.2, -51.2, -5.0, 51.2, 51.2, 5.4],
            )?,
            det_voxel_size: fixed_f32_array(
                source,
                PERCEPTION,
                "det_voxel_size",
                [0.512, 0.512, 10.4],
            )?,
            nuscenes_occ_pc_range: fixed_f32_array(
                source,
                PERCEPTION,
                "nuscenes_occ_pc_range",
                [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
            )?,
            nuscenes_occ_voxel_size: fixed_f32_array(
                source,
                PERCEPTION,
                "nuscenes_occ_voxel_size",
                [0.4, 0.4, 6.4],
            )?,
            nuplan_occ_pc_range: fixed_f32_array(
                source,
                PERCEPTION,
                "nuplan_occ_pc_range",
                [-50.0, -50.0, -4.0, 50.0, 50.0, 4.0],
            )?,
            nuplan_occ_voxel_size: fixed_f32_array(
                source,
                PERCEPTION,
                "nuplan_occ_voxel_size",
                [0.5, 0.5, 0.5],
            )?,
            map_xbound: fixed_f32_array(source, PERCEPTION, "map_xbound", [-30.0, 30.0, 0.15])?,
            map_ybound: fixed_f32_array(source, PERCEPTION, "map_ybound", [-15.0, 15.0, 0.15])?,
            frustum_range: fixed_f32_array(
                source,
                PERCEPTION,
                "frustum_range",
                [0.0, 0.0, 1.0, 896.0, 512.0, 60.0],
            )?,
            frustum_size: fixed_f32_array(source, PERCEPTION, "frustum_size", [16.0, 16.0, 0.5])?,
        })
    }
}
