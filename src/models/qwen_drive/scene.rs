use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Component, Path, PathBuf};

pub const CAMERA_VIEWS: [&str; 3] = ["<FRONT VIEW>", "<FRONT LEFT VIEW>", "<FRONT RIGHT VIEW>"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraFrame {
    pub image: PathBuf,
    pub resized_width: Option<usize>,
    pub resized_height: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraView {
    pub label: &'static str,
    pub frames: Vec<CameraFrame>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanningScene {
    pub token: String,
    pub views: Vec<CameraView>,
    pub instruction: String,
    pub history: Vec<[f32; 3]>,
    pub history_velocity: Vec<[f32; 2]>,
    pub history_acceleration: Vec<[f32; 2]>,
    pub nav_command: usize,
    pub ego_status: [f32; 8],
}

#[derive(Deserialize)]
struct SceneRecord {
    messages: Vec<SceneMessage>,
    trajectory: TrajectoryRecord,
    #[serde(default)]
    meta_info: SceneMeta,
}

#[derive(Deserialize)]
struct SceneMessage {
    role: String,
    content: Vec<SceneContent>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum SceneContent {
    Image {
        image: PathBuf,
        resized_width: Option<usize>,
        resized_height: Option<usize>,
    },
    Text {
        text: String,
    },
}

#[derive(Deserialize)]
struct TrajectoryRecord {
    hist_traj_1p5s_10hz: Option<Vec<[f32; 3]>>,
    hist_traj_10hz: Option<Vec<[f32; 3]>>,
    hist_vel_1p5s_10hz: Option<Vec<[f32; 2]>>,
    hist_vel_10hz: Option<Vec<[f32; 2]>>,
    hist_acc_1p5s_10hz: Option<Vec<[f32; 2]>>,
    hist_acc_10hz: Option<Vec<[f32; 2]>>,
    ego_status: EgoStatus,
    nav_command: usize,
}

#[derive(Deserialize)]
struct EgoStatus {
    ego_velocity: [f32; 2],
    ego_acceleration: [f32; 2],
    driving_command: [f32; 4],
}

#[derive(Default, Deserialize)]
struct SceneMeta {
    #[serde(default)]
    token: String,
}

fn finite<const N: usize>(values: &[[f32; N]], label: &str) -> Result<(), String> {
    if values.iter().flatten().any(|value| !value.is_finite()) {
        return Err(format!("{label} contains a non-finite value"));
    }
    Ok(())
}

fn history<const N: usize>(
    preferred: Option<Vec<[f32; N]>>,
    fallback: Option<Vec<[f32; N]>>,
    count: usize,
    label: &str,
) -> Result<Vec<[f32; N]>, String> {
    let mut values = preferred
        .filter(|values| !values.is_empty())
        .or(fallback)
        .ok_or_else(|| format!("{label} is missing"))?;
    if values.is_empty() {
        return Err(format!("{label} is empty"));
    }
    finite(&values, label)?;
    if values.len() < count {
        let first = values[0];
        let mut padded = vec![first; count - values.len()];
        padded.append(&mut values);
        values = padded;
    }
    Ok(values.split_off(values.len() - count))
}

fn resolve_image(root: &Path, relative: &Path) -> Result<PathBuf, String> {
    if relative.is_absolute()
        || relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_) | Component::CurDir))
    {
        return Err(format!(
            "Image path escapes image root: {}",
            relative.display()
        ));
    }
    let path = root.join(relative);
    let resolved = path
        .canonicalize()
        .map_err(|error| format!("Cannot resolve image {}: {error}", path.display()))?;
    if !resolved.starts_with(root) {
        return Err(format!(
            "Image path escapes image root: {}",
            relative.display()
        ));
    }
    Ok(resolved)
}

fn planning_scene(
    record: SceneRecord,
    image_root: &Path,
    history_points: usize,
) -> Result<PlanningScene, String> {
    let message = record
        .messages
        .first()
        .ok_or("Planning scene has no messages")?;
    if message.role != "user" {
        return Err("Planning scene first message must be user".into());
    }
    let images: Vec<_> = message
        .content
        .iter()
        .filter_map(|item| match item {
            SceneContent::Image {
                image,
                resized_width,
                resized_height,
            } => Some((image, *resized_width, *resized_height)),
            SceneContent::Text { .. } => None,
        })
        .collect();
    let (per_view, remainder) = (
        images.len() / CAMERA_VIEWS.len(),
        images.len() % CAMERA_VIEWS.len(),
    );
    if per_view == 0 || remainder != 0 {
        return Err(format!(
            "Expected a whole number of frames per camera view, got {} images",
            images.len()
        ));
    }
    let mut views = Vec::with_capacity(CAMERA_VIEWS.len());
    for (index, label) in CAMERA_VIEWS.into_iter().enumerate() {
        let mut frames = Vec::with_capacity(per_view);
        for (image, width, height) in &images[index * per_view..(index + 1) * per_view] {
            if width.is_some() != height.is_some() || width == &Some(0) || height == &Some(0) {
                return Err(format!("Invalid resize dimensions for {}", image.display()));
            }
            frames.push(CameraFrame {
                image: resolve_image(image_root, image)?,
                resized_width: *width,
                resized_height: *height,
            });
        }
        views.push(CameraView { label, frames });
    }
    let instruction = message
        .content
        .iter()
        .filter_map(|item| match item {
            SceneContent::Text { text } => Some(text.as_str()),
            SceneContent::Image { .. } => None,
        })
        .next_back()
        .filter(|text| !text.is_empty())
        .ok_or("Planning scene has no instruction text")?
        .to_owned();
    let trajectory = record.trajectory;
    if trajectory.nav_command >= 3 {
        return Err(format!(
            "Invalid navigation command {}",
            trajectory.nav_command
        ));
    }
    let ego_status = [
        trajectory.ego_status.ego_velocity[0],
        trajectory.ego_status.ego_velocity[1],
        trajectory.ego_status.ego_acceleration[0],
        trajectory.ego_status.ego_acceleration[1],
        trajectory.ego_status.driving_command[0],
        trajectory.ego_status.driving_command[1],
        trajectory.ego_status.driving_command[2],
        trajectory.ego_status.driving_command[3],
    ];
    if ego_status.iter().any(|value| !value.is_finite()) {
        return Err("ego_status contains a non-finite value".into());
    }
    Ok(PlanningScene {
        token: record.meta_info.token,
        views,
        instruction,
        history: history(
            trajectory.hist_traj_1p5s_10hz,
            trajectory.hist_traj_10hz,
            history_points,
            "hist_traj",
        )?,
        history_velocity: history(
            trajectory.hist_vel_1p5s_10hz,
            trajectory.hist_vel_10hz,
            history_points,
            "hist_vel",
        )?,
        history_acceleration: history(
            trajectory.hist_acc_1p5s_10hz,
            trajectory.hist_acc_10hz,
            history_points,
            "hist_acc",
        )?,
        nav_command: trajectory.nav_command,
        ego_status,
    })
}

pub fn read_planning_scenes(
    path: &Path,
    image_root: &Path,
    limit: Option<usize>,
) -> Result<Vec<PlanningScene>, String> {
    let image_root = image_root.canonicalize().map_err(|error| {
        format!(
            "Cannot resolve image root {}: {error}",
            image_root.display()
        )
    })?;
    let file = File::open(path)
        .map_err(|error| format!("Cannot open planning scenes {}: {error}", path.display()))?;
    let mut scenes = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        if limit.is_some_and(|limit| scenes.len() >= limit) {
            break;
        }
        let line =
            line.map_err(|error| format!("Cannot read scene line {}: {error}", index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: SceneRecord = serde_json::from_str(&line)
            .map_err(|error| format!("Invalid scene line {}: {error}", index + 1))?;
        scenes.push(planning_scene(record, &image_root, 16)?);
    }
    Ok(scenes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen_drive/planning-scene.jsonl")
    }

    #[test]
    fn qwen_drive_scene_uses_trailing_sixteen_history_points_and_view_order() {
        let root =
            std::env::temp_dir().join(format!("rmi-qwen-drive-scene-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        for name in ["front.jpg", "front-left.jpg", "front-right.jpg"] {
            std::fs::write(root.join(name), []).unwrap();
        }
        let scene = read_planning_scenes(&fixture(), &root, Some(1))
            .unwrap()
            .remove(0);
        assert_eq!(scene.views.len(), 3);
        assert_eq!(scene.views[0].label, "<FRONT VIEW>");
        assert_eq!(scene.history.len(), 16);
        assert_eq!(scene.history[0][0], 1.0);
        assert_eq!(scene.history[15][0], 16.0);
        assert_eq!(scene.nav_command, 0);
        assert_eq!(scene.ego_status.len(), 8);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn qwen_drive_scene_rejects_image_path_traversal() {
        let root = std::env::temp_dir();
        assert!(resolve_image(&root, Path::new("../outside.jpg"))
            .unwrap_err()
            .contains("escapes image root"));
    }
}
