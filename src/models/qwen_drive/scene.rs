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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerceptionContent {
    Text(String),
    Image { camera: String, path: PathBuf },
}

#[derive(Debug, Clone, PartialEq)]
pub struct PerceptionFrame {
    pub token: String,
    pub dataset_type: String,
    pub cam_order: Vec<String>,
    pub content: Vec<PerceptionContent>,
    pub image_shapes: Vec<[usize; 3]>,
    pub lidar2img: Vec<[f32; 16]>,
    pub lidar2ego: [f32; 16],
    pub box_coord_system_ego: bool,
}

#[derive(Deserialize)]
struct PerceptionFrameRecord {
    dataset_type: String,
    cam_order: Vec<String>,
    content: Vec<PerceptionContentRecord>,
    image_shapes: Vec<[usize; 3]>,
    lidar2img: Vec<[[f32; 4]; 4]>,
    lidar2ego: [[f32; 4]; 4],
    box_coord_system: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PerceptionContentRecord {
    Text { text: String },
    Image { image: String },
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

pub fn read_perception_frame(root: &Path) -> Result<PerceptionFrame, String> {
    let root = root.canonicalize().map_err(|error| {
        format!(
            "Cannot resolve perception frame {}: {error}",
            root.display()
        )
    })?;
    if !root.is_dir() {
        return Err(format!(
            "Perception frame is not a directory: {}",
            root.display()
        ));
    }
    let manifest = root.join("frame-manifest.json");
    let record: PerceptionFrameRecord =
        serde_json::from_reader(File::open(&manifest).map_err(|error| {
            format!(
                "Cannot open perception frame {}: {error}",
                manifest.display()
            )
        })?)
        .map_err(|error| format!("Invalid perception frame {}: {error}", manifest.display()))?;
    if !matches!(record.dataset_type.as_str(), "nuscenes" | "nuplan") {
        return Err(format!(
            "Unsupported Qwen-Drive dataset type: {}",
            record.dataset_type
        ));
    }
    let cameras = record.cam_order.len();
    if cameras == 0
        || record.image_shapes.len() != cameras
        || record.lidar2img.len() != cameras
        || record
            .image_shapes
            .iter()
            .any(|shape| *shape != [512, 896, 3])
    {
        return Err("Qwen-Drive frame camera metadata is incomplete or not 896x512 RGB".into());
    }
    if record
        .cam_order
        .iter()
        .enumerate()
        .any(|(index, camera)| camera.is_empty() || record.cam_order[..index].contains(camera))
    {
        return Err("Qwen-Drive camera order contains an empty or duplicate name".into());
    }

    let flatten = |matrix: [[f32; 4]; 4]| matrix.into_iter().flatten().collect::<Vec<_>>();
    let mut lidar2img = Vec::with_capacity(cameras);
    for matrix in record.lidar2img {
        let matrix: [f32; 16] = flatten(matrix)
            .try_into()
            .expect("4x4 matrix has sixteen values");
        super::perception::fpn::inverse_4x4(&matrix)?;
        lidar2img.push(matrix);
    }
    let lidar2ego: [f32; 16] = flatten(record.lidar2ego)
        .try_into()
        .expect("4x4 matrix has sixteen values");
    super::perception::fpn::inverse_4x4(&lidar2ego)?;

    let mut image_index = 0usize;
    let mut content = Vec::with_capacity(record.content.len());
    for item in record.content {
        match item {
            PerceptionContentRecord::Text { text } => {
                if text.is_empty() {
                    return Err("Qwen-Drive frame contains empty text".into());
                }
                content.push(PerceptionContent::Text(text));
            }
            PerceptionContentRecord::Image { image } => {
                if record.cam_order.get(image_index) != Some(&image) {
                    return Err(format!(
                        "Qwen-Drive frame image order differs at camera {image_index}: {image}"
                    ));
                }
                let path = resolve_image(
                    &root,
                    Path::new("images").join(format!("{image}.jpg")).as_path(),
                )?;
                content.push(PerceptionContent::Image {
                    camera: image,
                    path,
                });
                image_index += 1;
            }
        }
    }
    if image_index != cameras {
        return Err(format!(
            "Qwen-Drive frame has {image_index} images for {cameras} cameras"
        ));
    }
    Ok(PerceptionFrame {
        token: root
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("Perception frame directory must have a UTF-8 name")?
            .to_owned(),
        dataset_type: record.dataset_type,
        cam_order: record.cam_order,
        content,
        image_shapes: record.image_shapes,
        lidar2img,
        lidar2ego,
        box_coord_system_ego: match record.box_coord_system.as_str() {
            "ego" => true,
            "lidar" => false,
            value => return Err(format!("Unsupported box coordinate system: {value}")),
        },
    })
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

    #[test]
    fn qwen_drive_perception_frame_validates_official_manifest() {
        let root = std::env::temp_dir().join(format!(
            "rmi-qwen-drive-perception-frame-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("images")).unwrap();
        let fixture = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen_drive/perception-frame.json"
        ));
        std::fs::write(root.join("frame-manifest.json"), fixture).unwrap();
        for camera in [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ] {
            std::fs::write(root.join("images").join(format!("{camera}.jpg")), []).unwrap();
        }
        let frame = read_perception_frame(&root).unwrap();
        assert_eq!(frame.dataset_type, "nuscenes");
        assert_eq!(frame.cam_order.len(), 6);
        assert_eq!(frame.image_shapes, vec![[512, 896, 3]; 6]);
        assert_eq!(
            frame
                .content
                .iter()
                .filter(|item| matches!(item, PerceptionContent::Image { .. }))
                .count(),
            6
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
