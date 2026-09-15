pub mod bev;
pub mod fpn;
pub mod heads;
pub mod ops;

use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;
use crate::models::qwen_drive::config::PerceptionConfig;
use bev::{BevFormer, BevInput};
use fpn::{ViewBackbone, ViewGeometry};
use heads::{PerceptionHeads, PerceptionResult};
use ops::Tensor4;

pub struct QwenDrivePerception<'a> {
    backbone: ViewBackbone,
    transformer: BevFormer<'a>,
    heads: PerceptionHeads<'a>,
}

impl<'a> QwenDrivePerception<'a> {
    pub fn from_source<S: TensorSource + ?Sized>(source: &'a S) -> Result<Self, String> {
        Ok(Self {
            backbone: ViewBackbone::from_source(source)?,
            transformer: BevFormer::from_source(source)?,
            heads: PerceptionHeads::from_source(source)?,
        })
    }

    pub fn config(&self) -> &PerceptionConfig {
        self.backbone.config()
    }

    pub fn infer(
        &self,
        vit_features: &Tensor4,
        llm_features: &Tensor4,
        geometry: &ViewGeometry<'_>,
        dataset_type: &str,
        box_coord_system_ego: bool,
        pool: &ComputePool,
    ) -> Result<PerceptionResult, String> {
        let view = self
            .backbone
            .forward(vit_features, llm_features, geometry, pool)?;
        let bev = self.transformer.forward(
            &BevInput {
                levels: &view.llm_levels,
                uvtr_bev: &view.bev_tokens,
                geometry,
            },
            pool,
        )?;
        let lidar2ego = geometry
            .lidar2ego
            .first()
            .ok_or("Qwen-Drive perception requires lidar2ego")?;
        self.heads.forward(
            &self.transformer,
            &bev,
            &view.voxel,
            view.voxel_shape,
            dataset_type,
            lidar2ego,
            box_coord_system_ego,
            pool,
        )
    }
}
