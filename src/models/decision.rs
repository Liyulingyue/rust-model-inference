//! Native decision engines share their entry points, not their model protocols.
use crate::{
    compute::ComputePolicy,
    core::{
        tensor::{MetaValue, TensorSource},
        thread_pool::ComputePool,
    },
};
use serde_json::Value;
use std::{path::Path, sync::Arc};

pub enum Engine {
    Thx02(super::thx02::Engine),
    D3Turbo(super::d3_turbo::Engine),
}

pub fn is_arch(arch: &str) -> bool {
    matches!(arch, "thx02" | "d3_turbo")
}

pub fn is_model_directory(path: &Path) -> bool {
    super::d3_turbo::matches_model(path)
}

pub fn validate_request(
    path: &Path,
    source: Option<&dyn TensorSource>,
    request: &Value,
) -> Result<(), String> {
    if super::d3_turbo::matches_model(path)
        || source.is_some_and(|s| {
            s.metadata("general.architecture")
                .and_then(MetaValue::to_string_val)
                == Some(super::d3_turbo::ARCH)
        })
    {
        super::d3_turbo::validate_request(request)
    } else {
        super::thx02::validate_request(request)
    }
}

impl Engine {
    pub fn from_path(
        path: &Path,
        thinker_path: Option<&Path>,
        threads: usize,
        context: usize,
        compute: ComputePolicy,
    ) -> Result<Self, String> {
        if super::d3_turbo::matches_model(path) {
            if thinker_path.is_some() {
                return Err("d3-turbo has no thinker component".into());
            }
            return Ok(Self::D3Turbo(super::d3_turbo::Engine::load(
                path, threads, compute, context,
            )?));
        }
        let source = Arc::from(
            crate::format::ggufrs::open_model_source(
                path,
                crate::format::ggufrs::ComponentRole::Llm,
            )
            .map_err(|e| e.to_string())?,
        );
        let thinker = thinker_path
            .map(|path| {
                crate::format::ggufrs::open_model_source(
                    path,
                    crate::format::ggufrs::ComponentRole::Llm,
                )
                .map(Arc::from)
                .map_err(|error| format!("decision thinker {}: {error}", path.display()))
            })
            .transpose()?;
        Self::from_source(
            path,
            source,
            thinker,
            Arc::new(ComputePool::new(threads)),
            threads,
            context,
            compute,
        )
    }
    pub fn from_source(
        path: &Path,
        source: Arc<dyn TensorSource>,
        thinker: Option<Arc<dyn TensorSource>>,
        pool: Arc<ComputePool>,
        threads: usize,
        context: usize,
        compute: ComputePolicy,
    ) -> Result<Self, String> {
        match source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            .as_deref()
        {
            Some(super::thx02::ARCH) => Ok(Self::Thx02(super::thx02::Engine::new(
                source, thinker, pool, context, compute,
            )?)),
            Some(super::d3_turbo::ARCH) if thinker.is_none() => Ok(Self::D3Turbo(
                super::d3_turbo::Engine::load(path, threads, compute, context)?,
            )),
            _ => Err("unsupported decision architecture or thinker component".into()),
        }
    }
    pub fn validate_request(&self, request: &Value) -> Result<(), String> {
        match self {
            Self::Thx02(_) => super::thx02::validate_request(request),
            Self::D3Turbo(_) => super::d3_turbo::validate_request(request),
        }
    }
    pub fn decide(&self, request: &Value) -> Result<Value, String> {
        match self {
            Self::Thx02(engine) => engine.decide(request),
            Self::D3Turbo(engine) => engine.decide(request),
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Self::Thx02(_) => "THX-02",
            Self::D3Turbo(_) => "vllm-sr/d3-turbo",
        }
    }
}

pub fn system_one(
    path: &Path,
    adapter: Option<&Path>,
    request: &Value,
    threads: usize,
    compute: ComputePolicy,
) -> Result<Value, String> {
    let native = adapter.is_none() && (super::d3_turbo::matches_model(path) || path.is_file());
    if native {
        if super::d3_turbo::matches_model(path) {
            super::d3_turbo::validate_request(request)?;
        }
        Engine::from_path(path, None, threads, usize::MAX, compute)?.decide(request)
    } else {
        super::vela3::system_one_with_policy(path, adapter, request, threads, compute)
    }
}
