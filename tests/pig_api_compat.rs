use rust_model_inference::app::run_pig_image;
use rust_model_inference::models::diffusion::pig::PigSession;
use rust_model_inference::{MetaValue, PigConfig, PigModel, PigVAE, TensorInfo, TensorSource};

struct EmptySource;

impl TensorSource for EmptySource {
    fn metadata(&self, _key: &str) -> Option<&MetaValue> {
        None
    }

    fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
        None
    }

    fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
        None
    }
}

#[test]
fn pig_public_api_remains_available_alongside_z_image() {
    let _config_from_source = PigConfig::from_source::<EmptySource>;
    let _model_from_source = PigModel::from_source;
    let _model_config = PigModel::config;
    let _model_pool = PigModel::pool;
    let _vae_from_source = PigVAE::from_source;
    let _vae_decode = PigVAE::decode;
    let _session_new = PigSession::new;
    let _session_set_vae = PigSession::set_vae;
    let _session_generate_image = PigSession::generate_image;
    let _run_pig_image = run_pig_image;
}
