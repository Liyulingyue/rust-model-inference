//! Spark 2.5 model family

pub mod trunk;

pub use trunk::{
    load_layers, run_inference, SparkConfig, SparkLayerWeights, SparkModel, SparkSession,
};
