//! Spark 2.5 model family

pub mod trunk;

pub use trunk::{
    run_inference, load_layers, SparkConfig, SparkLayerWeights, SparkModel, SparkSession,
};