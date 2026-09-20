//! Original, bidirectional candidate-scoring architecture. No generative decoder.
mod attention;
mod checkpoint;
mod config;
mod encoder;
mod model;

pub use checkpoint::*;
pub use config::ModelConfig;
pub use model::{DecisionModel, ModelOutput};
