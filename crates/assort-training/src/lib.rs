//! Runnable supervised training baseline with hard/soft labels and held-out evaluation.
mod objective;
mod synthetic;
mod trainer;

pub use objective::cross_entropy;
pub use synthetic::support_routing_data;
pub use trainer::*;
