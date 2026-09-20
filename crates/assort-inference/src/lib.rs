//! Local batched inference and heterogeneous, typed decision keys.
mod engine;
mod typed;

pub use engine::Engine;
pub use typed::{Decision, DecisionKey, DecisionSet, QuestionSet};
