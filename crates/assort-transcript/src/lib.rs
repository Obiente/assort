//! Timestamp-preserving, extractive transcript summaries from candidate-scoring models.
mod data;
mod evaluation;
mod selection;
mod synthetic;
mod windows;

pub use data::*;
pub use evaluation::*;
pub use selection::*;
pub use synthetic::meeting_corpus;
pub use windows::*;
