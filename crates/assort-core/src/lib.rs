//! Backend-independent request contracts and validated decision distributions.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid input: {0}")]
    Invalid(String),
    #[error("tokenization failed: {0}")]
    Tokenizer(String),
    #[error("checkpoint error: {0}")]
    Checkpoint(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub fn ensure(condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::Invalid(message.into()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub id: String,
    pub description: String,
}

impl Candidate {
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Question {
    pub id: String,
    pub text: String,
    pub candidates: Vec<Candidate>,
}

impl Question {
    pub fn boolean(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            candidates: vec![
                Candidate::new("true", "Yes. The statement is true."),
                Candidate::new("false", "No. The statement is false."),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub state: String,
    pub questions: Vec<Question>,
}

impl Request {
    pub fn validate(&self) -> Result<()> {
        ensure(!self.state.trim().is_empty(), "state must not be blank")?;
        ensure(
            !self.questions.is_empty(),
            "at least one question is required",
        )?;
        let mut ids = HashSet::new();
        for q in &self.questions {
            ensure(
                !q.id.trim().is_empty() && ids.insert(&q.id),
                "question IDs must be nonblank and unique",
            )?;
            ensure(
                !q.text.trim().is_empty(),
                format!("question {} has blank text", q.id),
            )?;
            ensure(
                (1..=255).contains(&q.candidates.len()),
                "each question needs 1..=255 candidates",
            )?;
            let mut candidates = HashSet::new();
            for c in &q.candidates {
                ensure(
                    !c.id.trim().is_empty() && candidates.insert(&c.id),
                    "candidate IDs must be nonblank and unique within a question",
                )?;
                ensure(
                    !c.description.trim().is_empty(),
                    "candidate descriptions must not be blank",
                )?;
            }
        }
        Ok(())
    }
}

/// Probabilities in the exact order of the supplied candidates.
/// Entropy is uncertainty, not a calibrated estimate of correctness.
#[derive(Debug, Clone, Serialize)]
pub struct Distribution {
    probabilities: Vec<f32>,
    selected: usize,
    entropy: f32,
    normalized_entropy: f32,
}

impl Distribution {
    pub fn from_logits(logits: &[f32], temperature: f32) -> Result<Self> {
        ensure(!logits.is_empty(), "empty logits")?;
        ensure(
            temperature.is_finite() && temperature > 0.0,
            "temperature must be finite and positive",
        )?;
        ensure(logits.iter().all(|x| x.is_finite()), "nonfinite logits")?;
        // f64 avoids overflow when subtracting extreme f32 logits or scaling by tiny temperatures.
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let weights: Vec<f64> = logits
            .iter()
            .map(|&x| ((x as f64 - max) / temperature as f64).exp())
            .collect();
        let sum: f64 = weights.iter().sum();
        let probabilities: Vec<f32> = weights.into_iter().map(|x| (x / sum) as f32).collect();
        let mut selected = 0;
        for i in 1..logits.len() {
            if logits[i] > logits[selected] {
                selected = i;
            }
        }
        let entropy = -probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f32>();
        let normalized_entropy = if logits.len() == 1 {
            0.0
        } else {
            (entropy / (logits.len() as f32).ln()).clamp(0.0, 1.0)
        };
        Ok(Self {
            probabilities,
            selected,
            entropy,
            normalized_entropy,
        })
    }

    pub fn probabilities(&self) -> &[f32] {
        &self.probabilities
    }
    pub fn selected(&self) -> usize {
        self.selected
    }
    pub fn entropy(&self) -> f32 {
        self.entropy
    }
    pub fn normalized_entropy(&self) -> f32 {
        self.normalized_entropy
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Answer {
    pub question_id: String,
    pub selected_id: String,
    pub candidate_ids: Vec<String>,
    pub distribution: Distribution,
}

#[derive(Debug, Clone, Serialize)]
pub struct Response {
    pub answers: Vec<Answer>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_probabilities_and_deterministic_ties() {
        let p = Distribution::from_logits(&[f32::MAX, -f32::MAX], f32::MIN_POSITIVE).unwrap();
        assert_eq!(p.probabilities(), &[1.0, 0.0]);
        let p = Distribution::from_logits(&[2.0, 2.0], 1.0).unwrap();
        assert_eq!(p.selected(), 0);
        assert!((p.normalized_entropy() - 1.0).abs() < 1e-6);
        assert_eq!(
            Distribution::from_logits(&[1.0], 1.0)
                .unwrap()
                .normalized_entropy(),
            0.0
        );
    }

    #[test]
    fn invalid_distributions_are_errors() {
        for logits in [vec![], vec![f32::NAN], vec![f32::INFINITY]] {
            assert!(Distribution::from_logits(&logits, 1.0).is_err());
        }
        assert!(Distribution::from_logits(&[1.0], 0.0).is_err());
    }

    #[test]
    fn rejects_duplicate_ids_and_empty_candidates() {
        let q = Question::boolean("urgent", "Urgent?");
        let mut request = Request {
            state: "A synthetic request".into(),
            questions: vec![q.clone(), q],
        };
        assert!(request.validate().is_err());
        request.questions.pop();
        request.questions[0].candidates.clear();
        assert!(request.validate().is_err());
    }
}
