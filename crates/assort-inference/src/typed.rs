use assort_core::{Candidate, Distribution, Question, Request, Response, Result, ensure};
use std::sync::Arc;

/// Owned heterogeneous questions. Each returned key carries its own Rust value type.
/// Keys are tied to this exact set, so an index from a different request cannot be reused.
pub struct QuestionSet {
    request: Request,
    identity: Arc<()>,
}

pub struct DecisionKey<T> {
    index: usize,
    values: Vec<T>,
    identity: Arc<()>,
}

#[derive(Debug)]
pub struct Decision<T> {
    pub value: T,
    pub probabilities: Vec<(T, f32)>,
    pub distribution: Distribution,
}

pub struct DecisionSet {
    response: Response,
    identity: Arc<()>,
}

impl QuestionSet {
    pub fn new(state: impl Into<String>) -> Self {
        Self {
            request: Request {
                state: state.into(),
                questions: Vec::new(),
            },
            identity: Arc::new(()),
        }
    }

    pub fn choice<T: PartialEq>(
        &mut self,
        id: impl Into<String>,
        text: impl Into<String>,
        candidates: impl IntoIterator<Item = (T, Candidate)>,
    ) -> Result<DecisionKey<T>> {
        let (values, candidates): (Vec<_>, Vec<_>) = candidates.into_iter().unzip();
        ensure(
            values
                .iter()
                .enumerate()
                .all(|(i, value)| !values[..i].contains(value)),
            "typed candidate values must be unique",
        )?;
        let question = Question {
            id: id.into(),
            text: text.into(),
            candidates,
        };
        let mut request = self.request.clone();
        request.questions.push(question.clone());
        request.validate()?;
        let index = self.request.questions.len();
        self.request.questions.push(question);
        Ok(DecisionKey {
            index,
            values,
            identity: self.identity.clone(),
        })
    }

    pub fn boolean(
        &mut self,
        id: impl Into<String>,
        text: impl Into<String>,
    ) -> Result<DecisionKey<bool>> {
        let question = Question::boolean(id, text);
        self.choice(
            question.id,
            question.text,
            [true, false].into_iter().zip(question.candidates),
        )
    }

    /// Scores are ordinal indices with caller-provided descriptions, not a regression head.
    pub fn score(
        &mut self,
        id: impl Into<String>,
        text: impl Into<String>,
        descriptions: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<DecisionKey<usize>> {
        self.choice(
            id,
            text,
            descriptions
                .into_iter()
                .enumerate()
                .map(|(i, description)| (i, Candidate::new(i.to_string(), description))),
        )
    }

    pub(crate) fn into_parts(self) -> (Request, Arc<()>) {
        (self.request, self.identity)
    }
}

impl DecisionSet {
    pub(crate) fn new(response: Response, identity: Arc<()>) -> Self {
        Self { response, identity }
    }
    pub fn response(&self) -> &Response {
        &self.response
    }

    pub fn get<T: Clone>(&self, key: &DecisionKey<T>) -> Result<Decision<T>> {
        ensure(
            Arc::ptr_eq(&key.identity, &self.identity),
            "decision key belongs to a different question set",
        )?;
        let distribution = self.response.answers[key.index].distribution.clone();
        Ok(Decision {
            value: key.values[distribution.selected()].clone(),
            probabilities: key
                .values
                .iter()
                .cloned()
                .zip(distribution.probabilities().iter().copied())
                .collect(),
            distribution,
        })
    }
}
