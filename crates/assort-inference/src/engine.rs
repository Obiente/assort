use assort_core::{Answer, Distribution, Error, Request, Response, Result, ensure};
use assort_data::{BatchLimits, collate};
use assort_model::DecisionModel;
use assort_tokenizer::TextTokenizer;
use burn::tensor::backend::Backend;

pub struct Engine<B: Backend, T: TextTokenizer> {
    model: DecisionModel<B>,
    tokenizer: T,
    limits: BatchLimits,
    temperature: f32,
}

impl<B: Backend, T: TextTokenizer> Engine<B, T> {
    /// Use a non-autodiff backend for inference. For trained autodiff modules, call `.valid()`.
    pub fn new(model: DecisionModel<B>, tokenizer: T, limits: BatchLimits) -> Result<Self> {
        limits.validate()?;
        ensure(
            !B::ad_enabled(&model.device()),
            "inference requires a non-autodiff model; use .valid() after training",
        )?;
        ensure(
            model.config().vocab_size == tokenizer.spec().vocab_size,
            "tokenizer/model vocabulary mismatch",
        )?;
        ensure(
            limits
                .max_state_tokens
                .max(limits.max_question_tokens)
                .max(limits.max_candidate_tokens)
                <= model.config().max_positions,
            "batch token limits exceed model positions",
        )?;
        Ok(Self {
            model,
            tokenizer,
            limits,
            temperature: 1.0,
        })
    }

    /// Temperature is a caller-supplied scalar, not an automatically fitted calibrator.
    pub fn with_temperature(mut self, temperature: f32) -> Result<Self> {
        ensure(
            temperature.is_finite() && temperature > 0.0,
            "temperature must be finite and positive",
        )?;
        self.temperature = temperature;
        Ok(self)
    }

    pub fn model(&self) -> &DecisionModel<B> {
        &self.model
    }

    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    pub fn limits(&self) -> &BatchLimits {
        &self.limits
    }

    pub fn evaluate(&self, requests: &[Request]) -> Result<Vec<Response>> {
        let batch = collate(&self.tokenizer, requests, &self.limits)?;
        let shape = batch.shape();
        let output = self.model.forward(&batch)?;
        let logits = output
            .logits
            .to_data()
            .to_vec::<f32>()
            .map_err(|e| Error::Invalid(format!("cannot read model logits: {e}")))?;
        requests
            .iter()
            .enumerate()
            .map(|(b, request)| {
                let answers = request
                    .questions
                    .iter()
                    .enumerate()
                    .map(|(q, question)| {
                        let offset = (b * shape.questions + q) * shape.candidates;
                        let distribution = Distribution::from_logits(
                            &logits[offset..offset + question.candidates.len()],
                            self.temperature,
                        )?;
                        Ok(Answer {
                            question_id: question.id.clone(),
                            selected_id: question.candidates[distribution.selected()].id.clone(),
                            candidate_ids: question
                                .candidates
                                .iter()
                                .map(|c| c.id.clone())
                                .collect(),
                            distribution,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Response { answers })
            })
            .collect()
    }

    pub fn evaluate_set(&self, questions: crate::QuestionSet) -> Result<crate::DecisionSet> {
        let (request, identity) = questions.into_parts();
        let response = self.evaluate(&[request])?.remove(0);
        Ok(crate::DecisionSet::new(response, identity))
    }
}
