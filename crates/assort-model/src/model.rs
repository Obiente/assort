use crate::{
    ModelConfig,
    attention::Attention,
    encoder::{TextEncoder, mean_pool},
};
use assort_core::Result;
use assort_data::TokenBatch;
use burn::{
    module::{Module, ModuleVisitor, Param},
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{
        Bool, Int, Tensor, TensorData,
        activation::{gelu, softmax},
        backend::Backend,
    },
};

#[derive(Module, Debug)]
pub struct DecisionModel<B: Backend> {
    pub(crate) embeddings: Embedding<B>,
    state_encoder: TextEncoder<B>,
    query_encoder: TextEncoder<B>,
    question_attention: Attention<B>,
    candidate_attention: Option<Attention<B>>,
    question_norm: LayerNorm<B>,
    candidate_norm: LayerNorm<B>,
    scorer_hidden: Linear<B>,
    scorer_output: Linear<B>,
    #[module(skip)]
    config: ModelConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use assort_core::{Question, Request};
    use assort_data::{BatchLimits, collate};
    use assort_tokenizer::ByteTokenizer;
    use burn::backend::{Autodiff, Flex};

    #[test]
    fn gradients_reach_shared_embeddings_and_scorer() {
        let config = ModelConfig {
            hidden_size: 8,
            num_heads: 2,
            state_layers: 1,
            ffn_size: 16,
            ..ModelConfig::tiny(259)
        };
        let model = DecisionModel::<Autodiff<Flex>>::new(config, &Default::default()).unwrap();
        let request = Request {
            state: "Card failed".into(),
            questions: vec![Question::boolean("q", "Urgent?")],
        };
        let batch = collate(&ByteTokenizer, &[request], &BatchLimits::default()).unwrap();
        let output = model.forward(&batch).unwrap();
        let loss = -burn::tensor::activation::log_softmax(output.logits, 2)
            .slice([0..1, 0..1, 0..1])
            .sum();
        let gradients = loss.backward();
        for values in [
            model
                .embeddings
                .weight
                .val()
                .grad(&gradients)
                .unwrap()
                .to_data()
                .to_vec::<f32>()
                .unwrap(),
            model
                .scorer_hidden
                .weight
                .val()
                .grad(&gradients)
                .unwrap()
                .to_data()
                .to_vec::<f32>()
                .unwrap(),
        ] {
            assert!(values.iter().all(|x| x.is_finite()));
            assert!(values.iter().any(|x| x.abs() > 1e-8));
        }
    }
}

#[derive(Debug)]
pub struct ModelOutput<B: Backend> {
    /// [batch, questions, candidates]; padded entries are a finite negative sentinel.
    pub logits: Tensor<B, 3>,
    /// true = invalid. Use this for all losses, including padded question rows.
    pub candidate_mask: Tensor<B, 3, Bool>,
    pub question_mask: Tensor<B, 2, Bool>,
}

impl<B: Backend> ModelOutput<B> {
    /// Padded entries and fully padded question rows have probability zero.
    /// Host inference separately applies temperature in f64 for numeric robustness.
    pub fn probabilities(&self) -> Tensor<B, 3> {
        softmax(self.logits.clone(), 2).mask_fill(self.candidate_mask.clone(), 0.0)
    }
}

impl<B: Backend> DecisionModel<B> {
    pub fn new(config: ModelConfig, device: &B::Device) -> Result<Self> {
        let model = Self::uninitialized(config, device)?;
        // Burn parameters are lazy. Materialize before cloning, .valid(), or saving so
        // all views refer to the same seeded initialization rather than drawing again.
        struct Initialize;
        impl<B: Backend> ModuleVisitor<B> for Initialize {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                let _ = param.val();
            }
        }
        model.visit(&mut Initialize);
        Ok(model)
    }

    pub(crate) fn uninitialized(config: ModelConfig, device: &B::Device) -> Result<Self> {
        config.validate()?;
        let d = config.hidden_size;
        Ok(Self {
            embeddings: EmbeddingConfig::new(config.vocab_size, d).init(device),
            state_encoder: TextEncoder::new(&config, config.state_layers, device),
            query_encoder: TextEncoder::new(&config, config.query_layers, device),
            question_attention: Attention::new(d, config.num_heads, device),
            candidate_attention: config
                .candidate_state_attention
                .then(|| Attention::new(d, config.num_heads, device)),
            question_norm: LayerNormConfig::new(d).init(device),
            candidate_norm: LayerNormConfig::new(d).init(device),
            scorer_hidden: LinearConfig::new(d * 4, d).init(device),
            scorer_output: LinearConfig::new(d, 1).init(device),
            config,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn device(&self) -> B::Device {
        self.embeddings.weight.device()
    }

    /// Encodes each state once, then evaluates all questions/candidates in tensor batches.
    /// The only neural-network loops iterate over transformer depth.
    pub fn forward(&self, batch: &TokenBatch) -> Result<ModelOutput<B>> {
        self.config.validate_batch(batch)?;
        let s = batch.shape();
        let (b, q, c, d) = (s.batch, s.questions, s.candidates, self.config.hidden_size);
        let device = self.embeddings.weight.device();
        let state_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(batch.states().to_vec(), [b, s.state_tokens]),
            &device,
        );
        let state_padding = Tensor::<B, 2, Bool>::from_data(
            TensorData::new(batch.state_padding().to_vec(), [b, s.state_tokens]),
            &device,
        );
        let question_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(batch.questions().to_vec(), [b * q, s.question_tokens]),
            &device,
        );
        let question_padding = Tensor::<B, 2, Bool>::from_data(
            TensorData::new(
                batch.question_padding().to_vec(),
                [b * q, s.question_tokens],
            ),
            &device,
        );
        let candidate_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(batch.candidates().to_vec(), [b * q * c, s.candidate_tokens]),
            &device,
        );
        let candidate_padding = Tensor::<B, 2, Bool>::from_data(
            TensorData::new(
                batch.candidate_padding().to_vec(),
                [b * q * c, s.candidate_tokens],
            ),
            &device,
        );

        let state = self
            .state_encoder
            .forward(self.embeddings.forward(state_ids), state_padding.clone());
        let questions = mean_pool(
            self.query_encoder.forward(
                self.embeddings.forward(question_ids),
                question_padding.clone(),
            ),
            question_padding,
        )
        .reshape([b, q, d]);
        let candidates = mean_pool(
            self.query_encoder.forward(
                self.embeddings.forward(candidate_ids),
                candidate_padding.clone(),
            ),
            candidate_padding,
        )
        .reshape([b, q, c, d]);
        let questions = self.question_norm.forward(
            questions.clone()
                + self
                    .question_attention
                    .forward(questions, state.clone(), state_padding.clone()),
        );
        let questions = questions.unsqueeze_dim::<4>(2).expand([b, q, c, d]);
        let candidates = if let Some(attention) = &self.candidate_attention {
            let query = (candidates.clone() + questions.clone()).reshape([b, q * c, d]);
            self.candidate_norm.forward(
                candidates
                    + attention
                        .forward(query, state, state_padding)
                        .reshape([b, q, c, d]),
            )
        } else {
            candidates
        };
        let features = Tensor::cat(
            vec![
                questions.clone(),
                candidates.clone(),
                questions.clone() * candidates.clone(),
                (questions - candidates).abs(),
            ],
            3,
        );
        let logits = self
            .scorer_output
            .forward(gelu(self.scorer_hidden.forward(features)))
            .reshape([b, q, c]);
        let candidate_mask = Tensor::<B, 3, Bool>::from_data(
            TensorData::new(batch.candidate_mask().to_vec(), [b, q, c]),
            &device,
        );
        let question_mask = Tensor::<B, 2, Bool>::from_data(
            TensorData::new(batch.question_mask().to_vec(), [b, q]),
            &device,
        );
        // Finite sentinel avoids 0 * -inf in user-supplied soft-target losses.
        Ok(ModelOutput {
            logits: logits.mask_fill(candidate_mask.clone(), -1.0e9),
            candidate_mask,
            question_mask,
        })
    }
}
