use crate::cross_entropy;
use assort_core::{Distribution, Error, Result, ensure};
use assort_data::{BatchLimits, Example, Target, collate};
use assort_model::{DecisionModel, ModelConfig};
use assort_tokenizer::TextTokenizer;
use burn::{
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{
        Tensor,
        backend::{AutodiffBackend, Backend},
    },
};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub gradient_clip_norm: f32,
    pub shuffle_candidates: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 20,
            batch_size: 16,
            learning_rate: 0.003,
            seed: 42,
            gradient_clip_norm: 1.0,
            shuffle_candidates: true,
        }
    }
}

impl TrainingConfig {
    pub fn validate(&self, limits: &BatchLimits) -> Result<()> {
        ensure(
            self.epochs > 0 && self.batch_size > 0 && self.batch_size <= limits.max_batch_size,
            "invalid epochs/batch size",
        )?;
        ensure(
            self.learning_rate.is_finite() && self.learning_rate > 0.0,
            "learning rate must be finite and positive",
        )?;
        ensure(
            self.gradient_clip_norm.is_finite() && self.gradient_clip_norm > 0.0,
            "gradient clipping norm must be finite and positive",
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    pub cross_entropy: f64,
    /// Agreement with hard target or argmax of soft target; not soft-label calibration.
    pub accuracy: f64,
    /// Sum of squared probability errors, averaged per real question.
    pub brier: f64,
    pub questions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_cross_entropy: f64,
    pub validation: Metrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingReport {
    pub training: TrainingConfig,
    pub model: ModelConfig,
    pub train_examples: usize,
    pub validation_examples: usize,
    pub initial_validation: Metrics,
    pub epochs: Vec<EpochMetrics>,
    pub best_epoch: usize,
    pub best_validation: Metrics,
    pub optimizer_steps: usize,
}

pub struct TrainingResult<B: AutodiffBackend> {
    /// Best epoch by validation cross entropy, converted to an inference backend.
    pub model: DecisionModel<B::InnerBackend>,
    pub report: TrainingReport,
}

/// Starts from random weights. The caller owns data loading and checkpoint publication.
/// Validation runs without autodiff/dropout. No optimizer state or resume policy is hidden here.
#[allow(clippy::too_many_arguments)]
pub fn train<B: AutodiffBackend>(
    model_config: ModelConfig,
    tokenizer: &impl TextTokenizer,
    train_data: &[Example],
    validation_data: &[Example],
    config: TrainingConfig,
    limits: &BatchLimits,
    device: &B::Device,
    mut on_epoch: impl FnMut(&EpochMetrics) -> Result<()>,
) -> Result<TrainingResult<B>> {
    limits.validate()?;
    config.validate(limits)?;
    ensure(
        !train_data.is_empty() && !validation_data.is_empty(),
        "training and validation data must both be nonempty",
    )?;
    let normalized = |state: &str| {
        state
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    };
    let train_states: HashSet<_> = train_data
        .iter()
        .map(|e| normalized(&e.request.state))
        .collect();
    ensure(
        !validation_data
            .iter()
            .any(|e| train_states.contains(&normalized(&e.request.state))),
        "training and validation contain overlapping states",
    )?;
    for example in train_data.iter().chain(validation_data) {
        example.validate()?;
    }
    ensure(
        model_config.vocab_size == tokenizer.spec().vocab_size,
        "training tokenizer/model mismatch",
    )?;
    B::seed(device, config.seed);
    let mut model = DecisionModel::<B>::new(model_config.clone(), device)?;
    let mut optimizer = AdamConfig::new()
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(
            config.gradient_clip_norm,
        )))
        .init();
    let mut rng = SmallRng::seed_from_u64(config.seed);
    let initial = evaluate(
        &model.valid(),
        tokenizer,
        validation_data,
        config.batch_size,
        limits,
    )?;
    let mut best_model = model.valid();
    let mut best = initial.clone();
    let mut best_epoch = 0;
    let mut history = Vec::new();
    let mut steps = 0;
    let mut indices: Vec<_> = (0..train_data.len()).collect();
    for epoch in 1..=config.epochs {
        indices.shuffle(&mut rng);
        let mut total_loss = 0.0;
        let mut questions = 0;
        for chunk in indices.chunks(config.batch_size) {
            let mut examples: Vec<_> = chunk.iter().map(|&i| train_data[i].clone()).collect();
            if config.shuffle_candidates {
                for example in &mut examples {
                    permute_candidates(example, &mut rng);
                }
            }
            let requests: Vec<_> = examples.iter().map(|e| e.request.clone()).collect();
            let batch = collate(tokenizer, &requests, limits)?;
            let output = model.forward(&batch)?;
            let loss = cross_entropy(&output, &examples)?;
            let value = scalar(&loss)?;
            ensure(
                value.is_finite(),
                format!("nonfinite loss at epoch {epoch}, step {steps}"),
            )?;
            let count: usize = examples.iter().map(|e| e.targets.len()).sum();
            total_loss += value * count as f64;
            questions += count;
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(config.learning_rate, model, gradients);
            steps += 1;
        }
        let validation = evaluate(
            &model.valid(),
            tokenizer,
            validation_data,
            config.batch_size,
            limits,
        )?;
        // Always retain a trained epoch, even when the random baseline happened to score better.
        if epoch == 1 || validation.cross_entropy < best.cross_entropy {
            best_model = model.valid();
            best = validation.clone();
            best_epoch = epoch;
        }
        let metrics = EpochMetrics {
            epoch,
            train_cross_entropy: total_loss / questions as f64,
            validation,
        };
        on_epoch(&metrics)?;
        history.push(metrics);
    }
    Ok(TrainingResult {
        model: best_model,
        report: TrainingReport {
            training: config,
            model: model_config,
            train_examples: train_data.len(),
            validation_examples: validation_data.len(),
            initial_validation: initial,
            epochs: history,
            best_epoch,
            best_validation: best,
            optimizer_steps: steps,
        },
    })
}

pub fn evaluate<B: Backend>(
    model: &DecisionModel<B>,
    tokenizer: &impl TextTokenizer,
    examples: &[Example],
    batch_size: usize,
    limits: &BatchLimits,
) -> Result<Metrics> {
    ensure(
        !B::ad_enabled(&model.device()),
        "evaluation requires a non-autodiff model",
    )?;
    ensure(
        !examples.is_empty() && batch_size > 0,
        "evaluation needs examples and a positive batch size",
    )?;
    let mut loss = 0.0;
    let mut correct = 0usize;
    let mut brier = 0.0;
    let mut questions = 0usize;
    for chunk in examples.chunks(batch_size) {
        let requests: Vec<_> = chunk.iter().map(|e| e.request.clone()).collect();
        let batch = collate(tokenizer, &requests, limits)?;
        let s = batch.shape();
        let output = model.forward(&batch)?;
        let count: usize = chunk.iter().map(|e| e.targets.len()).sum();
        loss += scalar(&cross_entropy(&output, chunk)?)? * count as f64;
        let logits = output
            .logits
            .to_data()
            .to_vec::<f32>()
            .map_err(|e| Error::Invalid(e.to_string()))?;
        for (bi, example) in chunk.iter().enumerate() {
            for (qi, target) in example.targets.iter().enumerate() {
                let n = example.request.questions[qi].candidates.len();
                let start = (bi * s.questions + qi) * s.candidates;
                let distribution = Distribution::from_logits(&logits[start..start + n], 1.0)?;
                let expected = match target {
                    Target::Hard(index) => {
                        let mut p = vec![0.0; n];
                        p[*index] = 1.0;
                        p
                    }
                    Target::Soft(p) => p.clone(),
                };
                let mut target_index = 0;
                for i in 1..n {
                    if expected[i] > expected[target_index] {
                        target_index = i;
                    }
                }
                correct += usize::from(distribution.selected() == target_index);
                brier += distribution
                    .probabilities()
                    .iter()
                    .zip(expected)
                    .map(|(&p, y)| (p as f64 - y as f64).powi(2))
                    .sum::<f64>();
                questions += 1;
            }
        }
    }
    ensure(loss.is_finite(), "nonfinite evaluation loss")?;
    Ok(Metrics {
        cross_entropy: loss / questions as f64,
        accuracy: correct as f64 / questions as f64,
        brier: brier / questions as f64,
        questions,
    })
}

fn scalar<B: Backend>(tensor: &Tensor<B, 1>) -> Result<f64> {
    Ok(tensor
        .to_data()
        .to_vec::<f32>()
        .map_err(|e| Error::Invalid(e.to_string()))?[0] as f64)
}

fn permute_candidates(example: &mut Example, rng: &mut SmallRng) {
    for (question, target) in example
        .request
        .questions
        .iter_mut()
        .zip(&mut example.targets)
    {
        let mut indices: Vec<_> = (0..question.candidates.len()).collect();
        indices.shuffle(rng);
        question.candidates = indices
            .iter()
            .map(|&i| question.candidates[i].clone())
            .collect();
        *target = match target {
            Target::Hard(old) => Target::Hard(indices.iter().position(|i| i == old).unwrap()),
            Target::Soft(values) => Target::Soft(indices.iter().map(|&i| values[i]).collect()),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assort_core::{Question, Request};
    use assort_tokenizer::ByteTokenizer;
    use burn::backend::{Autodiff, Flex};

    #[test]
    fn candidate_shuffling_remaps_hard_and_soft_targets() {
        let mut hard = Example {
            request: Request {
                state: "A".into(),
                questions: vec![Question::boolean("q", "Yes?")],
            },
            targets: vec![Target::Hard(0)],
        };
        let mut soft = hard.clone();
        soft.targets = vec![Target::Soft(vec![0.8, 0.2])];
        let mut rng = SmallRng::seed_from_u64(123);
        for _ in 0..20 {
            permute_candidates(&mut hard, &mut rng);
            permute_candidates(&mut soft, &mut rng);
            let Target::Hard(index) = hard.targets[0] else {
                panic!()
            };
            assert_eq!(hard.request.questions[0].candidates[index].id, "true");
            let Target::Soft(values) = &soft.targets[0] else {
                panic!()
            };
            let index = soft.request.questions[0]
                .candidates
                .iter()
                .position(|c| c.id == "true")
                .unwrap();
            assert_eq!(values[index], 0.8);
        }
    }

    #[test]
    fn real_optimizer_updates_reduce_a_small_supervised_loss() {
        let sample = |state: &str| Example {
            request: Request {
                state: state.into(),
                questions: vec![Question::boolean("q", "Payment issue?")],
            },
            targets: vec![Target::Hard(0)],
        };
        let training = vec![sample("Card failed"), sample("Payment failed")];
        let validation = vec![sample("Another card failed")];
        let config = ModelConfig {
            hidden_size: 8,
            num_heads: 2,
            state_layers: 1,
            ffn_size: 16,
            ..ModelConfig::tiny(259)
        };
        let options = TrainingConfig {
            epochs: 5,
            batch_size: 2,
            learning_rate: 0.01,
            ..Default::default()
        };
        let result = train::<Autodiff<Flex>>(
            config.clone(),
            &ByteTokenizer,
            &training,
            &validation,
            options.clone(),
            &BatchLimits::default(),
            &Default::default(),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(result.report.optimizer_steps, 5);
        assert!(
            result.report.best_validation.cross_entropy
                < result.report.initial_validation.cross_entropy
        );
        assert!(result.report.best_epoch > 0);
        assert!(
            train::<Autodiff<Flex>>(
                config,
                &ByteTokenizer,
                &training,
                &training,
                options,
                &BatchLimits::default(),
                &Default::default(),
                |_| Ok(())
            )
            .is_err()
        );
    }
}
