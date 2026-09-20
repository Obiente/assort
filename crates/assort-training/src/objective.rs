use assort_core::{Result, ensure};
use assort_data::{Example, Target};
use assort_model::ModelOutput;
use burn::tensor::{Tensor, TensorData, activation::log_softmax, backend::Backend};

/// Mean cross entropy per real question, supporting hard labels and soft teacher targets.
/// Soft-target CE and KL have identical parameter gradients (target entropy is constant).
/// Padding contributes neither loss nor normalization count.
pub fn cross_entropy<B: Backend>(
    output: &ModelOutput<B>,
    examples: &[Example],
) -> Result<Tensor<B, 1>> {
    let [b, q, c] = output.logits.dims();
    ensure(b == examples.len(), "label batch size differs from logits")?;
    let mut targets = vec![0.0f32; b * q * c];
    let mut count = 0usize;
    for (bi, example) in examples.iter().enumerate() {
        example.validate()?;
        ensure(
            example.request.questions.len() <= q,
            "too many target questions",
        )?;
        for (qi, target) in example.targets.iter().enumerate() {
            let n = example.request.questions[qi].candidates.len();
            ensure(n <= c, "too many target candidates")?;
            let offset = (bi * q + qi) * c;
            match target {
                Target::Hard(index) => targets[offset + index] = 1.0,
                Target::Soft(values) => targets[offset..offset + n].copy_from_slice(values),
            }
            count += 1;
        }
    }
    ensure(count > 0, "no supervised questions")?;
    let targets =
        Tensor::<B, 3>::from_data(TensorData::new(targets, [b, q, c]), &output.logits.device());
    let log_probabilities =
        log_softmax(output.logits.clone(), 2).mask_fill(output.candidate_mask.clone(), 0.0);
    Ok(-(targets * log_probabilities).sum() / count as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use assort_core::{Candidate, Question, Request};
    use burn::{
        backend::{Autodiff, Flex},
        tensor::Bool,
    };

    #[test]
    fn mixed_targets_ignore_padding_and_average_per_question() {
        type B = Autodiff<Flex>;
        let device = Default::default();
        let logits = Tensor::<B, 3>::from_data(
            TensorData::new(
                vec![
                    2.0f32.ln(),
                    0.0,
                    -1e9,
                    -1e9,
                    -1e9,
                    -1e9,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1e9,
                    -1e9,
                ],
                [2, 2, 3],
            ),
            &device,
        )
        .require_grad();
        let output = ModelOutput {
            logits: logits.clone(),
            candidate_mask: Tensor::<B, 3, Bool>::from_data(
                TensorData::new(
                    vec![
                        false, false, true, true, true, true, false, false, false, false, true,
                        true,
                    ],
                    [2, 2, 3],
                ),
                &device,
            ),
            question_mask: Tensor::<B, 2, Bool>::from_data(
                [[false, true], [false, false]],
                &device,
            ),
        };
        let question = |id: &str, n: usize| Question {
            id: id.into(),
            text: "Choose".into(),
            candidates: (0..n)
                .map(|i| Candidate::new(i.to_string(), format!("Option {i}")))
                .collect(),
        };
        let examples = [
            Example {
                request: Request {
                    state: "First".into(),
                    questions: vec![question("q", 2)],
                },
                targets: vec![Target::Hard(0)],
            },
            Example {
                request: Request {
                    state: "Second".into(),
                    questions: vec![question("q", 3), question("singleton", 1)],
                },
                targets: vec![Target::Soft(vec![0.25, 0.25, 0.5]), Target::Hard(0)],
            },
        ];
        let loss = cross_entropy(&output, &examples).unwrap();
        let value = loss.to_data().to_vec::<f32>().unwrap()[0];
        let expected = (-(2.0f32 / 3.0).ln() + 3.0f32.ln()) / 3.0;
        assert!((value - expected).abs() < 1e-6);
        let gradients = logits
            .grad(&loss.backward())
            .unwrap()
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(gradients.iter().all(|x| x.is_finite()));
        for index in [2, 3, 4, 5, 10, 11] {
            assert_eq!(gradients[index], 0.0);
        }
        assert!(gradients[0] < 0.0);
    }
}
