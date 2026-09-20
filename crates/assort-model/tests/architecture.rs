use assort_core::{Candidate, Question, Request};
use assort_data::{BatchLimits, collate};
use assort_model::{DecisionModel, ModelConfig, WeightsStatus, load_checkpoint, save_checkpoint};
use assort_tokenizer::{ByteTokenizer, TextTokenizer};
use burn::backend::Flex;

fn config() -> ModelConfig {
    ModelConfig {
        hidden_size: 8,
        num_heads: 2,
        state_layers: 1,
        ffn_size: 16,
        max_positions: 128,
        ..ModelConfig::tiny(259)
    }
}

fn request() -> Request {
    Request {
        state: "My card failed".into(),
        questions: vec![Question {
            id: "route".into(),
            text: "Which team?".into(),
            candidates: vec![
                Candidate::new("billing", "Payments"),
                Candidate::new("tech", "Software errors"),
            ],
        }],
    }
}

fn logits(model: &DecisionModel<Flex>, requests: &[Request]) -> Vec<f32> {
    model
        .forward(&collate(&ByteTokenizer, requests, &BatchLimits::default()).unwrap())
        .unwrap()
        .logits
        .to_data()
        .to_vec()
        .unwrap()
}

fn near(a: f32, b: f32) {
    assert!((a - b).abs() < 1e-4, "{a} != {b}");
}

#[test]
fn batching_and_candidate_permutation_preserve_decisions() {
    let model = DecisionModel::<Flex>::new(config(), &Default::default()).unwrap();
    let a = request();
    let mut b = a.clone();
    b.state = "A longer state with additional padding in the batch".into();
    b.questions
        .push(Question::boolean("second", "An additional question?"));
    b.questions[0]
        .candidates
        .push(Candidate::new("sales", "Purchasing and upgrades"));
    let alone = logits(&model, std::slice::from_ref(&a));
    let together = logits(&model, &[a.clone(), b]);
    near(alone[0], together[0]);
    near(alone[1], together[1]);
    assert!(together[2] < -1e8);
    let mut reversed = a;
    reversed.questions[0].candidates.reverse();
    let permuted = logits(&model, &[reversed]);
    near(alone[0], permuted[1]);
    near(alone[1], permuted[0]);
}

#[test]
fn padded_rows_have_zero_probability_and_real_rows_sum_to_one() {
    let model = DecisionModel::<Flex>::new(config(), &Default::default()).unwrap();
    let a = request();
    let mut b = a.clone();
    b.questions.push(Question::boolean("second", "Another?"));
    let batch = collate(&ByteTokenizer, &[a, b], &BatchLimits::default()).unwrap();
    let probabilities = model
        .forward(&batch)
        .unwrap()
        .probabilities()
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    near(probabilities[0] + probabilities[1], 1.0);
    assert_eq!(&probabilities[2..4], &[0.0, 0.0]);
    assert!(probabilities.iter().all(|p| p.is_finite()));
}

#[test]
fn state_changes_scores_and_optional_attention_path_runs() {
    for enabled in [true, false] {
        let model = DecisionModel::<Flex>::new(
            ModelConfig {
                candidate_state_attention: enabled,
                ..config()
            },
            &Default::default(),
        )
        .unwrap();
        let a = logits(&model, &[request()]);
        let mut other = request();
        other.state = "App crashes on launch".into();
        let b = logits(&model, &[other]);
        assert!(a.iter().zip(b).any(|(a, b)| (a - b).abs() > 1e-6));
    }
}

#[test]
fn checkpoint_roundtrip_rejects_wrong_tokenizer_and_corruption() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model");
    let model = DecisionModel::<Flex>::new(config(), &Default::default()).unwrap();
    let spec = ByteTokenizer.spec();
    save_checkpoint(&model, &spec, WeightsStatus::Random, &path).unwrap();
    let (loaded, manifest) = load_checkpoint::<Flex>(&path, &spec, &Default::default()).unwrap();
    assert_eq!(manifest.weights_status, WeightsStatus::Random);
    assert_eq!(logits(&model, &[request()]), logits(&loaded, &[request()]));
    assert!(save_checkpoint(&model, &spec, WeightsStatus::Random, &path).is_err());
    let mut wrong = spec.clone();
    wrong.fingerprint = "different-vocabulary".into();
    assert!(load_checkpoint::<Flex>(&path, &wrong, &Default::default()).is_err());
    std::fs::write(path.join("weights.mpk"), b"corrupted").unwrap();
    assert!(load_checkpoint::<Flex>(&path, &spec, &Default::default()).is_err());
}

#[test]
fn invalid_config_and_attention_budget_fail_before_forward() {
    assert!(
        DecisionModel::<Flex>::new(
            ModelConfig {
                num_heads: 3,
                ..config()
            },
            &Default::default()
        )
        .is_err()
    );
    let model = DecisionModel::<Flex>::new(
        ModelConfig {
            max_attention_elements: 1,
            ..config()
        },
        &Default::default(),
    )
    .unwrap();
    let batch = collate(&ByteTokenizer, &[request()], &BatchLimits::default()).unwrap();
    assert!(model.forward(&batch).is_err());
}
