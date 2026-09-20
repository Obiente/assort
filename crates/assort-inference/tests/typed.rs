use assort_core::Candidate;
use assort_data::BatchLimits;
use assort_inference::{Engine, QuestionSet};
use assort_model::{DecisionModel, ModelConfig};
use assort_tokenizer::ByteTokenizer;
use burn::backend::Flex;

#[derive(Debug, Clone, PartialEq)]
enum Department {
    Billing,
    Technical,
}

#[test]
fn heterogeneous_keys_preserve_rust_types_and_request_identity() {
    let model = DecisionModel::<Flex>::new(
        ModelConfig {
            hidden_size: 8,
            num_heads: 2,
            state_layers: 1,
            ffn_size: 16,
            ..ModelConfig::tiny(259)
        },
        &Default::default(),
    )
    .unwrap();
    let engine = Engine::new(model, ByteTokenizer, BatchLimits::default()).unwrap();
    let mut questions = QuestionSet::new("A failed payment");
    let urgent = questions.boolean("urgent", "Urgent?").unwrap();
    let team = questions
        .choice(
            "team",
            "Which team?",
            [
                (Department::Billing, Candidate::new("billing", "Payments")),
                (
                    Department::Technical,
                    Candidate::new("technical", "Software"),
                ),
            ],
        )
        .unwrap();
    let score = questions
        .score("score", "How frustrated?", ["Calm", "Angry"])
        .unwrap();
    let result = engine.evaluate_set(questions).unwrap();
    let _: bool = result.get(&urgent).unwrap().value;
    let _: Department = result.get(&team).unwrap().value;
    assert!(result.get(&score).unwrap().value < 2);
    let mut other = QuestionSet::new("Another request");
    let foreign = other.boolean("urgent", "Urgent?").unwrap();
    assert!(result.get(&foreign).is_err());
    assert!(
        other
            .choice(
                "dup",
                "Duplicate?",
                [
                    (true, Candidate::new("a", "A")),
                    (true, Candidate::new("b", "B"))
                ]
            )
            .is_err()
    );
}
