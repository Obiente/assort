use assort_core::{Candidate, Result};
use assort_data::BatchLimits;
use assort_inference::{Engine, QuestionSet};
use assort_model::{DecisionModel, ModelConfig};
use assort_tokenizer::ByteTokenizer;
use burn::backend::Flex;

#[derive(Debug, Clone, PartialEq)]
enum Department {
    Billing,
    Technical,
    Sales,
}

fn main() -> Result<()> {
    let model = DecisionModel::<Flex>::new(ModelConfig::tiny(259), &Default::default())?;
    let engine = Engine::new(model, ByteTokenizer, BatchLimits::default())?;
    let mut questions =
        QuestionSet::new("The customer says their card payment failed three times.");
    let urgent = questions.boolean("urgent", "Does this require urgent attention?")?;
    let department = questions.choice(
        "department",
        "Which department should handle this?",
        [
            (
                Department::Billing,
                Candidate::new("billing", "Payments and invoices"),
            ),
            (
                Department::Technical,
                Candidate::new("technical", "Software failures"),
            ),
            (
                Department::Sales,
                Candidate::new("sales", "Purchasing and upgrades"),
            ),
        ],
    )?;
    let frustration = questions.score(
        "frustration",
        "How frustrated is the customer?",
        ["Calm", "Slightly frustrated", "Very frustrated"],
    )?;
    let results = engine.evaluate_set(questions)?;
    eprintln!("Random model: these values demonstrate the typed API, not learned behavior.");
    println!("Urgent: {:?}", results.get(&urgent)?.value);
    println!("Department: {:?}", results.get(&department)?);
    println!("Frustration: {:?}", results.get(&frustration)?.value);
    Ok(())
}
