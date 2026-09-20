use assort_core::{Candidate, Question, Request};
use assort_data::{Example, Target};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};

/// A deliberately narrow support-routing exercise with disjoint train/validation phrases.
/// Labels are synthetic, balanced, and NOT a benchmark of general language understanding.
pub fn support_routing_data(seed: u64) -> (Vec<Example>, Vec<Example>) {
    let descriptions = [
        ("billing", "Payments invoices charges refunds"),
        ("technical", "Software errors crashes bugs"),
        ("delivery", "Packages shipping tracking delivery"),
        ("account", "Account login passwords access"),
    ];
    let train = [
        [
            "My card payment was declined",
            "I need a copy of my invoice",
            "I was charged twice for my order",
            "Please refund the extra payment",
            "The invoice total is wrong",
            "My refund has not been paid",
            "There is an unknown charge on my card",
            "Can I pay this invoice by card",
            "The payment went through twice",
            "I want to dispute this charge",
            "Please explain the charges on my invoice",
            "I need help with a failed payment",
        ],
        [
            "The software crashes when I open it",
            "An error appears after the update",
            "The app freezes on the loading screen",
            "There is a bug in the export feature",
            "The software update will not install",
            "My app crashes every time I save",
            "The installation fails with an error",
            "The search feature has a software bug",
            "The app is stuck and will not respond",
            "I found an error in the dashboard",
            "The software keeps freezing",
            "I need help fixing an app crash",
        ],
        [
            "My package has not arrived",
            "The tracking number does not work",
            "The courier delivered to the wrong address",
            "My shipment is delayed",
            "Where is my parcel",
            "The package was damaged during shipping",
            "The tracking status has not changed",
            "I need to change my delivery address",
            "My order is missing from the delivery",
            "The courier lost my package",
            "Please check the shipping status",
            "I need help finding my shipment",
        ],
        [
            "I forgot my account password",
            "I cannot log in to my account",
            "Please reset my password",
            "My account has been locked",
            "I lost access to my account",
            "How can I change my login email",
            "The password reset link has expired",
            "I need to recover my account",
            "My login code is not arriving",
            "I want to update my account email",
            "Please unlock my login",
            "I need help signing into my account",
        ],
    ];
    let validation = [
        [
            "Please send the invoice for my last payment",
            "A duplicate charge appears on my card",
            "When will my refund reach my card",
            "The payment amount on this invoice is incorrect",
        ],
        [
            "The latest software version crashes on launch",
            "Saving a file causes an app error",
            "The app freezes after I open the dashboard",
            "This update introduced a software bug",
        ],
        [
            "Can you find the tracking status for my parcel",
            "My package is late and the courier has no update",
            "The shipping address on my package is incorrect",
            "The shipment never reached my delivery address",
        ],
        [
            "My password no longer lets me log in",
            "Can you restore access to my locked account",
            "The account password reset does not arrive",
            "I need a new login code for my account",
        ],
    ];
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut build = |phrases: &[[&str; 4]], wrappers: &[(&str, &str)]| -> Vec<Example> {
        let mut data = Vec::new();
        for (category, phrases) in phrases.iter().enumerate() {
            for phrase in phrases {
                for &(prefix, suffix) in wrappers {
                    let mut candidates: Vec<_> = descriptions
                        .iter()
                        .map(|&(id, description)| Candidate::new(id, description))
                        .collect();
                    candidates.shuffle(&mut rng);
                    let target = candidates
                        .iter()
                        .position(|c| c.id == descriptions[category].0)
                        .unwrap();
                    data.push(Example {
                        request: Request {
                            state: format!("{prefix}{phrase}{suffix}"),
                            questions: vec![Question {
                                id: "department".into(),
                                text: "Which team should handle this request?".into(),
                                candidates,
                            }],
                        },
                        targets: vec![Target::Hard(target)],
                    });
                }
            }
        }
        data.shuffle(&mut rng);
        data
    };
    // Keep phrase groups split before adding wrappers; paraphrases are never randomly split.
    let mut train_data = Vec::new();
    for chunk in 0..3 {
        let group = std::array::from_fn::<_, 4, _>(|category| {
            std::array::from_fn(|i| train[category][chunk * 4 + i])
        });
        train_data.extend(build(
            &group,
            &[
                ("", "."),
                ("Hello, ", ". Please help."),
                ("Support request: ", "."),
                ("", ". Thank you."),
            ],
        ));
    }
    let validation_data = build(&validation, &[("", "."), ("Hello, ", ". Please help.")]);
    (train_data, validation_data)
}
