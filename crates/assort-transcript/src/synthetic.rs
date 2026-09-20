use crate::{LabeledTranscript, MEETING_GOAL, PassageKind, Segment, SegmentLabel, Transcript};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};

/// Three disjoint sets of authored phrase templates. The test split is not used in fitting
/// the tokenizer, optimizer, epoch selection, thresholds, or other hyperparameters.
pub fn meeting_corpus(
    seed: u64,
) -> (
    Vec<LabeledTranscript>,
    Vec<LabeledTranscript>,
    Vec<LabeledTranscript>,
) {
    let train: [&[&str]; 4] = [
        &[
            "We decided to launch {topic} on Friday.",
            "The final decision is to delay {topic} by one week.",
            "We agreed to use the simpler design for {topic}.",
            "The team approved the new plan for {topic}.",
            "We have chosen the first option for {topic}.",
            "Our agreed approach is to test {topic} before release.",
            "We are going ahead with the revised scope for {topic}.",
            "The decision is final: keep {topic} in the current release.",
            "We agreed that {topic} needs a staged rollout.",
            "We approved a smaller first release of {topic}.",
        ],
        &[
            "Alex will send the {topic} report by Tuesday.",
            "Sam owns the next {topic} review and will schedule it today.",
            "Jordan will update the {topic} plan before Friday.",
            "Taylor is assigned to test {topic} this week.",
            "Alex will contact the customer about {topic} tomorrow.",
            "Sam will prepare the {topic} checklist by Monday.",
            "Jordan is responsible for fixing the {topic} error.",
            "Taylor will confirm the {topic} budget with finance today.",
            "Alex will share the final {topic} notes after this call.",
            "Sam will book the {topic} review for next week.",
        ],
        &[
            "The {topic} budget is limited to 5000 euros.",
            "The {topic} deadline is Friday and cannot move.",
            "The latest {topic} test failed on three devices.",
            "We cannot release {topic} until the security review is complete.",
            "The {topic} service is unavailable for two hours each day.",
            "The customer reported a critical error in {topic}.",
            "The {topic} release has a dependency on the vendor update.",
            "We have two engineers available for {topic} this month.",
            "The {topic} test results show a 20 percent improvement.",
            "The main blocker for {topic} is missing access to the server.",
        ],
        &[
            "Hello everyone, thanks for joining the {topic} call.",
            "Can everyone hear me on the {topic} call?",
            "I was thinking about {topic} over the weekend.",
            "Could we launch {topic} on Friday? This is only a suggestion.",
            "Maybe Alex could own {topic}, but nobody has been assigned.",
            "No decision has been made about the {topic} proposal.",
            "Let me find the {topic} document on my screen.",
            "Thanks for explaining the background to {topic}.",
            "We can discuss possible {topic} ideas later.",
            "Sorry, I was muted during the {topic} discussion.",
            "That is all I wanted to ask about {topic}.",
            "I hope everyone has a good afternoon after the {topic} call.",
        ],
    ];
    let validation: [&[&str]; 4] = [
        &[
            "The team agreed to postpone {topic} until next week.",
            "Our final choice for {topic} is the simpler option.",
            "We approved the updated {topic} schedule.",
            "We decided that {topic} will use a staged launch.",
        ],
        &[
            "Jordan will send the revised {topic} report on Monday.",
            "Alex is responsible for testing {topic} before launch.",
            "Taylor will contact finance to confirm the {topic} costs.",
            "Sam will schedule the next {topic} review tomorrow.",
        ],
        &[
            "The {topic} release is blocked by the security review.",
            "The confirmed {topic} budget is 8000 euros.",
            "The {topic} tests failed on two devices this morning.",
            "Only three engineers are available for {topic} next week.",
        ],
        &[
            "Thanks everyone for making time for {topic} today.",
            "Perhaps we could delay {topic}; nothing is agreed yet.",
            "I cannot find the {topic} document on my screen.",
            "We have not assigned anyone to the proposed {topic} review.",
        ],
    ];
    let test: [&[&str]; 4] = [
        &[
            "We agreed on Friday as the final launch date for {topic}.",
            "The approved decision is to reduce the scope of {topic}.",
            "We have decided to keep the current {topic} design.",
            "The team chose a staged rollout for {topic}.",
        ],
        &[
            "Sam will deliver the final {topic} checklist by Friday.",
            "Taylor owns the {topic} follow-up and will send notes tomorrow.",
            "Jordan will call the vendor about {topic} this afternoon.",
            "Alex is assigned to update the {topic} schedule today.",
        ],
        &[
            "The {topic} deadline cannot be moved beyond Monday.",
            "The vendor update is still blocking the {topic} release.",
            "The latest {topic} results show three critical errors.",
            "The available {topic} budget is 6000 euros this month.",
        ],
        &[
            "Before we begin {topic}, can you hear my audio?",
            "Friday might work for {topic}, but this is not a decision.",
            "No owner has been assigned to the {topic} suggestion.",
            "Thank you for the {topic} call, have a good evening.",
        ],
    ];
    let mut rng = SmallRng::seed_from_u64(seed);
    let train = build(
        "train",
        &train,
        &[
            "billing update",
            "mobile release",
            "search project",
            "support portal",
            "account migration",
            "payment integration",
        ],
        48,
        &mut rng,
    );
    let validation = build(
        "validation",
        &validation,
        &[
            "client dashboard",
            "delivery service",
            "customer onboarding",
        ],
        12,
        &mut rng,
    );
    let test = build(
        "test",
        &test,
        &["reporting module", "booking system", "inventory sync"],
        12,
        &mut rng,
    );
    (train, validation, test)
}

fn build(
    split: &str,
    phrases: &[&[&str]; 4],
    topics: &[&str],
    count: usize,
    rng: &mut SmallRng,
) -> Vec<LabeledTranscript> {
    (0..count)
        .map(|i| {
            let topic = topics[i % topics.len()];
            let mut turns = Vec::new();
            for kind in PassageKind::ALL {
                let mut pool = phrases[kind.index()].to_vec();
                pool.shuffle(rng);
                for phrase in pool.into_iter().take(if kind == PassageKind::Background {
                    4
                } else {
                    2
                }) {
                    turns.push((kind, phrase.replace("{topic}", topic)));
                }
            }
            turns.shuffle(rng);
            let mut segments = Vec::new();
            let mut labels = Vec::new();
            for (index, (kind, text)) in turns.into_iter().enumerate() {
                let id = format!("turn-{}", index + 1);
                labels.push(SegmentLabel {
                    segment_id: id.clone(),
                    kind,
                });
                segments.push(Segment {
                    id,
                    start_ms: index as u64 * 15_000,
                    end_ms: index as u64 * 15_000 + 12_000,
                    speaker: Some(["Alex", "Sam", "Jordan", "Taylor"][index % 4].into()),
                    text,
                });
            }
            LabeledTranscript {
                transcript: Transcript {
                    id: format!("{split}-{i}"),
                    title: format!("{topic} planning call"),
                    goal: MEETING_GOAL.into(),
                    segments,
                },
                labels,
            }
        })
        .collect()
}
