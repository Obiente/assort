use assort_core::Result;
use assort_tokenizer::{ByteTokenizer, TextTokenizer, TokenizerSpec};
use assort_transcript::*;

fn transcript(texts: &[&str]) -> Transcript {
    Transcript {
        id: "example".into(),
        title: "Example call".into(),
        goal: MEETING_GOAL.into(),
        segments: texts
            .iter()
            .enumerate()
            .map(|(i, text)| Segment {
                id: format!("s{i}"),
                start_ms: i as u64 * 1000,
                end_ms: i as u64 * 1000 + 900,
                speaker: Some("Alex".into()),
                text: (*text).into(),
            })
            .collect(),
    }
}

fn scores(t: &Transcript) -> Vec<SegmentScore> {
    t.segments
        .iter()
        .map(|s| SegmentScore {
            segment_id: s.id.clone(),
            kind: PassageKind::Decision,
            importance: 0.95,
        })
        .collect()
}

#[test]
fn srt_preserves_multiline_text_and_real_timestamps() {
    let t = Transcript::from_srt("srt", "Call", MEETING_GOAL, "\u{feff}1\r\n00:00:01,250 --> 00:00:03,500\r\nWe agreed to launch.\r\nOn Friday.\r\n\r\n2\r\n00:00:04,000 --> 00:00:05,000\r\nThank you.\r\n").unwrap();
    assert_eq!(t.segments[0].start_ms, 1250);
    assert_eq!(t.segments[0].end_ms, 3500);
    assert_eq!(t.segments[0].text, "We agreed to launch.\nOn Friday.");
    assert!(t.segments[0].speaker.is_none());
    assert_eq!(timestamp(3_601_250), "01:00:01.250");
    for malformed in [
        "1\n00:60:00,000 --> 00:61:00,000\nText",
        "1\n00:00:03,000 --> 00:00:02,000\nText",
        "1\ninvalid\nText",
        "",
    ] {
        assert!(Transcript::from_srt("srt", "Call", MEETING_GOAL, malformed).is_err());
    }
}

#[test]
fn validates_id_coverage_and_chronological_timestamps() {
    let mut t = transcript(&["Hello", "Decision"]);
    t.segments[1].id = t.segments[0].id.clone();
    assert!(t.validate().is_err());
    let mut t = transcript(&["Hello", "Decision"]);
    t.segments[0].start_ms = 2000;
    t.segments[0].end_ms = 3000;
    assert!(t.validate().is_err());
    let labeled = LabeledTranscript {
        transcript: transcript(&["Hello"]),
        labels: vec![],
    };
    assert!(labeled.validate().is_err());
}

struct Words;
impl TextTokenizer for Words {
    fn spec(&self) -> TokenizerSpec {
        TokenizerSpec {
            kind: "test".into(),
            fingerprint: "test".into(),
            vocab_size: 2,
            pad_id: 0,
        }
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(vec![1; text.split_whitespace().count().max(1)])
    }
}

#[test]
fn long_transcripts_cover_every_turn_once_with_context() {
    let t = transcript(&[
        "First passage",
        "Second passage",
        "Third passage",
        "Fourth passage",
        "Fifth passage",
    ]);
    let limits = assort_data::BatchLimits {
        max_questions: 2,
        max_state_tokens: 40,
        ..transcript_limits(2)
    };
    let windows = prepare_windows(&t, &Words, &limits).unwrap();
    assert_eq!(
        windows
            .iter()
            .flat_map(|w| w.segment_indices.clone())
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4]
    );
    assert!(windows[0].request.state.contains("Third passage"));
    assert!(windows[1].request.state.contains("Second passage"));
    assert_eq!(windows[0].request.questions[0].id, "s0");
    // Large single turns must fail, not be clipped or assigned invented sub-turn times.
    let long = transcript(&[&"long ".repeat(100)]);
    assert!(prepare_windows(&long, &Words, &limits).is_err());
    assert!(
        prepare_windows(
            &t,
            &ByteTokenizer,
            &assort_data::BatchLimits {
                max_question_tokens: 8,
                ..transcript_limits(1)
            }
        )
        .is_err()
    );
}

#[test]
fn summary_is_verbatim_chronological_bounded_and_deduplicated() {
    let t = transcript(&[
        "Hello everyone",
        "We decided to launch on Friday.",
        "We decided to launch on Friday.",
        "Alex will send the report.",
        "The release is blocked by security.",
    ]);
    let mut s = scores(&t);
    s[0].kind = PassageKind::Background;
    s[0].importance = 0.1;
    s[3].importance = 0.99;
    let options = SummaryOptions {
        max_words: 12,
        max_highlights: 3,
        ..Default::default()
    };
    let result = select_summary(&t, &s, &options).unwrap();
    assert_eq!(
        result
            .highlights
            .iter()
            .map(|h| h.source.id.as_str())
            .collect::<Vec<_>>(),
        vec!["s1", "s3"]
    );
    assert_eq!(result.highlights[0].source, t.segments[1]);
    assert!(result.word_count <= options.max_words);
    assert!(result.markdown().contains("00:00:01.000"));
    assert!(
        result
            .markdown()
            .contains("> We decided to launch on Friday.")
    );
}

#[test]
fn empty_summary_does_not_force_low_scoring_content() {
    let t = transcript(&["Hello everyone"]);
    let mut s = scores(&t);
    s[0].importance = 0.1;
    assert!(
        select_summary(&t, &s, &SummaryOptions::default())
            .unwrap()
            .highlights
            .is_empty()
    );
    s[0].importance = f32::NAN;
    assert!(select_summary(&t, &s, &SummaryOptions::default()).is_err());
    assert!(select_summary(&t, &[], &SummaryOptions::default()).is_err());
}

#[test]
fn differing_numbers_and_negation_are_not_removed_as_duplicates() {
    let t = transcript(&[
        "The budget is 5000 euros for this project.",
        "The budget is 6000 euros for this project.",
        "We agreed to launch this project on Friday.",
        "We agreed not to launch this project on Friday.",
    ]);
    let result = select_summary(&t, &scores(&t), &SummaryOptions::default()).unwrap();
    assert_eq!(result.highlights.len(), 4);
}

#[test]
fn labels_map_by_id_and_selection_metrics_match_known_counts() {
    let t = transcript(&["Hello", "A final decision", "An assigned action"]);
    let item = LabeledTranscript {
        transcript: t.clone(),
        labels: vec![
            SegmentLabel {
                segment_id: "s2".into(),
                kind: PassageKind::Action,
            },
            SegmentLabel {
                segment_id: "s0".into(),
                kind: PassageKind::Background,
            },
            SegmentLabel {
                segment_id: "s1".into(),
                kind: PassageKind::Decision,
            },
        ],
    };
    let examples = training_examples(&item, &Words, &transcript_limits(1)).unwrap();
    assert!(matches!(
        examples[0].targets[0],
        assort_data::Target::Hard(3)
    ));
    let mut predicted = scores(&t);
    predicted[0].kind = PassageKind::Background;
    predicted[2].kind = PassageKind::Action;
    let options = SummaryOptions {
        max_highlights: 2,
        ..Default::default()
    };
    let report = evaluate_transcripts(&[item], &[predicted], &options).unwrap();
    assert_eq!(report.category_accuracy, 1.0);
    assert_eq!(report.selected_highlights.f1, 1.0);
    assert_eq!(report.lead_baseline.f1, 0.5);
}

#[test]
fn demo_splits_are_disjoint_by_phrase_and_transcript_id() {
    let (train, validation, test) = meeting_corpus(42);
    assert_eq!((train.len(), validation.len(), test.len()), (48, 12, 12));
    let training_text: std::collections::HashSet<_> = train
        .iter()
        .flat_map(|t| t.transcript.segments.iter().map(|s| s.text.as_str()))
        .collect();
    for item in validation.iter().chain(&test) {
        item.validate().unwrap();
        assert!(
            item.transcript
                .segments
                .iter()
                .all(|s| !training_text.contains(s.text.as_str()))
        );
    }
}
