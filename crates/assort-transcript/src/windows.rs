use crate::{LabeledTranscript, PassageKind, Transcript};
use assort_core::{Candidate, Question, Request, Result, ensure};
use assort_data::{BatchLimits, Example, Target, collate};
use assort_inference::Engine;
use assort_tokenizer::TextTokenizer;
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct TranscriptWindow {
    pub request: Request,
    /// Original indices, excluding overlapping context-only turns.
    pub segment_indices: Vec<usize>,
}

pub fn transcript_limits(batch_size: usize) -> BatchLimits {
    BatchLimits {
        max_batch_size: batch_size,
        max_questions: 8,
        max_candidates: 4,
        max_state_tokens: 384,
        max_question_tokens: 128,
        max_candidate_tokens: 64,
        ..Default::default()
    }
}

/// Token-aware windows with one neighboring turn on each side when it fits.
/// Each original segment is scored exactly once. Individual overlong turns are rejected.
pub fn prepare_windows(
    transcript: &Transcript,
    tokenizer: &impl TextTokenizer,
    limits: &BatchLimits,
) -> Result<Vec<TranscriptWindow>> {
    transcript.validate()?;
    limits.validate()?;
    ensure(
        limits.max_candidates >= 4,
        "transcript classification requires four candidates",
    )?;
    let mut windows = Vec::new();
    let mut start = 0;
    while start < transcript.segments.len() {
        let mut end = start;
        while end < transcript.segments.len() && end - start < limits.max_questions {
            let state = state_text(transcript, start, end + 1);
            if tokenizer.encode(&state)?.len() > limits.max_state_tokens {
                break;
            }
            end += 1;
        }
        ensure(
            end > start,
            format!(
                "segment {} exceeds the state token limit; split it into smaller timed segments",
                transcript.segments[start].id
            ),
        )?;
        let mut context_start = start;
        let mut context_end = end;
        if start > 0
            && tokenizer
                .encode(&state_text(transcript, start - 1, end))?
                .len()
                <= limits.max_state_tokens
        {
            context_start -= 1;
        }
        if end < transcript.segments.len()
            && tokenizer
                .encode(&state_text(transcript, context_start, end + 1))?
                .len()
                <= limits.max_state_tokens
        {
            context_end += 1;
        }
        let questions = (start..end)
            .map(|index| {
                let segment = &transcript.segments[index];
                Question {
                    id: segment.id.clone(),
                    text: format!("Classify this passage: {}", segment.text),
                    candidates: PassageKind::ALL
                        .into_iter()
                        .map(|kind| Candidate::new(kind.id(), kind.description()))
                        .collect(),
                }
            })
            .collect();
        let request = Request {
            state: state_text(transcript, context_start, context_end),
            questions,
        };
        // Same tokenizer and validation contract for training and inference.
        collate(tokenizer, std::slice::from_ref(&request), limits)?;
        windows.push(TranscriptWindow {
            request,
            segment_indices: (start..end).collect(),
        });
        start = end;
    }
    Ok(windows)
}

fn state_text(transcript: &Transcript, start: usize, end: usize) -> String {
    let mut text = format!("Goal: {}\nTranscript:\n", transcript.goal);
    for segment in &transcript.segments[start..end] {
        if let Some(speaker) = &segment.speaker {
            text.push_str(speaker);
            text.push_str(": ");
        }
        text.push_str(&segment.text);
        text.push('\n');
    }
    text
}

pub fn training_examples(
    transcript: &LabeledTranscript,
    tokenizer: &impl TextTokenizer,
    limits: &BatchLimits,
) -> Result<Vec<Example>> {
    transcript.validate()?;
    let labels: std::collections::HashMap<_, _> = transcript
        .labels
        .iter()
        .map(|l| (l.segment_id.as_str(), l.kind))
        .collect();
    prepare_windows(&transcript.transcript, tokenizer, limits)?
        .into_iter()
        .map(|window| {
            let targets = window
                .request
                .questions
                .iter()
                .map(|q| Target::Hard(labels[q.id.as_str()].index()))
                .collect();
            Ok(Example {
                request: window.request,
                targets,
            })
        })
        .collect()
}

/// Text available for fitting a demo vocabulary. Call on the training split only.
pub fn tokenizer_corpus(transcripts: &[LabeledTranscript]) -> Vec<String> {
    let mut corpus = vec!["Goal Transcript Classify this passage".into()];
    corpus.extend(PassageKind::ALL.map(|kind| kind.description().to_owned()));
    for item in transcripts {
        corpus.push(item.transcript.goal.clone());
        for segment in &item.transcript.segments {
            corpus.push(segment.text.clone());
            if let Some(speaker) = &segment.speaker {
                corpus.push(speaker.clone());
            }
        }
    }
    corpus
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentScore {
    pub segment_id: String,
    pub kind: PassageKind,
    /// 1 - P(background). This is not calibrated correctness confidence.
    pub importance: f32,
}

pub fn score_transcript<B: Backend, T: TextTokenizer>(
    engine: &Engine<B, T>,
    transcript: &Transcript,
) -> Result<Vec<SegmentScore>> {
    let limits = engine.limits();
    let windows = prepare_windows(transcript, engine.tokenizer(), limits)?;
    let mut scores = Vec::with_capacity(transcript.segments.len());
    for chunk in windows.chunks(limits.max_batch_size) {
        let requests: Vec<_> = chunk.iter().map(|w| w.request.clone()).collect();
        let responses = engine.evaluate(&requests)?;
        for response in responses {
            for answer in response.answers {
                scores.push(SegmentScore {
                    segment_id: answer.question_id,
                    kind: PassageKind::ALL[answer.distribution.selected()],
                    importance: 1.0
                        - answer.distribution.probabilities()[PassageKind::Background.index()],
                });
            }
        }
    }
    Ok(scores)
}
