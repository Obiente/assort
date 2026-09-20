use crate::{PassageKind, Segment, SegmentScore, Transcript, timestamp};
use assort_core::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SummaryOptions {
    pub max_words: usize,
    pub max_highlights: usize,
    pub min_importance: f32,
    pub duplicate_similarity: f32,
    pub redundancy_penalty: f32,
}

impl Default for SummaryOptions {
    fn default() -> Self {
        Self {
            max_words: 120,
            max_highlights: 6,
            min_importance: 0.65,
            duplicate_similarity: 0.8,
            redundancy_penalty: 0.3,
        }
    }
}

impl SummaryOptions {
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.max_words > 0 && self.max_highlights > 0,
            "summary budgets must be positive",
        )?;
        ensure(
            [
                self.min_importance,
                self.duplicate_similarity,
                self.redundancy_penalty,
            ]
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
                && self.duplicate_similarity > 0.0,
            "summary scores must be finite in [0,1], with duplicate similarity > 0",
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Highlight {
    pub kind: PassageKind,
    pub importance: f32,
    /// Exact source text, speaker, ID, and timestamps. Never generated or shortened.
    pub source: Segment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    pub transcript_id: String,
    pub title: String,
    pub goal: String,
    pub word_count: usize,
    pub source_segments: usize,
    pub highlights: Vec<Highlight>,
}

impl Summary {
    pub fn markdown(&self) -> String {
        let mut text = format!("# {}\n\n", escape(&self.title));
        if self.highlights.is_empty() {
            text.push_str("No passages met the selection threshold and word budget.\n");
        }
        for highlight in &self.highlights {
            let source = &highlight.source;
            let speaker = source
                .speaker
                .as_ref()
                .map(|s| format!(" | {}", escape(s)))
                .unwrap_or_default();
            text.push_str(&format!(
                "**{} - {} | {}{}** (source: {})\n\n",
                timestamp(source.start_ms),
                timestamp(source.end_ms),
                highlight.kind.id(),
                speaker,
                escape(&source.id)
            ));
            for line in source.text.lines() {
                text.push_str("> ");
                text.push_str(&escape(line));
                text.push('\n');
            }
            text.push('\n');
        }
        text
    }
}

fn escape(text: &str) -> String {
    text.chars()
        .flat_map(|c| {
            if "\\`*_{}[]<>#|".contains(c) {
                vec!['\\', c]
            } else if c == '\n' || c == '\r' {
                vec![' ']
            } else {
                vec![c]
            }
        })
        .collect()
}

/// Global greedy selection over all scored windows, with lexical redundancy control.
/// It cannot resolve contradictions or infer omitted cross-window dependencies.
pub fn select_summary(
    transcript: &Transcript,
    scores: &[SegmentScore],
    options: &SummaryOptions,
) -> Result<Summary> {
    transcript.validate()?;
    options.validate()?;
    let lookup: HashMap<_, _> = scores.iter().map(|s| (s.segment_id.as_str(), s)).collect();
    ensure(
        scores.len() == transcript.segments.len()
            && lookup.len() == scores.len()
            && transcript
                .segments
                .iter()
                .all(|s| lookup.contains_key(s.id.as_str())),
        "scores must cover every transcript segment exactly once",
    )?;
    ensure(
        scores
            .iter()
            .all(|s| s.importance.is_finite() && (0.0..=1.0).contains(&s.importance)),
        "invalid segment importance",
    )?;
    let tokens: Vec<_> = transcript
        .segments
        .iter()
        .map(|s| token_set(&s.text))
        .collect();
    let mut chosen: Vec<usize> = Vec::new();
    let mut words = 0;
    while chosen.len() < options.max_highlights {
        let mut best: Option<(usize, f32)> = None;
        for (index, segment) in transcript.segments.iter().enumerate() {
            let score = lookup[segment.id.as_str()];
            let count = word_count(&segment.text);
            if chosen.contains(&index)
                || !score.kind.is_highlight()
                || score.importance < options.min_importance
                || count > options.max_words - words
            {
                continue;
            }
            let similarity = chosen
                .iter()
                .map(|&other| similarity(&tokens[index], &tokens[other]))
                .fold(0.0f32, f32::max);
            if similarity >= options.duplicate_similarity {
                continue;
            }
            let priority = score.importance - options.redundancy_penalty * similarity;
            if best.is_none_or(|(_, value)| priority > value) {
                best = Some((index, priority));
            }
        }
        let Some((index, _)) = best else {
            break;
        };
        words += word_count(&transcript.segments[index].text);
        chosen.push(index);
    }
    chosen.sort_unstable();
    let highlights = chosen
        .into_iter()
        .map(|i| {
            let source = transcript.segments[i].clone();
            let score = lookup[source.id.as_str()];
            Highlight {
                kind: score.kind,
                importance: score.importance,
                source,
            }
        })
        .collect();
    Ok(Summary {
        transcript_id: transcript.id.clone(),
        title: transcript.title.clone(),
        goal: transcript.goal.clone(),
        word_count: words,
        source_segments: transcript.segments.len(),
        highlights,
    })
}

pub fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

fn token_set(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

fn similarity(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    // Do not merge superficially similar statements that differ in numbers or negation.
    let sensitive = |set: &HashSet<String>| {
        set.iter()
            .filter(|s| {
                s.chars().any(|c| c.is_numeric())
                    || matches!(
                        s.as_str(),
                        "no" | "not" | "never" | "without" | "cancelled" | "canceled"
                    )
                    || s.ends_with("n't")
            })
            .cloned()
            .collect::<HashSet<_>>()
    };
    if sensitive(a) != sensitive(b) {
        return 0.0;
    }
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        a.intersection(b).count() as f32 / union as f32
    }
}
