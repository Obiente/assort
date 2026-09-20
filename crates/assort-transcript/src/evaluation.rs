use crate::{
    LabeledTranscript, PassageKind, SegmentScore, SummaryOptions, select_summary, word_count,
};
use assort_core::{Result, ensure};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default, Clone, Serialize)]
pub struct SelectionMetrics {
    pub selected: usize,
    pub relevant: usize,
    pub correct: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl SelectionMetrics {
    fn add(&mut self, selected: &HashSet<&str>, relevant: &HashSet<&str>) {
        self.selected += selected.len();
        self.relevant += relevant.len();
        self.correct += selected.intersection(relevant).count();
        self.precision = if self.selected == 0 {
            0.0
        } else {
            self.correct as f64 / self.selected as f64
        };
        self.recall = if self.relevant == 0 {
            0.0
        } else {
            self.correct as f64 / self.relevant as f64
        };
        self.f1 = if self.precision + self.recall == 0.0 {
            0.0
        } else {
            2.0 * self.precision * self.recall / (self.precision + self.recall)
        };
    }
}

#[derive(Debug, Serialize)]
pub struct TranscriptEvaluation {
    pub transcripts: usize,
    pub segments: usize,
    pub category_accuracy: f64,
    /// Rows = true class, columns = predicted class, in category_order.
    pub confusion: [[usize; 4]; 4],
    pub category_order: [&'static str; 4],
    pub selected_highlights: SelectionMetrics,
    /// Simple first-N source turns baseline, subject to the same word/turn budgets.
    pub lead_baseline: SelectionMetrics,
    pub options: SummaryOptions,
}

/// Micro-averaged metrics against annotated source spans, without generated-text metrics.
pub fn evaluate_transcripts(
    data: &[LabeledTranscript],
    predictions: &[Vec<SegmentScore>],
    options: &SummaryOptions,
) -> Result<TranscriptEvaluation> {
    ensure(
        !data.is_empty() && data.len() == predictions.len(),
        "evaluation requires one scored result per labeled transcript",
    )?;
    let mut result = TranscriptEvaluation {
        transcripts: data.len(),
        segments: 0,
        category_accuracy: 0.0,
        confusion: [[0; 4]; 4],
        category_order: PassageKind::ALL.map(|k| k.id()),
        selected_highlights: SelectionMetrics::default(),
        lead_baseline: SelectionMetrics::default(),
        options: options.clone(),
    };
    let mut correct = 0;
    for (item, scores) in data.iter().zip(predictions) {
        item.validate()?;
        let summary = select_summary(&item.transcript, scores, options)?;
        let labels: HashMap<_, _> = item
            .labels
            .iter()
            .map(|l| (l.segment_id.as_str(), l.kind))
            .collect();
        let relevant: HashSet<_> = item
            .labels
            .iter()
            .filter(|l| l.kind.is_highlight())
            .map(|l| l.segment_id.as_str())
            .collect();
        let selected: HashSet<_> = summary
            .highlights
            .iter()
            .map(|h| h.source.id.as_str())
            .collect();
        result.selected_highlights.add(&selected, &relevant);
        let mut lead = HashSet::new();
        let mut words = 0;
        for segment in &item.transcript.segments {
            let count = word_count(&segment.text);
            if lead.len() < options.max_highlights && count <= options.max_words - words {
                lead.insert(segment.id.as_str());
                words += count;
            }
        }
        result.lead_baseline.add(&lead, &relevant);
        for score in scores {
            let label = labels[score.segment_id.as_str()];
            result.confusion[label.index()][score.kind.index()] += 1;
            correct += usize::from(label == score.kind);
            result.segments += 1;
        }
    }
    result.category_accuracy = correct as f64 / result.segments as f64;
    Ok(result)
}
