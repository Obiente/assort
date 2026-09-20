use assort_core::{Request, Result, ensure};
use assort_tokenizer::TextTokenizer;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BatchLimits {
    pub max_batch_size: usize,
    pub max_questions: usize,
    pub max_candidates: usize,
    pub max_state_tokens: usize,
    pub max_question_tokens: usize,
    pub max_candidate_tokens: usize,
    /// Total padded token slots, checked before allocating dense buffers.
    pub max_padded_tokens: usize,
}

impl Default for BatchLimits {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_questions: 32,
            max_candidates: 255,
            max_state_tokens: 512,
            max_question_tokens: 128,
            max_candidate_tokens: 128,
            max_padded_tokens: 262_144,
        }
    }
}

impl BatchLimits {
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.max_batch_size > 0
                && self.max_questions > 0
                && (1..=255).contains(&self.max_candidates),
            "invalid batch/question/candidate limits",
        )?;
        ensure(
            self.max_state_tokens > 0
                && self.max_question_tokens > 0
                && self.max_candidate_tokens > 0
                && self.max_padded_tokens > 0,
            "token limits must be positive",
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchShape {
    pub batch: usize,
    pub questions: usize,
    pub candidates: usize,
    pub state_tokens: usize,
    pub question_tokens: usize,
    pub candidate_tokens: usize,
}

/// Row-major tokens and masks. All masks use true = padding/invalid.
/// Padded text rows expose one dummy key to keep attention finite;
/// their question/candidate mask still excludes them from decisions and losses.
#[derive(Debug, Clone)]
pub struct TokenBatch {
    shape: BatchShape,
    states: Vec<i64>,
    state_padding: Vec<bool>,
    questions: Vec<i64>,
    question_padding: Vec<bool>,
    candidates: Vec<i64>,
    candidate_padding: Vec<bool>,
    question_mask: Vec<bool>,
    candidate_mask: Vec<bool>,
    vocab_size: usize,
}

impl TokenBatch {
    pub fn shape(&self) -> BatchShape {
        self.shape
    }
    pub fn states(&self) -> &[i64] {
        &self.states
    }
    pub fn state_padding(&self) -> &[bool] {
        &self.state_padding
    }
    pub fn questions(&self) -> &[i64] {
        &self.questions
    }
    pub fn question_padding(&self) -> &[bool] {
        &self.question_padding
    }
    pub fn candidates(&self) -> &[i64] {
        &self.candidates
    }
    pub fn candidate_padding(&self) -> &[bool] {
        &self.candidate_padding
    }
    pub fn question_mask(&self) -> &[bool] {
        &self.question_mask
    }
    pub fn candidate_mask(&self) -> &[bool] {
        &self.candidate_mask
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

pub fn collate(
    tokenizer: &impl TextTokenizer,
    requests: &[Request],
    limits: &BatchLimits,
) -> Result<TokenBatch> {
    limits.validate()?;
    ensure(
        !requests.is_empty() && requests.len() <= limits.max_batch_size,
        "batch size outside configured limits",
    )?;
    let spec = tokenizer.spec();
    ensure(
        spec.vocab_size > 0 && (spec.pad_id as usize) < spec.vocab_size,
        "invalid tokenizer vocabulary/pad ID",
    )?;
    let mut shape = BatchShape {
        batch: requests.len(),
        questions: 1,
        candidates: 1,
        state_tokens: 1,
        question_tokens: 1,
        candidate_tokens: 1,
    };
    let mut states = Vec::new();
    let mut questions = Vec::new();
    let mut candidates = Vec::new();
    let mut raw_tokens = 0usize;
    let mut encode = |text: &str, max: usize| -> Result<Vec<u32>> {
        let tokens = tokenizer.encode(text)?;
        ensure(
            !tokens.is_empty() && tokens.len() <= max,
            format!(
                "encoded length {} outside 1..={max}; input is not truncated",
                tokens.len()
            ),
        )?;
        ensure(
            tokens.iter().all(|&id| (id as usize) < spec.vocab_size),
            "token ID outside vocabulary",
        )?;
        raw_tokens = raw_tokens.saturating_add(tokens.len());
        ensure(
            raw_tokens <= limits.max_padded_tokens,
            "unpadded tokens exceed batch token budget",
        )?;
        Ok(tokens)
    };
    for request in requests {
        request.validate()?;
        ensure(
            request.questions.len() <= limits.max_questions,
            "too many questions",
        )?;
        shape.questions = shape.questions.max(request.questions.len());
        let state = encode(&request.state, limits.max_state_tokens)?;
        shape.state_tokens = shape.state_tokens.max(state.len());
        states.push(state);
        let mut qs = Vec::new();
        let mut cs = Vec::new();
        for q in &request.questions {
            ensure(
                q.candidates.len() <= limits.max_candidates,
                "too many candidates",
            )?;
            shape.candidates = shape.candidates.max(q.candidates.len());
            let text = encode(&q.text, limits.max_question_tokens)?;
            shape.question_tokens = shape.question_tokens.max(text.len());
            qs.push(text);
            let mut row = Vec::new();
            for c in &q.candidates {
                // IDs are output metadata, not semantic input. Describe all meaning in description.
                let text = encode(&c.description, limits.max_candidate_tokens)?;
                shape.candidate_tokens = shape.candidate_tokens.max(text.len());
                row.push(text);
            }
            cs.push(row);
        }
        questions.push(qs);
        candidates.push(cs);
    }
    let bq = shape.batch.saturating_mul(shape.questions);
    let bqc = bq.saturating_mul(shape.candidates);
    let ns = shape.batch.saturating_mul(shape.state_tokens);
    let nq = bq.saturating_mul(shape.question_tokens);
    let nc = bqc.saturating_mul(shape.candidate_tokens);
    ensure(
        ns.saturating_add(nq).saturating_add(nc) <= limits.max_padded_tokens,
        "padded batch exceeds token budget; use smaller batches or bucket similar lengths",
    )?;
    let mut batch = TokenBatch {
        shape,
        states: vec![spec.pad_id as i64; ns],
        state_padding: vec![true; ns],
        questions: vec![spec.pad_id as i64; nq],
        question_padding: vec![true; nq],
        candidates: vec![spec.pad_id as i64; nc],
        candidate_padding: vec![true; nc],
        question_mask: vec![true; bq],
        candidate_mask: vec![true; bqc],
        vocab_size: spec.vocab_size,
    };
    // Dummy rows avoid all-masked softmax. Real rows overwrite their complete valid prefix.
    for row in batch.question_padding.chunks_mut(shape.question_tokens) {
        row[0] = false;
    }
    for row in batch.candidate_padding.chunks_mut(shape.candidate_tokens) {
        row[0] = false;
    }
    for b in 0..shape.batch {
        copy_tokens(
            &states[b],
            &mut batch.states,
            &mut batch.state_padding,
            b * shape.state_tokens,
        );
        for q in 0..questions[b].len() {
            let qi = b * shape.questions + q;
            batch.question_mask[qi] = false;
            copy_tokens(
                &questions[b][q],
                &mut batch.questions,
                &mut batch.question_padding,
                qi * shape.question_tokens,
            );
            for (c, candidate) in candidates[b][q].iter().enumerate() {
                let ci = qi * shape.candidates + c;
                batch.candidate_mask[ci] = false;
                copy_tokens(
                    candidate,
                    &mut batch.candidates,
                    &mut batch.candidate_padding,
                    ci * shape.candidate_tokens,
                );
            }
        }
    }
    Ok(batch)
}

fn copy_tokens(source: &[u32], ids: &mut [i64], padding: &mut [bool], offset: usize) {
    for (i, &token) in source.iter().enumerate() {
        ids[offset + i] = token as i64;
        padding[offset + i] = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assort_core::Question;
    use assort_tokenizer::ByteTokenizer;

    #[test]
    fn ragged_batches_preserve_masks_and_order() {
        let a = Request {
            state: "a".into(),
            questions: vec![Question::boolean("a", "Short?")],
        };
        let mut b = a.clone();
        b.state = "A much longer state".into();
        b.questions.push(Question::boolean("b", "Longer question?"));
        b.questions[1].candidates.pop();
        let batch = collate(&ByteTokenizer, &[a, b], &BatchLimits::default()).unwrap();
        assert_eq!(batch.shape().questions, 2);
        assert_eq!(batch.question_mask(), &[false, true, false, false]);
        assert_eq!(
            batch.candidate_mask(),
            &[false, false, true, true, false, false, false, true]
        );
        assert_eq!(batch.states()[1], b'a' as i64 + 3);
        assert!(batch.state_padding()[3]);
        assert!(!batch.question_padding()[batch.shape().question_tokens]);
    }

    #[test]
    fn rejects_overlong_inputs_and_padding_explosion() {
        let r = Request {
            state: "abcd".into(),
            questions: vec![Question::boolean("a", "Question?")],
        };
        let limits = BatchLimits {
            max_state_tokens: 3,
            ..Default::default()
        };
        assert!(collate(&ByteTokenizer, std::slice::from_ref(&r), &limits).is_err());
        let limits = BatchLimits {
            max_padded_tokens: 4,
            ..Default::default()
        };
        assert!(collate(&ByteTokenizer, &[r], &limits).is_err());
    }
}
