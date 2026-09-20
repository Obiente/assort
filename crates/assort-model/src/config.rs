use assort_core::{Result, ensure};
use assort_data::TokenBatch;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub state_layers: usize,
    pub query_layers: usize,
    pub ffn_size: usize,
    pub max_positions: usize,
    pub dropout: f64,
    pub candidate_state_attention: bool,
    /// Per-attention-operation score element budget, not a total device memory guarantee.
    pub max_attention_elements: usize,
}

impl ModelConfig {
    /// Small enough for CPU smoke tests. Weights are always random at initialization.
    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size: 32,
            num_heads: 4,
            state_layers: 2,
            query_layers: 1,
            ffn_size: 64,
            max_positions: 512,
            dropout: 0.0,
            candidate_state_attention: true,
            max_attention_elements: 16_777_216,
        }
    }

    /// A larger research starting point. This is a configuration, not a quality claim.
    pub fn small(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size: 512,
            num_heads: 8,
            state_layers: 8,
            query_layers: 2,
            ffn_size: 2048,
            max_positions: 4096,
            dropout: 0.1,
            candidate_state_attention: true,
            max_attention_elements: 134_217_728,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure(
            self.vocab_size > 0 && self.vocab_size <= i32::MAX as usize,
            "invalid vocabulary size",
        )?;
        ensure(
            self.hidden_size > 0
                && self.num_heads > 0
                && self.hidden_size.is_multiple_of(self.num_heads),
            "hidden size must be divisible by a positive head count",
        )?;
        ensure(
            self.hidden_size.checked_mul(4).is_some()
                && self.ffn_size > 0
                && self.state_layers > 0
                && self.query_layers > 0,
            "invalid encoder dimensions",
        )?;
        ensure(
            self.max_positions > 0
                && self.max_positions <= i32::MAX as usize
                && self.max_attention_elements > 0,
            "invalid position/attention limits",
        )?;
        ensure(
            self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout),
            "dropout must be in [0, 1)",
        )
    }

    pub(crate) fn validate_batch(&self, batch: &TokenBatch) -> Result<()> {
        let s = batch.shape();
        ensure(
            batch.vocab_size() == self.vocab_size,
            "tokenizer and model vocabulary sizes differ",
        )?;
        ensure(
            s.state_tokens
                .max(s.question_tokens)
                .max(s.candidate_tokens)
                <= self.max_positions,
            "batch exceeds model position capacity",
        )?;
        let b = s.batch as u128;
        let q = s.questions as u128;
        let c = s.candidates as u128;
        let h = self.num_heads as u128;
        let scores = [
            b * h * (s.state_tokens as u128).pow(2),
            b * q * h * (s.question_tokens as u128).pow(2),
            b * q * c * h * (s.candidate_tokens as u128).pow(2),
            b * h * q * s.state_tokens as u128,
            if self.candidate_state_attention {
                b * h * q * c * s.state_tokens as u128
            } else {
                0
            },
        ];
        ensure(
            scores.into_iter().max().unwrap_or(0) <= self.max_attention_elements as u128,
            "attention matrix exceeds configured element budget; reduce batch or sequence lengths",
        )
    }
}
