use crate::attention::Attention;
use burn::{
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig,
    },
    tensor::{Bool, Int, Tensor, activation::gelu, backend::Backend},
};

#[derive(Module, Debug)]
pub(crate) struct EncoderBlock<B: Backend> {
    attention: Attention<B>,
    attention_norm: LayerNorm<B>,
    ffn_norm: LayerNorm<B>,
    up: Linear<B>,
    down: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> EncoderBlock<B> {
    pub fn new(d: usize, heads: usize, ffn: usize, dropout: f64, device: &B::Device) -> Self {
        Self {
            attention: Attention::new(d, heads, device),
            attention_norm: LayerNormConfig::new(d).init(device),
            ffn_norm: LayerNormConfig::new(d).init(device),
            up: LinearConfig::new(d, ffn).init(device),
            down: LinearConfig::new(ffn, d).init(device),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, padding: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let norm = self.attention_norm.forward(x.clone());
        let x = x + self
            .dropout
            .forward(self.attention.forward(norm.clone(), norm, padding));
        let ffn = self
            .down
            .forward(gelu(self.up.forward(self.ffn_norm.forward(x.clone()))));
        x + self.dropout.forward(ffn)
    }
}

#[derive(Module, Debug)]
pub(crate) struct TextEncoder<B: Backend> {
    positions: Embedding<B>,
    blocks: Vec<EncoderBlock<B>>,
    norm: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> TextEncoder<B> {
    pub fn new(config: &crate::ModelConfig, layers: usize, device: &B::Device) -> Self {
        Self {
            positions: EmbeddingConfig::new(config.max_positions, config.hidden_size).init(device),
            blocks: (0..layers)
                .map(|_| {
                    EncoderBlock::new(
                        config.hidden_size,
                        config.num_heads,
                        config.ffn_size,
                        config.dropout,
                        device,
                    )
                })
                .collect(),
            norm: LayerNormConfig::new(config.hidden_size).init(device),
            dropout: DropoutConfig::new(config.dropout).init(),
        }
    }

    pub fn forward(&self, embeddings: Tensor<B, 3>, padding: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let [b, tokens, _] = embeddings.dims();
        let positions = Tensor::<B, 1, Int>::arange(0..tokens as i64, &embeddings.device())
            .reshape([1, tokens])
            .expand([b, tokens]);
        let mut x = self
            .dropout
            .forward(embeddings + self.positions.forward(positions));
        for block in &self.blocks {
            x = block.forward(x, padding.clone());
        }
        self.norm.forward(x)
    }
}

pub(crate) fn mean_pool<B: Backend>(x: Tensor<B, 3>, padding: Tensor<B, 2, Bool>) -> Tensor<B, 2> {
    let [b, _, d] = x.dims();
    let valid = padding.bool_not().float().unsqueeze_dim::<3>(2);
    ((x * valid.clone()).sum_dim(1) / valid.sum_dim(1).clamp_min(1.0)).reshape([b, d])
}
