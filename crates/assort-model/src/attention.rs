use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Bool, Tensor, activation::softmax, backend::Backend},
};

/// Explicit multi-head attention keeps the research surface small and inspectable.
/// Queries never attend to other questions or candidates.
#[derive(Module, Debug)]
pub(crate) struct Attention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    heads: usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(hidden: usize, heads: usize, device: &B::Device) -> Self {
        let linear = LinearConfig::new(hidden, hidden);
        Self {
            query: linear.init(device),
            key: linear.init(device),
            value: linear.init(device),
            output: linear.init(device),
            heads,
        }
    }

    pub fn forward(
        &self,
        query: Tensor<B, 3>,
        memory: Tensor<B, 3>,
        padding: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3> {
        let [b, q, d] = query.dims();
        let [_, k, _] = memory.dims();
        let h = self.heads;
        let dh = d / h;
        let query = self
            .query
            .forward(query)
            .reshape([b, q, h, dh])
            .swap_dims(1, 2);
        let key = self
            .key
            .forward(memory.clone())
            .reshape([b, k, h, dh])
            .swap_dims(1, 2);
        let value = self
            .value
            .forward(memory)
            .reshape([b, k, h, dh])
            .swap_dims(1, 2);
        let scores = query.matmul(key.swap_dims(2, 3)) / (dh as f32).sqrt();
        let scores = scores.mask_fill(
            padding.reshape([b, 1, 1, k]).expand([b, h, q, k]),
            f32::NEG_INFINITY,
        );
        let context = softmax(scores, 3)
            .matmul(value)
            .swap_dims(1, 2)
            .reshape([b, q, d]);
        self.output.forward(context)
    }
}
