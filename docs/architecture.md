# Architecture

## Forward pass

```text
state tokens [B,S] -> shared token embeddings -> state transformer [B,S,D]
                                                           |
question tokens [B,Q,Tq] -> query transformer -> mean pool [B,Q,D]
                                                           |
                                 question-to-state attention
                                                           |
candidate tokens [B,Q,C,Tc] -> query transformer -> mean pool [B,Q,C,D]
                                                           |
                           optional candidate-to-state attention
                                                           |
                         concat(q, c, q*c, abs(q-c)) -> MLP
                                                           |
                                                 logits [B,Q,C]
```

`B` is states per batch, `S` state tokens, `Q` questions per state, `C` candidates per question, `Tq` and `Tc` text lengths, and `D` hidden size. All dimensions except hidden size are dynamically padded within the batch.

Each state's transformer runs once per forward pass. Question text is flattened to `[B*Q,Tq]` and candidate text to `[B*Q*C,Tc]` for their respective encoder calls. They share a query encoder and the same token embedding table as the state encoder. State and query encoders have separate learned position tables.

Encoder blocks use pre-layer normalization, multi-head bidirectional attention, residual connections, and GELU feed-forward layers. A final layer norm follows the stack. Pooling averages valid text tokens only. No causal mask is used.

The question cross-attention operates on `[B,Q,D]` queries against `[B,S,D]` state memory. If enabled, candidate queries combine their own embedding with the contextual question and attend to state as `[B,Q*C,D]`. Candidates and questions do not self-attend to one another. Candidate ordering therefore permutes the output ordering without changing a candidate's score, apart from numerical tolerance. Questions can share evidence, but the model does not impose cross-question logical constraints.

The scorer concatenates the contextual question, candidate, elementwise product, and absolute difference. A shared two-layer MLP returns one scalar for every candidate. There are no category-specific output heads and no generation loop.

## Public surfaces

`DecisionModel<B>::new(config, device)` creates random parameters. `forward(&TokenBatch)` returns `ModelOutput<B>` with differentiable logits and masks. All operations remain on the backend until an explicit data read. The public forward method includes host-to-device tensor creation; it does not yet cache device batches or encoded state across calls.

`ModelOutput::probabilities()` returns normalized probabilities for real candidate rows and zeros for padding. The example trainer consumes logits through masked cross entropy. Inference reads real candidate logits only and applies a numerically stable host softmax with optional temperature.

The tiny preset uses hidden size 32, 4 heads, 2 state layers, 1 query layer, and FFN size 64. The support demo uses 1 state layer. The small research preset uses hidden size 512, 8 heads, 8 state layers, 2 query layers, and FFN size 2048. Vocabulary and position capacity also contribute to parameter count. Presets express shapes, not expected capability.

## Padding and cost

All masks use `true = invalid/padding`. Attention masks padding keys; pooling separately excludes padded text positions. Fully padded question/candidate text rows expose one dummy key to prevent all-masked attention. Their outputs are excluded by question and candidate masks. Padded candidate logits use `-1e9`, avoiding `0 * -infinity` in soft-target losses.

Attention is implemented explicitly with dense score matrices. It is not Flash Attention and is not designed to make a 4096-token CPU run cheap. Memory cost includes:

- state attention: `B * heads * S²`;
- question encoding: `B * Q * heads * Tq²`;
- candidate encoding: `B * Q * C * heads * Tc²`;
- candidate-to-state attention: `B * heads * Q * C * S`.

`max_attention_elements` bounds each individual score matrix before device allocation. It does not bound total model, optimizer, activation, or autodiff memory. Bucket similar lengths and reduce batch size before raising budgets. Candidate text encoding can become the dominant cost.

## Research extension points

- Replace `TextEncoder` with another bidirectional block or position scheme.
- Replace mean pooling with a learned pooling token.
- Change candidate evidence attention independently of the text encoders.
- Add state/candidate encoding caches with explicit model/tokenizer identity.
- Replace the scorer without changing the typed API or data format.
- Add GPU batching, fused attention, and mixed precision after backend-specific tests.

The optional `assort-model/wgpu` feature exposes Burn's WGPU backend through dependency features. The supplied CLI and verification baseline use float32 CPU. GPU performance and float16 behavior are not claimed; in particular, revisit the finite padding sentinel before using half precision.
