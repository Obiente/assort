# Training and evaluation

## Runnable support-routing example

```sh
cargo run --release --locked -p assort-cli -- train-demo --output .local/support-demo
```

The task routes short requests to billing, technical support, delivery, or account support. Data is generated locally from authored synthetic phrases. Each category has 12 training phrases with 4 wrappers, producing 192 examples total. Each has 4 separate validation phrases with 2 wrappers, producing 32 validation examples. Splitting occurs before adding wrappers, so the same base phrase cannot appear in both sets. These examples share category vocabulary intentionally: this is a narrow supervised exercise, not an out-of-domain benchmark.

The example fits a lowercase, whitespace/punctuation-aware WordLevel tokenizer from training states, questions, and candidate descriptions only. Unknown validation words map to `[UNK]`. This keeps the CPU example fast. Replace it with a more expressive BPE/Unigram tokenizer for broader work.

Training performs:

1. Seeded random initialization and seeded data shuffling.
2. Candidate permutations with matching hard-label/soft-target permutations.
3. Batched forward passes and mean cross entropy per real question.
4. Backpropagation, per-parameter norm clipping, and Adam updates.
5. Validation after every epoch on a non-autodiff model, with dropout disabled.
6. Selection of the best trained epoch by validation cross entropy.

Defaults: 20 epochs, batch size 16, learning rate 0.003, clipping norm 1, seed 42. Change them directly:

```sh
cargo run --release --locked -p assort-cli -- train-demo --output .local/support-experiment --epochs 40 --batch-size 8 --learning-rate 0.001 --seed 7
```

The initial random validation score is retained for comparison. The selected checkpoint is always a trained epoch; if training worsens validation, the report exposes that rather than silently returning random weights. Matching seeds should make runs reproducible on the same dependency versions/backend, but exact cross-platform floating-point identity is not promised.

## Saved run

```text
run/
  train.jsonl              exact input training split
  validation.jsonl         exact validation split
  tokenizer.json           present when using a JSON tokenizer
  model-config.json
  training-config.json
  batch-limits.json
  metrics.jsonl            flushed after each completed epoch
  report.json              baseline, epoch history, best epoch, optimizer steps
  checkpoint/
    manifest.json
    weights.mpk
```

Output must be a new directory. Interrupted runs retain their input snapshots and completed epoch metrics, but this baseline saves model weights only after the full training run. It does not save Adam state, schedulers, RNG state, or resumable epoch checkpoints. A saved inference checkpoint is not an exact training-resume checkpoint.

## Metrics

`cross_entropy` is the mean per-question negative log probability for hard targets, or soft-target cross entropy. Lower is better. Soft-target CE and KL divergence have the same parameter gradients; their numeric values differ by target entropy.

`accuracy` measures agreement with the hard label or argmax of a soft target, with first-candidate tie breaking. It can hide uncertainty and class imbalance.

`brier` is the sum of squared probability errors across candidates, averaged over real questions. Lower is better. The example reports it but optimizes only cross entropy. No calibration fitting, ECE, teacher distillation pipeline, or reinforcement-learning objective is implemented.

Re-evaluate the selected checkpoint:

```sh
cargo run --release --locked -p assort-cli -- evaluate --checkpoint .local/support-demo/checkpoint --tokenizer .local/support-demo/tokenizer.json --input .local/support-demo/validation.jsonl
```

Validation is used for model selection. For a meaningful final quality estimate, supply a third untouched test split and evaluate it once after choosing your setup. The generated demo does not provide an independent test benchmark.

## Train on your own data

Prepare separate JSONL files following [the data contract](data.md), then run:

```sh
cargo run --release --locked -p assort-cli -- train --train data/train.jsonl --validation data/validation.jsonl --tokenizer data/tokenizer.json --pad-id 0 --output .local/custom-run --epochs 20 --batch-size 8
```

If `--tokenizer` is omitted, the byte baseline is used. It covers all UTF-8 bytes but consumes longer sequences and is not pretrained. If a model config is omitted, a tiny model with the tokenizer's vocabulary size is created. All `train` commands start from random weights.

To edit architecture dimensions:

```sh
cargo run --locked -p assort-cli -- config --preset small --vocab-size 32000
```

Save the printed JSON, edit it, and pass `--config path/to/config.json`. The vocabulary size must equal the tokenizer's maximum token ID plus one, including added tokens. Use the actual tokenizer capacity, not an assumed vocabulary size. Pass `--limits configs/batch-limits.json` or an edited local copy to control token/batch budgets. Configured lengths cannot exceed the model's positional capacity at inference.

The trainer accepts hard and soft labels in the same dataset and supports multiple questions and ragged candidate counts. It rejects overlapping normalized state strings between train and validation; grouping related conversations and near-duplicate examples is still your responsibility. Candidate IDs and question IDs are metadata, so semantic meaning must be in the text descriptions.

The CLI reads at most 100,000 examples per split into memory. JSONL validation itself is streaming, but the simple training loop is not an out-of-core trainer. Adapt loading, tokenization caching, bucketing, optimizer scheduling, and checkpoint cadence for large datasets.

## Extend training in Rust

`assort_training::train::<Autodiff<Flex>>(...)` accepts model config, tokenizer, two slices of `Example`, training config, batch limits, device, and an epoch callback. It returns an inference model plus `TrainingReport`. The callback can log metrics or stop the run by returning an error.

For a custom objective or optimizer, reuse `assort_data::collate`, `DecisionModel::forward`, and the masks on `ModelOutput`. `assort_training::cross_entropy` is an inspectable hard/soft-target baseline. Use `.valid()` when evaluating an autodiff model. Preserve target order when permuting candidates; exclude padded question rows from both loss and normalization.

Changing the architecture requires a new checkpoint architecture identifier/version when its parameter or forward contracts cease to be compatible. Checkpoints currently store inference weights in Burn's named MessagePack format, not SafeTensors.
