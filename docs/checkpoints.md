# Checkpoints and inference

## Format

A version-1 checkpoint is a directory containing `manifest.json` and `weights.mpk`. The manifest stores:

- checkpoint version, architecture identifier, and Burn version;
- all model dimensions and forward options;
- tokenizer kind, exact content fingerprint, pad ID, and vocabulary capacity;
- producer-declared weight status, `random` or `trained`;
- SHA-256 of the weight file.

The weight format is Burn 0.21 named MessagePack with full-precision recording. It is not SafeTensors and does not currently provide Python/PyTorch import/export. Checkpoint loading validates compatibility, tokenizer identity, and the weight checksum before constructing the module. Hashes detect accidental corruption or mismatches; they are not an authenticity signature.

Saving stages files in a sibling temporary directory before renaming to a new destination. Existing destinations are rejected. This avoids exposing an incomplete checkpoint during a normal save, but is not a power-loss durability guarantee. A run's `report.json` is separate from the inference checkpoint. Tokenizer JSON is stored at the run root and is passed explicitly when loading.

## Initialize without training

```sh
cargo run --locked -p assort-cli -- init --output .local/random-checkpoint
cargo run --locked -p assort-cli -- inspect .local/random-checkpoint
cargo run --release --locked -p assort-cli -- infer --checkpoint .local/random-checkpoint --input examples/requests.json --allow-random
```

`init` uses the byte tokenizer by default and creates no learned capability. `infer` rejects a random checkpoint unless `--allow-random` is supplied. `inspect` reads metadata only; it does not verify the weight file. `trained` is a producer declaration that updates occurred, not evidence of quality or calibration.

## Inference behavior

The `infer` command loads a checkpoint and a JSON array of requests, collates them, and executes one batched model forward. Use `--input -` for standard input. JSON requests/config files accept a UTF-8 BOM and are bounded to 8 MiB at the CLI; larger datasets should use JSONL training/evaluation paths.

Outputs preserve question and candidate order. Each answer includes `question_id`, `selected_id`, `candidate_ids`, and a `distribution` containing probabilities, selected index, entropy, and normalized entropy. The first candidate wins exact score ties. Nonfinite logits cause an error rather than a fabricated answer.

The optional `--temperature` must be finite and positive. Temperature is applied uniformly to real candidate logits before host softmax. It is not learned or automatically calibrated. Entropy measures distribution concentration, not the probability that the model is correct.

`Engine::new` accepts only a non-autodiff model. Convert a training module with `.valid()` to disable dropout and gradients. The low-level `DecisionModel::forward` remains generic so custom trainers can use autodiff directly.

The typed API uses request-bound `DecisionKey<T>` values. It can combine enums, booleans, and ordinal scores in one `QuestionSet`. Trying to read another set's key returns an error. String-based JSON requests return selected IDs instead of pretending a dynamic lookup has an arbitrary compile-time Rust type.
