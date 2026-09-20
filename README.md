# assort

A Rust model for scoring explicit choices against shared context. Assort encodes the context once and answers multiple questions in one forward pass, with typed Rust results, supervised training, and local inference.

The **meeting and call summary** example trains a passage classifier and assembles summaries from source quotations. It supports JSON/SRT input, timestamps, word budgets, duplicate filtering, and separate training, validation, and test splits:

```powershell
./scripts/train-transcript-demo.ps1
```

See [the transcript guide](docs/transcripts.md) to train on your own calls or run the individual commands.

This is an experimental foundation with runnable training examples. No pretrained weights are included. The model selects from supplied candidates; transcript summaries preserve source text rather than generating rewritten prose. The synthetic demos demonstrate learning on narrow tasks and do not establish general language understanding or calibrated confidence.

## Start here

Install Rust 1.92 or newer with Cargo. Windows needs the MSVC build tools used by Rust. The default backend is Burn Flex on CPU; no Python, CUDA, LibTorch, downloaded weights, or external service is required. The first build downloads Rust dependencies.

```sh
git clone https://github.com/obiente/assort.git
cd assort
```

Train a meeting model and summarize the example transcript:

```sh
cargo run --release --locked -p assort-cli -- train-transcript-demo --output .local/meeting-demo
cargo run --release --locked -p assort-cli -- summarize --checkpoint .local/meeting-demo/checkpoint --tokenizer .local/meeting-demo/tokenizer.json --input examples/meeting.json
```

Use a new output directory for each training run. See [meeting and call summaries](docs/transcripts.md) for annotation formats and training on your own calls.

## Support-routing example

Train a model to route support requests from the repository root:

```sh
cargo run --release --locked -p assort-cli -- train-demo --output .local/support-demo
```

This creates 192 training examples and 32 validation examples, fits a small tokenizer on training text only, and performs 20 epochs of Adam updates. Candidate order is randomized and labels are remapped. It saves the best trained epoch by validation cross entropy. Use a new output directory for each run.

Evaluate the saved model and then run it on new requests:

```sh
cargo run --release --locked -p assort-cli -- evaluate --checkpoint .local/support-demo/checkpoint --tokenizer .local/support-demo/tokenizer.json --input .local/support-demo/validation.jsonl
cargo run --release --locked -p assort-cli -- infer --checkpoint .local/support-demo/checkpoint --tokenizer .local/support-demo/tokenizer.json --input examples/support-requests.json
```

On Windows, this script trains and runs inference, automatically choosing a fresh local output directory:

```powershell
./scripts/train-demo.ps1
```

For a quick wiring check without training:

```sh
cargo run --release --locked -p assort-cli -- demo
cargo run --release --locked -p assort-inference --example typed
```

Both print explicit random-weight notices. See [training](docs/training.md) for data replacement, command options, and what the metrics mean.

## Workspace

| Crate | Responsibility |
| --- | --- |
| `assort-core` | Validated requests, candidate IDs, distributions, error types |
| `assort-tokenizer` | Byte baseline, local Hugging Face JSON loading, example vocabulary fitting |
| `assort-data` | Hard/soft-target JSONL, streaming validation, ragged batch collation |
| `assort-model` | Shared embeddings, bidirectional encoders, cross-attention, scorer, checkpoints |
| `assort-inference` | Batched evaluation and heterogeneous typed question keys |
| `assort-training` | Differentiable objective, Adam loop, candidate shuffling, held-out metrics |
| `assort-transcript` | Timestamped transcript windows, labeled meetings, highlight selection, evaluation |
| `assort-cli` | Training, evaluation, inference, initialization, inspection, validation |

The neural implementation uses [Burn 0.21](https://docs.rs/burn/0.21.0/burn/) and local tokenizer JSON files use [tokenizers 0.23.1](https://docs.rs/tokenizers/0.23.1/tokenizers/). `Cargo.lock` fixes the dependency graph. The model is generic over Burn backends; the runnable CLI uses CPU.

## Typed Rust decisions

Use `QuestionSet` to send unrelated Rust result types through one model evaluation:

```rust
use assort_core::Candidate;
use assort_inference::QuestionSet;

#[derive(Debug, Clone, PartialEq)]
enum Department { Billing, Technical }

// Inside a function returning assort_core::Result, with an Engine named engine:
let mut questions = QuestionSet::new("My card payment failed.");
let urgent = questions.boolean("urgent", "Does this require urgent attention?")?;
let department = questions.choice("department", "Which team should handle this?", [
    (Department::Billing, Candidate::new("billing", "Payments and invoices")),
    (Department::Technical, Candidate::new("technical", "Software failures")),
])?;

let results = engine.evaluate_set(questions)?;
let urgent: bool = results.get(&urgent)?.value;
let department: Department = results.get(&department)?.value;
```

The complete runnable version is [typed.rs](crates/assort-inference/examples/typed.rs). A key from another request is rejected. Every returned value comes from the supplied candidate set. Type safety constrains the output domain, not whether the selected answer is correct.

## Documentation

- [Architecture and tensor contracts](docs/architecture.md)
- [Training and evaluation](docs/training.md)
- [Meeting and call summaries](docs/transcripts.md)
- [Data, tokenizers, masks, and limits](docs/data.md)
- [Checkpoint format and inference](docs/checkpoints.md)
- [Development and verification](docs/development.md)

The checked-in examples are synthetic. Keep datasets, checkpoints, and run output under `.local/`, `data/`, `models/`, or `runs/`, all ignored by Git. Training and inference run locally and upload nothing.
