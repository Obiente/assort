# Data and tokenization

## Inference request

The CLI accepts a JSON **array** of requests. Each request has a nonblank `state` and one or more questions:

```json
[
  {
    "state": "My card was charged twice.",
    "questions": [
      {
        "id": "department",
        "text": "Which team should handle this request?",
        "candidates": [
          {"id": "billing", "description": "Payments invoices charges refunds"},
          {"id": "technical", "description": "Software errors crashes bugs"}
        ]
      }
    ]
  }
]
```

Question IDs must be unique within each request. Candidate IDs must be unique within their question. IDs, text, and descriptions cannot be blank. Each question has 1 through 255 candidates. A singleton always has probability 1, regardless of evidence. The API supports larger question counts subject to caller limits; it does not silently split oversized requests into separate forwards.

IDs are mapping metadata and are not tokenized. Descriptions must contain the meaning the model needs. Scores use ordered candidate descriptions; an ordinal score has no continuous regression assumption. Boolean helpers use explicit true/false descriptions.

## Supervised JSONL

Each nonblank line contains `request` plus `targets`, with one target per question in question order. Candidate positions are zero-based.

Hard target:

```json
{"kind":"hard","value":0}
```

Soft target:

```json
{"kind":"soft","value":[0.85,0.15]}
```

Soft vectors must match candidate count, contain finite values in `[0,1]`, and sum to one within `1e-5`. Nothing is silently renormalized. Unknown schema fields are rejected to expose spelling mistakes. See [the checked-in dataset](../examples/dataset.jsonl) for complete records.

```sh
cargo run --locked -p assort-cli -- validate-dataset examples/dataset.jsonl
```

Pass `--tokenizer` and `--pad-id` to validate against a real tokenizer, and `--limits` for custom limits. This command checks schema and per-example collation, not whether a future mixed batch fits your GPU memory. JSONL is UTF-8, one JSON value per line, with an 8 MiB line limit. Errors include line numbers and terminate reading. Blank lines are skipped. Write JSONL without a BOM.

## Tokenizers

`ByteTokenizer` emits BOS `1`, every UTF-8 byte plus `3`, and EOS `2`; PAD is `0`, so vocabulary size is 259. Empty raw text still has BOS/EOS, but request validation rejects blank state/question/description fields.

`HfTokenizer::from_file(path, pad_id)` loads a local tokenizer JSON without network access. It includes added tokens when determining embedding capacity. It disables the JSON's padding and truncation policies so the collator owns them, while retaining normalization, tokenization, and special-token postprocessing. The specified pad ID must exist in the vocabulary. The tokenizer's exact JSON bytes, pad ID, kind, and vocabulary capacity are checked against checkpoint metadata.

`train_wordlevel` fits and saves the demo vocabulary; it does not download a pretrained tokenizer. For broader work, produce your own tokenizer JSON using Hugging Face's Rust `tokenizers` tooling and keep the vocabulary fixed for the entire checkpoint family.

## Collation contract

`TokenBatch` stores immutable row-major host buffers with private fields and read-only accessors:

| Buffer | Logical shape |
| --- | --- |
| State IDs and state padding | `[B,S]` |
| Question IDs and text padding | `[B,Q,Tq]`, flattened to `[B*Q,Tq]` at the model boundary |
| Candidate IDs and text padding | `[B,Q,C,Tc]`, flattened to `[B*Q*C,Tc]` |
| Question mask | `[B,Q]` |
| Candidate mask | `[B,Q,C]` |

Every mask uses `true = padding/invalid`. Actual token positions are identified by sequence length, not by comparing token IDs with the pad ID. Padded text rows receive one dummy valid attention key; their output masks remain fully invalid. Loss and inference must use the output masks.

Token sequences are never silently truncated. Overlong text, invalid IDs, excessive batch/question/candidate counts, and padded token budget overruns return errors. `BatchLimits` defaults are in [configs/batch-limits.json](../configs/batch-limits.json). Attention budgets are a separate model-level constraint described in [architecture](architecture.md).
