# Meeting and call summaries

Assort classifies transcript passages and assembles a short summary from exact source quotations. This example supports timestamped JSON and SRT. It does not transcribe audio or generate rewritten prose.

## Run the complete example

From PowerShell:

```powershell
./scripts/train-transcript-demo.ps1
```

The script creates a fresh ignored run directory, trains the model, evaluates the selected checkpoint, and summarizes the checked-in meeting. To run the stages separately:

```sh
cargo run --release --locked -p assort-cli -- train-transcript-demo --output .local/meeting-demo
cargo run --release --locked -p assort-cli -- summarize --checkpoint .local/meeting-demo/checkpoint --tokenizer .local/meeting-demo/tokenizer.json --input examples/meeting.json
cargo run --release --locked -p assort-cli -- summarize --checkpoint .local/meeting-demo/checkpoint --tokenizer .local/meeting-demo/tokenizer.json --input examples/meeting.srt --format json
cargo run --release --locked -p assort-cli -- evaluate-transcripts --checkpoint .local/meeting-demo/checkpoint --tokenizer .local/meeting-demo/tokenizer.json --input .local/meeting-demo/transcripts-test.json
```

Use `--output .local/summary.md` to save Markdown or `--format json --output .local/summary.json` for structured output. Output files must be new and their parent directory must exist. JSON can also come from `--input -`; SRT is selected by the `.srt` filename extension. Input files are bounded to 8 MiB by the CLI.

The original support-routing checkpoint is not a transcript model. Transcript commands require `checkpoint/transcript-pipeline.json`, which pins the prompt/category contract. The tokenizer must also match the checkpoint's exact fingerprint.

## Passage labels

| Label | Intended meaning |
| --- | --- |
| `decision` | A final decision or agreed choice |
| `action` | An assigned action or concrete next step |
| `key_fact` | An important constraint, result, fact, or blocker |
| `background` | Small talk, background, repetition, or unconfirmed proposals |

These labels are mutually exclusive. Split turns with different functions before annotation, and retain enough context to avoid orphaned statements such as “yes, that one.” Action owners and deadlines are quoted in source text; they are not separately extracted. A highlight's speaker identifies who spoke, which can differ from who owns the action.

Every passage receives its own four-candidate question. Passages do not compete in one winner-takes-all choice. Importance is `1 - P(background)`, while the category is the highest-probability class. Selection requires a non-background category and importance above the threshold. Importance is not a calibrated confidence estimate.

## Long transcripts and selection

Preparation packs turns into token-aware windows with at most 8 target passages and 384 state tokens by default. One preceding and one following turn are added as context when they fit. Every source passage is scored exactly once. Questions are limited to 128 tokens and candidate descriptions to 64. The meeting goal is in the shared state; each question identifies its target passage.

One state encoding is shared by the questions in each window. All window scores then enter a global selection step:

1. Filter background and low-importance passages.
2. Prefer high-importance passages, penalizing lexical overlap with selected text.
3. Reject close lexical duplicates and passages exceeding the remaining word budget.
4. Return quotations in source order with unchanged text, IDs, speakers, and timestamps.

Defaults are 120 words, 6 highlights, and minimum importance 0.65. Change them with `--max-words`, `--max-highlights`, and `--min-importance`. Word counts use whitespace-separated source words, excluding Markdown headers. Entire passages are retained or omitted; nothing is clipped to fill a budget. Empty selection returns an empty summary.

Duplicate filtering is lexical, not semantic. It preserves statements that differ in numeric values or common English negations. It does not resolve contradictions, superseded decisions, pronouns, or dependencies across distant windows. Missing context can still make an extracted quotation misleading, so treat output as a reviewable draft with source references.

Use `--limits` for custom `BatchLimits`. Overlong individual turns fail clearly. Split them into shorter timed units upstream, or raise limits and model position capacity together. No timestamps are interpolated and no text is silently truncated. SRT requires numbered cues and `HH:MM:SS,mmm` timestamps; overlapping cues are allowed when sorted by start time. Styling tags and speaker labels embedded in SRT text remain literal text.

## Input format

`summarize` takes one transcript object:

```json
{
  "id": "planning-call-001",
  "title": "Release planning call",
  "goal": "Capture final decisions, assigned actions, and important facts.",
  "segments": [
    {
      "id": "turn-1",
      "start_ms": 12000,
      "end_ms": 18000,
      "speaker": "Alex",
      "text": "We decided to delay the release by one week."
    }
  ]
}
```

The speaker is optional. IDs must be unique and nonblank. Times are nonnegative integer milliseconds, end must not precede start, and turns must be sorted by start. The model receives the goal, but the demo only trains the default meeting goal. Changing `--goal` does not establish that it understands a new summarization task.

## Train on your own meetings

Each split file is a JSON array of labeled transcripts:

```json
[
  {
    "transcript": {
      "id": "planning-call-001",
      "title": "Release planning call",
      "goal": "Capture final decisions, assigned actions, and important facts.",
      "segments": [
        {"id":"turn-1","start_ms":0,"end_ms":5000,"speaker":"Sam","text":"Hello everyone."},
        {"id":"turn-2","start_ms":6000,"end_ms":12000,"speaker":"Alex","text":"We decided to delay the release by one week."}
      ]
    },
    "labels": [
      {"segment_id":"turn-1","kind":"background"},
      {"segment_id":"turn-2","kind":"decision"}
    ]
  }
]
```

Labels match by segment ID and must cover every segment exactly once. Split complete calls, not windows. Related calls, paraphrases, and repeated agendas can leak across splits. The loader rejects identical normalized transcript content and repeated transcript IDs across splits. The underlying trainer also rejects identical generated state windows across train/validation. Common greetings alone are allowed, and semantic overlap is not automatically detected.

```sh
cargo run --release --locked -p assort-cli -- train-transcripts --train data/calls-train.json --validation data/calls-validation.json --test data/calls-test.json --output .local/my-meeting-model
```

The command fits a WordLevel tokenizer on training text only unless you provide `--tokenizer` and `--pad-id`. Use your own BPE/Unigram tokenizer for broader vocabulary. `--config` accepts a model config. The transcript default is hidden size 32, one state layer, one query layer, dropout 0.1, and candidate-to-state attention disabled. Question-to-state attention remains enabled.

Training uses 0.1 label smoothing on the training set only: 0.9 for the true class and 0.1/3 for the others. Validation and test labels remain hard. The existing trainer supplies Adam, shuffling, and best-epoch selection by validation cross entropy. Defaults are 20 epochs, batch size 16, learning rate 0.003, seed 42. Use the existing training flags to change them.

The test split is optional for custom training. When supplied, it is evaluated after model selection and produces a sample summary. If you repeatedly inspect test results while developing the pipeline, reserve another untouched real-call set for final assessment.

## Artifacts and evaluation

In addition to the standard training files, each run saves:

```text
transcripts-train.json
transcripts-validation.json
transcripts-test.json          when supplied
preprocessing-limits.json
checkpoint/transcript-pipeline.json
transcript-test-report.json    when test data was supplied
sample-summary.json
sample-summary.md
```

The synthetic demo has 48 training, 12 validation, and 12 test calls, with 10 passages per call: 2 decisions, 2 actions, 2 key facts, and 4 background passages. Each split uses separate authored sentence templates and topic names. Phrases are reused within each split in different contexts. This is a small English learning example, not a real-world benchmark.

Evaluation reports four-way category accuracy and a confusion matrix, plus micro-averaged highlight precision, recall, and F1 after applying the actual summary policy. The lead baseline takes the first passages that fit the same word and highlight budgets, without scores or duplicate filtering. Perfect highlight selection can coexist with incorrect category labels: calling a decision a fact still preserves the relevant quotation.

The demo's six relevant passages match its default six-highlight budget, and its strong lexical cues make selection relatively easy. Real-call evaluation should include varied numbers of key points, long-distance corrections, tentative versus agreed actions, domain vocabulary, transcription errors, and different speaker styles.

Keep real transcripts and generated artifacts in ignored local directories or outside the repository. These commands upload nothing.
