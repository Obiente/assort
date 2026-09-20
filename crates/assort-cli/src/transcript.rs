use super::training::{self, TrainOptions};
use anyhow::{Context, Result};
use assort_inference::Engine;
use assort_model::{ModelConfig, WeightsStatus, load_checkpoint};
use assort_tokenizer::{TextTokenizer, train_wordlevel};
use assort_transcript::{
    LabeledTranscript, MEETING_GOAL, SummaryOptions, Transcript, evaluate_transcripts,
    meeting_corpus, score_transcript, select_summary, tokenizer_corpus, training_examples,
    transcript_limits,
};
use burn::backend::Flex;
use clap::{Args, ValueEnum};
use std::{collections::HashSet, fs::File, io::Write, path::PathBuf};

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PipelineMetadata {
    version: u32,
    category_ids: [String; 4],
    training_label_smoothing: f32,
}

impl PipelineMetadata {
    fn current() -> Self {
        Self {
            version: 1,
            category_ids: assort_transcript::PassageKind::ALL.map(|k| k.id().into()),
            training_label_smoothing: 0.1,
        }
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.version == 1 && self.category_ids == Self::current().category_ids,
            "unsupported transcript prompt or category contract"
        );
        anyhow::ensure!(
            self.training_label_smoothing.is_finite()
                && (0.0..1.0).contains(&self.training_label_smoothing),
            "invalid transcript training metadata"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    Markdown,
    Json,
}

#[derive(Args)]
pub struct ModelArgs {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    pad_id: u32,
    #[arg(long, default_value_t = 4)]
    batch_size: usize,
    #[arg(long)]
    limits: Option<PathBuf>,
    #[arg(long)]
    allow_random: bool,
}

#[derive(Args)]
pub struct SelectionArgs {
    #[arg(long, default_value_t = 120)]
    max_words: usize,
    #[arg(long, default_value_t = 6)]
    max_highlights: usize,
    #[arg(long, default_value_t = 0.65)]
    min_importance: f32,
}

impl SelectionArgs {
    fn options(self) -> SummaryOptions {
        SummaryOptions {
            max_words: self.max_words,
            max_highlights: self.max_highlights,
            min_importance: self.min_importance,
            ..Default::default()
        }
    }
}

fn engine(args: ModelArgs) -> Result<Engine<Flex, Box<dyn TextTokenizer>>> {
    let pipeline: PipelineMetadata = serde_json::from_slice(&super::read_input(&args.checkpoint.join("transcript-pipeline.json"))
        .context("checkpoint is missing transcript pipeline metadata; use train-transcript-demo or train-transcripts")?)?;
    pipeline.validate()?;
    let tokenizer = super::make_tokenizer(args.tokenizer, args.pad_id)?;
    let (model, manifest) =
        load_checkpoint::<Flex>(&args.checkpoint, &tokenizer.spec(), &Default::default())?;
    anyhow::ensure!(
        manifest.weights_status == WeightsStatus::Trained || args.allow_random,
        "random checkpoint requires --allow-random"
    );
    if manifest.weights_status == WeightsStatus::Random {
        eprintln!("Random weights: highlights have no learned meaning.");
    }
    let limits = if args.limits.is_some() {
        super::load_limits(args.limits)?
    } else {
        transcript_limits(args.batch_size)
    };
    Ok(Engine::new(model, tokenizer, limits)?)
}

pub fn summarize(
    input: PathBuf,
    goal: Option<String>,
    format: OutputFormat,
    output: Option<PathBuf>,
    model: ModelArgs,
    selection: SelectionArgs,
) -> Result<()> {
    let bytes = super::read_input(&input)?;
    let mut transcript: Transcript = if input
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("srt"))
    {
        Transcript::from_srt(
            "srt-input",
            "Call highlights",
            goal.clone().unwrap_or_else(|| MEETING_GOAL.into()),
            std::str::from_utf8(&bytes)?,
        )?
    } else {
        serde_json::from_slice(&bytes)
            .context("expected one Transcript JSON object, or use an .srt file")?
    };
    if let Some(goal) = goal {
        transcript.goal = goal;
    }
    let engine = engine(model)?;
    let scores = score_transcript(&engine, &transcript)?;
    let summary = select_summary(&transcript, &scores, &selection.options())?;
    let rendered = match format {
        OutputFormat::Markdown => summary.markdown(),
        OutputFormat::Json => serde_json::to_string_pretty(&summary)?,
    };
    if let Some(path) = output {
        let mut file = File::create_new(path)
            .context("summary output must be a new file in an existing directory")?;
        file.write_all(rendered.as_bytes())?;
    } else {
        println!("{rendered}");
    }
    Ok(())
}

pub fn evaluate(input: PathBuf, model: ModelArgs, selection: SelectionArgs) -> Result<()> {
    let corpus = read_corpus(&input)?;
    let engine = engine(model)?;
    let scores = corpus
        .iter()
        .map(|item| score_transcript(&engine, &item.transcript))
        .collect::<assort_core::Result<Vec<_>>>()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&evaluate_transcripts(
            &corpus,
            &scores,
            &selection.options()
        )?)?
    );
    Ok(())
}

pub fn demo(options: TrainOptions) -> Result<()> {
    let (train, validation, test) = meeting_corpus(options.seed);
    eprintln!(
        "Synthetic meetings/calls: {} training, {} validation, {} held-out test transcripts. Exact-span summaries; no generated prose.",
        train.len(),
        validation.len(),
        test.len()
    );
    fit(train, validation, Some(test), None, 0, None, options)
}

#[allow(clippy::too_many_arguments)]
pub fn custom(
    train: PathBuf,
    validation: PathBuf,
    test: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    pad_id: u32,
    config: Option<PathBuf>,
    options: TrainOptions,
) -> Result<()> {
    fit(
        read_corpus(&train)?,
        read_corpus(&validation)?,
        test.map(|p| read_corpus(&p)).transpose()?,
        tokenizer,
        pad_id,
        config,
        options,
    )
}

#[allow(clippy::too_many_arguments)]
fn fit(
    train: Vec<LabeledTranscript>,
    validation: Vec<LabeledTranscript>,
    test: Option<Vec<LabeledTranscript>>,
    tokenizer_path: Option<PathBuf>,
    pad_id: u32,
    config: Option<PathBuf>,
    mut options: TrainOptions,
) -> Result<()> {
    validate_splits(&train, &validation, test.as_deref().unwrap_or_default())?;
    training::create_run(&options.output)?;
    let output = options.output.clone();
    training::write_json(&output.join("transcripts-train.json"), &train)?;
    training::write_json(&output.join("transcripts-validation.json"), &validation)?;
    if let Some(test) = &test {
        training::write_json(&output.join("transcripts-test.json"), test)?;
    }
    let tokenizer: Box<dyn TextTokenizer> = if let Some(path) = tokenizer_path {
        std::fs::copy(path, output.join("tokenizer.json"))?;
        super::make_tokenizer(Some(output.join("tokenizer.json")), pad_id)?
    } else {
        anyhow::ensure!(pad_id == 0, "the fitted example tokenizer uses pad ID 0");
        Box::new(train_wordlevel(
            &tokenizer_corpus(&train),
            output.join("tokenizer.json"),
        )?)
    };
    let limits = if options.limits.is_some() {
        super::load_limits(options.limits.clone())?
    } else {
        transcript_limits(options.batch_size)
    };
    training::write_json(&output.join("preprocessing-limits.json"), &limits)?;
    options.limits = Some(output.join("preprocessing-limits.json"));
    let prepare = |corpus: &[LabeledTranscript]| -> Result<Vec<assort_data::Example>> {
        let mut examples = Vec::new();
        for item in corpus {
            examples.extend(training_examples(item, &tokenizer, &limits)?);
        }
        Ok(examples)
    };
    let mut training_data = prepare(&train)?;
    // Mild label smoothing reduces overconfidence on a small corpus. Validation/test
    // labels remain hard; the tokenizer and selection policy never see test labels.
    for example in &mut training_data {
        for target in &mut example.targets {
            if let assort_data::Target::Hard(index) = target {
                let mut probabilities = vec![0.1 / 3.0; 4];
                probabilities[*index] = 0.9;
                *target = assort_data::Target::Soft(probabilities);
            }
        }
    }
    let validation_data = prepare(&validation)?;
    training::write_examples(&output.join("train.jsonl"), &training_data)?;
    training::write_examples(&output.join("validation.jsonl"), &validation_data)?;
    let config = if let Some(path) = config {
        serde_json::from_slice(&super::read_input(&path)?)?
    } else {
        ModelConfig {
            state_layers: 1,
            dropout: 0.1,
            candidate_state_attention: false,
            ..ModelConfig::tiny(tokenizer.spec().vocab_size)
        }
    };
    training::run(options, training_data, validation_data, tokenizer, config)?;
    training::write_json(
        &output.join("checkpoint").join("transcript-pipeline.json"),
        &PipelineMetadata::current(),
    )?;
    if let Some(test) = test {
        let engine = engine(ModelArgs {
            checkpoint: output.join("checkpoint"),
            tokenizer: Some(output.join("tokenizer.json")),
            pad_id,
            batch_size: limits.max_batch_size,
            limits: Some(output.join("preprocessing-limits.json")),
            allow_random: false,
        })?;
        let predictions = test
            .iter()
            .map(|item| score_transcript(&engine, &item.transcript))
            .collect::<assort_core::Result<Vec<_>>>()?;
        let report = evaluate_transcripts(&test, &predictions, &SummaryOptions::default())?;
        training::write_json(&output.join("transcript-test-report.json"), &report)?;
        let sample = select_summary(
            &test[0].transcript,
            &predictions[0],
            &SummaryOptions::default(),
        )?;
        training::write_json(&output.join("sample-summary.json"), &sample)?;
        std::fs::write(output.join("sample-summary.md"), sample.markdown())?;
        eprintln!(
            "Test span accuracy {:.1}%; selected-highlight precision {:.1}%, recall {:.1}%, F1 {:.3}; lead baseline F1 {:.3}.",
            report.category_accuracy * 100.0,
            report.selected_highlights.precision * 100.0,
            report.selected_highlights.recall * 100.0,
            report.selected_highlights.f1,
            report.lead_baseline.f1
        );
    }
    Ok(())
}

fn read_corpus(path: &PathBuf) -> Result<Vec<LabeledTranscript>> {
    let corpus: Vec<LabeledTranscript> = serde_json::from_slice(&super::read_input(path)?)
        .context("expected a JSON array of labeled transcripts")?;
    anyhow::ensure!(!corpus.is_empty(), "transcript corpus is empty");
    for item in &corpus {
        item.validate()?;
    }
    Ok(corpus)
}

fn validate_splits(
    train: &[LabeledTranscript],
    validation: &[LabeledTranscript],
    test: &[LabeledTranscript],
) -> Result<()> {
    anyhow::ensure!(
        !train.is_empty() && !validation.is_empty(),
        "train and validation transcripts are required"
    );
    let mut previous_ids = HashSet::new();
    let mut previous_texts = HashSet::new();
    for split in [train, validation, test] {
        let mut ids = HashSet::new();
        let mut texts = HashSet::new();
        for item in split {
            item.validate()?;
            anyhow::ensure!(
                !previous_ids.contains(&item.transcript.id)
                    && ids.insert(item.transcript.id.clone()),
                "transcript IDs must be unique and disjoint across splits"
            );
            // Common greetings across unrelated calls are not leakage by themselves.
            let text = item
                .transcript
                .segments
                .iter()
                .map(|segment| {
                    segment
                        .text
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ")
                        .to_lowercase()
                })
                .collect::<Vec<_>>()
                .join("\n");
            anyhow::ensure!(
                !previous_texts.contains(&text),
                "identical normalized transcript found in different splits; group related calls before splitting"
            );
            texts.insert(text);
        }
        previous_ids.extend(ids);
        previous_texts.extend(texts);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_greetings_are_allowed_but_duplicate_calls_are_rejected() {
        let (mut train, mut validation, _) = meeting_corpus(42);
        train[0].transcript.segments[0].text = "Hello everyone".into();
        validation[0].transcript.segments[0].text = "Hello everyone".into();
        validate_splits(&train, &validation, &[]).unwrap();
        let mut duplicate = train[0].clone();
        duplicate.transcript.id = "different-id".into();
        assert!(validate_splits(&train, &[duplicate], &[]).is_err());
    }

    #[test]
    fn incompatible_transcript_pipeline_is_rejected() {
        let mut metadata = PipelineMetadata::current();
        metadata.validate().unwrap();
        metadata.category_ids.swap(0, 1);
        assert!(metadata.validate().is_err());
        metadata = PipelineMetadata::current();
        metadata.version = 2;
        assert!(metadata.validate().is_err());
    }
}
