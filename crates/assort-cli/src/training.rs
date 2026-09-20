use anyhow::{Context, Result};
use assort_data::{BatchLimits, Example, JsonlReader};
use assort_model::{ModelConfig, WeightsStatus, save_checkpoint};
use assort_tokenizer::{TextTokenizer, train_wordlevel};
use assort_training::{TrainingConfig, support_routing_data, train};
use burn::backend::{Autodiff, Flex};
use clap::Args;
use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

#[derive(Args)]
pub struct TrainOptions {
    /// New run directory. Existing runs are never overwritten.
    #[arg(long)]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = 20)]
    epochs: usize,
    #[arg(long, default_value_t = 16)]
    pub(crate) batch_size: usize,
    #[arg(long, default_value_t = 0.003)]
    learning_rate: f64,
    #[arg(long, default_value_t = 42)]
    pub(crate) seed: u64,
    /// Optional JSON BatchLimits; defaults are expanded to accommodate batch-size.
    #[arg(long)]
    pub(crate) limits: Option<PathBuf>,
}

pub fn demo(options: TrainOptions) -> Result<()> {
    let (training, validation) = support_routing_data(options.seed);
    create_run(&options.output)?;
    write_examples(&options.output.join("train.jsonl"), &training)?;
    write_examples(&options.output.join("validation.jsonl"), &validation)?;
    let corpus: Vec<String> = training
        .iter()
        .flat_map(|e| {
            std::iter::once(e.request.state.clone()).chain(e.request.questions.iter().flat_map(
                |q| {
                    std::iter::once(q.text.clone())
                        .chain(q.candidates.iter().map(|c| c.description.clone()))
                },
            ))
        })
        .collect();
    let tokenizer = train_wordlevel(&corpus, options.output.join("tokenizer.json"))?;
    let mut model_config = ModelConfig::tiny(tokenizer.spec().vocab_size);
    model_config.state_layers = 1;
    model_config.max_positions = 512;
    eprintln!(
        "Training from random weights on synthetic support routing: {} train / {} validation examples. This is a narrow learning baseline, not a general semantic model.",
        training.len(),
        validation.len()
    );
    run(options, training, validation, tokenizer, model_config)
}

pub fn custom(
    training: PathBuf,
    validation: PathBuf,
    tokenizer: Option<PathBuf>,
    pad_id: u32,
    config: Option<PathBuf>,
    options: TrainOptions,
) -> Result<()> {
    let training_data = load_examples(&training)?;
    let validation_data = load_examples(&validation)?;
    let tokenizer_impl = super::make_tokenizer(tokenizer.clone(), pad_id)?;
    let config = if let Some(path) = config {
        serde_json::from_slice(&super::read_input(&path)?)?
    } else {
        ModelConfig::tiny(tokenizer_impl.spec().vocab_size)
    };
    create_run(&options.output)?;
    if let Some(path) = tokenizer {
        std::fs::copy(path, options.output.join("tokenizer.json"))?;
    }
    // Keep explicit split snapshots with the local run so it remains reproducible.
    write_examples(&options.output.join("train.jsonl"), &training_data)?;
    write_examples(&options.output.join("validation.jsonl"), &validation_data)?;
    run(
        options,
        training_data,
        validation_data,
        tokenizer_impl,
        config,
    )
}

pub(crate) fn run(
    options: TrainOptions,
    training_data: Vec<Example>,
    validation_data: Vec<Example>,
    tokenizer: impl TextTokenizer,
    model_config: ModelConfig,
) -> Result<()> {
    let limits = if options.limits.is_some() {
        super::load_limits(options.limits.clone())?
    } else {
        BatchLimits {
            max_batch_size: options.batch_size,
            ..Default::default()
        }
    };
    let config = TrainingConfig {
        epochs: options.epochs,
        batch_size: options.batch_size,
        learning_rate: options.learning_rate,
        seed: options.seed,
        ..Default::default()
    };
    config.validate(&limits)?;
    write_json(&options.output.join("training-config.json"), &config)?;
    write_json(&options.output.join("model-config.json"), &model_config)?;
    write_json(&options.output.join("batch-limits.json"), &limits)?;
    let mut metrics = BufWriter::new(File::create_new(options.output.join("metrics.jsonl"))?);
    let started = std::time::Instant::now();
    let result = train::<Autodiff<Flex>>(
        model_config,
        &tokenizer,
        &training_data,
        &validation_data,
        config,
        &limits,
        &Default::default(),
        |epoch| {
            eprintln!(
                "Epoch {:>3}: train CE {:.4} | val CE {:.4} | accuracy {:>5.1}% | Brier {:.4}",
                epoch.epoch,
                epoch.train_cross_entropy,
                epoch.validation.cross_entropy,
                epoch.validation.accuracy * 100.0,
                epoch.validation.brier
            );
            serde_json::to_writer(&mut metrics, epoch)
                .map_err(|e| assort_core::Error::Invalid(e.to_string()))?;
            writeln!(metrics)?;
            metrics.flush()?;
            Ok(())
        },
    )?;
    save_checkpoint(
        &result.model,
        &tokenizer.spec(),
        WeightsStatus::Trained,
        options.output.join("checkpoint"),
    )?;
    write_json(&options.output.join("report.json"), &result.report)?;
    let best = &result.report.best_validation;
    eprintln!(
        "Saved best epoch {} after {} optimizer steps ({:.1}s). Validation CE {:.4} -> {:.4}; accuracy {:.1}% -> {:.1}%.",
        result.report.best_epoch,
        result.report.optimizer_steps,
        started.elapsed().as_secs_f64(),
        result.report.initial_validation.cross_entropy,
        best.cross_entropy,
        result.report.initial_validation.accuracy * 100.0,
        best.accuracy * 100.0
    );
    println!("{}", serde_json::to_string_pretty(&result.report)?);
    Ok(())
}

pub fn load_examples(path: &Path) -> Result<Vec<Example>> {
    let mut examples = Vec::new();
    for example in JsonlReader::new(BufReader::new(File::open(path)?)) {
        anyhow::ensure!(
            examples.len() < 100_000,
            "example trainer supports at most 100,000 examples per split; adapt the loader for larger datasets"
        );
        examples.push(example.with_context(|| format!("invalid dataset {}", path.display()))?);
    }
    anyhow::ensure!(!examples.is_empty(), "dataset is empty");
    Ok(examples)
}

pub(crate) fn create_run(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::create_dir(path).context("run directory must be new; choose a different --output")?;
    Ok(())
}

pub(crate) fn write_examples(path: &Path, examples: &[Example]) -> Result<()> {
    let mut file = BufWriter::new(File::create_new(path)?);
    for example in examples {
        serde_json::to_writer(&mut file, example)?;
        writeln!(file)?;
    }
    file.flush()?;
    Ok(())
}

pub(crate) fn write_json(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    let mut file = BufWriter::new(File::create_new(path)?);
    serde_json::to_writer_pretty(&mut file, value)?;
    file.flush()?;
    Ok(())
}
