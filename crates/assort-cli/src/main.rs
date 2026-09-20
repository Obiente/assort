use anyhow::{Context, Result, bail};
use assort_core::{Candidate, Question, Request};
use assort_data::{BatchLimits, JsonlReader, collate};
use assort_inference::Engine;
use assort_model::{
    DecisionModel, Manifest, ModelConfig, WeightsStatus, load_checkpoint, save_checkpoint,
};
use assort_tokenizer::{ByteTokenizer, HfTokenizer, TextTokenizer};
use burn::{backend::Flex, module::Module, tensor::backend::Backend};
use clap::{Parser, Subcommand, ValueEnum};
use std::{
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
};

mod training;
mod transcript;

#[derive(Parser)]
#[command(
    name = "assort",
    version,
    about = "Local candidate-scoring model workbench"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train a meeting-highlights model on synthetic train/validation/test transcripts.
    TrainTranscriptDemo {
        #[command(flatten)]
        options: training::TrainOptions,
    },
    /// Train on your annotated meetings/calls, optionally with an untouched test set.
    TrainTranscripts {
        #[arg(long)]
        train: PathBuf,
        #[arg(long)]
        validation: PathBuf,
        #[arg(long)]
        test: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        pad_id: u32,
        #[arg(long)]
        config: Option<PathBuf>,
        #[command(flatten)]
        options: training::TrainOptions,
    },
    /// Produce a timestamped extractive summary from transcript JSON or an SRT file.
    Summarize {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        goal: Option<String>,
        #[arg(long, value_enum, default_value_t = transcript::OutputFormat::Markdown)]
        format: transcript::OutputFormat,
        #[arg(long)]
        output: Option<PathBuf>,
        #[command(flatten)]
        model: transcript::ModelArgs,
        #[command(flatten)]
        selection: transcript::SelectionArgs,
    },
    /// Evaluate span categories and selected highlights on annotated transcripts.
    EvaluateTranscripts {
        #[arg(long)]
        input: PathBuf,
        #[command(flatten)]
        model: transcript::ModelArgs,
        #[command(flatten)]
        selection: transcript::SelectionArgs,
    },
    /// Train a useful, narrow support-routing example and save a complete local run.
    TrainDemo {
        #[command(flatten)]
        options: training::TrainOptions,
    },
    /// Train on your own JSONL data, with a separate validation split.
    Train {
        #[arg(long)]
        train: PathBuf,
        #[arg(long)]
        validation: PathBuf,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        pad_id: u32,
        #[arg(long)]
        config: Option<PathBuf>,
        #[command(flatten)]
        options: training::TrainOptions,
    },
    /// Evaluate a saved model against labeled JSONL examples.
    Evaluate {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        pad_id: u32,
        #[arg(long)]
        limits: Option<PathBuf>,
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
    },
    /// Run a tiny random model on a synthetic request. This does not train anything.
    Demo {
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Print an editable model configuration as JSON.
    Config {
        #[arg(long, value_enum, default_value_t = Preset::Tiny)]
        preset: Preset,
        #[arg(long, default_value_t = 259)]
        vocab_size: usize,
    },
    /// Save newly initialized random weights to a NEW checkpoint directory.
    Init {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        pad_id: u32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Print checkpoint metadata without allocating a model or verifying the weight file.
    Inspect { checkpoint: PathBuf },
    /// Evaluate a JSON array of requests, from a file or '-' for stdin.
    Infer {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        pad_id: u32,
        #[arg(long)]
        limits: Option<PathBuf>,
        #[arg(long, default_value_t = 1.0)]
        temperature: f32,
        #[arg(long)]
        allow_random: bool,
    },
    /// Stream and validate JSONL examples, including token lengths for each example.
    ValidateDataset {
        input: PathBuf,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        pad_id: u32,
        #[arg(long)]
        limits: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Preset {
    Tiny,
    Small,
}

fn main() -> Result<()> {
    let device = Default::default();
    match Cli::parse().command {
        Command::TrainTranscriptDemo { options } => transcript::demo(options)?,
        Command::TrainTranscripts {
            train,
            validation,
            test,
            tokenizer,
            pad_id,
            config,
            options,
        } => transcript::custom(train, validation, test, tokenizer, pad_id, config, options)?,
        Command::Summarize {
            input,
            goal,
            format,
            output,
            model,
            selection,
        } => transcript::summarize(input, goal, format, output, model, selection)?,
        Command::EvaluateTranscripts {
            input,
            model,
            selection,
        } => transcript::evaluate(input, model, selection)?,
        Command::TrainDemo { options } => training::demo(options)?,
        Command::Train {
            train,
            validation,
            tokenizer,
            pad_id,
            config,
            options,
        } => training::custom(train, validation, tokenizer, pad_id, config, options)?,
        Command::Evaluate {
            checkpoint,
            input,
            tokenizer,
            pad_id,
            limits,
            batch_size,
        } => {
            let tokenizer = make_tokenizer(tokenizer, pad_id)?;
            let (model, manifest) =
                load_checkpoint::<Flex>(checkpoint, &tokenizer.spec(), &device)?;
            if manifest.weights_status == WeightsStatus::Random {
                eprintln!("Evaluating a randomly initialized model.");
            }
            let data = training::load_examples(&input)?;
            let metrics = assort_training::evaluate(
                &model,
                &tokenizer,
                &data,
                batch_size,
                &load_limits(limits)?,
            )?;
            println!("{}", serde_json::to_string_pretty(&metrics)?);
        }
        Command::Demo { seed } => {
            Flex::seed(&device, seed);
            let model = DecisionModel::<Flex>::new(ModelConfig::tiny(259), &device)?;
            eprintln!(
                "Random weights: outputs demonstrate wiring only and have no learned meaning. Parameters: {}",
                model.num_params()
            );
            let engine = Engine::new(model, ByteTokenizer, BatchLimits::default())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&engine.evaluate(&[synthetic_request()])?)?
            );
        }
        Command::Config { preset, vocab_size } => {
            let config = match preset {
                Preset::Tiny => ModelConfig::tiny(vocab_size),
                Preset::Small => ModelConfig::small(vocab_size),
            };
            config.validate()?;
            println!("{}", serde_json::to_string_pretty(&config)?);
        }
        Command::Init {
            output,
            config,
            tokenizer,
            pad_id,
            seed,
        } => {
            let tokenizer = make_tokenizer(tokenizer, pad_id)?;
            let config = if let Some(path) = config {
                serde_json::from_slice(&read_input(&path)?)?
            } else {
                ModelConfig::tiny(tokenizer.spec().vocab_size)
            };
            anyhow::ensure!(
                config.vocab_size == tokenizer.spec().vocab_size,
                "configuration vocabulary differs from tokenizer"
            );
            Flex::seed(&device, seed);
            let model = DecisionModel::<Flex>::new(config, &device)?;
            let manifest =
                save_checkpoint(&model, &tokenizer.spec(), WeightsStatus::Random, output)?;
            eprintln!(
                "Saved {} randomly initialized parameters. No training was performed.",
                model.num_params()
            );
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        Command::Inspect { checkpoint } => println!(
            "{}",
            serde_json::to_string_pretty(&Manifest::read(checkpoint)?)?
        ),
        Command::Infer {
            checkpoint,
            input,
            tokenizer,
            pad_id,
            limits,
            temperature,
            allow_random,
        } => {
            let manifest = Manifest::read(&checkpoint)?;
            if manifest.weights_status == WeightsStatus::Random && !allow_random {
                bail!("checkpoint contains random weights; pass --allow-random to test wiring");
            }
            let tokenizer = make_tokenizer(tokenizer, pad_id)?;
            let (model, _) = load_checkpoint::<Flex>(&checkpoint, &tokenizer.spec(), &device)?;
            let limits = load_limits(limits)?;
            let requests: Vec<Request> = serde_json::from_slice(&read_input(&input)?)
                .context("expected a JSON array of requests")?;
            let engine = Engine::new(model, tokenizer, limits)?.with_temperature(temperature)?;
            if manifest.weights_status == WeightsStatus::Random {
                eprintln!("Random weights: outputs have no learned meaning.");
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&engine.evaluate(&requests)?)?
            );
        }
        Command::ValidateDataset {
            input,
            tokenizer,
            pad_id,
            limits,
        } => {
            let tokenizer = make_tokenizer(tokenizer, pad_id)?;
            let limits = load_limits(limits)?;
            let mut count = 0;
            let reader: Box<dyn Read> = if input.as_os_str() == "-" {
                Box::new(std::io::stdin())
            } else {
                Box::new(File::open(input)?)
            };
            for (index, example) in JsonlReader::new(BufReader::new(reader)).enumerate() {
                let example = example?;
                collate(&tokenizer, &[example.request], &limits).with_context(|| {
                    format!("token validation failed for example {}", index + 1)
                })?;
                count += 1;
            }
            anyhow::ensure!(count > 0, "dataset contains no examples");
            println!("Validated {count} examples. Training was not run.");
        }
    }
    Ok(())
}

fn make_tokenizer(path: Option<PathBuf>, pad_id: u32) -> Result<Box<dyn TextTokenizer>> {
    if let Some(path) = path {
        Ok(Box::new(HfTokenizer::from_file(path, pad_id)?))
    } else {
        anyhow::ensure!(pad_id == 0, "byte tokenizer requires pad ID 0");
        Ok(Box::new(ByteTokenizer))
    }
}

fn load_limits(path: Option<PathBuf>) -> Result<BatchLimits> {
    let limits: BatchLimits = if let Some(path) = path {
        serde_json::from_slice(&read_input(&path)?)?
    } else {
        BatchLimits::default()
    };
    limits.validate()?;
    Ok(limits)
}

fn read_input(path: &PathBuf) -> Result<Vec<u8>> {
    let reader: Box<dyn Read> = if path.as_os_str() == "-" {
        Box::new(std::io::stdin())
    } else {
        Box::new(File::open(path)?)
    };
    let mut bytes = Vec::new();
    reader.take(8 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    anyhow::ensure!(bytes.len() <= 8 * 1024 * 1024, "input exceeds 8 MiB limit");
    // Accept a UTF-8 BOM written by Windows text editors/shells.
    if bytes.starts_with(&[0xef, 0xbb, 0xbf]) {
        bytes.drain(..3);
    }
    Ok(bytes)
}

fn synthetic_request() -> Request {
    Request {
        state: "A customer reports a failed card payment and asks for help.".into(),
        questions: vec![
            Question::boolean("urgent", "Does this require urgent attention?"),
            Question {
                id: "department".into(),
                text: "Which department should handle this?".into(),
                candidates: vec![
                    Candidate::new("billing", "Payments and invoices"),
                    Candidate::new("technical", "Software failures"),
                    Candidate::new("sales", "Purchasing and upgrades"),
                ],
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_arguments_have_no_conflicts() {
        Cli::command().debug_assert();
    }
}
