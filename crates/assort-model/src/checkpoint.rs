use crate::{DecisionModel, ModelConfig};
use assort_core::{Error, Result, ensure};
use assort_tokenizer::TokenizerSpec;
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::backend::Backend,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};

pub const CHECKPOINT_VERSION: u32 = 1;
const ARCHITECTURE: &str = "assort-candidate-scoring-v1";

/// Declared by the checkpoint producer; not a certification of model quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightsStatus {
    Random,
    Trained,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub format_version: u32,
    pub architecture: String,
    pub burn_version: String,
    pub config: ModelConfig,
    pub tokenizer: TokenizerSpec,
    pub weights_status: WeightsStatus,
    pub weights_sha256: String,
}

impl Manifest {
    pub fn read(directory: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(directory.as_ref().join("manifest.json"))?;
        let manifest: Self =
            serde_json::from_slice(&bytes).map_err(|e| Error::Checkpoint(e.to_string()))?;
        ensure(
            manifest.format_version == CHECKPOINT_VERSION
                && manifest.architecture == ARCHITECTURE
                && manifest.burn_version == "0.21.0",
            "unsupported checkpoint format, architecture, or Burn version",
        )?;
        manifest.config.validate()?;
        ensure(
            manifest.config.vocab_size == manifest.tokenizer.vocab_size
                && (manifest.tokenizer.pad_id as usize) < manifest.tokenizer.vocab_size,
            "checkpoint tokenizer/model mismatch",
        )?;
        Ok(manifest)
    }
}

/// Writes a new checkpoint directory via a sibling staging directory. Never overwrites.
/// Save the tokenizer JSON alongside your experiment separately; its content hash is pinned here.
pub fn save_checkpoint<B: Backend>(
    model: &DecisionModel<B>,
    tokenizer: &TokenizerSpec,
    status: WeightsStatus,
    directory: impl AsRef<Path>,
) -> Result<Manifest> {
    let directory = directory.as_ref();
    ensure(!directory.exists(), "checkpoint destination already exists")?;
    ensure(
        tokenizer.vocab_size == model.config().vocab_size
            && (tokenizer.pad_id as usize) < tokenizer.vocab_size,
        "tokenizer/model mismatch",
    )?;
    let parent = directory
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let staging = tempfile::Builder::new()
        .prefix(".assort-checkpoint-")
        .tempdir_in(parent)?;
    model
        .clone()
        .save_file(
            staging.path().join("weights"),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
        )
        .map_err(|e| Error::Checkpoint(e.to_string()))?;
    let manifest = Manifest {
        format_version: CHECKPOINT_VERSION,
        architecture: ARCHITECTURE.into(),
        burn_version: "0.21.0".into(),
        config: model.config().clone(),
        tokenizer: tokenizer.clone(),
        weights_status: status,
        weights_sha256: file_hash(&staging.path().join("weights.mpk"))?,
    };
    let bytes =
        serde_json::to_vec_pretty(&manifest).map_err(|e| Error::Checkpoint(e.to_string()))?;
    std::fs::write(staging.path().join("manifest.json"), bytes)?;
    std::fs::rename(staging.path(), directory)?;
    Ok(manifest)
}

pub fn load_checkpoint<B: Backend>(
    directory: impl AsRef<Path>,
    tokenizer: &TokenizerSpec,
    device: &B::Device,
) -> Result<(DecisionModel<B>, Manifest)> {
    let directory = directory.as_ref();
    let manifest = Manifest::read(directory)?;
    ensure(
        &manifest.tokenizer == tokenizer,
        "checkpoint requires a different tokenizer (including exact JSON hash and pad ID)",
    )?;
    ensure(
        file_hash(&directory.join("weights.mpk"))? == manifest.weights_sha256,
        "checkpoint weight checksum mismatch",
    )?;
    let model = DecisionModel::uninitialized(manifest.config.clone(), device)?
        .load_file(
            directory.join("weights"),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
            device,
        )
        .map_err(|e| Error::Checkpoint(e.to_string()))?;
    Ok((model, manifest))
}

fn file_hash(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65_536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
