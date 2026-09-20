//! Local-only tokenizers. ByteTokenizer is a wiring baseline, not a pretrained vocabulary.

use assort_core::{Error, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerSpec {
    pub kind: String,
    pub fingerprint: String,
    /// Embedding capacity required (max token ID + 1), including added tokens.
    pub vocab_size: usize,
    pub pad_id: u32,
}

pub trait TextTokenizer: Send + Sync {
    fn spec(&self) -> TokenizerSpec;
    /// Return an unpadded, nonempty sequence. No silent truncation is allowed.
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
}

impl<T: TextTokenizer + ?Sized> TextTokenizer for Box<T> {
    fn spec(&self) -> TokenizerSpec {
        (**self).spec()
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        (**self).encode(text)
    }
}

#[derive(Debug, Clone, Default)]
pub struct ByteTokenizer;

impl TextTokenizer for ByteTokenizer {
    fn spec(&self) -> TokenizerSpec {
        TokenizerSpec {
            kind: "byte-v1".into(),
            fingerprint: "byte-v1:pad=0,bos=1,eos=2,byte-offset=3".into(),
            vocab_size: 259,
            pad_id: 0,
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(std::iter::once(1)
            .chain(text.bytes().map(|b| b as u32 + 3))
            .chain(std::iter::once(2))
            .collect())
    }
}

pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
    spec: TokenizerSpec,
}

/// Fit a small word-level baseline using training text only and save a local tokenizer JSON.
/// This is for the runnable example; use BPE/Unigram for a broader vocabulary.
pub fn train_wordlevel(texts: &[String], path: impl AsRef<Path>) -> Result<HfTokenizer> {
    use tokenizers::{
        AddedToken,
        models::{
            TrainerWrapper,
            wordlevel::{WordLevel, WordLevelTrainer},
        },
        normalizers::Lowercase,
        pre_tokenizers::whitespace::Whitespace,
    };
    ensure(!texts.is_empty(), "tokenizer training corpus is empty")?;
    ensure(
        !path.as_ref().exists(),
        "tokenizer destination already exists",
    )?;
    let mut tokenizer = tokenizers::Tokenizer::new(
        WordLevel::builder()
            .unk_token("[UNK]".into())
            .build()
            .map_err(|e| Error::Tokenizer(e.to_string()))?,
    );
    tokenizer
        .with_normalizer(Some(Lowercase))
        .map_err(|e| Error::Tokenizer(e.to_string()))?;
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    let trainer = WordLevelTrainer::builder()
        .show_progress(false)
        .vocab_size(32_000)
        .special_tokens(vec![
            AddedToken::from("[PAD]", true),
            AddedToken::from("[UNK]", true),
        ])
        .build()
        .map_err(|e| Error::Tokenizer(e.to_string()))?;
    let mut trainer = TrainerWrapper::from(trainer);
    tokenizer
        .train(&mut trainer, texts.iter())
        .map_err(|e| Error::Tokenizer(e.to_string()))?;
    tokenizer
        .save(path.as_ref(), true)
        .map_err(|e| Error::Tokenizer(e.to_string()))?;
    HfTokenizer::from_file(path, 0)
}

impl HfTokenizer {
    pub fn from_file(path: impl AsRef<Path>, pad_id: u32) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let mut inner = tokenizers::Tokenizer::from_bytes(&bytes)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        // Collation owns these policies so file defaults cannot silently change the input.
        inner.with_padding(None);
        inner
            .with_truncation(None)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let vocab = inner.get_vocab(true);
        ensure(
            vocab.values().any(|&id| id == pad_id),
            "pad ID is absent from tokenizer vocabulary",
        )?;
        let vocab_size = vocab
            .values()
            .copied()
            .max()
            .ok_or_else(|| Error::Tokenizer("empty vocabulary".into()))?
            as usize
            + 1;
        let spec = TokenizerSpec {
            kind: "huggingface-json-v1".into(),
            fingerprint: format!("{:x}", Sha256::digest(&bytes)),
            vocab_size,
            pad_id,
        };
        Ok(Self { inner, spec })
    }
}

impl TextTokenizer for HfTokenizer {
    fn spec(&self) -> TokenizerSpec {
        self.spec.clone()
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let ids = encoding.get_ids().to_vec();
        ensure(!ids.is_empty(), "tokenizer produced an empty sequence")?;
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bytes_cover_unicode_without_unknown_tokens() {
        let tokenizer = ByteTokenizer;
        let ids = tokenizer.encode("été 🦀").unwrap();
        assert_eq!(ids.len(), "été 🦀".len() + 2);
        assert_eq!(ids[0], 1);
        assert_eq!(*ids.last().unwrap(), 2);
        assert!(ids.iter().all(|&id| id > 0 && id < 259));
    }

    #[test]
    fn local_wordlevel_roundtrip_preserves_vocabulary_and_fingerprint() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tokenizer.json");
        let tokenizer =
            train_wordlevel(&["My card failed".into(), "Payment issue".into()], &path).unwrap();
        let loaded = HfTokenizer::from_file(&path, 0).unwrap();
        assert_eq!(tokenizer.spec(), loaded.spec());
        assert_eq!(
            tokenizer.encode("MY CARD").unwrap(),
            loaded.encode("my card").unwrap()
        );
        assert_eq!(loaded.encode("unseenword").unwrap(), vec![1]);
        assert!(HfTokenizer::from_file(&path, 10000).is_err());
        assert!(train_wordlevel(&["do not overwrite".into()], &path).is_err());
    }
}
