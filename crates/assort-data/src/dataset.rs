use assort_core::{Error, Request, Result, ensure};
use serde::{Deserialize, Serialize};
use std::io::BufRead;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum Target {
    Hard(usize),
    Soft(Vec<f32>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Example {
    pub request: Request,
    /// One target per question, in request order. Candidate order is authoritative.
    pub targets: Vec<Target>,
}

impl Example {
    pub fn validate(&self) -> Result<()> {
        self.request.validate()?;
        ensure(
            self.targets.len() == self.request.questions.len(),
            "one target is required per question",
        )?;
        for (q, target) in self.request.questions.iter().zip(&self.targets) {
            match target {
                Target::Hard(index) => ensure(
                    *index < q.candidates.len(),
                    "hard target index outside candidate set",
                )?,
                Target::Soft(values) => {
                    ensure(
                        values.len() == q.candidates.len(),
                        "soft target length differs from candidate count",
                    )?;
                    ensure(
                        values
                            .iter()
                            .all(|p| p.is_finite() && *p >= 0.0 && *p <= 1.0),
                        "invalid target probability",
                    )?;
                    ensure(
                        (values.iter().map(|&v| v as f64).sum::<f64>() - 1.0).abs() <= 1e-5,
                        "soft target probabilities must sum to one",
                    )?;
                }
            }
        }
        Ok(())
    }
}

/// Streaming JSONL reader with bounded line allocation and contextual errors.
/// Blank lines are skipped. An error terminates the iterator.
pub struct JsonlReader<R> {
    reader: R,
    line: usize,
    max_line_bytes: usize,
    finished: bool,
}

impl<R: BufRead> JsonlReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            line: 0,
            max_line_bytes: 8 * 1024 * 1024,
            finished: false,
        }
    }
    pub fn with_max_line_bytes(mut self, max: usize) -> Self {
        self.max_line_bytes = max;
        self
    }
}

impl<R: BufRead> Iterator for JsonlReader<R> {
    type Item = Result<Example>;

    fn next(&mut self) -> Option<Self::Item> {
        use std::io::Read;
        while !self.finished {
            let mut bytes = Vec::new();
            let read = self
                .reader
                .by_ref()
                .take(self.max_line_bytes.saturating_add(1) as u64)
                .read_until(b'\n', &mut bytes);
            self.line += 1;
            let result = match read {
                Ok(0) => {
                    self.finished = true;
                    return None;
                }
                Ok(n) if n > self.max_line_bytes => {
                    Err(Error::Invalid("line exceeds byte limit".into()))
                }
                Ok(_) if bytes.iter().all(u8::is_ascii_whitespace) => continue,
                Ok(_) => serde_json::from_slice::<Example>(&bytes)
                    .map_err(|e| Error::Invalid(e.to_string()))
                    .and_then(|example| {
                        example.validate()?;
                        Ok(example)
                    }),
                Err(e) => Err(e.into()),
            };
            return Some(result.map_err(|e| {
                self.finished = true;
                Error::Invalid(format!("dataset line {}: {e}", self.line))
            }));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assort_core::Question;

    #[test]
    fn validates_soft_and_hard_targets() {
        let mut e = Example {
            request: Request {
                state: "Synthetic".into(),
                questions: vec![Question::boolean("q", "Question?")],
            },
            targets: vec![Target::Soft(vec![0.25, 0.75])],
        };
        e.validate().unwrap();
        e.targets = vec![Target::Soft(vec![0.1, 0.1])];
        assert!(e.validate().is_err());
        e.targets = vec![Target::Hard(2)];
        assert!(e.validate().is_err());
    }

    #[test]
    fn errors_include_line_and_stop_reading() {
        let mut reader = JsonlReader::new(&b"\ninvalid\n{}\n"[..]);
        assert!(
            reader
                .next()
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("line 2")
        );
        assert!(reader.next().is_none());
        let mut reader = JsonlReader::new(&b"123456789"[..]).with_max_line_bytes(4);
        assert!(
            reader
                .next()
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("byte limit")
        );
    }
}
