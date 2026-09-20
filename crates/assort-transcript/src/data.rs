use assort_core::{Error, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const MEETING_GOAL: &str = "Capture final decisions, assigned actions, and important facts.";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Segment {
    pub id: String,
    pub start_ms: u64,
    pub end_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transcript {
    pub id: String,
    pub title: String,
    pub goal: String,
    pub segments: Vec<Segment>,
}

impl Transcript {
    pub fn validate(&self) -> Result<()> {
        ensure(
            !self.id.trim().is_empty()
                && !self.title.trim().is_empty()
                && !self.goal.trim().is_empty(),
            "transcript ID, title and goal must not be blank",
        )?;
        ensure(!self.segments.is_empty(), "transcript contains no segments")?;
        let mut ids = HashSet::new();
        let mut previous = 0;
        for segment in &self.segments {
            ensure(
                !segment.id.trim().is_empty() && ids.insert(&segment.id),
                "segment IDs must be nonblank and unique",
            )?;
            ensure(
                segment.start_ms <= segment.end_ms && segment.start_ms >= previous,
                "timestamps must have end >= start and be ordered by start time",
            )?;
            ensure(
                !segment.text.trim().is_empty(),
                "segment text must not be blank",
            )?;
            ensure(
                segment
                    .speaker
                    .as_ref()
                    .is_none_or(|s| !s.trim().is_empty()),
                "speaker must be nonblank when supplied",
            )?;
            previous = segment.start_ms;
        }
        Ok(())
    }

    /// SRT cue text is preserved, with no invented speaker names or sub-cue timing.
    /// Standard numbered cues only; reject malformed timing instead of skipping content.
    pub fn from_srt(
        id: impl Into<String>,
        title: impl Into<String>,
        goal: impl Into<String>,
        source: &str,
    ) -> Result<Self> {
        let normalized = source
            .trim_start_matches('\u{feff}')
            .replace("\r\n", "\n")
            .replace('\r', "\n");
        let mut segments = Vec::new();
        let mut blocks: Vec<Vec<&str>> = vec![Vec::new()];
        for line in normalized.lines() {
            if line.trim().is_empty() {
                if !blocks.last().unwrap().is_empty() {
                    blocks.push(Vec::new());
                }
            } else {
                blocks.last_mut().unwrap().push(line);
            }
        }
        for block in blocks.into_iter().filter(|b| !b.is_empty()) {
            ensure(
                block.len() >= 3 && block[0].trim().parse::<u64>().is_ok(),
                "SRT cue requires an integer ID, timing line, and text",
            )?;
            let (start, end) = block[1]
                .split_once("-->")
                .ok_or_else(|| Error::Invalid("invalid SRT timing line".into()))?;
            segments.push(Segment {
                id: format!("cue-{}", block[0].trim()),
                start_ms: parse_timestamp(start.trim())?,
                end_ms: parse_timestamp(end.trim())?,
                speaker: None,
                text: block[2..].join("\n"),
            });
        }
        let transcript = Self {
            id: id.into(),
            title: title.into(),
            goal: goal.into(),
            segments,
        };
        transcript.validate()?;
        Ok(transcript)
    }
}

fn parse_timestamp(text: &str) -> Result<u64> {
    let parts: Vec<_> = text.split([':', ',']).collect();
    ensure(
        parts.len() == 4
            && parts[0].len() >= 2
            && parts[1].len() == 2
            && parts[2].len() == 2
            && parts[3].len() == 3
            && parts.iter().all(|p| p.bytes().all(|c| c.is_ascii_digit())),
        "SRT timestamp must be HH:MM:SS,mmm",
    )?;
    let parts = parts
        .iter()
        .map(|p| {
            p.parse::<u64>()
                .map_err(|_| Error::Invalid("invalid timestamp number".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    ensure(
        parts[1] < 60 && parts[2] < 60,
        "invalid SRT minutes or seconds",
    )?;
    parts[0]
        .checked_mul(3_600_000)
        .and_then(|n| n.checked_add(parts[1] * 60_000 + parts[2] * 1000 + parts[3]))
        .ok_or_else(|| Error::Invalid("timestamp overflow".into()))
}

pub fn timestamp(ms: u64) -> String {
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        ms / 3_600_000,
        (ms / 60_000) % 60,
        (ms / 1000) % 60,
        ms % 1000
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PassageKind {
    Decision,
    Action,
    KeyFact,
    Background,
}

impl PassageKind {
    pub const ALL: [Self; 4] = [
        Self::Decision,
        Self::Action,
        Self::KeyFact,
        Self::Background,
    ];
    pub fn id(self) -> &'static str {
        match self {
            Self::Decision => "decision",
            Self::Action => "action",
            Self::KeyFact => "key_fact",
            Self::Background => "background",
        }
    }
    pub fn description(self) -> &'static str {
        match self {
            Self::Decision => "A final decision or agreed choice to retain in the summary",
            Self::Action => "An assigned action with an owner or concrete next step",
            Self::KeyFact => "An important fact, constraint, result, or blocker",
            Self::Background => "Background, small talk, repetition, or an unconfirmed suggestion",
        }
    }
    pub fn is_highlight(self) -> bool {
        self != Self::Background
    }
    pub fn index(self) -> usize {
        Self::ALL.iter().position(|&kind| kind == self).unwrap()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentLabel {
    pub segment_id: String,
    pub kind: PassageKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabeledTranscript {
    pub transcript: Transcript,
    pub labels: Vec<SegmentLabel>,
}

impl LabeledTranscript {
    pub fn validate(&self) -> Result<()> {
        self.transcript.validate()?;
        let expected: HashSet<_> = self
            .transcript
            .segments
            .iter()
            .map(|s| s.id.as_str())
            .collect();
        let actual: HashSet<_> = self.labels.iter().map(|s| s.segment_id.as_str()).collect();
        ensure(
            self.labels.len() == actual.len() && expected == actual,
            "labels must cover every segment ID exactly once",
        )
    }
}
