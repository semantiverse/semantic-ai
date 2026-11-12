//! Speech-to-text with [Kyutai STT](https://kyutai.org/stt).
//!
//! The model is streaming: text comes out a fixed delay (0.5 s for the 1B model) after the
//! audio goes in, with word-level timestamps.
//!
//! ```no_run
//! use semantic_voice_local::{Audio, stt::{Stt, SttEvent}};
//!
//! # fn main() -> semantic_voice_local::Result<()> {
//! let stt = Stt::load(Default::default())?;
//!
//! // A whole recording...
//! let transcript = stt.transcribe(&Audio::load("speech.wav")?)?;
//! println!("{}", transcript.text);
//!
//! // ...or a live stream, fed as samples arrive (at `stt.sample_rate()`).
//! let mut stream = stt.stream()?;
//! # let samples: Vec<f32> = vec![];
//! for event in stream.push(&samples)? {
//!     if let SttEvent::Word { text, .. } = event {
//!         print!("{text} ");
//!     }
//! }
//! stream.finish()?;
//! # Ok(())
//! # }
//! ```

mod model;

use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};

use crate::hub::HubFile;
use crate::nn::mimi::{MimiEncoder, MimiEncoderState};
use crate::nn::transformer::TransformerState;
use crate::nn::{load_mimi_weights, mmap_weights};
use crate::tokenizer::Tokenizer;
use crate::{Audio, Error, Result};

pub use model::SttConfig;

use model::{LanguageModel, Quantizer};

/// The published Kyutai STT models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SttModel {
    /// `kyutai/stt-1b-en_fr`: English and French, 0.5 s delay, with semantic VAD.
    #[default]
    EnFr1b,
    /// `kyutai/stt-2.6b-en`: English only, more accurate, 2.5 s delay.
    En2b6,
}

impl SttModel {
    fn hub_file(self, file: &str) -> HubFile {
        let (repo, revision) = match self {
            Self::EnFr1b => (
                "kyutai/stt-1b-en_fr-candle",
                "095e38f6242006a93c2541149b181988397f5c7c",
            ),
            Self::En2b6 => (
                "kyutai/stt-2.6b-en-candle",
                "8d380a4d647cd8f67b22e58f50424c8bf04c182b",
            ),
        };
        HubFile {
            repo: repo.into(),
            file: file.into(),
            revision: Some(revision.into()),
        }
    }
}

impl std::str::FromStr for SttModel {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "en_fr-1b" | "stt-1b-en_fr" | "1b" => Ok(Self::EnFr1b),
            "en-2.6b" | "stt-2.6b-en" | "2.6b" => Ok(Self::En2b6),
            _ => Err(Error::invalid_input(format!(
                "unknown STT model {s:?}, expected 1b or 2.6b"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
enum ModelSource {
    Hub(SttModel),
    Directory(PathBuf),
}

/// Configures and loads an [`Stt`].
#[derive(Debug, Clone)]
pub struct SttBuilder {
    source: ModelSource,
    device: Device,
    quantize: bool,
}

impl Default for SttBuilder {
    fn default() -> Self {
        Self {
            source: ModelSource::Hub(SttModel::default()),
            device: Device::Cpu,
            quantize: true,
        }
    }
}

impl SttBuilder {
    pub fn model(mut self, model: SttModel) -> Self {
        self.source = ModelSource::Hub(model);
        self
    }

    /// A local copy of a model repository (`config.json`, `model.safetensors`, Mimi weights
    /// and tokenizer).
    pub fn directory(mut self, path: impl Into<PathBuf>) -> Self {
        self.source = ModelSource::Directory(path.into());
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Whether to run the language model with 8-bit weights on CPU (the default): without,
    /// the 1B model does not keep up with real time on most CPUs. No effect on GPUs.
    pub fn quantize(mut self, quantize: bool) -> Self {
        self.quantize = quantize;
        self
    }

    pub fn build(self) -> Result<Stt> {
        let fetch = |file: &str| -> Result<PathBuf> {
            match &self.source {
                ModelSource::Hub(model) => model.hub_file(file).fetch(),
                ModelSource::Directory(dir) => Ok(dir.join(file)),
            }
        };
        let config: SttConfig = serde_json::from_slice(&std::fs::read(fetch("config.json")?)?)
            .map_err(|e| Error::config(format!("invalid STT config.json: {e}")))?;
        Stt::new(
            config.clone(),
            &fetch("model.safetensors")?,
            &fetch(&config.mimi_name)?,
            &fetch(&config.tokenizer_name)?,
            self.device,
            self.quantize,
        )
    }
}

/// A loaded Kyutai STT model. It is `Send + Sync`: streams can run concurrently.
pub struct Stt {
    encoder: MimiEncoder,
    quantizer: Quantizer,
    lm: LanguageModel,
    tokenizer: Tokenizer,
    config: SttConfig,
    device: Device,
}

/// The text tokens that are not words.
const END_OF_PADDING: u32 = 0;
const PADDING: u32 = 3;
/// Model frame rate.
const FRAME_RATE: f64 = 12.5;

impl Stt {
    pub fn builder() -> SttBuilder {
        SttBuilder::default()
    }

    /// Downloads (if needed) and loads a published model on the CPU.
    pub fn load(model: SttModel) -> Result<Self> {
        Self::builder().model(model).build()
    }

    fn new(
        config: SttConfig,
        lm_path: &Path,
        mimi_path: &Path,
        tokenizer_path: &Path,
        device: Device,
        quantize: bool,
    ) -> Result<Self> {
        let dtype = device.bf16_default_to_f32();
        let mimi_vb = load_mimi_weights(mimi_path, dtype, &device)?;
        let encoder = MimiEncoder::new(&model::mimi_config(), mimi_vb.clone())?;
        let quantizer = Quantizer::new(config.n_q, config.card, mimi_vb.pp("quantizer"))?;
        let lm = LanguageModel::new(&config, quantize, mmap_weights(lm_path, dtype, &device)?)?;
        let tokenizer = Tokenizer::load(tokenizer_path)?;
        Ok(Self {
            encoder,
            quantizer,
            lm,
            tokenizer,
            config,
            device,
        })
    }

    /// The sample rate [`SttStream::push`] expects.
    pub fn sample_rate(&self) -> u32 {
        24_000
    }

    pub fn config(&self) -> &SttConfig {
        &self.config
    }

    /// How long after being spoken a word is recognized.
    pub fn delay(&self) -> std::time::Duration {
        std::time::Duration::from_secs_f64(self.config.stt_config.audio_delay_seconds)
    }

    /// Transcribes a whole recording.
    pub fn transcribe(&self, audio: &Audio) -> Result<Transcript> {
        let audio = audio.resampled(self.sample_rate());
        let mut stream = self.stream()?;
        let mut events = stream.push(audio.samples())?;
        events.extend(stream.finish()?);
        Ok(Transcript::from_events(&events))
    }

    /// Starts a streaming transcription.
    pub fn stream(&self) -> Result<SttStream<'_>> {
        let stt_config = &self.config.stt_config;
        let frame_size = self.encoder.frame_size();
        let mut stream = SttStream {
            stt: self,
            encoder_state: self.encoder.init_state(),
            pending: Vec::with_capacity(frame_size),
            inserted: 0,
            recognizer: Recognizer {
                lm_state: self.lm.init_state(),
                previous_text_token: None,
                steps: 0,
                time_offset: stt_config.audio_silence_prefix_seconds
                    + stt_config.audio_delay_seconds,
                word_tokens: Vec::new(),
                word_start: 0.0,
                unended_word: false,
            },
        };
        // Training prefixed the audio with silence; timestamps are shifted back accordingly.
        // Whole frames, as upstream, which still offsets timestamps by the prefix in seconds.
        let prefix_frames = (stt_config.audio_silence_prefix_seconds * FRAME_RATE).ceil() as usize;
        stream.push(&vec![0.0; prefix_frames * frame_size])?;
        stream.inserted = prefix_frames * frame_size;
        Ok(stream)
    }
}

/// Something recognized in the audio.
#[derive(Debug, Clone, PartialEq)]
pub enum SttEvent {
    /// A word, starting `start` seconds into the stream. Its end follows in a [`SttEvent::WordEnd`].
    Word { text: String, start: f64 },
    /// The end of the last word, in seconds.
    WordEnd { end: f64 },
    /// Semantic voice activity, once per frame (80 ms) on models that have it: for several
    /// horizons, the probability that there is no more speech, i.e. that the turn ended.
    VoiceActivity {
        time: f64,
        pause_probabilities: Vec<f32>,
    },
}

/// Detects the ends of the speaker's turns from the [`SttEvent::VoiceActivity`] of a stream,
/// on models that have voice activity heads (the 1B one).
///
/// A turn ends when, after some words, the probability that the speaker is done stays above
/// a threshold for a number of consecutive frames. Call [`SttStream::flush`] then, to get the
/// turn's last words right away.
#[derive(Debug, Clone)]
pub struct TurnDetector {
    threshold: f32,
    frames: usize,
    /// Consecutive frames above the threshold.
    above: usize,
    /// Whether words came since the last end of turn.
    in_turn: bool,
}

impl Default for TurnDetector {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            frames: 1,
            above: 0,
            in_turn: false,
        }
    }
}

impl TurnDetector {
    /// The voice activity head used: its horizon is 2 seconds.
    const HEAD: usize = 2;

    pub fn new() -> Self {
        Self::default()
    }

    /// Probability of the end of the turn above which a frame counts (default 0.5). Higher
    /// values wait for more certainty.
    pub fn threshold(self, threshold: f32) -> Self {
        Self { threshold, ..self }
    }

    /// Consecutive frames (80 ms each) above the threshold that end a turn (default 1). More
    /// frames ignore short hesitations, at the cost of latency.
    pub fn frames(self, frames: usize) -> Self {
        Self {
            frames: frames.max(1),
            ..self
        }
    }

    /// Feeds an event of the stream; `true` when it ends the current turn.
    pub fn update(&mut self, event: &SttEvent) -> bool {
        match event {
            SttEvent::Word { .. } => {
                self.in_turn = true;
                false
            }
            SttEvent::WordEnd { .. } => false,
            SttEvent::VoiceActivity {
                pause_probabilities,
                ..
            } => {
                let pause = pause_probabilities
                    .get(Self::HEAD)
                    .is_some_and(|&p| p > self.threshold);
                self.above = if pause { self.above + 1 } else { 0 };
                if self.in_turn && self.above >= self.frames {
                    self.in_turn = false;
                    true
                } else {
                    false
                }
            }
        }
    }
}

/// A word with its time span, in seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Transcript {
    pub text: String,
    pub words: Vec<Word>,
}

impl Transcript {
    /// Assembles the words of a stream of events.
    pub fn from_events(events: &[SttEvent]) -> Self {
        let mut words: Vec<Word> = Vec::new();
        for event in events {
            match event {
                SttEvent::Word { text, start } => {
                    // Words sharing a segment end where the next one starts.
                    if let Some(previous) = words.last_mut()
                        && previous.end < *start
                    {
                        previous.end = *start;
                    }
                    words.push(Word {
                        text: text.clone(),
                        start: *start,
                        end: *start,
                    })
                }
                SttEvent::WordEnd { end } => {
                    if let Some(word) = words.last_mut() {
                        word.end = *end;
                    }
                }
                SttEvent::VoiceActivity { .. } => {}
            }
        }
        let text = words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Self { text, words }
    }
}

/// Frames encoded at once when more audio than one frame is available: encoding is much
/// cheaper per frame in batches.
const ENCODER_BATCH_FRAMES: usize = 16;

/// A streaming transcription, see [`Stt::stream`].
pub struct SttStream<'a> {
    stt: &'a Stt,
    encoder_state: MimiEncoderState,
    /// Samples not yet making up a whole frame.
    pending: Vec<f32>,
    /// Samples of silence the stream inserted itself: the prefix, and the flushes.
    inserted: usize,
    recognizer: Recognizer,
}

/// The language model side of a stream: audio codes in, words out.
struct Recognizer {
    lm_state: TransformerState,
    previous_text_token: Option<u32>,
    steps: usize,
    /// Text lags behind the audio by the model delay, after the silence prefix and the silence
    /// inserted by flushes.
    time_offset: f64,
    word_tokens: Vec<u32>,
    word_start: f64,
    unended_word: bool,
}

impl SttStream<'_> {
    /// Feeds samples at [`Stt::sample_rate`]; returns what was recognized meanwhile.
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<SttEvent>> {
        let frame_size = self.stt.encoder.frame_size();
        let mut events = Vec::new();
        let mut samples = samples;
        if !self.pending.is_empty() {
            let take = (frame_size - self.pending.len()).min(samples.len());
            self.pending.extend_from_slice(&samples[..take]);
            samples = &samples[take..];
            if self.pending.len() == frame_size {
                let frame = std::mem::take(&mut self.pending);
                self.process(&frame, &mut events)?;
                self.pending = frame;
                self.pending.clear();
            }
        }
        let whole = samples.len() / frame_size * frame_size;
        self.process(&samples[..whole], &mut events)?;
        self.pending.extend_from_slice(&samples[whole..]);
        Ok(events)
    }

    /// Brings out the last words without waiting for more audio: feeds the model silence
    /// for its delay, as if the speaker paused, and returns what it recognizes. For the end of
    /// a turn (see [`TurnDetector`]), where the voice activity reacts to the audio while the
    /// text lags behind it: the last words would otherwise come only with the next audio,
    /// or after the delay in real time.
    ///
    /// The stream goes on afterwards. Timestamps leave out the inserted silence, and the
    /// voice activity during it is dropped.
    pub fn flush(&mut self) -> Result<Vec<SttEvent>> {
        let stt = self.stt;
        let frame_size = stt.encoder.frame_size();
        let end = self.audio_time();
        let delay_frames = (stt.config.stt_config.audio_delay_seconds * FRAME_RATE).ceil() as usize;
        let mut padding = delay_frames * frame_size;
        if !self.pending.is_empty() {
            padding += frame_size - self.pending.len();
        }
        let mut events = self.push(&vec![0.0; padding])?;
        events.retain(|event| !matches!(event, SttEvent::VoiceActivity { .. }));
        let recognizer = &mut self.recognizer;
        if !recognizer.word_tokens.is_empty() {
            recognizer.emit_words(stt, &mut events);
        }
        if recognizer.unended_word {
            recognizer.unended_word = false;
            events.push(SttEvent::WordEnd {
                end: end.max(recognizer.word_start),
            });
        }
        recognizer.time_offset += padding as f64 / stt.sample_rate() as f64;
        self.inserted += padding;
        Ok(events)
    }

    /// Ends the stream: [flushes](Self::flush) the last words out.
    pub fn finish(mut self) -> Result<Vec<SttEvent>> {
        self.flush()
    }

    /// Seconds of audio pushed so far by the caller.
    fn audio_time(&self) -> f64 {
        let stt = self.stt;
        let samples = self.recognizer.steps * stt.encoder.frame_size() + self.pending.len();
        (samples - self.inserted) as f64 / stt.sample_rate() as f64
    }

    /// Runs whole frames through the model, encoding the next batch of frames while the
    /// language model goes through the current one.
    fn process(&mut self, frames: &[f32], events: &mut Vec<SttEvent>) -> Result<()> {
        let stt = self.stt;
        let (encoder_state, recognizer) = (&mut self.encoder_state, &mut self.recognizer);
        let mut batches = frames.chunks(ENCODER_BATCH_FRAMES * stt.encoder.frame_size());
        let mut codes = batches
            .next()
            .map(|batch| stt.encode_frames(batch, encoder_state))
            .transpose()?;
        while let Some(current) = codes.take() {
            let next = batches.next();
            let (stepped, next_codes) = rayon::join(
                || {
                    current
                        .iter()
                        .try_for_each(|codes| recognizer.step(stt, codes, events))
                },
                || {
                    next.map(|batch| stt.encode_frames(batch, encoder_state))
                        .transpose()
                },
            );
            stepped?;
            codes = next_codes?;
        }
        Ok(())
    }
}

impl Stt {
    /// The audio codes of each frame of `samples`, a whole number of frames.
    fn encode_frames(
        &self,
        samples: &[f32],
        state: &mut MimiEncoderState,
    ) -> Result<Vec<Vec<u32>>> {
        let pcm = Tensor::from_slice(samples, (1, 1, samples.len()), &self.device)?
            .to_dtype(self.device.bf16_default_to_f32())?;
        let latents = self.encoder.encode_step(&pcm, state)?;
        Ok(self.quantizer.encode(&latents)?.squeeze(0)?.to_vec2()?)
    }
}

impl Recognizer {
    /// Audio time of the text predicted at step `index`.
    fn text_time(&self, index: usize) -> f64 {
        let seconds = (index as f64 / FRAME_RATE - self.time_offset).max(0.0);
        (seconds * 1000.0).round() / 1000.0
    }

    fn step(&mut self, stt: &Stt, codes: &[u32], events: &mut Vec<SttEvent>) -> Result<()> {
        let index = self.steps;
        // As in training, the first step sees the initial tokens instead of the audio.
        let initial;
        let audio_tokens = if index == 0 {
            initial = vec![stt.config.card as u32; stt.config.n_q];
            &initial
        } else {
            codes
        };
        let text_token = self
            .previous_text_token
            .unwrap_or(stt.config.text_card as u32);
        let prediction = stt.lm.step(text_token, audio_tokens, &mut self.lm_state)?;
        self.previous_text_token = Some(prediction.text_token);
        self.steps += 1;
        if !prediction.extra_head_probabilities.is_empty() {
            events.push(SttEvent::VoiceActivity {
                time: self.text_time(index),
                pause_probabilities: prediction.extra_head_probabilities,
            });
        }

        // Words are separated by end-of-padding tokens and followed by padding.
        let token = prediction.text_token;
        if token == END_OF_PADDING || token == PADDING {
            if !self.word_tokens.is_empty() {
                self.emit_words(stt, events);
            }
        } else if token > PADDING {
            self.word_tokens.push(token);
        }
        if token == END_OF_PADDING {
            if self.unended_word {
                self.unended_word = false;
                events.push(SttEvent::WordEnd {
                    end: self.text_time(index),
                });
            }
            self.word_start = self.text_time(index + 1);
        }
        Ok(())
    }

    /// Emits the pending word tokens. Words spoken quickly can share a segment: each then gets
    /// as many frames as it has tokens.
    fn emit_words(&mut self, stt: &Stt, events: &mut Vec<SttEvent>) {
        let tokenizer = &stt.tokenizer;
        let text = tokenizer.decode(&std::mem::take(&mut self.word_tokens));
        let mut start = self.word_start;
        for word in text.split_whitespace() {
            events.push(SttEvent::Word {
                text: word.to_string(),
                start,
            });
            let n_tokens = tokenizer.encode(word).map_or(1, |t| t.len());
            start = ((start + n_tokens as f64 / FRAME_RATE) * 1000.0).round() / 1000.0;
        }
        self.unended_word = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn activity(pause: f32) -> SttEvent {
        SttEvent::VoiceActivity {
            time: 0.0,
            pause_probabilities: vec![0.0, 0.0, pause, 0.0],
        }
    }

    fn word() -> SttEvent {
        SttEvent::Word {
            text: "word".into(),
            start: 0.0,
        }
    }

    fn ends(detector: &mut TurnDetector, events: &[SttEvent]) -> Vec<bool> {
        events.iter().map(|e| detector.update(e)).collect()
    }

    #[test]
    fn turns_end_after_words_only() {
        let mut detector = TurnDetector::new();
        // Pauses before any word are not ends of turns.
        assert_eq!(
            ends(&mut detector, &[activity(0.9), activity(0.9)]),
            [false; 2]
        );
        let events = [word(), activity(0.2), activity(0.9), activity(0.9)];
        assert_eq!(ends(&mut detector, &events), [false, false, true, false]);
    }

    #[test]
    fn turns_end_after_consecutive_frames() {
        let mut detector = TurnDetector::new().frames(3).threshold(0.6);
        let events = [
            word(),
            activity(0.7),
            activity(0.7),
            activity(0.5), // A hesitation: starts over.
            activity(0.7),
            activity(0.7),
            activity(0.7),
        ];
        assert_eq!(
            ends(&mut detector, &events),
            [false, false, false, false, false, false, true]
        );
    }
}
