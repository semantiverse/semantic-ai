//! Local text-to-speech and speech-to-text in pure Rust, on [candle].
//!
//! - [`tts`]: [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts), a small streaming
//!   TTS with voice cloning that runs faster than real time on a CPU.
//! - [`stt`]: [Kyutai STT](https://kyutai.org/stt), streaming speech recognition with word
//!   timestamps.
//!
//! Models are downloaded from the Hugging Face Hub on first use (feature `hub`, on by default)
//! or loaded from local files.
//!
//! ```no_run
//! use semantic_voice_local::{Audio, stt::Stt, tts::Tts};
//!
//! # fn main() -> semantic_voice_local::Result<()> {
//! let tts = Tts::load("english")?;
//! let voice = tts.default_voice()?;
//! let speech = tts.synthesize("Hello from Rust!", &voice)?;
//! speech.write_wav("hello.wav")?;
//!
//! let stt = Stt::load(Default::default())?;
//! let transcript = stt.transcribe(&Audio::load("hello.wav")?)?;
//! println!("{}", transcript.text);
//! # Ok(())
//! # }
//! ```
//!
//! [candle]: https://github.com/huggingface/candle

mod audio;
mod error;
mod hub;
mod nn;
mod tokenizer;

pub mod stt;
pub mod tts;

pub use audio::{Audio, Resampler, resample};
pub use candle_core::Device;
pub use error::{Error, Result};
pub use tokenizer::Tokenizer;

/// The examples of the README compile.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
