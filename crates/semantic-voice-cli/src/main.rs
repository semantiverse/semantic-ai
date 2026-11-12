//! `semantic-voice`: a demo command line for local text-to-speech and speech-to-text.

#[cfg(feature = "audio-device")]
mod device;
mod roundtrip;
mod stt;
mod tts;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use semantic_voice_local::Device;

#[derive(Parser)]
#[command(
    version,
    about = "Local text-to-speech and speech-to-text (Kyutai Pocket TTS and STT)"
)]
struct Cli {
    /// Device to run the models on.
    #[arg(long, global = true, value_enum, default_value_t = DeviceKind::Cpu)]
    device: DeviceKind,

    /// CPU threads. Small models run fastest on few threads; defaults to `RAYON_NUM_THREADS`,
    /// or 4.
    #[arg(long, global = true)]
    threads: Option<usize>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Synthesize speech from text.
    Tts(tts::Args),
    /// Transcribe speech from an audio file or the microphone.
    Stt(stt::Args),
    /// Listen with streaming speech-to-text, and say each turn back with streaming
    /// text-to-speech.
    Roundtrip(roundtrip::Args),
    /// List the available models and voices.
    Models,
}

#[derive(Clone, Copy, ValueEnum)]
enum DeviceKind {
    Cpu,
    Cuda,
    Metal,
}

impl DeviceKind {
    fn device(self) -> Result<Device> {
        Ok(match self {
            Self::Cpu => Device::Cpu,
            Self::Cuda => Device::new_cuda(0)?,
            Self::Metal => Device::new_metal(0)?,
        })
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let threads = cli
        .threads
        .or_else(|| std::env::var("RAYON_NUM_THREADS").ok()?.parse().ok());
    // SAFETY: no other thread exists yet. candle sizes its thread pools from this variable.
    unsafe { std::env::set_var("RAYON_NUM_THREADS", threads.unwrap_or(4).to_string()) };
    let device = cli.device.device()?;
    match cli.command {
        Command::Tts(args) => tts::run(args, device),
        Command::Stt(args) => stt::run(args, device),
        Command::Roundtrip(args) => roundtrip::run(args, device),
        Command::Models => {
            println!("Text-to-speech models (--model):");
            for model in semantic_voice_local::tts::builtin_models() {
                println!("  {model}");
            }
            println!("\nPredefined voices (--voice), or pass an audio file to clone:");
            println!(
                "  {}",
                semantic_voice_local::tts::PREDEFINED_VOICES.join(", ")
            );
            println!("\nSpeech-to-text models (--model):");
            println!("  1b    kyutai/stt-1b-en_fr, English and French, 0.5 s delay");
            println!("  2.6b  kyutai/stt-2.6b-en, English, 2.5 s delay");
            Ok(())
        }
    }
}

/// Error for the device options of a build without the `audio-device` feature.
#[cfg(not(feature = "audio-device"))]
fn no_audio_device<T>() -> Result<T> {
    anyhow::bail!("this build has no audio device support, rebuild with `--features audio-device`")
}
