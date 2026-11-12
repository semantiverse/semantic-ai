use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use semantic_voice_local::tts::{DEFAULT_MODEL, Tts, Voice};
use semantic_voice_local::{Audio, Device};

#[derive(clap::Args)]
pub struct Args {
    /// Text to speak; read from standard input if omitted.
    text: Option<String>,

    /// Built-in model (see `models`), or a path to an upstream YAML config.
    #[arg(long, default_value = DEFAULT_MODEL)]
    model: String,

    /// Predefined voice name, saved voice (.safetensors) or audio file to clone.
    #[arg(long)]
    voice: Option<String>,

    /// WAV file to write. Defaults to `tts.wav` unless streaming.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Stream raw 32-bit float PCM (mono, 24 kHz) to standard output while generating,
    /// e.g. `| aplay -f FLOAT_LE -r 24000 -c 1`.
    #[arg(long, conflicts_with = "play")]
    stream: bool,

    /// Play through the speakers while generating (needs the `audio-device` feature).
    #[arg(long)]
    play: bool,

    /// Sampling temperature (model default if omitted).
    #[arg(long)]
    temperature: Option<f32>,

    /// Seed for reproducible output.
    #[arg(long)]
    seed: Option<u64>,

    /// Run the flow LM with 8-bit weights (faster for the larger models).
    #[arg(long)]
    quantize: bool,

    /// Also save the voice state, to reuse a cloned voice without encoding it again.
    #[arg(long)]
    save_voice: Option<PathBuf>,
}

/// Where generated audio goes while streaming.
pub enum Sink {
    Stdout(std::io::Stdout),
    #[cfg(feature = "audio-device")]
    Speaker(crate::device::Speaker),
}

impl Sink {
    /// The speakers if `play`, raw samples on standard output if `stream`, else none.
    pub fn open(play: bool, stream: bool, sample_rate: u32) -> Result<Option<Self>> {
        if play {
            #[cfg(feature = "audio-device")]
            return Ok(Some(Self::Speaker(crate::device::Speaker::open(
                sample_rate,
            )?)));
            #[cfg(not(feature = "audio-device"))]
            return crate::no_audio_device();
        }
        let _ = sample_rate;
        Ok(stream.then(|| Self::Stdout(std::io::stdout())))
    }

    pub fn write(&mut self, chunk: &Audio) -> Result<()> {
        match self {
            Self::Stdout(stdout) => {
                let bytes: Vec<u8> = chunk
                    .samples()
                    .iter()
                    .flat_map(|s| s.to_le_bytes())
                    .collect();
                stdout.write_all(&bytes)?;
                stdout.flush()?;
            }
            #[cfg(feature = "audio-device")]
            Self::Speaker(speaker) => speaker.play(chunk.samples()),
        }
        Ok(())
    }

    /// Waits until the audio written so far was played.
    pub fn drain(&mut self) {
        #[cfg(feature = "audio-device")]
        if let Self::Speaker(speaker) = self {
            speaker.drain();
        }
    }
}

/// Loads `model` (built-in name or YAML config) and `voice` (the model's default if `None`).
pub fn load(
    model: &str,
    voice: Option<&str>,
    quantize: bool,
    device: Device,
) -> Result<(Tts, Voice)> {
    let builder = Tts::builder().device(device).quantize(quantize);
    let builder = if model.ends_with(".yaml") || model.ends_with(".yml") {
        builder.config_file(model)
    } else {
        builder.model(model)
    };
    let tts = builder.build()?;
    let voice = match voice {
        Some(voice) => tts.voice(voice)?,
        None => tts.default_voice()?,
    };
    Ok((tts, voice))
}

pub fn run(args: Args, device: Device) -> Result<()> {
    let text = match args.text {
        Some(text) => text,
        None => std::io::read_to_string(std::io::stdin()).context("reading text from stdin")?,
    };

    let started = Instant::now();
    let (tts, voice) = load(&args.model, args.voice.as_deref(), args.quantize, device)?;
    eprintln!("Loaded model and voice in {:.1?}", started.elapsed());
    if let Some(path) = &args.save_voice {
        voice.save(path)?;
        eprintln!("Saved voice to {}", path.display());
    }

    let mut options = tts.default_options();
    if let Some(temperature) = args.temperature {
        options.temperature = temperature;
    }
    options.seed = args.seed;

    let mut sink = Sink::open(args.play, args.stream, tts.sample_rate())?;

    let started = Instant::now();
    let mut first_chunk = None;
    let mut audio = Audio::new(Vec::new(), tts.sample_rate());
    for chunk in tts.stream_with(&text, &voice, &options)? {
        let chunk = chunk?;
        first_chunk.get_or_insert_with(|| started.elapsed());
        if let Some(sink) = &mut sink {
            sink.write(&chunk)?;
        }
        audio.append(&chunk)?;
    }
    let elapsed = started.elapsed();
    if audio.is_empty() {
        bail!("no audio was generated");
    }
    eprintln!(
        "Generated {:.2}s of audio in {:.2?} ({:.1}x real time, first audio after {:.0?})",
        audio.duration().as_secs_f64(),
        elapsed,
        audio.duration().as_secs_f64() / elapsed.as_secs_f64(),
        first_chunk.unwrap_or_default(),
    );

    let output = args
        .output
        .or_else(|| sink.is_none().then(|| "tts.wav".into()));
    if let Some(mut sink) = sink {
        sink.drain();
    }
    if let Some(path) = output {
        audio.write_wav(&path)?;
        eprintln!("Wrote {}", path.display());
    }
    Ok(())
}
