use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use semantic_voice_local::stt::{Stt, SttEvent, SttModel, SttStream, Transcript, TurnDetector};
use semantic_voice_local::{Audio, Device};

#[derive(clap::Args)]
pub struct Args {
    /// Audio file to transcribe (WAV, FLAC, MP3, Ogg).
    input: Option<PathBuf>,

    /// Transcribe the microphone live (needs the `audio-device` feature).
    #[arg(long, conflicts_with = "input")]
    mic: bool,

    /// Stop listening to the microphone after this many seconds.
    #[arg(long, requires = "mic")]
    duration: Option<f64>,

    /// `1b` (English and French) or `2.6b` (English).
    #[arg(long, default_value = "1b")]
    model: SttModel,

    /// Print words as they are recognized instead of the final transcript.
    #[arg(long)]
    stream: bool,

    /// When streaming a file, feed it at real-time pace, as if it was live.
    #[arg(long, requires = "stream")]
    realtime: bool,

    /// Show word timestamps.
    #[arg(long)]
    timestamps: bool,

    /// Show end-of-turn detections while streaming (semantic VAD, 1b model only).
    #[arg(long)]
    vad: bool,

    /// Run the language model with full-precision weights instead of 8-bit ones (slower).
    #[arg(long)]
    no_quantize: bool,

    /// Print the transcript as JSON.
    #[arg(long, conflicts_with = "stream")]
    json: bool,
}

/// Samples fed per step when streaming: one model frame (80 ms at 24 kHz).
pub const CHUNK: usize = 1920;

pub fn run(args: Args, device: Device) -> Result<()> {
    let started = Instant::now();
    let stt = Stt::builder()
        .model(args.model)
        .device(device)
        .quantize(!args.no_quantize)
        .build()?;
    eprintln!("Loaded model in {:.1?}", started.elapsed());

    if args.mic {
        #[cfg(feature = "audio-device")]
        return transcribe_microphone(&stt, &args);
        #[cfg(not(feature = "audio-device"))]
        return crate::no_audio_device();
    }
    let Some(input) = &args.input else {
        bail!("pass an audio file, or --mic")
    };
    let audio = Audio::load(input)?.resampled(stt.sample_rate());

    let started = Instant::now();
    if args.stream {
        let mut printer = EventPrinter::new(&args);
        let mut stream = stt.stream()?;
        for chunk in audio.samples().chunks(CHUNK) {
            let chunk_started = Instant::now();
            let events = stream.push(chunk)?;
            printer.print(&events, &mut stream)?;
            if args.realtime {
                let period = Duration::from_secs_f64(chunk.len() as f64 / stt.sample_rate() as f64);
                std::thread::sleep(period.saturating_sub(chunk_started.elapsed()));
            }
        }
        printer.print_words(&stream.finish()?)?;
        println!();
    } else {
        let transcript = stt.transcribe(&audio)?;
        print_transcript(&transcript, &args)?;
    }
    let elapsed = started.elapsed();
    eprintln!(
        "Transcribed {:.1}s of audio in {:.1?} ({:.1}x real time)",
        audio.duration().as_secs_f64(),
        elapsed,
        audio.duration().as_secs_f64() / elapsed.as_secs_f64()
    );
    Ok(())
}

#[cfg(feature = "audio-device")]
fn transcribe_microphone(stt: &Stt, args: &Args) -> Result<()> {
    let mut microphone = crate::device::Microphone::open(stt.sample_rate())?;
    eprintln!(
        "Listening{}...",
        if args.duration.is_some() {
            ""
        } else {
            ", Ctrl-C to stop"
        }
    );
    let deadline = args
        .duration
        .map(|secs| Instant::now() + Duration::from_secs_f64(secs));
    let mut printer = EventPrinter::new(args);
    let mut stream = stt.stream()?;
    while deadline.is_none_or(|d| Instant::now() < d) {
        let Some(samples) = microphone.recv() else {
            break;
        };
        let events = stream.push(&samples)?;
        printer.print(&events, &mut stream)?;
    }
    printer.print_words(&stream.finish()?)?;
    println!();
    Ok(())
}

/// Prints streaming events as they arrive.
struct EventPrinter {
    timestamps: bool,
    /// Marks the ends of turns, if enabled.
    turns: Option<TurnDetector>,
}

impl EventPrinter {
    fn new(args: &Args) -> Self {
        Self {
            timestamps: args.timestamps,
            turns: args.vad.then(TurnDetector::new),
        }
    }

    /// Prints `events`; at the end of a turn, flushes the turn's last words out of `stream`
    /// first.
    fn print(&mut self, events: &[SttEvent], stream: &mut SttStream) -> Result<()> {
        for event in events {
            self.print_words(std::slice::from_ref(event))?;
            if self.turns.as_mut().is_some_and(|turns| turns.update(event)) {
                self.print_words(&stream.flush()?)?;
                writeln!(std::io::stdout(), "[end of turn]")?;
            }
        }
        Ok(())
    }

    /// Prints the words of `events`.
    fn print_words(&mut self, events: &[SttEvent]) -> Result<()> {
        let mut stdout = std::io::stdout().lock();
        for event in events {
            if let SttEvent::Word { text, start } = event {
                if self.timestamps {
                    write!(stdout, "[{start:.2}] {text} ")?;
                } else {
                    write!(stdout, "{text} ")?;
                }
            }
        }
        stdout.flush()?;
        Ok(())
    }
}

fn print_transcript(transcript: &Transcript, args: &Args) -> Result<()> {
    if args.json {
        let words: Vec<_> = transcript
            .words
            .iter()
            .map(|w| serde_json::json!({ "text": w.text, "start": w.start, "end": w.end }))
            .collect();
        let json = serde_json::json!({ "text": transcript.text, "words": words });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else if args.timestamps {
        for word in &transcript.words {
            println!("{:7.2} {:7.2}  {}", word.start, word.end, word.text);
        }
    } else {
        println!("{}", transcript.text);
    }
    Ok(())
}
