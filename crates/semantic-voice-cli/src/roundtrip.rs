//! `roundtrip`: listens with streaming speech-to-text, and speaks each turn back with streaming
//! text-to-speech, a skeleton of a voice conversation.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use semantic_voice_local::stt::{Stt, SttEvent, SttModel, SttStream, TurnDetector};
use semantic_voice_local::tts::DEFAULT_MODEL;
use semantic_voice_local::{Audio, Device};

use crate::stt::CHUNK;
use crate::tts::Sink;

#[derive(clap::Args)]
pub struct Args {
    /// Audio file to listen to (WAV, FLAC, MP3, Ogg).
    input: Option<PathBuf>,

    /// Listen to the microphone (needs the `audio-device` feature). It is muted while
    /// speaking, so that the speech is not heard back.
    #[arg(long, conflicts_with = "input")]
    mic: bool,

    /// Stop listening to the microphone after this many seconds.
    #[arg(long, requires = "mic")]
    duration: Option<f64>,

    /// Feed the input file at real-time pace, as if it was live.
    #[arg(long, requires = "input")]
    realtime: bool,

    /// Speech-to-text model: `1b` (English and French), or `2.6b` (English).
    #[arg(long, default_value = "1b")]
    stt_model: SttModel,

    /// Text-to-speech model (see `models`).
    #[arg(long, default_value = DEFAULT_MODEL)]
    tts_model: String,

    /// Predefined voice name, saved voice (.safetensors) or audio file to clone.
    #[arg(long)]
    voice: Option<String>,

    /// WAV file to write the speech to. Defaults to `roundtrip.wav` unless playing or
    /// streaming.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Play the speech through the speakers (needs the `audio-device` feature).
    #[arg(long)]
    play: bool,

    /// Stream the speech as raw 32-bit float PCM (mono, 24 kHz) to standard output.
    #[arg(long, conflicts_with = "play")]
    stream: bool,
}

/// What the speaker said in one turn.
struct Turn {
    text: String,
    /// When the end of the turn was detected.
    ended: Instant,
}

pub fn run(args: Args, device: Device) -> Result<()> {
    let started = Instant::now();
    let (stt, tts) = std::thread::scope(|scope| {
        let stt = scope.spawn(|| {
            Stt::builder()
                .model(args.stt_model)
                .device(device.clone())
                .build()
        });
        let tts = crate::tts::load(
            &args.tts_model,
            args.voice.as_deref(),
            false,
            device.clone(),
        );
        (stt.join().expect("loading panicked"), tts)
    });
    let (stt, (tts, voice)) = (stt?, tts?);
    eprintln!("Loaded models in {:.1?}", started.elapsed());

    let mut sink = Sink::open(args.play, args.stream, tts.sample_rate())?;
    let output = args
        .output
        .clone()
        .or_else(|| sink.is_none().then(|| "roundtrip.wav".into()));
    let mut speech = Audio::new(Vec::new(), tts.sample_rate());
    let speaking = AtomicBool::new(false);
    let (sender, turns) = channel();

    std::thread::scope(|scope| -> Result<()> {
        let listener = scope.spawn(|| listen(&args, &stt, &speaking, sender));
        // Speaks the turns as they come, while the listener goes on.
        for turn in turns {
            speaking.store(true, Ordering::SeqCst);
            let mut first_audio = None;
            let mut duration = Duration::ZERO;
            for frame in tts.stream(&turn.text, &voice)? {
                let frame = frame?;
                first_audio.get_or_insert_with(|| turn.ended.elapsed());
                duration += frame.duration();
                if let Some(sink) = &mut sink {
                    sink.write(&frame)?;
                }
                speech.append(&frame)?;
            }
            eprintln!(
                "  spoke {:.1}s, first audio {:.0?} after the end of the turn",
                duration.as_secs_f64(),
                first_audio.unwrap_or_default()
            );
            if let Some(sink) = &mut sink {
                sink.drain();
            }
            speaking.store(false, Ordering::SeqCst);
        }
        listener.join().expect("listener panicked")
    })?;

    if let Some(path) = output {
        if speech.is_empty() {
            bail!("nothing was said");
        }
        speech.write_wav(&path)?;
        eprintln!("Wrote {}", path.display());
    }
    Ok(())
}

/// Runs speech-to-text over the input, sending each completed turn.
fn listen(args: &Args, stt: &Stt, speaking: &AtomicBool, sender: Sender<Turn>) -> Result<()> {
    let mut stream = stt.stream()?;
    let mut turns = TurnSplitter::new(sender);
    if args.mic {
        #[cfg(feature = "audio-device")]
        listen_to_microphone(args, stt, speaking, &mut stream, &mut turns)?;
        #[cfg(not(feature = "audio-device"))]
        return crate::no_audio_device();
    } else {
        let _ = speaking;
        let Some(input) = &args.input else {
            bail!("pass an audio file, or --mic")
        };
        let audio = Audio::load(input)?.resampled(stt.sample_rate());
        eprintln!("Listening to {}...", input.display());
        let period = Duration::from_secs_f64(CHUNK as f64 / stt.sample_rate() as f64);
        for chunk in audio.samples().chunks(CHUNK) {
            let chunk_started = Instant::now();
            let events = stream.push(chunk)?;
            if !turns.handle(&events, &mut stream)? {
                break;
            }
            if args.realtime {
                std::thread::sleep(period.saturating_sub(chunk_started.elapsed()));
            }
        }
    }
    for event in stream.finish()? {
        turns.add(&event);
    }
    turns.send();
    Ok(())
}

#[cfg(feature = "audio-device")]
fn listen_to_microphone(
    args: &Args,
    stt: &Stt,
    speaking: &AtomicBool,
    stream: &mut SttStream,
    turns: &mut TurnSplitter,
) -> Result<()> {
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
    while deadline.is_none_or(|d| Instant::now() < d) {
        let Some(mut samples) = microphone.recv() else {
            break;
        };
        if speaking.load(Ordering::SeqCst) {
            // Silence instead of our own voice: the stream keeps its pace.
            samples.fill(0.0);
        }
        let events = stream.push(&samples)?;
        if !turns.handle(&events, stream)? {
            break;
        }
    }
    Ok(())
}

/// Groups recognized words into turns: up to an end of turn of the semantic voice activity
/// detection, or up to the end of a sentence for models without it.
struct TurnSplitter {
    sender: Sender<Turn>,
    text: String,
    detector: TurnDetector,
    has_vad: bool,
}

impl TurnSplitter {
    fn new(sender: Sender<Turn>) -> Self {
        Self {
            sender,
            text: String::new(),
            detector: TurnDetector::new(),
            has_vad: false,
        }
    }

    /// Adds `events` from `stream`, sending the turns they complete. `false` once turns can no
    /// longer be sent.
    fn handle(&mut self, events: &[SttEvent], stream: &mut SttStream) -> Result<bool> {
        for event in events {
            self.add(event);
            let ended = if self.detector.update(event) {
                // The last words of the turn, without waiting for them.
                stream.flush()?.iter().for_each(|event| self.add(event));
                true
            } else {
                self.has_vad |= matches!(event, SttEvent::VoiceActivity { .. });
                // Models without voice activity: one sentence at a time.
                !self.has_vad
                    && matches!(event, SttEvent::Word { text, .. } if text.ends_with(['.', '?', '!']))
            };
            if ended && !self.send() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn add(&mut self, event: &SttEvent) {
        if let SttEvent::Word { text, .. } = event {
            if !self.text.is_empty() {
                self.text.push(' ');
            }
            self.text.push_str(text);
        }
    }

    /// Sends the words so far as a turn, if any; `false` once turns can no longer be sent.
    fn send(&mut self) -> bool {
        if self.text.is_empty() {
            return true;
        }
        let text = std::mem::take(&mut self.text);
        eprintln!("heard: {text}");
        self.sender
            .send(Turn {
                text,
                ended: Instant::now(),
            })
            .is_ok()
    }
}
