# semantic-voice-local

Local text-to-speech and speech-to-text in pure Rust, on [candle](https://github.com/huggingface/candle):

- **TTS**: [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts). Streaming, voice
  cloning, 7 languages, models from 1.9M to 300M+ parameters. Every upstream model config is
  built in.
- **STT**: [Kyutai STT](https://kyutai.org/stt). Streaming recognition with word timestamps
  and semantic end-of-turn detection, in English and French (1B) or English (2.6B).

No Python and no C/C++ dependencies: the SentencePiece tokenizer, resampling and audio
decoding are Rust too. The library processes sample buffers only; recording and playback are
up to the application (the [CLI](../semantic-voice-cli) shows how with `cpal`).

```toml
[dependencies]
semantic-voice-local = { git = "https://github.com/semantiverse/pocket-voice-rs" }
```

## Text-to-speech

```rust,no_run
use semantic_voice_local::tts::Tts;

fn main() -> semantic_voice_local::Result<()> {
    let tts = Tts::load("english")?; // a built-in model, see `tts::builtin_models()`
    let voice = tts.voice("alba")?; // predefined, saved (.safetensors), or audio to clone
    tts.synthesize("Hello world.", &voice)?.write_wav("hello.wav")?;
    Ok(())
}
```

For low latency, stream: frames of 80 ms (24 kHz mono) come as they are generated, the first
after a few tens of milliseconds. Dropping the stream stops the generation.

```rust,no_run
use semantic_voice_local::tts::Tts;

fn speak(tts: &Tts, text: &str, mut play: impl FnMut(&[f32])) -> semantic_voice_local::Result<()> {
    let voice = tts.default_voice()?;
    for frame in tts.stream(text, &voice)? {
        play(frame?.samples());
    }
    Ok(())
}
```

- **Voices**: [`PREDEFINED_VOICES`](src/tts/voice.rs), or `tts.clone_voice(&audio)` from a few
  seconds of speech. Keep a voice with `voice.save(path)` and `tts.load_voice(path)` (upstream's
  `export-voice` format). Cloning needs the gated weights: accept the terms at
  <https://huggingface.co/kyutai/pocket-tts> and set `HF_TOKEN`. Without them the public weights
  are used, which work with the predefined voices.
- **Options**: `synthesize_with` and `stream_with` take `SynthesisOptions`, from
  `tts.default_options()`: temperature, seed (for reproducible output), end-of-speech
  threshold, chunk size of long texts, and so on.
- **Long texts** are split into sentences, generated one after the other in one stream.

## Speech-to-text

```rust,no_run
use semantic_voice_local::{Audio, stt::{Stt, SttModel}};

fn main() -> semantic_voice_local::Result<()> {
    let stt = Stt::load(SttModel::EnFr1b)?;
    let transcript = stt.transcribe(&Audio::load("speech.mp3")?)?; // any sample rate
    println!("{}", transcript.text);
    for word in &transcript.words {
        println!("{:.2}-{:.2} {}", word.start, word.end, word.text);
    }
    Ok(())
}
```

Live, push samples at `stt.sample_rate()` (24 kHz) in chunks of any size. Words come out a
fixed delay after they are spoken: 0.5 s with the 1B model, 2.5 s with the 2.6B one.

```rust,no_run
use semantic_voice_local::stt::{Stt, SttEvent, TurnDetector};
use semantic_voice_local::Resampler;

fn listen(
    stt: &Stt,
    microphone: impl Iterator<Item = Vec<f32>>,
    microphone_rate: u32,
) -> semantic_voice_local::Result<()> {
    let mut resampler = Resampler::new(microphone_rate, stt.sample_rate());
    let mut stream = stt.stream()?;
    let mut turns = TurnDetector::new();
    for chunk in microphone {
        for event in stream.push(&resampler.process(&chunk))? {
            if let SttEvent::Word { text, .. } = &event {
                print!("{text} ");
            }
            if turns.update(&event) {
                // The speaker is done: get the turn's last words without waiting for them.
                for event in stream.flush()? {
                    if let SttEvent::Word { text, .. } = event {
                        print!("{text} ");
                    }
                }
                println!("[end of turn]");
            }
        }
    }
    stream.finish()?; // the last words, at the end of the audio
    Ok(())
}
```

- **Events**: `Word { text, start }`, then `WordEnd { end }`; with the 1B model also
  `VoiceActivity` every frame, the probabilities that the speaker's turn is over.
  `Transcript::from_events` assembles them.
- **End of turn**: `TurnDetector` reads the voice activity; `.threshold()` and `.frames()` trade
  latency for fewer false ends on hesitations. The 2.6B model has no voice activity.
- **Flush**: the voice activity reacts to the audio, while the text lags behind it by the model
  delay. `flush()` feeds the delay's worth of silence at once (a fraction of a second of
  compute) so the last words come out. The stream then goes on; timestamps leave the inserted
  silence out.
- **Resampling**: `Resampler` resamples a stream chunk by chunk without artifacts at the chunk
  boundaries; `resample` and `Audio::resampled` do whole signals.

## Models, devices, threads

- Models are downloaded from the Hugging Face Hub on first use (feature `hub`, on by default)
  into the standard cache. Offline, use `Tts::builder().config_file(...)` with local paths in
  the config, and `Stt::builder().directory(...)`.
- `Tts` and `Stt` are `Send + Sync`: load once, run streams concurrently from several threads.
- The builders select the device (features `cuda` and `metal` enable candle's GPU backends) and
  8-bit weights: on by default for STT, off for TTS (`quantize(true)` speeds up its larger
  models, at a small cost in quality).
- On CPU, set `RAYON_NUM_THREADS` to a few threads (4 here): these small models get slower with
  many.

## Performance

Measured on an AMD Ryzen 7 7840HS laptop, CPU, 4 threads:

| Model | Speed | First audio / delay |
|---|---|---|
| TTS `english` (default) | 8x real time | ~70 ms |
| TTS `english_2m_2026-10` | 50x real time | ~12 ms |
| STT 1B, Q8 | 3.4x real time | 0.5 s |
| STT 2.6B, Q8 | 1.1x real time | 2.5 s |

Streaming steps are small: CPU kernels ([`src/nn/kernels`](src/nn/kernels)) replace candle's
paths where those are dominated by overheads (single-row products on every core, attention
without masks, convolution data movement, vectorized activations). The language models then
run at the speed of memory: each step reads all of their weights, 300 MB for the default TTS
model in f32, 1 GB for the 1B STT model in Q8. STT encodes the audio in batches of frames,
concurrently with the language model.

## Correctness

The port is checked against the upstream Python implementations
(`tests/parity.rs`, fixtures from [`scripts/reference`](../../scripts/reference)):

- TTS audio at temperature 0 matches sample for sample (max difference ~3e-4).
- Tokenization and text chunking are identical on tricky inputs.
- The STT transcript is identical.

STT follows the Python reference (`moshi.models.LMGen`) rather than moshi's Rust `asr`
module, which feeds the audio one frame late.

## License

Code: MIT or Apache-2.0. The built-in configs come from upstream Pocket TTS (MIT). The model
weights are CC-BY-4.0 (Kyutai); the voices have [their own
licenses](https://huggingface.co/kyutai/tts-voices).
