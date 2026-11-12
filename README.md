# semantic-ai

Pure-Rust local voice: Kyutai Pocket TTS and Kyutai STT on [candle](https://github.com/huggingface/candle).

- [`semantic-voice-local`](crates/semantic-voice-local): the library, with batch and streaming
  APIs for text-to-speech and speech-to-text.
- [`semantic-voice-cli`](crates/semantic-voice-cli): `semantic-voice`, a demo command line.

## CLI

```sh
cargo build --release -p semantic-voice-cli
alias sv=target/release/semantic-voice

sv models                                          # models and voices
sv tts "Hello world." -o hello.wav                 # default model and voice
sv tts --model french --voice estelle "Bonjour !"
sv tts --voice ./me.wav --save-voice me.safetensors "Cloned."   # needs the gated weights
echo "Streamed text." | sv tts --stream | aplay -f FLOAT_LE -r 24000 -c 1

sv stt speech.wav                                  # transcript
sv stt --timestamps speech.wav                     # one word per line, with times
sv stt --json speech.wav
sv stt --stream --realtime --vad speech.wav        # words as recognized, end-of-turn marks
sv stt --model 2.6b speech.mp3

# Streaming STT, then each turn said back with streaming TTS (a turn ends when the semantic
# VAD detects it, or at the end of a sentence with the 2.6b model).
sv roundtrip --realtime speech.wav -o said.wav
```

Live microphone input and speaker output need the `audio-device` feature, which links the
system audio library through `cpal` (ALSA on Linux, found with `pkg-config`). With Nix,
`nix develop` provides them along with the Rust toolchain:

```sh
cargo build --release -p semantic-voice-cli --features audio-device
sv tts --play "Played while it is generated."
sv stt --mic --vad
sv roundtrip --mic --play                          # repeats what you say; muted while talking
```

Global options: `--device cpu|cuda|metal` (the GPU backends need the `cuda` or `metal` build
feature) and `--threads` (default 4). Logs go to stderr; set `RUST_LOG=info` for more.

## Tests

```sh
cargo test                                                   # unit tests
cargo test --release --test models -- --ignored              # end to end, downloads models
scripts/reference/generate.sh /tmp/fixtures                  # Python reference outputs
PARITY_FIXTURES=/tmp/fixtures cargo test --release --test parity -- --ignored
```
