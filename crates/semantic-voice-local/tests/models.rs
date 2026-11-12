//! End-to-end behavior of the public API with real models (downloaded on first run):
//! `cargo test --release --test models -- --ignored`.

use semantic_voice_local::Audio;
use semantic_voice_local::stt::{Stt, SttEvent, SttModel, Transcript};
use semantic_voice_local::tts::Tts;

const TEXT: &str = "The quick brown fox jumps over the lazy dog. It was not amused.";

fn deterministic(tts: &Tts) -> semantic_voice_local::tts::SynthesisOptions {
    let mut options = tts.default_options();
    options.temperature = 0.0;
    options
}

#[test]
#[ignore = "downloads models"]
fn saved_voice_reproduces_the_original() {
    let tts = Tts::load("english_2m_2026-10").unwrap();
    let voice = tts.voice("alba").unwrap();
    let path = std::env::temp_dir().join(format!(
        "semantic-voice-test-{}.safetensors",
        std::process::id()
    ));
    voice.save(&path).unwrap();
    let reloaded = tts.load_voice(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    let options = deterministic(&tts);
    let a = tts.synthesize_with(TEXT, &voice, &options).unwrap();
    let b = tts.synthesize_with(TEXT, &reloaded, &options).unwrap();
    assert_eq!(a, b);
}

#[test]
#[ignore = "downloads models"]
fn seeded_generation_is_reproducible() {
    let tts = Tts::load("english_2m_2026-10").unwrap();
    let voice = tts.default_voice().unwrap();
    let mut options = tts.default_options();
    options.seed = Some(42);
    let a = tts.synthesize_with(TEXT, &voice, &options).unwrap();
    let b = tts.synthesize_with(TEXT, &voice, &options).unwrap();
    assert_eq!(a, b);
    assert!(a.duration().as_secs_f64() > 1.0);
}

/// Speech from the TTS is transcribed back to the same words, whether the audio is given all
/// at once or streamed in uneven chunks.
#[test]
#[ignore = "downloads models"]
fn tts_to_stt_roundtrip() {
    let tts = Tts::load("english").unwrap();
    let voice = tts.default_voice().unwrap();
    let speech = tts
        .synthesize_with(TEXT, &voice, &deterministic(&tts))
        .unwrap();
    // A sample rate the STT model does not run at, to exercise resampling.
    let speech = speech.resampled(16_000);

    let stt = Stt::load(SttModel::EnFr1b).unwrap();
    let transcript = stt.transcribe(&speech).unwrap();
    let normalize = |s: &str| {
        s.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
    };
    assert_eq!(
        normalize(&transcript.text),
        normalize(TEXT),
        "{transcript:?}"
    );

    let words = &transcript.words;
    assert!(
        words.windows(2).all(|w| w[0].start <= w[1].start),
        "{words:?}"
    );
    assert!(words.iter().all(|w| w.start <= w.end));
    assert!(words.last().unwrap().end <= speech.duration().as_secs_f64() + 0.1);

    let speech = speech.resampled(stt.sample_rate());
    let mut stream = stt.stream().unwrap();
    let mut events = Vec::new();
    let mut samples = speech.samples();
    for size in [100, 5000, 1920, 1, 30_000].iter().cycle() {
        if samples.is_empty() {
            break;
        }
        let n = (*size).min(samples.len());
        events.extend(stream.push(&samples[..n]).unwrap());
        samples = &samples[n..];
    }
    events.extend(stream.finish().unwrap());
    assert!(
        events
            .iter()
            .any(|e| matches!(e, SttEvent::VoiceActivity { .. }))
    );
    assert_eq!(Transcript::from_events(&events), transcript);
}

/// A flush brings the words of a turn out without more audio, and timestamps of the speech
/// after it stay on the timeline of the audio pushed.
#[test]
#[ignore = "downloads models"]
fn flush_ends_a_turn() {
    let tts = Tts::load("english").unwrap();
    let voice = tts.default_voice().unwrap();
    let say = |text: &str| {
        tts.synthesize_with(text, &voice, &deterministic(&tts))
            .unwrap()
            .resampled(24_000)
    };
    let (first, second) = (say("The weather is nice today."), say("Let us go outside."));
    let stt = Stt::load(SttModel::EnFr1b).unwrap();

    let mut stream = stt.stream().unwrap();
    let mut events = stream.push(first.samples()).unwrap();
    events.extend(stream.flush().unwrap());
    let turn = Transcript::from_events(&events);
    assert_eq!(turn.text, "The weather is nice today.", "{turn:?}");
    assert!(
        events
            .iter()
            .all(|e| !matches!(e, SttEvent::VoiceActivity { time, .. } if *time > first.duration().as_secs_f64())),
    );

    // The second sentence, as heard alone, shifted by the first one.
    let alone = stt.transcribe(&second).unwrap();
    let mut events = stream.push(second.samples()).unwrap();
    events.extend(stream.finish().unwrap());
    let next = Transcript::from_events(&events);
    assert_eq!(next.text, alone.text);
    let offset = first.duration().as_secs_f64();
    for (word, expected) in next.words.iter().zip(&alone.words) {
        assert!(
            (word.start - offset - expected.start).abs() < 0.2,
            "{word:?} vs {expected:?} + {offset}"
        );
    }
}

#[test]
#[ignore = "downloads models"]
fn silence_has_no_words() {
    let stt = Stt::load(SttModel::EnFr1b).unwrap();
    let transcript = stt
        .transcribe(&Audio::new(vec![0.0; 24_000 * 2], 24_000))
        .unwrap();
    assert!(transcript.words.is_empty(), "{transcript:?}");
}
