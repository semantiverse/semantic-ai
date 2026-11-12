//! Speaker output and microphone input through cpal.
//!
//! Devices are opened at the model's sample rate when they support it, at their default rate
//! otherwise, with the audio resampled as a stream.

use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig, SupportedStreamConfig, SupportedStreamConfigRange};
use semantic_voice_local::Resampler;

/// The f32 configuration closest to `sample_rate` among `configs`, else `default`.
fn pick_config(
    configs: impl Iterator<Item = SupportedStreamConfigRange>,
    default: SupportedStreamConfig,
    sample_rate: u32,
) -> Result<StreamConfig> {
    let config = configs
        .filter(|c| c.sample_format() == SampleFormat::F32)
        .find_map(|c| c.try_with_sample_rate(sample_rate))
        .unwrap_or(default);
    if config.sample_format() != SampleFormat::F32 {
        bail!("the audio device does not support 32-bit float samples");
    }
    Ok(config.config())
}

/// Plays audio pushed while it is being generated.
pub struct Speaker {
    _stream: cpal::Stream,
    queue: Arc<Mutex<VecDeque<f32>>>,
    resampler: Resampler,
}

impl Speaker {
    pub fn open(sample_rate: u32) -> Result<Self> {
        let device = cpal::default_host()
            .default_output_device()
            .context("no output device")?;
        let config = pick_config(
            device.supported_output_configs()?,
            device.default_output_config()?,
            sample_rate,
        )?;
        let channels = config.channels as usize;
        let queue = Arc::new(Mutex::new(VecDeque::<f32>::new()));
        let source = queue.clone();
        let stream = device.build_output_stream(
            config,
            move |data: &mut [f32], _| {
                let mut queue = source.lock().unwrap();
                for frame in data.chunks_mut(channels) {
                    // Silence on underrun: generation is lagging behind playback.
                    frame.fill(queue.pop_front().unwrap_or(0.0));
                }
            },
            |err| eprintln!("audio output error: {err}"),
            None,
        )?;
        stream.play()?;
        Ok(Self {
            _stream: stream,
            queue,
            resampler: Resampler::new(sample_rate, config.sample_rate),
        })
    }

    pub fn play(&mut self, samples: &[f32]) {
        let samples = self.resampler.process(samples);
        self.queue.lock().unwrap().extend(samples);
    }

    /// Waits until everything queued was played.
    pub fn drain(&mut self) {
        let tail = self.resampler.finish();
        self.queue.lock().unwrap().extend(tail);
        while !self.queue.lock().unwrap().is_empty() {
            std::thread::sleep(Duration::from_millis(20));
        }
        // Let the device play out its own buffer.
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// Captures mono audio at the model's sample rate.
pub struct Microphone {
    _stream: cpal::Stream,
    chunks: Receiver<Vec<f32>>,
    resampler: Resampler,
}

impl Microphone {
    pub fn open(sample_rate: u32) -> Result<Self> {
        let device = cpal::default_host()
            .default_input_device()
            .context("no input device")?;
        let config = pick_config(
            device.supported_input_configs()?,
            device.default_input_config()?,
            sample_rate,
        )?;
        let channels = config.channels as usize;
        let (sender, chunks) = channel();
        let stream = device.build_input_stream(
            config,
            move |data: &[f32], _| {
                let mono = data
                    .chunks(channels)
                    .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
                    .collect();
                // The receiver is gone once the transcription stopped.
                let _ = sender.send(mono);
            },
            |err| eprintln!("audio input error: {err}"),
            None,
        )?;
        stream.play()?;
        Ok(Self {
            _stream: stream,
            chunks,
            resampler: Resampler::new(config.sample_rate, sample_rate),
        })
    }

    /// The next captured samples, `None` if the device stopped.
    pub fn recv(&mut self) -> Option<Vec<f32>> {
        let samples = self.chunks.recv().ok()?;
        Some(self.resampler.process(&samples))
    }
}
