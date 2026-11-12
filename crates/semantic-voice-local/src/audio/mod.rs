//! Mono audio buffers: decoding, resampling and WAV output.

mod resample;

use std::io::Cursor;
use std::path::Path;
use std::time::Duration;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::{Error, Result};

pub use resample::{Resampler, resample};

/// A mono audio clip with `f32` samples in `[-1, 1]`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Audio {
    samples: Vec<f32>,
    sample_rate: u32,
}

impl Audio {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    /// Reads and decodes an audio file (WAV, FLAC, MP3, Ogg/Vorbis), mixing it down to mono.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        let extension = path.extension().and_then(|e| e.to_str());
        Self::decode(bytes, extension)
    }

    /// Decodes an in-memory audio file. `extension` (e.g. `"wav"`) helps detecting the format.
    pub fn decode(bytes: Vec<u8>, extension: Option<&str>) -> Result<Self> {
        let audio_err = |e: SymphoniaError| Error::Audio(e.to_string());
        let mut hint = Hint::new();
        if let Some(extension) = extension {
            hint.with_extension(extension);
        }
        let stream = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
        let probed = symphonia::default::get_probe()
            .format(
                &hint,
                stream,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .map_err(audio_err)?;
        let mut format = probed.format;
        let track = format
            .default_track()
            .ok_or_else(|| Error::Audio("no audio track".into()))?;
        let track_id = track.id;
        let sample_rate = track
            .codec_params
            .sample_rate
            .ok_or_else(|| Error::Audio("unknown sample rate".into()))?;
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .map_err(audio_err)?;

        let mut samples = Vec::new();
        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(e) => return Err(audio_err(e)),
            };
            if packet.track_id() != track_id {
                continue;
            }
            let decoded = match decoder.decode(&packet) {
                Ok(decoded) => decoded,
                // Corrupted frames are skipped, as other decoders do.
                Err(SymphoniaError::DecodeError(_)) => continue,
                Err(e) => return Err(audio_err(e)),
            };
            let channels = decoded.spec().channels.count();
            let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
            buffer.copy_interleaved_ref(decoded);
            samples.extend(
                buffer
                    .samples()
                    .chunks_exact(channels)
                    .map(|frame| frame.iter().sum::<f32>() / channels as f32),
            );
        }
        Ok(Self {
            samples,
            sample_rate,
        })
    }

    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    pub fn into_samples(self) -> Vec<f32> {
        self.samples
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn duration(&self) -> Duration {
        if self.sample_rate == 0 {
            return Duration::ZERO;
        }
        Duration::from_secs_f64(self.samples.len() as f64 / self.sample_rate as f64)
    }

    /// Returns this clip at `sample_rate`.
    pub fn resampled(&self, sample_rate: u32) -> Self {
        Self {
            samples: resample(&self.samples, self.sample_rate, sample_rate),
            sample_rate,
        }
    }

    /// Keeps only the first `duration` of the clip.
    pub fn truncate(&mut self, duration: Duration) {
        let len = (duration.as_secs_f64() * self.sample_rate as f64) as usize;
        self.samples.truncate(len);
    }

    /// Appends `other`, which must have the same sample rate.
    pub fn append(&mut self, other: &Audio) -> Result<()> {
        if self.samples.is_empty() && self.sample_rate == 0 {
            self.sample_rate = other.sample_rate;
        }
        if other.sample_rate != self.sample_rate {
            return Err(Error::invalid_input(format!(
                "cannot append {} Hz audio to {} Hz audio",
                other.sample_rate, self.sample_rate
            )));
        }
        self.samples.extend_from_slice(&other.samples);
        Ok(())
    }

    /// Writes the clip as a 16-bit PCM WAV file.
    pub fn write_wav(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = std::io::BufWriter::new(std::fs::File::create(path)?);
        self.write_wav_to(file)
    }

    /// Writes the clip as 16-bit PCM WAV to `writer`.
    pub fn write_wav_to<W: std::io::Write + std::io::Seek>(&self, writer: W) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let wav_err = |e: hound::Error| Error::Audio(e.to_string());
        let mut wav = hound::WavWriter::new(writer, spec).map_err(wav_err)?;
        for &sample in &self.samples {
            let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
            wav.write_sample(value).map_err(wav_err)?;
        }
        wav.finalize().map_err(wav_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_roundtrip() {
        let audio = Audio::new(vec![0.0, 0.5, -0.5, 0.25], 24_000);
        let mut bytes = Cursor::new(Vec::new());
        audio.write_wav_to(&mut bytes).unwrap();
        let decoded = Audio::decode(bytes.into_inner(), Some("wav")).unwrap();
        assert_eq!(decoded.sample_rate(), 24_000);
        for (a, b) in audio.samples().iter().zip(decoded.samples()) {
            assert!((a - b).abs() < 1e-3);
        }
    }
}
