//! Voices: the model state after the voice prompt, reusable across generations.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};

use crate::nn::transformer::TransformerState;
use crate::{Audio, Error, Result};

/// The voices published by Kyutai, as precomputed states for each built-in model.
pub const PREDEFINED_VOICES: &[&str] = &[
    "alba",
    "anna",
    "azelma",
    "bill_boerst",
    "caro_davy",
    "charles",
    "cosette",
    "eponine",
    "eve",
    "fantine",
    "george",
    "jane",
    "javert",
    "jean",
    "marius",
    "mary",
    "michael",
    "paul",
    "peter_yearsley",
    "stuart_bell",
    "vera",
    "giovanni",
    "lola",
    "juergen",
    "rafael",
    "daan",
    "estelle",
];

/// Repository and revision of the precomputed voice states.
pub(crate) const VOICES_REPO: &str = "kyutai/pocket-tts-without-voice-cloning";
pub(crate) const VOICES_REVISION: &str = "a7056f505a4a8ca72d8b5fb1440071d6f1a35739";

/// The voice used by default for a built-in model.
pub(crate) fn default_voice_for(model: &str) -> &'static str {
    [
        ("italian", "giovanni"),
        ("spanish", "lola"),
        ("german", "juergen"),
        ("portuguese", "rafael"),
        ("french", "estelle"),
        ("dutch", "daan"),
    ]
    .into_iter()
    .find(|(language, _)| model.contains(language))
    .map_or("alba", |(_, voice)| voice)
}

/// A speaker, as conditioning of the model. Obtain one from [`super::Tts`], then pass it to
/// every generation: it is not modified.
#[derive(Debug, Clone)]
pub struct Voice {
    pub(crate) state: TransformerState,
}

const LAYER_PREFIX: &str = "transformer.layers.";
const CACHE_SUFFIX: &str = ".self_attn/cache";

impl Voice {
    pub(crate) fn new(state: TransformerState) -> Self {
        Self { state }
    }

    /// Saves the voice in the safetensors format of upstream Pocket TTS (`export-voice`), so
    /// that it loads without encoding the audio again.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut tensors = HashMap::new();
        for (i, cache) in self.state.layer_caches()?.into_iter().enumerate() {
            let Some((k, v)) = cache else { continue };
            // Upstream stores `[2, batch, time, heads, head_dim]`.
            let cache = Tensor::stack(&[k.transpose(1, 2)?, v.transpose(1, 2)?], 0)?
                .to_dtype(DType::F32)?
                .contiguous()?;
            let len = cache.dim(2)? as i64;
            let module = format!("{LAYER_PREFIX}{i}.self_attn");
            tensors.insert(format!("{module}/cache"), cache);
            tensors.insert(
                format!("{module}/offset"),
                Tensor::new(&[len], &Device::Cpu)?,
            );
            tensors.insert(format!("{module}/pad"), Tensor::new(&[0i64], &Device::Cpu)?);
        }
        candle_core::safetensors::save(&tensors, path)?;
        Ok(())
    }

    /// Loads a voice saved by [`Voice::save`] or upstream Pocket TTS.
    pub(crate) fn load(path: &Path, dtype: DType, device: &Device) -> Result<Self> {
        let tensors = candle_core::safetensors::load(path, device)?;
        let invalid = |msg: String| Error::invalid_input(format!("{}: {msg}", path.display()));
        let mut layers = Vec::new();
        for (name, cache) in &tensors {
            let Some(index) = name
                .strip_prefix(LAYER_PREFIX)
                .and_then(|s| s.strip_suffix(CACHE_SUFFIX))
            else {
                continue;
            };
            let index: usize = index
                .parse()
                .map_err(|_| invalid(format!("bad tensor name {name}")))?;
            let module = &name[..name.len() - "/cache".len()];
            let len = match tensors.get(&format!("{module}/offset")) {
                Some(offset) => offset.flatten_all()?.to_vec1::<i64>()?[0] as usize,
                // Older exports encoded the length in the shape of `current_end`.
                None => match tensors.get(&format!("{module}/current_end")) {
                    Some(end) => end.dim(0)?,
                    None => cache.dim(2)?,
                },
            };
            if let Some(pad) = tensors.get(&format!("{module}/pad"))
                && pad.flatten_all()?.to_vec1::<i64>()?.iter().any(|&p| p != 0)
            {
                return Err(invalid(
                    "padded (batched) voice states are not supported".into(),
                ));
            }
            let cache = cache.narrow(2, 0, len)?.to_dtype(dtype)?;
            let k = cache.i(0)?.transpose(1, 2)?;
            let v = cache.i(1)?.transpose(1, 2)?;
            layers.push((index, (k, v)));
        }
        if layers.is_empty() {
            return Err(invalid("not a voice state".into()));
        }
        layers.sort_by_key(|(index, _)| *index);
        let layers = layers.into_iter().map(|(_, kv)| kv).collect();
        Ok(Self::new(TransformerState::from_layer_caches(
            layers, None,
        )?))
    }
}

/// Ends a voice prompt on exactly `PAUSE` of silence.
///
/// A prompt that stops on speech makes the model continue that utterance, and one that ends
/// on a long silence delays the onset. The trailing silence (20 ms frames 35 dB below the
/// loudest one) is cut, the last 20 ms faded out, and 80 ms of silence appended.
pub(crate) fn end_on_pause(audio: &Audio) -> Audio {
    const PAUSE_SECS: f64 = 0.08;
    const FADE_SECS: f64 = 0.02;
    const FLOOR_DB: f32 = 35.0;

    let rate = audio.sample_rate();
    let samples = audio.samples();
    let frame = ((0.02 * rate as f64) as usize).max(1);
    if samples.len() < frame {
        return audio.clone();
    }
    let db: Vec<f32> = samples
        .chunks_exact(frame)
        .map(|f| {
            let rms = (f.iter().map(|x| x * x).sum::<f32>() / frame as f32).sqrt();
            20.0 * (rms + 1e-12).log10()
        })
        .collect();
    let loudest = db.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let last_loud = db
        .iter()
        .rposition(|&d| d > loudest - FLOOR_DB)
        .unwrap_or(0);
    let end = (last_loud + 1) * frame;

    let mut out = samples[..end].to_vec();
    let fade = ((FADE_SECS * rate as f64) as usize).min(end);
    for (i, sample) in out[end - fade..].iter_mut().enumerate() {
        *sample *= 1.0 - i as f32 / (fade.max(2) - 1) as f32;
    }
    out.resize(end + (PAUSE_SECS * rate as f64) as usize, 0.0);
    Audio::new(out, rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn end_on_pause_trims_trailing_silence() {
        let rate = 1000;
        let mut samples = vec![0.5f32; 100];
        samples.extend(vec![0.0; 500]);
        let out = end_on_pause(&Audio::new(samples, rate));
        assert_eq!(out.len(), 100 + 80);
        assert_eq!(out.samples()[0], 0.5);
        assert_eq!(out.samples()[99], 0.0);
        assert!(out.samples()[100..].iter().all(|&s| s == 0.0));
    }
}
