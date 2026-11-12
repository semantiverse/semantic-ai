//! Text-to-speech with [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts).
//!
//! ```no_run
//! use semantic_voice_local::tts::Tts;
//!
//! # fn main() -> semantic_voice_local::Result<()> {
//! let tts = Tts::load("english")?;
//! let voice = tts.voice("alba")?;
//!
//! // All at once...
//! tts.synthesize("Hello world.", &voice)?.write_wav("hello.wav")?;
//!
//! // ...or streamed, 80 ms at a time, as soon as each frame is generated.
//! for chunk in tts.stream("A longer text, played while it is generated.", &voice)? {
//!     let chunk = chunk?;
//!     // play `chunk.samples()`...
//! }
//! # Ok(())
//! # }
//! ```

mod config;
mod flow_lm;
mod text;
mod voice;

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::hub::{self, HubFile};
use crate::nn::mimi::{MimiDecoder, MimiDecoderState, MimiEncoder};
use crate::nn::transformer::TransformerState;
use crate::nn::{load_mimi_weights, mmap_weights};
use crate::tokenizer::Tokenizer;
use crate::{Audio, Error, Result};

pub use config::{TtsConfig, builtin_models};
pub use voice::{PREDEFINED_VOICES, Voice};

use flow_lm::{FlowLm, Sampling};
use text::TextRules;

/// The default built-in model.
pub const DEFAULT_MODEL: &str = "english";

/// Generation parameters. [`Tts::default_options`] holds the values recommended by the model.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthesisOptions {
    /// Sampling temperature: higher is more expressive but less stable.
    pub temperature: f32,
    /// Generation stops once the end-of-speech logit exceeds this.
    pub eos_threshold: f32,
    /// Flow decoding steps per frame.
    pub decode_steps: usize,
    /// Clamp of the sampling noise, unclamped if `None`.
    pub noise_clamp: Option<f32>,
    /// Frames generated after the end of speech is detected; guessed from the text if `None`.
    pub frames_after_eos: Option<usize>,
    /// Long texts are generated in chunks of at most this many tokens.
    pub max_tokens_per_chunk: usize,
    /// Seed of the sampling noise, random if `None`.
    pub seed: Option<u64>,
}

/// Where a model comes from.
#[derive(Debug, Clone)]
enum ModelSource {
    Builtin(String),
    ConfigFile(PathBuf),
    Config(Box<TtsConfig>),
}

/// Configures and loads a [`Tts`].
#[derive(Debug, Clone)]
pub struct TtsBuilder {
    source: ModelSource,
    device: Device,
    quantize: bool,
}

impl Default for TtsBuilder {
    fn default() -> Self {
        Self {
            source: ModelSource::Builtin(DEFAULT_MODEL.into()),
            device: Device::Cpu,
            quantize: false,
        }
    }
}

impl TtsBuilder {
    /// A built-in model, see [`builtin_models`].
    pub fn model(mut self, name: impl Into<String>) -> Self {
        self.source = ModelSource::Builtin(name.into());
        self
    }

    /// A model described by an upstream-format YAML config file.
    pub fn config_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.source = ModelSource::ConfigFile(path.into());
        self
    }

    pub fn config(mut self, config: TtsConfig) -> Self {
        self.source = ModelSource::Config(Box::new(config));
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Whether to run the flow LM transformer with 8-bit weights on CPU. Off by default:
    /// only the larger models benefit. No effect on GPUs.
    pub fn quantize(mut self, quantize: bool) -> Self {
        self.quantize = quantize;
        self
    }

    pub fn build(self) -> Result<Tts> {
        let (config, name) = match self.source {
            ModelSource::Builtin(name) => (config::builtin_config(&name)?, Some(name)),
            ModelSource::ConfigFile(path) => {
                (TtsConfig::from_yaml(&std::fs::read_to_string(path)?)?, None)
            }
            ModelSource::Config(config) => (*config, None),
        };
        Tts::new(config, name, self.device, self.quantize)
    }
}

/// A loaded Pocket TTS model. It is `Send + Sync`: generations can run concurrently.
pub struct Tts {
    flow_lm: FlowLm,
    encoder: MimiEncoder,
    decoder: MimiDecoder,
    /// Maps generated latents to the decoder input (`quantizer.output_proj`).
    latent_projection: Linear,
    tokenizer: Tokenizer,
    rules: TextRules,
    config: TtsConfig,
    /// Built-in model name, needed for the predefined voices.
    model_name: Option<String>,
    voice_cloning: bool,
    dtype: DType,
    device: Device,
}

/// The voice-cloning weights are gated, so models fall back to the public ones.
fn resolve_bundle(config: &TtsConfig) -> Result<Option<(PathBuf, bool)>> {
    let Some(primary) = &config.weights_path else {
        return Ok(None);
    };
    match (
        hub::resolve(primary),
        &config.weights_path_without_voice_cloning,
    ) {
        (Ok(path), _) => Ok(Some((path, true))),
        (Err(err), Some(fallback)) => {
            tracing::info!("voice cloning weights unavailable ({err}), using {fallback}");
            Ok(Some((hub::resolve(fallback)?, false)))
        }
        (Err(err), None) => Err(err),
    }
}

impl Tts {
    pub fn builder() -> TtsBuilder {
        TtsBuilder::default()
    }

    /// Loads a built-in model on the CPU, see [`builtin_models`].
    pub fn load(model: &str) -> Result<Self> {
        Self::builder().model(model).build()
    }

    fn new(
        config: TtsConfig,
        model_name: Option<String>,
        device: Device,
        quantize: bool,
    ) -> Result<Self> {
        let dtype = flow_lm::parse_dtype(&config.flow_lm.dtype);
        let bundle = resolve_bundle(&config)?;
        let voice_cloning = bundle.as_ref().is_none_or(|(_, cloning)| *cloning);
        let bundle = bundle
            .map(|(path, _)| mmap_weights(&path, dtype, &device))
            .transpose()?;

        let flow_vb = match (&config.flow_lm.weights_path, &bundle) {
            (Some(path), _) => load_flow_lm_weights(&hub::resolve(path)?, dtype, &device)?,
            (None, Some(bundle)) => bundle.pp("flow_lm"),
            (None, None) => return Err(Error::config("no weights_path for the flow LM")),
        };
        let codec_vb = match (&config.mimi.weights_path, &bundle) {
            (Some(path), _) => load_mimi_weights(&hub::resolve(path)?, dtype, &device)?,
            (None, Some(bundle)) => bundle.pp("mimi"),
            (None, None) => return Err(Error::config("no weights_path for the codec")),
        };

        let mimi = config.mimi.mimi();
        let latent_dim = config.mimi.quantizer.dimension;
        let flow_lm = FlowLm::new(
            &config.flow_lm,
            latent_dim,
            mimi.inner_dim,
            quantize,
            flow_vb,
        )?;
        let encoder = MimiEncoder::new(&mimi, codec_vb.clone())?;
        let decoder = MimiDecoder::new(&mimi, codec_vb.clone())?;
        let projection = codec_vb
            .pp("quantizer.output_proj")
            .get(
                (config.mimi.quantizer.output_dimension, latent_dim, 1),
                "weight",
            )?
            .squeeze(2)?;

        let lut = &config.flow_lm.lookup_table;
        let tokenizer = Tokenizer::load(hub::resolve(&lut.tokenizer_path)?)?;
        if tokenizer.vocab_size() != lut.n_bins {
            return Err(Error::config(format!(
                "tokenizer has {} tokens, the model {}",
                tokenizer.vocab_size(),
                lut.n_bins
            )));
        }
        let rules = TextRules {
            pad_with_spaces_for_short_inputs: config.pad_with_spaces_for_short_inputs,
            remove_semicolons: config.remove_semicolons,
            append_terminal_punctuation: config.append_terminal_punctuation,
            capitalize_first_letter: config.capitalize_first_letter,
            replace_characters: config.replace_characters.clone(),
        };
        Ok(Self {
            voice_cloning: voice_cloning && flow_lm.supports_voice_cloning(),
            flow_lm,
            encoder,
            decoder,
            latent_projection: Linear::new(projection, None),
            tokenizer,
            rules,
            config,
            model_name,
            dtype,
            device,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.mimi.sample_rate
    }

    pub fn config(&self) -> &TtsConfig {
        &self.config
    }

    /// Whether [`Tts::clone_voice`] is available: it needs the gated weights.
    pub fn supports_voice_cloning(&self) -> bool {
        self.voice_cloning
    }

    /// The generation parameters recommended by the model config.
    pub fn default_options(&self) -> SynthesisOptions {
        SynthesisOptions {
            temperature: self.config.default_temperature,
            eos_threshold: self.config.default_eos_threshold,
            decode_steps: 1,
            noise_clamp: None,
            frames_after_eos: self.config.model_recommended_frames_after_eos,
            max_tokens_per_chunk: 50,
            seed: None,
        }
    }

    /// The model's default predefined voice.
    pub fn default_voice(&self) -> Result<Voice> {
        let name = self.model_name.as_deref().unwrap_or(DEFAULT_MODEL);
        self.predefined_voice(voice::default_voice_for(name))
    }

    /// A voice by predefined name (see [`PREDEFINED_VOICES`]), saved state (`.safetensors`),
    /// or audio file to clone. Files may be local paths or `hf://` references.
    pub fn voice(&self, voice: &str) -> Result<Voice> {
        if PREDEFINED_VOICES.contains(&voice) {
            return self.predefined_voice(voice);
        }
        let path = hub::resolve(voice)?;
        if path.extension().is_some_and(|e| e == "safetensors") {
            self.load_voice(&path)
        } else {
            self.clone_voice(&Audio::load(&path)?)
        }
    }

    /// A precomputed voice of a built-in model, downloaded from the Hub.
    pub fn predefined_voice(&self, name: &str) -> Result<Voice> {
        let model = self.model_name.as_deref().ok_or_else(|| {
            Error::invalid_input("predefined voices need a built-in model, clone a voice instead")
        })?;
        if !PREDEFINED_VOICES.contains(&name) {
            return Err(Error::invalid_input(format!("unknown voice {name:?}")));
        }
        let file = HubFile {
            repo: voice::VOICES_REPO.into(),
            file: format!("languages/{model}/embeddings/{name}.safetensors"),
            revision: Some(voice::VOICES_REVISION.into()),
        };
        self.load_voice(&file.fetch()?)
    }

    /// Loads a voice saved with [`Voice::save`] (or upstream's `export-voice`).
    pub fn load_voice(&self, path: impl AsRef<Path>) -> Result<Voice> {
        let voice = Voice::load(path.as_ref(), self.dtype, &self.device)?;
        let expected = self.flow_lm.init_state().num_layers();
        if voice.state.num_layers() != expected {
            return Err(Error::invalid_input(format!(
                "voice has {} layers, the model {expected}: it was made for another model",
                voice.state.num_layers()
            )));
        }
        Ok(voice)
    }

    /// Makes a voice from a recording of the speaker; a few seconds of clean speech suffice.
    /// Recordings longer than 30 seconds are truncated.
    pub fn clone_voice(&self, audio: &Audio) -> Result<Voice> {
        if !self.voice_cloning {
            return Err(Error::VoiceCloningUnavailable);
        }
        if audio.is_empty() {
            return Err(Error::invalid_input("empty voice recording"));
        }
        let mut audio = audio.resampled(self.sample_rate());
        audio.truncate(std::time::Duration::from_secs(30));
        let audio = voice::end_on_pause(&audio);
        let samples = Tensor::from_slice(audio.samples(), (1, 1, audio.len()), &self.device)?
            .to_dtype(self.dtype)?;
        let latents = self.encoder.encode(&samples)?.transpose(1, 2)?;
        let mut state = self.flow_lm.init_state();
        self.flow_lm.prompt_voice(&latents, &mut state)?;
        Ok(Voice::new(state))
    }

    /// The chunks `text` is generated in: its normalized sentences, grouped up to `max_tokens`
    /// tokens per chunk.
    pub fn split_text(&self, text: &str, max_tokens: usize) -> Result<Vec<String>> {
        self.rules
            .split_into_chunks(&self.tokenizer, text, max_tokens)
    }

    /// Synthesizes `text` with the default options.
    pub fn synthesize(&self, text: &str, voice: &Voice) -> Result<Audio> {
        self.synthesize_with(text, voice, &self.default_options())
    }

    pub fn synthesize_with(
        &self,
        text: &str,
        voice: &Voice,
        options: &SynthesisOptions,
    ) -> Result<Audio> {
        let mut audio = Audio::new(Vec::new(), self.sample_rate());
        for chunk in self.stream_with(text, voice, options)? {
            audio.append(&chunk?)?;
        }
        Ok(audio)
    }

    /// Streams the speech for `text`, one codec frame (80 ms) at a time, with the default
    /// options. Dropping the stream stops the generation.
    pub fn stream<'a>(&'a self, text: &str, voice: &'a Voice) -> Result<SpeechStream<'a>> {
        self.stream_with(text, voice, &self.default_options())
    }

    pub fn stream_with<'a>(
        &'a self,
        text: &str,
        voice: &'a Voice,
        options: &SynthesisOptions,
    ) -> Result<SpeechStream<'a>> {
        let chunks =
            self.rules
                .split_into_chunks(&self.tokenizer, text, options.max_tokens_per_chunk)?;
        let rng = match options.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };
        Ok(SpeechStream {
            tts: self,
            voice,
            options: options.clone(),
            sampling: Sampling {
                temperature: options.temperature,
                noise_clamp: options.noise_clamp,
                decode_steps: options.decode_steps,
                eos_threshold: options.eos_threshold,
            },
            chunks: chunks.into(),
            current: None,
            rng,
        })
    }

    /// Generation state of one text chunk.
    fn start_chunk(
        &self,
        chunk: &str,
        voice: &Voice,
        options: &SynthesisOptions,
    ) -> Result<ChunkGeneration> {
        let prompt = self.rules.prepare(chunk)?;
        let tokens = self.tokenizer.encode(&prompt.text)?;
        let mut lm_state = voice.state.duplicate()?;
        self.flow_lm.prompt_text(&tokens, &mut lm_state)?;
        // Generous upper bound on the speech duration, from the token count.
        let max_secs = tokens.len() as f64 / 3.0 + 2.0;
        Ok(ChunkGeneration {
            lm: LatentGenerator {
                state: lm_state,
                steps: 0,
                max_steps: (max_secs * self.config.mimi.frame_rate).ceil() as usize,
                eos_step: None,
                frames_after_eos: options
                    .frames_after_eos
                    .unwrap_or(prompt.frames_after_eos + 2),
            },
            decoder_state: self.decoder.init_state(),
            next: None,
            decoded_frames: 0,
        })
    }

    fn decode_latent(&self, latent: &Tensor, state: &mut MimiDecoderState) -> Result<Vec<f32>> {
        let latent = latent
            .broadcast_mul(&self.flow_lm.emb_std)?
            .broadcast_add(&self.flow_lm.emb_mean)?;
        let xs = self.latent_projection.forward(&latent)?.transpose(1, 2)?;
        let audio = self.decoder.decode_step(&xs, state)?;
        Ok(audio.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?)
    }
}

/// Before speech starts, the end-of-speech logit of some voices can spike: it is ignored on
/// the first frames.
const MIN_FRAMES_BEFORE_EOS: usize = 6;

/// Generation of one text chunk: the flow LM produces latents, the decoder turns them into
/// audio one step behind, so that both run in parallel.
struct ChunkGeneration {
    lm: LatentGenerator,
    decoder_state: MimiDecoderState,
    /// The latent to decode next, `None` once the chunk is complete.
    next: Option<Tensor>,
    decoded_frames: usize,
}

/// The flow LM side of a chunk generation.
struct LatentGenerator {
    state: TransformerState,
    steps: usize,
    max_steps: usize,
    eos_step: Option<usize>,
    frames_after_eos: usize,
}

impl LatentGenerator {
    /// The latent following `previous`, `None` once the speech of the chunk is complete.
    fn advance(
        &mut self,
        flow_lm: &FlowLm,
        previous: Option<&Tensor>,
        sampling: &Sampling,
        rng: &mut StdRng,
    ) -> Result<Option<Tensor>> {
        let past_end = |step: usize, eos: Option<usize>| {
            eos.is_some_and(|e| step >= e + self.frames_after_eos)
        };
        if past_end(self.steps, self.eos_step) {
            return Ok(None);
        }
        if self.steps >= self.max_steps {
            tracing::warn!("maximum length reached without end of speech");
            return Ok(None);
        }
        let step = self.steps;
        let (latent, eos) = flow_lm.step(previous, &mut self.state, sampling, rng)?;
        self.steps += 1;
        if eos && self.eos_step.is_none() && step >= MIN_FRAMES_BEFORE_EOS {
            self.eos_step = Some(step);
        }
        Ok((!past_end(step, self.eos_step)).then_some(latent))
    }
}

/// Speech generated frame by frame, see [`Tts::stream`].
pub struct SpeechStream<'a> {
    tts: &'a Tts,
    voice: &'a Voice,
    options: SynthesisOptions,
    sampling: Sampling,
    chunks: VecDeque<String>,
    current: Option<ChunkGeneration>,
    rng: StdRng,
}

impl SpeechStream<'_> {
    pub fn sample_rate(&self) -> u32 {
        self.tts.sample_rate()
    }

    /// The next frame of audio, `None` once the speech is complete.
    fn next_frame(&mut self) -> Result<Option<Audio>> {
        let tts = self.tts;
        loop {
            let generation = match &mut self.current {
                Some(generation) => generation,
                None => {
                    let Some(chunk) = self.chunks.pop_front() else {
                        return Ok(None);
                    };
                    let mut generation = tts.start_chunk(&chunk, self.voice, &self.options)?;
                    generation.next =
                        generation
                            .lm
                            .advance(&tts.flow_lm, None, &self.sampling, &mut self.rng)?;
                    self.current.insert(generation)
                }
            };
            let Some(latent) = generation.next.take() else {
                self.current = None;
                continue;
            };
            let (lm, decoder_state) = (&mut generation.lm, &mut generation.decoder_state);
            let (sampling, rng) = (&self.sampling, &mut self.rng);
            let (samples, next) = rayon::join(
                || tts.decode_latent(&latent, decoder_state),
                || lm.advance(&tts.flow_lm, Some(&latent), sampling, rng),
            );
            let mut samples = samples?;
            generation.next = next?;
            if generation.decoded_frames == 0 {
                // A fresh decoder state starts with a small step: fade the first 5 ms in.
                let n = (tts.sample_rate() as usize / 200).min(samples.len());
                for (i, sample) in samples[..n].iter_mut().enumerate() {
                    *sample *= i as f32 / (n.max(2) - 1) as f32;
                }
            }
            generation.decoded_frames += 1;
            return Ok(Some(Audio::new(samples, tts.sample_rate())));
        }
    }
}

impl Iterator for SpeechStream<'_> {
    type Item = Result<Audio>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_frame() {
            Ok(frame) => frame.map(Ok),
            Err(err) => {
                // Stop after an error.
                self.chunks.clear();
                self.current = None;
                Some(Err(err))
            }
        }
    }
}

/// Loads a standalone flow LM checkpoint (training layout) with the names of the bundles.
fn load_flow_lm_weights(path: &Path, dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
    let tensors: HashMap<String, Tensor> = candle_core::safetensors::load(path, device)?
        .into_iter()
        .filter_map(|(name, tensor)| {
            let name = match name.as_str() {
                "condition_provider.conditioners.transcript_in_segment.embed.weight" => {
                    "conditioner.embed.weight".to_string()
                }
                "condition_provider.conditioners.speaker_wavs.output_proj.weight" => {
                    "speaker_proj_weight".to_string()
                }
                "fuser.padding_value" => "bos_before_voice".to_string(),
                n if n.starts_with("flow.w_s_t.") || n.ends_with("learnt_padding") => return None,
                n => n.replace(".self_attn.in_proj_weight", ".self_attn.in_proj.weight"),
            };
            Some((name, tensor))
        })
        .collect();
    Ok(VarBuilder::from_tensors(tensors, dtype, device))
}
