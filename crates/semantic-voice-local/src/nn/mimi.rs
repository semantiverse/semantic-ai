//! The Mimi neural audio codec (without quantizer): SEANet plus a transformer on each side, and
//! a learnt resampling between the SEANet frame rate and the codec frame rate.
//!
//! Pocket TTS and Kyutai STT use different Mimi variants; both are described by [`MimiConfig`].

use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

use super::Linear;
use super::conv::{
    Conv1dConfig, Conv1dState, ConvTranspose1dState, PadMode, StreamingConv1d,
    StreamingConvTranspose1d,
};
use super::seanet::{SeanetConfig, SeanetDecoder, SeanetEncoder, SeanetState};
use super::transformer::{FeedForward, Norm, Transformer, TransformerConfig, TransformerState};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ProjectedTransformerConfig {
    pub d_model: usize,
    pub input_dimension: usize,
    pub output_dimensions: Vec<usize>,
    pub num_heads: usize,
    pub num_layers: usize,
    pub layer_scale: f64,
    pub context: usize,
    #[serde(default = "default_max_period")]
    pub max_period: f64,
    pub dim_feedforward: usize,
}

fn default_max_period() -> f64 {
    10_000.0
}

#[derive(Debug, Clone, PartialEq)]
pub struct MimiConfig {
    pub sample_rate: u32,
    pub frame_rate: f64,
    pub encoder: SeanetConfig,
    pub decoder: SeanetConfig,
    pub encoder_transformer: ProjectedTransformerConfig,
    pub decoder_transformer: ProjectedTransformerConfig,
    /// Channels after downsampling to the codec frame rate.
    pub inner_dim: usize,
    /// Channels before upsampling from the codec frame rate.
    pub outer_dim: usize,
    /// Erf GELU in the transformers (moshi); the tanh approximation otherwise (Pocket TTS).
    pub exact_gelu: bool,
}

impl MimiConfig {
    /// Audio samples per codec frame.
    pub fn frame_size(&self) -> usize {
        (self.sample_rate as f64 / self.frame_rate).round() as usize
    }

    /// SEANet frames per codec frame.
    fn resample_stride(&self, seanet: &SeanetConfig) -> usize {
        let seanet_rate = self.sample_rate as f64 / seanet.hop_length() as f64;
        (seanet_rate / self.frame_rate).round() as usize
    }
}

/// The transformer between SEANet and the resampling, in channel-first layout.
#[derive(Debug, Clone)]
struct ProjectedTransformer {
    input_proj: Option<Linear>,
    transformer: Transformer,
    output_proj: Option<Linear>,
}

impl ProjectedTransformer {
    fn new(cfg: &ProjectedTransformerConfig, exact_gelu: bool, vb: VarBuilder) -> Result<Self> {
        let [output_dim] = cfg.output_dimensions[..] else {
            candle_core::bail!("mimi transformers must have exactly one output");
        };
        let linear = |vb: VarBuilder, out: usize, inp: usize| -> Result<Linear> {
            Ok(Linear::new(vb.get((out, inp), "weight")?, None))
        };
        let input_proj = (cfg.input_dimension != cfg.d_model)
            .then(|| linear(vb.pp("input_proj"), cfg.d_model, cfg.input_dimension))
            .transpose()?;
        let output_proj = (output_dim != cfg.d_model)
            .then(|| linear(vb.pp("output_projs").pp(0), output_dim, cfg.d_model))
            .transpose()?;
        let transformer_cfg = TransformerConfig {
            d_model: cfg.d_model,
            num_layers: cfg.num_layers,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_heads,
            head_dim: cfg.d_model / cfg.num_heads,
            feed_forward: FeedForward::Plain {
                hidden: cfg.dim_feedforward,
                exact_gelu,
            },
            norm: Norm::LayerNorm { eps: 1e-5 },
            layer_scale: true,
            context: Some(cfg.context),
            max_period: cfg.max_period,
            quantize: false,
        };
        let transformer = Transformer::new(&transformer_cfg, vb.pp("transformer"))?;
        Ok(Self {
            input_proj,
            transformer,
            output_proj,
        })
    }

    fn forward(&self, xs: &Tensor, state: &mut TransformerState) -> Result<Tensor> {
        let mut xs = xs.transpose(1, 2)?.contiguous()?;
        if let Some(proj) = &self.input_proj {
            xs = proj.forward(&xs)?;
        }
        let mut ys = self.transformer.forward(&xs, state)?;
        if let Some(proj) = &self.output_proj {
            ys = proj.forward(&ys)?;
        }
        ys.transpose(1, 2)
    }
}

/// Waveform to unquantized latents at the codec frame rate.
#[derive(Debug, Clone)]
pub struct MimiEncoder {
    seanet: SeanetEncoder,
    transformer: ProjectedTransformer,
    downsample: Option<StreamingConv1d>,
    frame_size: usize,
}

#[derive(Debug, Clone)]
pub struct MimiEncoderState {
    seanet: SeanetState,
    transformer: TransformerState,
    downsample: Conv1dState,
}

impl MimiEncoder {
    /// Loads `encoder`, `encoder_transformer` and `downsample` under `vb`.
    pub fn new(cfg: &MimiConfig, vb: VarBuilder) -> Result<Self> {
        let stride = cfg.resample_stride(&cfg.encoder);
        let downsample = (stride > 1)
            .then(|| {
                StreamingConv1d::new(
                    cfg.encoder.dimension,
                    cfg.inner_dim,
                    Conv1dConfig::new(2 * stride)
                        .stride(stride)
                        .no_bias()
                        .pad_mode(PadMode::Replicate),
                    vb.pp("downsample").pp("conv"),
                )
            })
            .transpose()?;
        Ok(Self {
            seanet: SeanetEncoder::new(&cfg.encoder, vb.pp("encoder"))?,
            transformer: ProjectedTransformer::new(
                &cfg.encoder_transformer,
                cfg.exact_gelu,
                vb.pp("encoder_transformer"),
            )?,
            downsample,
            frame_size: cfg.frame_size(),
        })
    }

    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    pub fn init_state(&self) -> MimiEncoderState {
        MimiEncoderState {
            seanet: self.seanet.init_state(),
            transformer: self.transformer.transformer.init_state(),
            downsample: Conv1dState::default(),
        }
    }

    /// Encodes `[batch, channels, samples]`, `samples` being a multiple of the frame size, into
    /// `[batch, dim, frames]`, continuing the stream in `state`.
    pub fn encode_step(&self, xs: &Tensor, state: &mut MimiEncoderState) -> Result<Tensor> {
        let xs = self.seanet.forward(xs, &mut state.seanet)?;
        let xs = self.transformer.forward(&xs, &mut state.transformer)?;
        match &self.downsample {
            Some(conv) => conv.forward(&xs, &mut state.downsample),
            None => Ok(xs),
        }
    }

    /// Encodes a whole waveform, zero-padded to a multiple of the frame size.
    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        /// Codec frames processed at once, bounding memory use on long inputs.
        const FRAMES_PER_CHUNK: usize = 64;
        let len = xs.dim(2)?;
        let padded = len.div_ceil(self.frame_size).max(1) * self.frame_size;
        let xs = xs.pad_with_zeros(2, 0, padded - len)?;
        let mut state = self.init_state();
        let chunk = FRAMES_PER_CHUNK * self.frame_size;
        let frames = (0..padded)
            .step_by(chunk)
            .map(|start| {
                self.encode_step(&xs.narrow(2, start, chunk.min(padded - start))?, &mut state)
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&frames, 2)
    }
}

/// Unquantized latents at the codec frame rate to waveform.
#[derive(Debug, Clone)]
pub struct MimiDecoder {
    upsample: Option<StreamingConvTranspose1d>,
    transformer: ProjectedTransformer,
    seanet: SeanetDecoder,
}

#[derive(Debug, Clone)]
pub struct MimiDecoderState {
    upsample: ConvTranspose1dState,
    transformer: TransformerState,
    seanet: SeanetState,
}

impl MimiDecoder {
    /// Loads `upsample`, `decoder_transformer` and `decoder` under `vb`.
    pub fn new(cfg: &MimiConfig, vb: VarBuilder) -> Result<Self> {
        let stride = cfg.resample_stride(&cfg.decoder);
        let dim = cfg.decoder.dimension;
        let upsample = (stride > 1)
            .then(|| {
                StreamingConvTranspose1d::new(
                    cfg.outer_dim,
                    dim,
                    2 * stride,
                    stride,
                    dim,
                    false,
                    vb.pp("upsample").pp("convtr"),
                )
            })
            .transpose()?;
        Ok(Self {
            upsample,
            transformer: ProjectedTransformer::new(
                &cfg.decoder_transformer,
                cfg.exact_gelu,
                vb.pp("decoder_transformer"),
            )?,
            seanet: SeanetDecoder::new(&cfg.decoder, vb.pp("decoder"))?,
        })
    }

    pub fn init_state(&self) -> MimiDecoderState {
        MimiDecoderState {
            upsample: ConvTranspose1dState::default(),
            transformer: self.transformer.transformer.init_state(),
            seanet: self.seanet.init_state(),
        }
    }

    /// Decodes `[batch, outer_dim, frames]` into `[batch, channels, samples]`.
    pub fn decode_step(&self, xs: &Tensor, state: &mut MimiDecoderState) -> Result<Tensor> {
        let xs = match &self.upsample {
            Some(conv) => conv.forward(xs, &mut state.upsample)?,
            None => xs.clone(),
        };
        let xs = self.transformer.forward(&xs, &mut state.transformer)?;
        self.seanet.forward(&xs, &mut state.seanet)
    }
}
