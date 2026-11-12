//! Kyutai STT networks: the Mimi encoder with its residual vector quantizer, and the
//! language model predicting text tokens from audio tokens. Ported from Kyutai's `moshi`
//! crate (MIT/Apache-2.0).

use candle_core::{D, IndexOp, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Embedding, VarBuilder};
use serde::Deserialize;

use crate::nn::Linear;
use crate::nn::conv::PadMode;
use crate::nn::mimi::{MimiConfig, ProjectedTransformerConfig};
use crate::nn::seanet::SeanetConfig;
use crate::nn::transformer::{FeedForward, Norm, Transformer, TransformerConfig, TransformerState};

/// `config.json` of the Kyutai STT model repositories.
#[derive(Debug, Clone, Deserialize)]
pub struct SttConfig {
    /// Audio codebook size.
    pub card: usize,
    /// Audio codebooks fed to the model.
    pub n_q: usize,
    pub dim: usize,
    pub text_card: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub hidden_scale: f64,
    pub context: usize,
    pub max_period: f64,
    pub extra_heads_num_heads: Option<usize>,
    pub extra_heads_dim: Option<usize>,
    pub stt_config: StreamingConfig,
    pub mimi_name: String,
    pub tokenizer_name: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamingConfig {
    /// How far the text stream lags behind the audio.
    pub audio_delay_seconds: f64,
    /// Silence fed before the audio, as during training.
    #[serde(default)]
    pub audio_silence_prefix_seconds: f64,
}

/// Mimi as trained for moshi (`mimi-pytorch-e351c8d8@125`).
pub(crate) fn mimi_config() -> MimiConfig {
    let seanet = SeanetConfig {
        dimension: 512,
        channels: 1,
        n_filters: 64,
        n_residual_layers: 1,
        ratios: vec![8, 6, 5, 4],
        kernel_size: 7,
        residual_kernel_size: 3,
        last_kernel_size: 3,
        dilation_base: 2,
        pad_mode: PadMode::Constant,
        compress: 2,
    };
    let transformer = ProjectedTransformerConfig {
        d_model: 512,
        input_dimension: 512,
        output_dimensions: vec![512],
        num_heads: 8,
        num_layers: 8,
        layer_scale: 0.01,
        context: 250,
        max_period: 10_000.0,
        dim_feedforward: 2048,
    };
    MimiConfig {
        sample_rate: 24_000,
        frame_rate: 12.5,
        encoder: seanet.clone(),
        decoder: seanet,
        encoder_transformer: transformer.clone(),
        decoder_transformer: transformer,
        inner_dim: 512,
        outer_dim: 512,
        exact_gelu: true,
    }
}

/// Mimi's split residual vector quantizer, encoding only: one semantic codebook, then the
/// acoustic codebooks quantizing the same input residually.
#[derive(Debug, Clone)]
pub(crate) struct Quantizer {
    first: ResidualQuantizer,
    rest: ResidualQuantizer,
}

impl Quantizer {
    pub fn new(n_q: usize, bins: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            first: ResidualQuantizer::new(1, bins, vb.pp("rvq_first"))?,
            rest: ResidualQuantizer::new(n_q - 1, bins, vb.pp("rvq_rest"))?,
        })
    }

    /// `[batch, 512, frames]` latents to `[batch, frames, n_q]` codes.
    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        Tensor::cat(&[self.first.encode(xs)?, self.rest.encode(xs)?], 2)
    }
}

#[derive(Debug, Clone)]
struct ResidualQuantizer {
    input_proj: Conv1d,
    codebooks: Vec<Codebook>,
}

impl ResidualQuantizer {
    const DIM: usize = 256;

    fn new(n_q: usize, bins: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.pp("input_proj").get((Self::DIM, 512, 1), "weight")?;
        let codebooks = (0..n_q)
            .map(|i| Codebook::new(bins, vb.pp("vq.layers").pp(i).pp("_codebook")))
            .collect::<Result<_>>()?;
        Ok(Self {
            input_proj: Conv1d::new(weight, None, Conv1dConfig::default()),
            codebooks,
        })
    }

    fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let mut residual = self.input_proj.forward(xs)?.transpose(1, 2)?.contiguous()?;
        let mut codes = Vec::with_capacity(self.codebooks.len());
        for codebook in &self.codebooks {
            let indices = codebook.encode(&residual)?;
            residual = (residual - codebook.decode(&indices)?)?;
            codes.push(indices);
        }
        Tensor::stack(&codes, 2)
    }
}

#[derive(Debug, Clone)]
struct Codebook {
    embedding: Tensor,
    /// Half squared norms of the entries.
    half_norms: Tensor,
}

impl Codebook {
    fn new(bins: usize, vb: VarBuilder) -> Result<Self> {
        let usage = vb.get(bins, "cluster_usage")?.maximum(1e-5)?.unsqueeze(1)?;
        let embedding = vb
            .get((bins, ResidualQuantizer::DIM), "embedding_sum")?
            .broadcast_div(&usage)?;
        let half_norms = (embedding.sqr()?.sum(D::Minus1)? / 2.0)?;
        Ok(Self {
            embedding,
            half_norms,
        })
    }

    /// Nearest entries of `[batch, frames, dim]` vectors.
    fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, d) = xs.dims3()?;
        let dot = xs.reshape((b * t, d))?.matmul(&self.embedding.t()?)?;
        self.half_norms
            .broadcast_sub(&dot)?
            .argmin(D::Minus1)?
            .reshape((b, t))
    }

    fn decode(&self, indices: &Tensor) -> Result<Tensor> {
        let (b, t) = indices.dims2()?;
        self.embedding
            .index_select(&indices.flatten_all()?, 0)?
            .reshape((b, t, ()))
    }
}

/// The language model: per step, the previous text token and the audio tokens in, the next
/// text token's logits out.
#[derive(Debug, Clone)]
pub(crate) struct LanguageModel {
    text_embedding: Embedding,
    audio_embeddings: Vec<Embedding>,
    transformer: Transformer,
    out_norm: Tensor,
    text_linear: Linear,
    extra_heads: Vec<Linear>,
}

/// What the language model predicts for one step.
pub(crate) struct Prediction {
    pub text_token: u32,
    /// Per extra head, the probability of its first class (for the VAD heads: no speech).
    pub extra_head_probabilities: Vec<f32>,
}

impl LanguageModel {
    pub fn new(cfg: &SttConfig, quantize: bool, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.dim;
        let embedding = |vb: VarBuilder, n: usize| -> Result<Embedding> {
            Ok(Embedding::new(vb.get((n, dim), "weight")?, dim))
        };
        let feed_forward = (cfg.hidden_scale * dim as f64) as usize;
        let transformer_cfg = TransformerConfig {
            d_model: dim,
            num_layers: cfg.num_layers,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_heads,
            head_dim: dim / cfg.num_heads,
            feed_forward: FeedForward::GatedSilu {
                hidden: 2 * feed_forward / 3,
            },
            norm: Norm::RmsNorm { eps: 1e-8 },
            layer_scale: false,
            context: Some(cfg.context),
            max_period: cfg.max_period,
            quantize,
        };
        let extra_heads = match (cfg.extra_heads_num_heads, cfg.extra_heads_dim) {
            (Some(n), Some(head_dim)) => (0..n)
                .map(|i| Linear::load(vb.pp("extra_heads").pp(i), head_dim, dim, false))
                .collect::<Result<_>>()?,
            _ => Vec::new(),
        };
        Ok(Self {
            text_embedding: embedding(vb.pp("text_emb"), cfg.text_card + 1)?,
            audio_embeddings: (0..cfg.n_q)
                .map(|i| embedding(vb.pp("emb").pp(i), cfg.card + 1))
                .collect::<Result<_>>()?,
            transformer: Transformer::new(&transformer_cfg, vb.pp("transformer"))?,
            out_norm: vb.pp("out_norm").get((1, 1, dim), "alpha")?.reshape(dim)?,
            text_linear: Linear::load(vb.pp("text_linear"), cfg.text_card, dim, quantize)?,
            extra_heads,
        })
    }

    pub fn init_state(&self) -> TransformerState {
        self.transformer.init_state()
    }

    pub fn step(
        &self,
        text_token: u32,
        audio_tokens: &[u32],
        state: &mut TransformerState,
    ) -> Result<Prediction> {
        let device = self.out_norm.device();
        let token = |t: u32| Tensor::new(&[[t]], device);
        let mut xs = self.text_embedding.forward(&token(text_token)?)?;
        for (embedding, &t) in self.audio_embeddings.iter().zip(audio_tokens) {
            xs = (xs + embedding.forward(&token(t)?)?)?;
        }
        let ys = self.transformer.forward(&xs, state)?;
        let ys = candle_nn::ops::rms_norm(&ys, &self.out_norm, 1e-8)?;
        let text_token = self
            .text_linear
            .forward(&ys)?
            .i((0, 0))?
            .argmax(D::Minus1)?
            .to_scalar::<u32>()?;
        let extra_head_probabilities = self
            .extra_heads
            .iter()
            .map(|head| {
                let logits = head
                    .forward(&ys)?
                    .i((0, 0))?
                    .to_dtype(candle_core::DType::F32)?;
                candle_nn::ops::softmax_last_dim(&logits)?
                    .i(0)?
                    .to_scalar::<f32>()
            })
            .collect::<Result<_>>()?;
        Ok(Prediction {
            text_token,
            extra_head_probabilities,
        })
    }
}
