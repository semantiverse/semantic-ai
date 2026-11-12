//! The Pocket TTS flow language model: a causal transformer over text tokens, voice
//! conditioning and audio latents, with a small flow-matching head sampling the next latent.

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use super::config::{FlowKind, FlowLmConfig};
use crate::nn::Linear;
use crate::nn::kernels::Activation;
use crate::nn::transformer::{FeedForward, Norm, Transformer, TransformerConfig, TransformerState};

/// Sampling parameters of one generated frame.
#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    pub temperature: f32,
    /// Clamp of the noise, in standard deviations of the unit normal; none if `None`.
    pub noise_clamp: Option<f32>,
    pub decode_steps: usize,
    pub eos_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct FlowLm {
    text_embedding: Embedding,
    text_projection: Option<Linear>,
    input_linear: Linear,
    transformer: Transformer,
    out_norm: (Tensor, Tensor),
    z_proj: Option<Linear>,
    out_eos: Linear,
    head: FlowHead,
    bos: Tensor,
    bos_before_voice: Option<Tensor>,
    speaker_projection: Option<Linear>,
    /// Latent normalization: the codec consumes `latent * std + mean`.
    pub emb_mean: Tensor,
    pub emb_std: Tensor,
    latent_dim: usize,
    device: Device,
}

impl FlowLm {
    pub fn new(
        cfg: &FlowLmConfig,
        latent_dim: usize,
        voice_dim: usize,
        quantize: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let t = &cfg.transformer;
        let d_model = t.d_model;
        let attn_dim = t.attn_dim.unwrap_or(d_model);
        let num_kv_heads = t.num_kv_heads.unwrap_or(t.num_heads);
        let transformer_cfg = TransformerConfig {
            d_model,
            num_layers: t.num_layers,
            num_heads: t.num_heads,
            num_kv_heads,
            head_dim: attn_dim / t.num_heads,
            feed_forward: FeedForward::Plain {
                hidden: t.dim_feedforward.unwrap_or(d_model * t.hidden_scale),
                exact_gelu: false,
            },
            norm: Norm::LayerNorm { eps: 1e-5 },
            layer_scale: false,
            context: None,
            max_period: t.max_period,
            quantize,
        };
        let linear = |vb: VarBuilder, out: usize, inp: usize| -> Result<Linear> {
            Ok(Linear::new(vb.get((out, inp), "weight")?, None))
        };

        let lut = &cfg.lookup_table;
        let table_width = lut.rank.unwrap_or(lut.dim);
        let text_embedding = Embedding::new(
            vb.pp("conditioner.embed")
                .get((lut.n_bins + 1, table_width), "weight")?,
            table_width,
        );
        let text_projection = lut
            .rank
            .map(|rank| linear(vb.pp("conditioner.proj"), lut.dim, rank))
            .transpose()?;

        let z_dim = cfg.z_dim.unwrap_or(d_model);
        let out_eos = vb.pp("out_eos");
        let speaker_projection = vb
            .contains_tensor("speaker_proj_weight")
            .then(|| {
                vb.get((d_model, voice_dim), "speaker_proj_weight")
                    .map(|w| Linear::new(w, None))
            })
            .transpose()?;
        Ok(Self {
            text_embedding,
            text_projection,
            input_linear: linear(vb.pp("input_linear"), d_model, latent_dim)?,
            transformer: Transformer::new(&transformer_cfg, vb.pp("transformer"))?,
            out_norm: (
                vb.pp("out_norm").get(d_model, "weight")?,
                vb.pp("out_norm").get(d_model, "bias")?,
            ),
            z_proj: cfg
                .z_dim
                .map(|z| linear(vb.pp("z_proj"), z, d_model))
                .transpose()?,
            out_eos: Linear::new(
                out_eos.get((1, z_dim), "weight")?,
                Some(out_eos.get(1, "bias")?),
            ),
            head: FlowHead::new(cfg, latent_dim, z_dim, vb.pp("flow_net"))?,
            bos: vb.get(latent_dim, "bos_emb")?,
            bos_before_voice: cfg
                .insert_bos_before_voice
                .then(|| vb.get((1, 1, d_model), "bos_before_voice"))
                .transpose()?,
            speaker_projection,
            emb_mean: vb.get(latent_dim, "emb_mean")?,
            emb_std: vb.get(latent_dim, "emb_std")?,
            latent_dim,
            device: vb.device().clone(),
        })
    }

    pub fn init_state(&self) -> TransformerState {
        self.transformer.init_state()
    }

    pub fn supports_voice_cloning(&self) -> bool {
        self.speaker_projection.is_some()
    }

    /// Feeds encoded voice latents (`[1, frames, voice_dim]`) to the model.
    pub fn prompt_voice(&self, latents: &Tensor, state: &mut TransformerState) -> Result<()> {
        let projection = self.speaker_projection.as_ref().ok_or_else(|| {
            candle_core::Error::msg("the model weights have no speaker projection")
        })?;
        let mut conditioning = projection.forward(latents)?;
        if let Some(bos) = &self.bos_before_voice {
            conditioning = Tensor::cat(&[bos, &conditioning], 1)?;
        }
        self.transformer.forward(&conditioning, state)?;
        Ok(())
    }

    /// Feeds text tokens to the model.
    pub fn prompt_text(&self, tokens: &[u32], state: &mut TransformerState) -> Result<()> {
        let tokens = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let mut embeddings = self.text_embedding.forward(&tokens)?;
        if let Some(projection) = &self.text_projection {
            embeddings = projection.forward(&embeddings)?;
        }
        self.transformer.forward(&embeddings, state)?;
        Ok(())
    }

    /// Runs one autoregressive step on the previous latent (`None` at the start of speech).
    /// Returns the next latent (`[1, 1, latent_dim]`) and whether the model signals the end.
    pub fn step(
        &self,
        previous: Option<&Tensor>,
        state: &mut TransformerState,
        sampling: &Sampling,
        rng: &mut impl Rng,
    ) -> Result<(Tensor, bool)> {
        let input = match previous {
            Some(latent) => latent.clone(),
            None => self.bos.reshape((1, 1, self.latent_dim))?,
        };
        let xs = self.input_linear.forward(&input)?;
        let xs = self.transformer.forward(&xs, state)?;
        let (weight, bias) = &self.out_norm;
        let mut z = candle_nn::ops::layer_norm(&xs, weight, bias, 1e-5)?;
        if let Some(z_proj) = &self.z_proj {
            z = z_proj.forward(&z)?;
        }
        let z = z.squeeze(1)?;
        let eos_logit: f32 = self.out_eos.forward(&z)?.flatten_all()?.to_vec1::<f32>()?[0];
        let noise = self.sample_noise(sampling, rng)?;
        let latent = self.head.decode(&z, noise, sampling.decode_steps)?;
        Ok((latent.unsqueeze(1)?, eos_logit > sampling.eos_threshold))
    }

    fn sample_noise(&self, sampling: &Sampling, rng: &mut impl Rng) -> Result<Tensor> {
        let std = sampling.temperature.max(0.0).sqrt();
        let noise: Vec<f32> = (0..self.latent_dim)
            .map(|_| {
                loop {
                    let x: f32 = StandardNormal.sample(rng);
                    let x = x * std;
                    // Truncated normal by rejection, as `torch.nn.init.trunc_normal_`.
                    match sampling.noise_clamp {
                        Some(clamp) if x.abs() > clamp => continue,
                        _ => break x,
                    }
                }
            })
            .collect();
        Tensor::from_vec(noise, (1, self.latent_dim), &self.device)
    }
}

/// `SimpleMLPAdaLN`: an MLP with adaptive layer norm, conditioned on the backbone output and on
/// zero, one or two flow times.
#[derive(Debug, Clone)]
struct FlowHead {
    kind: FlowKind,
    time_embeddings: Vec<TimestepEmbedder>,
    cond_embed: Linear,
    input_proj: Linear,
    res_blocks: Vec<ResBlock>,
    final_modulation: Linear,
    final_linear: Linear,
}

fn linear_b(vb: VarBuilder, out: usize, inp: usize) -> Result<Linear> {
    Ok(Linear::new(
        vb.get((out, inp), "weight")?,
        Some(vb.get(out, "bias")?),
    ))
}

fn layer_norm_no_affine(xs: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = xs.mean_keepdim(D::Minus1)?;
    let centered = xs.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    centered.broadcast_div(&(var + eps)?.sqrt()?)
}

/// `x * (1 + scale) + shift`
fn modulate(xs: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    xs.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
}

impl FlowHead {
    fn new(cfg: &FlowLmConfig, latent_dim: usize, cond_dim: usize, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.flow.dim;
        let time_embeddings = (0..cfg.flow.kind.num_time_conditions())
            .map(|i| TimestepEmbedder::new(dim, vb.pp("time_embed").pp(i)))
            .collect::<Result<_>>()?;
        let res_blocks = (0..cfg.flow.depth)
            .map(|i| ResBlock::new(dim, vb.pp("res_blocks").pp(i)))
            .collect::<Result<_>>()?;
        let vb_final = vb.pp("final_layer");
        Ok(Self {
            kind: cfg.flow.kind,
            time_embeddings,
            cond_embed: linear_b(vb.pp("cond_embed"), dim, cond_dim)?,
            input_proj: linear_b(vb.pp("input_proj"), dim, latent_dim)?,
            res_blocks,
            final_modulation: linear_b(vb_final.pp("adaLN_modulation").pp(1), 2 * dim, dim)?,
            final_linear: linear_b(vb_final.pp("linear"), latent_dim, dim)?,
        })
    }

    fn forward(&self, cond: &Tensor, times: &[f32], xs: &Tensor) -> Result<Tensor> {
        let mut y = self.cond_embed.forward(cond)?;
        if !self.time_embeddings.is_empty() {
            let mut t_sum: Option<Tensor> = None;
            for (embedder, &t) in self.time_embeddings.iter().zip(times) {
                let e = embedder.forward(t, cond.device())?;
                t_sum = Some(match t_sum {
                    Some(sum) => (sum + e)?,
                    None => e,
                });
            }
            let t_mean = (t_sum.expect("non-empty") / self.time_embeddings.len() as f64)?;
            y = y.broadcast_add(&t_mean)?;
        }
        let mut xs = self.input_proj.forward(xs)?;
        for block in &self.res_blocks {
            xs = block.forward(&xs, &y)?;
        }
        let modulation = self
            .final_modulation
            .forward(&Activation::Silu.apply(&y)?)?
            .chunk(2, D::Minus1)?;
        let xs = modulate(
            &layer_norm_no_affine(&xs, 1e-6)?,
            &modulation[0],
            &modulation[1],
        )?;
        self.final_linear.forward(&xs)
    }

    /// Turns noise into a latent, conditioned on the backbone output `cond`.
    fn decode(&self, cond: &Tensor, noise: Tensor, steps: usize) -> Result<Tensor> {
        let steps = steps.max(1);
        match self.kind {
            FlowKind::Lsd => {
                let mut x = noise;
                for i in 0..steps {
                    let s = i as f32 / steps as f32;
                    let t = (i + 1) as f32 / steps as f32;
                    let flow = self.forward(cond, &[s, t], &x)?;
                    x = (x + (flow / steps as f64)?)?;
                }
                Ok(x)
            }
            FlowKind::FlowMatching => {
                let mut x = noise;
                for i in 0..steps {
                    let t = i as f32 / steps as f32;
                    let flow = self.forward(cond, &[t], &x)?;
                    x = (x + (flow / steps as f64)?)?;
                }
                Ok(x)
            }
            FlowKind::Drifting => self.forward(cond, &[], &noise),
            FlowKind::Pointwise => &noise + self.forward(cond, &[], &noise)?,
        }
    }
}

#[derive(Debug, Clone)]
struct TimestepEmbedder {
    freqs: Tensor,
    linear1: Linear,
    linear2: Linear,
    norm_alpha: Tensor,
}

impl TimestepEmbedder {
    const FREQUENCY_DIM: usize = 256;

    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let mlp = vb.pp("mlp");
        Ok(Self {
            freqs: vb.get(Self::FREQUENCY_DIM / 2, "freqs")?,
            linear1: linear_b(mlp.pp(0), dim, Self::FREQUENCY_DIM)?,
            linear2: linear_b(mlp.pp(2), dim, dim)?,
            norm_alpha: mlp.pp(3).get(dim, "alpha")?,
        })
    }

    fn forward(&self, t: f32, device: &Device) -> Result<Tensor> {
        let args = (self.freqs.to_device(device)? * t as f64)?;
        let embedding = Tensor::cat(&[args.cos()?, args.sin()?], 0)?.unsqueeze(0)?;
        let xs = self
            .linear2
            .forward(&Activation::Silu.apply(&self.linear1.forward(&embedding)?)?)?;
        // This RMSNorm divides by the (unbiased) variance, not the mean square.
        let n = xs.dim(D::Minus1)? as f64;
        let centered = xs.broadcast_sub(&xs.mean_keepdim(D::Minus1)?)?;
        let var = (centered.sqr()?.sum_keepdim(D::Minus1)? / (n - 1.0))?;
        xs.broadcast_mul(&(var + 1e-5)?.sqrt()?.recip()?)?
            .broadcast_mul(&self.norm_alpha)
    }
}

#[derive(Debug, Clone)]
struct ResBlock {
    norm: (Tensor, Tensor),
    mlp1: Linear,
    mlp2: Linear,
    modulation: Linear,
}

impl ResBlock {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let ln = vb.pp("in_ln");
        Ok(Self {
            norm: (ln.get(dim, "weight")?, ln.get(dim, "bias")?),
            mlp1: linear_b(vb.pp("mlp").pp(0), dim, dim)?,
            mlp2: linear_b(vb.pp("mlp").pp(2), dim, dim)?,
            modulation: linear_b(vb.pp("adaLN_modulation").pp(1), 3 * dim, dim)?,
        })
    }

    fn forward(&self, xs: &Tensor, y: &Tensor) -> Result<Tensor> {
        let m = self
            .modulation
            .forward(&Activation::Silu.apply(y)?)?
            .chunk(3, D::Minus1)?;
        let (shift, scale, gate) = (&m[0], &m[1], &m[2]);
        let (weight, bias) = &self.norm;
        let h = candle_nn::ops::layer_norm(xs, weight, bias, 1e-6)?;
        let h = modulate(&h, shift, scale)?;
        let h = self
            .mlp2
            .forward(&Activation::Silu.apply(&self.mlp1.forward(&h)?)?)?;
        xs + gate.broadcast_mul(&h)?
    }
}

/// The weights dtype of a config's `dtype` field.
pub(crate) fn parse_dtype(name: &str) -> DType {
    match name {
        "bfloat16" => DType::BF16,
        "float16" => DType::F16,
        _ => DType::F32,
    }
}
