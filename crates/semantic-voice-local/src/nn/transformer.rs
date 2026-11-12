//! A causal streaming transformer covering the variants used by Kyutai's models: LayerNorm or
//! RMSNorm, plain or gated feed-forward, optional layer scale, grouped-query attention, RoPE and
//! an optional attention window (`context`).

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use rayon::prelude::*;

use super::Linear;
use super::kernels::{self, Activation};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Norm {
    LayerNorm { eps: f32 },
    RmsNorm { eps: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeedForward {
    /// `linear2(gelu(linear1(x)))`; `exact_gelu` selects erf GELU over the tanh approximation.
    Plain { hidden: usize, exact_gelu: bool },
    /// `linear_out(silu(a) * b)` where `[a, b] = linear_in(x)`.
    GatedSilu { hidden: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransformerConfig {
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub feed_forward: FeedForward,
    pub norm: Norm,
    pub layer_scale: bool,
    /// How many past steps a query may attend to, unlimited when `None`.
    pub context: Option<usize>,
    pub max_period: f64,
    /// Q8_0 weights for the linear layers, on CPU.
    pub quantize: bool,
}

#[derive(Debug, Clone)]
pub struct Transformer {
    layers: Vec<Layer>,
    num_heads: usize,
    context: Option<usize>,
    inv_freq: Tensor,
}

/// Streaming state: the attention caches and the number of steps processed so far.
#[derive(Debug, Clone)]
pub struct TransformerState {
    caches: Vec<KvCache>,
    position: usize,
}

impl TransformerState {
    pub fn num_layers(&self) -> usize {
        self.caches.len()
    }

    /// Deep copy, so that the copy and `self` can be extended independently.
    pub fn duplicate(&self) -> Result<Self> {
        Ok(Self {
            caches: self
                .caches
                .iter()
                .map(KvCache::duplicate)
                .collect::<Result<_>>()?,
            position: self.position,
        })
    }

    /// Per layer `(keys, values)` as `[batch, kv_heads, time, head_dim]`.
    pub fn layer_caches(&self) -> Result<Vec<Option<(Tensor, Tensor)>>> {
        self.caches.iter().map(KvCache::current).collect()
    }

    /// Rebuilds a state from per-layer `(keys, values)` covering positions `0..position`.
    pub fn from_layer_caches(
        layers: Vec<(Tensor, Tensor)>,
        context: Option<usize>,
    ) -> Result<Self> {
        let mut position = None;
        let caches = layers
            .into_iter()
            .map(|(k, v)| {
                let len = k.dim(2)?;
                if position.is_some_and(|p| p != len) {
                    candle_core::bail!("inconsistent cache lengths across layers");
                }
                position = Some(len);
                let mut cache = KvCache::new(context);
                cache.append(&k.contiguous()?, &v.contiguous()?)?;
                Ok(cache)
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            caches,
            position: position.unwrap_or(0),
        })
    }
}

impl Transformer {
    pub fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let vb_layers = vb.pp("layers");
        // In parallel: converting and quantizing the weights dominates the loading time.
        let layers = (0..cfg.num_layers)
            .into_par_iter()
            .map(|i| Layer::new(cfg, vb_layers.pp(i)))
            .collect::<Result<_>>()?;
        let half = cfg.head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| (-(cfg.max_period.ln()) * 2.0 * i as f64 / cfg.head_dim as f64).exp() as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), vb.device())?;
        Ok(Self {
            layers,
            num_heads: cfg.num_heads,
            context: cfg.context,
            inv_freq,
        })
    }

    pub fn init_state(&self) -> TransformerState {
        TransformerState {
            caches: self
                .layers
                .iter()
                .map(|_| KvCache::new(self.context))
                .collect(),
            position: 0,
        }
    }

    /// Processes `xs` (`[batch, time, d_model]`) following the steps already in `state`.
    pub fn forward(&self, xs: &Tensor, state: &mut TransformerState) -> Result<Tensor> {
        let t = xs.dim(1)?;
        // Within a window, long inputs are processed in blocks to bound the attention matrix.
        match self.context {
            Some(context) if t > context => {
                let blocks = (0..t)
                    .step_by(context)
                    .map(|start| {
                        self.forward_block(&xs.narrow(1, start, context.min(t - start))?, state)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Tensor::cat(&blocks, 1)
            }
            _ => self.forward_block(xs, state),
        }
    }

    fn forward_block(&self, xs: &Tensor, state: &mut TransformerState) -> Result<Tensor> {
        let (_, t, _) = xs.dims3()?;
        if t == 0 {
            return Ok(xs.clone());
        }
        let positions = Tensor::arange(
            state.position as u32,
            (state.position + t) as u32,
            xs.device(),
        )?
        .to_dtype(DType::F32)?
        .reshape((t, 1))?;
        let freqs = positions.matmul(&self.inv_freq)?;
        let rope = Rope {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        };
        let mut xs = xs.clone();
        for (layer, cache) in self.layers.iter().zip(state.caches.iter_mut()) {
            xs = layer.forward(
                &xs,
                &rope,
                cache,
                state.position,
                self.num_heads,
                self.context,
            )?;
        }
        state.position += t;
        Ok(xs)
    }
}

struct Rope {
    cos: Tensor,
    sin: Tensor,
}

#[derive(Debug, Clone)]
struct Layer {
    norm1: NormLayer,
    norm2: NormLayer,
    attention: Attention,
    feed_forward: FeedForwardLayer,
    layer_scale: Option<(Tensor, Tensor)>,
}

impl Layer {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let layer_scale = if cfg.layer_scale {
            Some((
                vb.pp("layer_scale_1").get(cfg.d_model, "scale")?,
                vb.pp("layer_scale_2").get(cfg.d_model, "scale")?,
            ))
        } else {
            None
        };
        Ok(Self {
            norm1: NormLayer::new(cfg.norm, cfg.d_model, vb.pp("norm1"))?,
            norm2: NormLayer::new(cfg.norm, cfg.d_model, vb.pp("norm2"))?,
            attention: Attention::new(cfg, vb.pp("self_attn"))?,
            feed_forward: FeedForwardLayer::new(cfg, vb.clone())?,
            layer_scale,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        rope: &Rope,
        cache: &mut KvCache,
        position: usize,
        num_heads: usize,
        context: Option<usize>,
    ) -> Result<Tensor> {
        let update = self.attention.forward(
            &self.norm1.forward(xs)?,
            rope,
            cache,
            position,
            num_heads,
            context,
        )?;
        let update = match &self.layer_scale {
            Some((scale, _)) => update.broadcast_mul(scale)?,
            None => update,
        };
        let xs = (xs + update)?;
        let update = self.feed_forward.forward(&self.norm2.forward(&xs)?)?;
        let update = match &self.layer_scale {
            Some((_, scale)) => update.broadcast_mul(scale)?,
            None => update,
        };
        xs + update
    }
}

#[derive(Debug, Clone)]
enum NormLayer {
    LayerNorm {
        weight: Tensor,
        bias: Tensor,
        eps: f32,
    },
    RmsNorm {
        alpha: Tensor,
        eps: f32,
    },
}

impl NormLayer {
    fn new(norm: Norm, dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(match norm {
            Norm::LayerNorm { eps } => Self::LayerNorm {
                weight: vb.get(dim, "weight")?,
                bias: vb.get(dim, "bias")?,
                eps,
            },
            Norm::RmsNorm { eps } => Self::RmsNorm {
                alpha: vb.get((1, 1, dim), "alpha")?.reshape(dim)?,
                eps,
            },
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::LayerNorm { weight, bias, eps } => {
                candle_nn::ops::layer_norm(xs, weight, bias, *eps)
            }
            Self::RmsNorm { alpha, eps } => candle_nn::ops::rms_norm(xs, alpha, *eps),
        }
    }
}

#[derive(Debug, Clone)]
struct Attention {
    in_proj: Linear,
    out_proj: Linear,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let shape = (q_dim + 2 * kv_dim, cfg.d_model);
        // Pocket TTS checkpoints use `in_proj.weight`, moshi ones `in_proj_weight`.
        let in_proj = if vb.contains_tensor("in_proj_weight") {
            vb.get(shape, "in_proj_weight")?
        } else {
            vb.pp("in_proj").get(shape, "weight")?
        };
        Ok(Self {
            in_proj: Linear::from_weight(in_proj, cfg.quantize)?,
            out_proj: Linear::load(vb.pp("out_proj"), cfg.d_model, q_dim, cfg.quantize)?,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        rope: &Rope,
        cache: &mut KvCache,
        position: usize,
        num_heads: usize,
        context: Option<usize>,
    ) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let (hd, kvh) = (self.head_dim, self.num_kv_heads);
        let qkv = self.in_proj.forward(xs)?;
        let split = |offset: usize, heads: usize| -> Result<Tensor> {
            qkv.narrow(2, offset * hd, heads * hd)?
                .reshape((b, t, heads, hd))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(0, num_heads)?;
        let k = split(num_heads, kvh)?;
        let v = split(num_heads + kvh, kvh)?;
        let q = candle_nn::rotary_emb::rope_i(&q, &rope.cos, &rope.sin)?;
        let k = candle_nn::rotary_emb::rope_i(&k, &rope.cos, &rope.sin)?;

        let (k, v, first_key) = cache.append(&k, &v)?;
        if let Some(out) = kernels::attention(&q, &k, &v, position, first_key, context)? {
            return self.out_proj.forward(&out.reshape((b, t, num_heads * hd))?);
        }
        let (k, v) = if kvh != num_heads {
            (
                repeat_kv(&k, num_heads / kvh)?,
                repeat_kv(&v, num_heads / kvh)?,
            )
        } else {
            (k, v)
        };

        let scores = (q.matmul(&k.t()?)? * (hd as f64).powf(-0.5))?;
        let n_keys = k.dim(2)?;
        let scores = match attention_mask(position, t, first_key, n_keys, context, xs.device())? {
            Some(mask) => scores.broadcast_add(&mask)?,
            None => scores,
        };
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = weights.matmul(&v.contiguous()?)?;
        let out = out.transpose(1, 2)?.reshape((b, t, num_heads * hd))?;
        self.out_proj.forward(&out)
    }
}

fn repeat_kv(xs: &Tensor, n_rep: usize) -> Result<Tensor> {
    let (b, h, t, d) = xs.dims4()?;
    xs.unsqueeze(2)?
        .broadcast_as((b, h, n_rep, t, d))?
        .reshape((b, h * n_rep, t, d))
}

/// Additive mask letting query `q` see key `k` when `k <= q` and `q - k < context`.
/// `None` when every query may see every key.
fn attention_mask(
    position: usize,
    t: usize,
    first_key: usize,
    n_keys: usize,
    context: Option<usize>,
    device: &Device,
) -> Result<Option<Tensor>> {
    let visible = |q: usize, k: usize| k <= q && context.is_none_or(|c| q - k < c);
    let last_query = position + t - 1;
    if t == 1 && visible(last_query, first_key) {
        return Ok(None);
    }
    let mask: Vec<f32> = (position..position + t)
        .flat_map(|q| {
            (first_key..first_key + n_keys).map(move |k| {
                if visible(q, k) {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Tensor::from_vec(mask, (1, 1, t, n_keys), device).map(Some)
}

#[derive(Debug, Clone)]
enum FeedForwardLayer {
    Plain {
        linear1: Linear,
        linear2: Linear,
        exact_gelu: bool,
    },
    Gated {
        linear_in: Linear,
        linear_out: Linear,
    },
}

impl FeedForwardLayer {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let linear =
            |vb: VarBuilder, out: usize, inp: usize| Linear::load(vb, out, inp, cfg.quantize);
        Ok(match cfg.feed_forward {
            FeedForward::Plain { hidden, exact_gelu } => Self::Plain {
                linear1: linear(vb.pp("linear1"), hidden, d)?,
                linear2: linear(vb.pp("linear2"), d, hidden)?,
                exact_gelu,
            },
            FeedForward::GatedSilu { hidden } => {
                let vb = vb.pp("gating");
                Self::Gated {
                    linear_in: linear(vb.pp("linear_in"), 2 * hidden, d)?,
                    linear_out: linear(vb.pp("linear_out"), d, hidden)?,
                }
            }
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Plain {
                linear1,
                linear2,
                exact_gelu,
            } => {
                let hidden = linear1.forward(xs)?;
                let hidden = if *exact_gelu {
                    hidden.gelu_erf()?
                } else {
                    Activation::GeluTanh.apply(&hidden)?
                };
                linear2.forward(&hidden)
            }
            Self::Gated {
                linear_in,
                linear_out,
            } => {
                let hidden = linear_in.forward(xs)?;
                let half = hidden.dim(D::Minus1)? / 2;
                let gate = Activation::Silu.apply(&hidden.narrow(D::Minus1, 0, half)?)?;
                let value = hidden.narrow(D::Minus1, half, half)?;
                linear_out.forward(&(gate * value)?)
            }
        }
    }
}

/// Keys and values of past steps, preallocated and grown geometrically. With a window, entries
/// that fell out of it are dropped when the buffer is compacted.
#[derive(Debug, Clone)]
struct KvCache {
    buffers: Option<(Tensor, Tensor)>,
    len: usize,
    /// Absolute position of the first stored entry.
    start: usize,
    window: Option<usize>,
}

impl KvCache {
    fn new(window: Option<usize>) -> Self {
        Self {
            buffers: None,
            len: 0,
            start: 0,
            window,
        }
    }

    fn duplicate(&self) -> Result<Self> {
        let buffers = match &self.buffers {
            Some((k, v)) => Some((k.copy()?, v.copy()?)),
            None => None,
        };
        Ok(Self {
            buffers,
            ..self.clone()
        })
    }

    fn current(&self) -> Result<Option<(Tensor, Tensor)>> {
        match &self.buffers {
            Some((k, v)) => Ok(Some((k.narrow(2, 0, self.len)?, v.narrow(2, 0, self.len)?))),
            None => Ok(None),
        }
    }

    /// Appends `[batch, heads, t, dim]` keys and values. Returns all the cached keys and values
    /// and the absolute position of the first one.
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor, usize)> {
        let (b, h, t, d) = k.dims4()?;
        let capacity = self
            .buffers
            .as_ref()
            .map_or(0, |(k, _)| k.dim(2).unwrap_or(0));
        if self.len + t > capacity {
            // Only the last `window` entries can still be attended to.
            let keep = self.window.map_or(self.len, |w| self.len.min(w));
            let drop = self.len - keep;
            let new_capacity = match self.window {
                Some(w) => 2 * w + t,
                None => (2 * capacity).max(keep + t).max(64),
            };
            let new_k = Tensor::zeros((b, h, new_capacity, d), k.dtype(), k.device())?;
            let new_v = Tensor::zeros((b, h, new_capacity, d), v.dtype(), v.device())?;
            if let Some((old_k, old_v)) = &self.buffers
                && keep > 0
            {
                new_k.slice_set(&old_k.narrow(2, drop, keep)?.contiguous()?, 2, 0)?;
                new_v.slice_set(&old_v.narrow(2, drop, keep)?.contiguous()?, 2, 0)?;
            }
            self.buffers = Some((new_k, new_v));
            self.start += drop;
            self.len = keep;
        }
        let (cache_k, cache_v) = self.buffers.as_ref().expect("allocated above");
        cache_k.slice_set(&k.contiguous()?, 2, self.len)?;
        cache_v.slice_set(&v.contiguous()?, 2, self.len)?;
        self.len += t;

        // Hand out only the entries that the new queries may see.
        let visible = match self.window {
            Some(w) => self.len.min(w + t - 1),
            None => self.len,
        };
        let offset = self.len - visible;
        Ok((
            cache_k.narrow(2, offset, visible)?,
            cache_v.narrow(2, offset, visible)?,
            self.start + offset,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn random_transformer(cfg: &TransformerConfig) -> Transformer {
        let dev = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        // Materialize every weight with random values through the var map.
        let _ = Transformer::new(cfg, vb.clone());
        let tensors: HashMap<String, Tensor> = varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    Tensor::randn(0f32, 0.3, v.shape(), &dev).unwrap(),
                )
            })
            .collect();
        Transformer::new(cfg, VarBuilder::from_tensors(tensors, DType::F32, &dev)).unwrap()
    }

    fn config(context: Option<usize>) -> TransformerConfig {
        TransformerConfig {
            d_model: 16,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            feed_forward: FeedForward::Plain {
                hidden: 32,
                exact_gelu: false,
            },
            norm: Norm::LayerNorm { eps: 1e-5 },
            layer_scale: true,
            context,
            max_period: 10_000.0,
            quantize: false,
        }
    }

    #[test]
    fn streaming_matches_full_sequence() {
        for context in [None, Some(5)] {
            let model = random_transformer(&config(context));
            let xs = Tensor::randn(0f32, 1.0, (1, 23, 16), &Device::Cpu).unwrap();
            let full = model.forward(&xs, &mut model.init_state()).unwrap();
            let mut state = model.init_state();
            let mut chunks = vec![];
            let mut start = 0;
            for len in [1, 3, 1, 7, 1, 1, 9] {
                chunks.push(
                    model
                        .forward(&xs.narrow(1, start, len).unwrap(), &mut state)
                        .unwrap(),
                );
                start += len;
            }
            let streamed = Tensor::cat(&chunks, 1).unwrap();
            let diff: f32 = (full - streamed)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar()
                .unwrap();
            assert!(diff < 1e-4, "context {context:?}: diff {diff}");
        }
    }
}
