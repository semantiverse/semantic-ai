//! SEANet convolutional encoder and decoder, the waveform side of the Mimi codec.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

use super::conv::{
    Conv1dConfig, Conv1dState, ConvTranspose1dState, PadMode, StreamingConv1d,
    StreamingConvTranspose1d,
};
use super::kernels::Activation;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SeanetConfig {
    /// Channels of the latent side.
    pub dimension: usize,
    /// Audio channels.
    pub channels: usize,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    /// Up-sampling ratios, from latent to waveform.
    pub ratios: Vec<usize>,
    pub kernel_size: usize,
    pub residual_kernel_size: usize,
    pub last_kernel_size: usize,
    pub dilation_base: usize,
    pub pad_mode: PadMode,
    pub compress: usize,
}

impl SeanetConfig {
    /// Waveform samples per latent frame.
    pub fn hop_length(&self) -> usize {
        self.ratios.iter().product()
    }
}

#[derive(Debug, Clone)]
struct ResnetBlock {
    convs: Vec<StreamingConv1d>,
}

impl ResnetBlock {
    fn new(cfg: &SeanetConfig, dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = dim / cfg.compress;
        let vb = vb.pp("block");
        // Layers alternate ELU and convolution, so the convolutions are at odd indices.
        let conv_a = StreamingConv1d::new(
            dim,
            hidden,
            Conv1dConfig::new(cfg.residual_kernel_size)
                .dilation(dilation)
                .pad_mode(cfg.pad_mode),
            vb.pp(1),
        )?;
        let conv_b = StreamingConv1d::new(
            hidden,
            dim,
            Conv1dConfig::new(1).pad_mode(cfg.pad_mode),
            vb.pp(3),
        )?;
        Ok(Self {
            convs: vec![conv_a, conv_b],
        })
    }

    fn forward(&self, xs: &Tensor, states: &mut [Conv1dState]) -> Result<Tensor> {
        let mut ys = xs.clone();
        for (conv, state) in self.convs.iter().zip(states.iter_mut()) {
            ys = conv.forward(&Activation::Elu.apply(&ys)?, state)?;
        }
        xs + ys
    }
}

#[derive(Debug, Clone)]
enum Layer {
    Elu,
    Conv(StreamingConv1d),
    ConvTranspose(StreamingConvTranspose1d),
    Resnet(ResnetBlock),
}

#[derive(Debug, Clone)]
enum LayerState {
    None,
    Conv(Conv1dState),
    ConvTranspose(ConvTranspose1dState),
    Resnet(Vec<Conv1dState>),
}

/// Streaming state of a [`SeanetEncoder`] or [`SeanetDecoder`].
#[derive(Debug, Clone)]
pub struct SeanetState(Vec<LayerState>);

fn init_state(layers: &[Layer]) -> SeanetState {
    SeanetState(
        layers
            .iter()
            .map(|layer| match layer {
                Layer::Elu => LayerState::None,
                Layer::Conv(_) => LayerState::Conv(Default::default()),
                Layer::ConvTranspose(_) => LayerState::ConvTranspose(Default::default()),
                Layer::Resnet(block) => {
                    LayerState::Resnet(vec![Default::default(); block.convs.len()])
                }
            })
            .collect(),
    )
}

fn forward(layers: &[Layer], xs: &Tensor, state: &mut SeanetState) -> Result<Tensor> {
    let mut xs = xs.clone();
    for (layer, state) in layers.iter().zip(state.0.iter_mut()) {
        xs = match (layer, state) {
            (Layer::Elu, _) => Activation::Elu.apply(&xs)?,
            (Layer::Conv(conv), LayerState::Conv(s)) => conv.forward(&xs, s)?,
            (Layer::ConvTranspose(conv), LayerState::ConvTranspose(s)) => conv.forward(&xs, s)?,
            (Layer::Resnet(block), LayerState::Resnet(s)) => block.forward(&xs, s)?,
            _ => candle_core::bail!("seanet state does not match the model"),
        };
    }
    Ok(xs)
}

/// Waveform `[batch, channels, samples]` to latent frames `[batch, dimension, frames]`.
#[derive(Debug, Clone)]
pub struct SeanetEncoder {
    layers: Vec<Layer>,
}

impl SeanetEncoder {
    pub fn new(cfg: &SeanetConfig, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("model");
        let conv = |inp, out, c: Conv1dConfig, i: usize| -> Result<Layer> {
            Ok(Layer::Conv(StreamingConv1d::new(
                inp,
                out,
                c.pad_mode(cfg.pad_mode),
                vb.pp(i),
            )?))
        };
        let mut layers = vec![conv(
            cfg.channels,
            cfg.n_filters,
            Conv1dConfig::new(cfg.kernel_size),
            0,
        )?];
        let mut mult = 1;
        for &ratio in cfg.ratios.iter().rev() {
            let dim = mult * cfg.n_filters;
            for j in 0..cfg.n_residual_layers {
                let dilation = cfg.dilation_base.pow(j as u32);
                layers.push(Layer::Resnet(ResnetBlock::new(
                    cfg,
                    dim,
                    dilation,
                    vb.pp(layers.len()),
                )?));
            }
            layers.push(Layer::Elu);
            layers.push(conv(
                dim,
                2 * dim,
                Conv1dConfig::new(2 * ratio).stride(ratio),
                layers.len(),
            )?);
            mult *= 2;
        }
        layers.push(Layer::Elu);
        layers.push(conv(
            mult * cfg.n_filters,
            cfg.dimension,
            Conv1dConfig::new(cfg.last_kernel_size),
            layers.len(),
        )?);
        Ok(Self { layers })
    }

    pub fn init_state(&self) -> SeanetState {
        init_state(&self.layers)
    }

    pub fn forward(&self, xs: &Tensor, state: &mut SeanetState) -> Result<Tensor> {
        forward(&self.layers, xs, state)
    }
}

/// Latent frames `[batch, dimension, frames]` to waveform `[batch, channels, samples]`.
#[derive(Debug, Clone)]
pub struct SeanetDecoder {
    layers: Vec<Layer>,
}

impl SeanetDecoder {
    pub fn new(cfg: &SeanetConfig, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("model");
        let mut mult = 1 << cfg.ratios.len();
        let mut layers = vec![Layer::Conv(StreamingConv1d::new(
            cfg.dimension,
            mult * cfg.n_filters,
            Conv1dConfig::new(cfg.kernel_size).pad_mode(cfg.pad_mode),
            vb.pp(0),
        )?)];
        for &ratio in &cfg.ratios {
            let dim = mult * cfg.n_filters;
            layers.push(Layer::Elu);
            layers.push(Layer::ConvTranspose(StreamingConvTranspose1d::new(
                dim,
                dim / 2,
                2 * ratio,
                ratio,
                1,
                true,
                vb.pp(layers.len()),
            )?));
            for j in 0..cfg.n_residual_layers {
                let dilation = cfg.dilation_base.pow(j as u32);
                layers.push(Layer::Resnet(ResnetBlock::new(
                    cfg,
                    dim / 2,
                    dilation,
                    vb.pp(layers.len()),
                )?));
            }
            mult /= 2;
        }
        layers.push(Layer::Elu);
        layers.push(Layer::Conv(StreamingConv1d::new(
            cfg.n_filters,
            cfg.channels,
            Conv1dConfig::new(cfg.last_kernel_size).pad_mode(cfg.pad_mode),
            vb.pp(layers.len()),
        )?));
        Ok(Self { layers })
    }

    pub fn init_state(&self) -> SeanetState {
        init_state(&self.layers)
    }

    pub fn forward(&self, xs: &Tensor, state: &mut SeanetState) -> Result<Tensor> {
        forward(&self.layers, xs, state)
    }
}
