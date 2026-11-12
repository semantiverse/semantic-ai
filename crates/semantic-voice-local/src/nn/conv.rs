//! Causal convolutions that can process a signal in consecutive chunks.
//!
//! Each layer keeps its streaming state in a separate value so that one set of weights can serve
//! many concurrent streams. Processing a whole signal at once is the same as processing it with
//! a fresh state.

use candle_core::{D, DType, Result, Tensor};

use super::kernels;
use candle_nn::VarBuilder;
use serde::Deserialize;

/// How a convolution is padded on the left before its first chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PadMode {
    #[default]
    Constant,
    Replicate,
}

#[derive(Debug, Clone)]
pub struct StreamingConv1d {
    weight: Tensor,
    /// For ungrouped convolutions, the kernel as `[out_channels, in_channels * kernel]`.
    matmul_weight: Option<Tensor>,
    bias: Option<Tensor>,
    stride: usize,
    dilation: usize,
    groups: usize,
    pad_mode: PadMode,
}

/// The input samples a convolution still needs from previous chunks.
#[derive(Debug, Clone, Default)]
pub struct Conv1dState {
    previous: Option<Tensor>,
}

#[derive(Debug, Clone, Copy)]
pub struct Conv1dConfig {
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub bias: bool,
    pub pad_mode: PadMode,
}

impl Conv1dConfig {
    pub fn new(kernel_size: usize) -> Self {
        Self {
            kernel_size,
            stride: 1,
            dilation: 1,
            groups: 1,
            bias: true,
            pad_mode: PadMode::Constant,
        }
    }

    pub fn stride(self, stride: usize) -> Self {
        Self { stride, ..self }
    }

    pub fn dilation(self, dilation: usize) -> Self {
        Self { dilation, ..self }
    }

    pub fn no_bias(self) -> Self {
        Self {
            bias: false,
            ..self
        }
    }

    pub fn pad_mode(self, pad_mode: PadMode) -> Self {
        Self { pad_mode, ..self }
    }
}

impl StreamingConv1d {
    /// Loads `conv.weight` (and `conv.bias`) under `vb`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        cfg: Conv1dConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = vb.pp("conv");
        let weight = vb.get(
            (out_channels, in_channels / cfg.groups, cfg.kernel_size),
            "weight",
        )?;
        let bias = if cfg.bias {
            Some(vb.get(out_channels, "bias")?)
        } else {
            None
        };
        let matmul_weight = (cfg.groups == 1)
            .then(|| weight.reshape((out_channels, ())))
            .transpose()?;
        Ok(Self {
            weight,
            matmul_weight,
            bias,
            stride: cfg.stride,
            dilation: cfg.dilation,
            groups: cfg.groups,
            pad_mode: cfg.pad_mode,
        })
    }

    /// Number of input steps carried over between chunks.
    fn context(&self) -> usize {
        let kernel = self.weight.dim(D::Minus1).unwrap_or(1);
        ((kernel - 1) * self.dilation + 1).saturating_sub(self.stride)
    }

    /// `xs` is `[batch, channels, time]` with `time` a multiple of the stride.
    pub fn forward(&self, xs: &Tensor, state: &mut Conv1dState) -> Result<Tensor> {
        let (b, c, t) = xs.dims3()?;
        if t == 0 || t % self.stride != 0 {
            candle_core::bail!(
                "conv input length {t} is not a positive multiple of {}",
                self.stride
            );
        }
        let context = self.context();
        let previous = match (context, state.previous.take()) {
            (0, _) => None,
            (_, Some(previous)) => Some(previous),
            (_, None) => Some(match self.pad_mode {
                PadMode::Constant => Tensor::zeros((b, c, context), xs.dtype(), xs.device())?,
                PadMode::Replicate => xs.narrow(2, 0, 1)?.broadcast_as((b, c, context))?,
            }),
        };
        let fast = xs.device().is_cpu() && xs.dtype() == DType::F32;
        let ys = match (&self.matmul_weight, &previous) {
            (Some(weight), previous) if fast => {
                let kernel = self.weight.dim(2)?;
                let columns =
                    kernels::im2col(previous.as_ref(), xs, kernel, self.stride, self.dilation)?;
                weight.broadcast_matmul(&columns)?
            }
            (weight, previous) => {
                let xs = match previous {
                    Some(previous) => Tensor::cat(&[previous, xs], 2)?,
                    None => xs.clone(),
                };
                match weight {
                    Some(weight) => self.im2col_conv(&xs, weight)?,
                    None => xs.conv1d(&self.weight, 0, self.stride, self.dilation, self.groups)?,
                }
            }
        };
        if let Some(previous) = previous {
            // The last `context` samples of the input with the previous ones before it.
            state.previous = Some(if t >= context {
                xs.narrow(2, t - context, context)?.contiguous()?
            } else {
                Tensor::cat(&[&previous, xs], 2)?
                    .narrow(2, t, context)?
                    .contiguous()?
            });
        }
        match &self.bias {
            Some(bias) => ys.broadcast_add(&bias.reshape((1, (), 1))?),
            None => Ok(ys),
        }
    }
}

impl StreamingConv1d {
    /// Unpadded convolution as one matrix multiplication over the gathered kernel taps, much
    /// faster than candle's CPU convolution for the small chunks of streaming.
    fn im2col_conv(&self, xs: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let (b, c, t) = xs.dims3()?;
        let kernel = self.weight.dim(2)?;
        let (stride, dilation) = (self.stride, self.dilation);
        let len = (t - (kernel - 1) * dilation - 1) / stride + 1;
        // Tap `j` of output `l` reads input `j * dilation + l * stride`: every `stride`-th sample
        // from `j * dilation`, taken by reshaping a slice padded to a whole number of strides.
        let xs = if stride > 1 {
            xs.pad_with_zeros(2, 0, stride - 1)?
        } else {
            xs.clone()
        };
        let taps = (0..kernel)
            .map(|j| {
                let tap = xs.narrow(2, j * dilation, len * stride)?;
                if stride > 1 {
                    tap.reshape((b, c, len, stride))?
                        .narrow(3, 0, 1)?
                        .squeeze(3)
                } else {
                    Ok(tap)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        // `[batch, channels * kernel, len]`, in the order of the kernel weights.
        let columns = Tensor::stack(&taps, 2)?.reshape((b, c * kernel, len))?;
        weight.broadcast_matmul(&columns)
    }
}

#[derive(Debug, Clone)]
pub struct StreamingConvTranspose1d {
    /// `[in_channels, out_channels / groups, kernel]`.
    weight: Tensor,
    /// For ungrouped and depthwise convolutions, the kernel as `[out_channels * kernel,
    /// in_channels / groups]`: the products of the inputs with each tap, before they are
    /// overlap-added.
    tap_weight: Option<Tensor>,
    bias: Option<Tensor>,
    stride: usize,
    groups: usize,
}

/// The overlapping output tail of the previous chunk.
#[derive(Debug, Clone, Default)]
pub struct ConvTranspose1dState {
    partial: Option<Tensor>,
}

impl StreamingConvTranspose1d {
    /// Loads `convtr.weight` (and `convtr.bias`) under `vb`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = vb.pp("convtr");
        let weight = vb.get((in_channels, out_channels / groups, kernel_size), "weight")?;
        let bias = if bias {
            Some(vb.get(out_channels, "bias")?)
        } else {
            None
        };
        let tap_weight = if groups == 1 {
            Some(
                weight
                    .permute((1, 2, 0))?
                    .reshape((out_channels * kernel_size, ()))?,
            )
        } else if groups == in_channels && out_channels == in_channels {
            Some(weight.reshape((out_channels * kernel_size, 1))?)
        } else {
            None
        };
        Ok(Self {
            weight,
            tap_weight: tap_weight
                .filter(|_| kernel_size >= stride)
                .map(|w| w.contiguous())
                .transpose()?,
            bias,
            stride,
            groups,
        })
    }

    pub fn forward(&self, xs: &Tensor, state: &mut ConvTranspose1dState) -> Result<Tensor> {
        if let Some(out) = self.forward_taps(xs, state)? {
            return Ok(out);
        }
        let kernel = self.weight.dim(2)?;
        let ys = self.transposed_conv(xs)?;
        let ys = match &self.bias {
            Some(bias) => ys.broadcast_add(&bias.reshape((1, (), 1))?)?,
            None => ys,
        };
        let overlap = kernel.saturating_sub(self.stride);
        if overlap == 0 {
            return Ok(ys);
        }
        let len = ys.dim(2)?;
        let tail = ys.narrow(2, len - overlap, overlap)?;
        let tail = match &self.bias {
            Some(bias) => tail.broadcast_sub(&bias.reshape((1, (), 1))?)?,
            None => tail,
        };
        let out = ys.narrow(2, 0, len - overlap)?;
        let out = match state.partial.replace(tail) {
            Some(partial) => {
                let head = (out.narrow(2, 0, overlap)? + partial)?;
                Tensor::cat(&[&head, &out.narrow(2, overlap, len - 2 * overlap)?], 2)?
            }
            None => out,
        };
        Ok(out)
    }

    /// On CPU: the products of the inputs with every tap in one matrix multiplication, then
    /// one pass adding them up with the bias and the previous tail. `None` elsewhere.
    fn forward_taps(
        &self,
        xs: &Tensor,
        state: &mut ConvTranspose1dState,
    ) -> Result<Option<Tensor>> {
        let Some(tap_weight) = &self.tap_weight else {
            return Ok(None);
        };
        if !xs.device().is_cpu() || xs.dtype() != DType::F32 {
            return Ok(None);
        }
        let (b, c, t) = xs.dims3()?;
        let kernel = self.weight.dim(2)?;
        let out_channels = tap_weight.dim(0)? / kernel;
        // `[batch, out_channels * kernel, time]`
        let products = if self.groups == 1 {
            tap_weight.broadcast_matmul(xs)?
        } else {
            xs.unsqueeze(2)?
                .broadcast_mul(&tap_weight.reshape((1, c, kernel, 1))?)?
                .reshape((b, c * kernel, t))?
        };
        let overlap = kernel - self.stride;
        let partial = match state.partial.take() {
            Some(partial) => partial,
            None => Tensor::zeros((b, out_channels, overlap), DType::F32, xs.device())?,
        };
        let full = kernels::overlap_add(&products, self.stride, self.bias.as_ref(), &partial)?;
        let len = t * self.stride;
        if overlap > 0 {
            state.partial = Some(full.narrow(2, len, overlap)?.contiguous()?);
        }
        Ok(Some(full.narrow(2, 0, len)?))
    }

    fn transposed_conv(&self, xs: &Tensor) -> Result<Tensor> {
        let (in_channels, out_per_group, kernel) = self.weight.dims3()?;
        let (b, c, t) = xs.dims3()?;
        let s = self.stride;
        if kernel % s != 0 {
            return xs.conv_transpose1d(&self.weight, 0, 0, s, 1, self.groups);
        }
        // With a kernel spanning whole strides (as in all Mimi layers), each input step adds
        // `kernel / stride` blocks of `stride` outputs: compute the blocks with one product,
        // then overlap-add them. Much faster than candle's CPU transposed convolution.
        let n_blocks = kernel / s;
        // `[batch, out_channels, time, kernel]`
        let products = if self.groups == 1 {
            let weight = self.weight.reshape((in_channels, out_per_group * kernel))?;
            xs.transpose(1, 2)?
                .broadcast_matmul(&weight)?
                .reshape((b, t, out_per_group, kernel))?
                .permute((0, 2, 1, 3))?
        } else if self.groups == in_channels && out_per_group == 1 {
            xs.unsqueeze(3)?
                .broadcast_mul(&self.weight.reshape((1, c, 1, kernel))?)?
        } else {
            return xs.conv_transpose1d(&self.weight, 0, 0, s, 1, self.groups);
        };
        let out_channels = products.dim(1)?;
        let mut out: Option<Tensor> = None;
        for j in 0..n_blocks {
            let block = products
                .narrow(3, j * s, s)?
                .reshape((b, out_channels, t * s))?;
            let block = block.pad_with_zeros(2, j * s, (n_blocks - 1 - j) * s)?;
            out = Some(match out {
                Some(out) => (out + block)?,
                None => block,
            });
        }
        Ok(out.expect("kernel is at least one stride"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    fn vb(tensors: Vec<(&str, Tensor)>) -> VarBuilder<'static> {
        let map: HashMap<String, Tensor> = tensors
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
    }

    fn randn(shape: &[usize]) -> Tensor {
        Tensor::randn(0f32, 1.0, shape, &Device::Cpu).unwrap()
    }

    fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap()
    }

    #[test]
    fn conv_streaming_matches_full() {
        for pad_mode in [PadMode::Constant, PadMode::Replicate] {
            let conv = StreamingConv1d::new(
                3,
                4,
                Conv1dConfig::new(4).stride(2).pad_mode(pad_mode),
                vb(vec![
                    ("conv.weight", randn(&[4, 3, 4])),
                    ("conv.bias", randn(&[4])),
                ]),
            )
            .unwrap();
            let xs = randn(&[1, 3, 12]);
            let full = conv.forward(&xs, &mut Default::default()).unwrap();
            let mut state = Conv1dState::default();
            let chunks: Vec<Tensor> = (0..3)
                .map(|i| {
                    conv.forward(&xs.narrow(2, i * 4, 4).unwrap(), &mut state)
                        .unwrap()
                })
                .collect();
            let streamed = Tensor::cat(&chunks, 2).unwrap();
            assert_eq!(full.dims(), &[1, 4, 6]);
            assert!(max_diff(&full, &streamed) < 1e-5);
        }
    }

    #[test]
    fn im2col_conv_matches_candle() {
        for (stride, dilation, kernel) in [(1, 1, 7), (1, 2, 3), (4, 1, 8), (2, 1, 4), (1, 1, 1)] {
            let weight = randn(&[5, 3, kernel]);
            let conv = StreamingConv1d::new(
                3,
                5,
                Conv1dConfig::new(kernel)
                    .stride(stride)
                    .dilation(dilation)
                    .no_bias(),
                vb(vec![("conv.weight", weight.clone())]),
            )
            .unwrap();
            let xs = randn(&[2, 3, 37]);
            let fast = conv
                .im2col_conv(&xs, conv.matmul_weight.as_ref().unwrap())
                .unwrap();
            let reference = xs.conv1d(&weight, 0, stride, dilation, 1).unwrap();
            assert_eq!(fast.dims(), reference.dims());
            assert!(max_diff(&fast, &reference) < 1e-4);
        }
    }

    #[test]
    fn transposed_conv_matches_candle() {
        let weight = randn(&[3, 4, 6]);
        let conv = StreamingConvTranspose1d::new(
            3,
            4,
            6,
            3,
            1,
            false,
            vb(vec![("convtr.weight", weight.clone())]),
        )
        .unwrap();
        let xs = randn(&[2, 3, 5]);
        let fast = conv.transposed_conv(&xs).unwrap();
        // Item by item: candle 0.11's CPU transposed convolution is wrong for batches above 1.
        let reference: Vec<Tensor> = (0..2)
            .map(|i| {
                xs.narrow(0, i, 1)
                    .unwrap()
                    .conv_transpose1d(&weight, 0, 0, 3, 1, 1)
                    .unwrap()
            })
            .collect();
        assert!(max_diff(&fast, &Tensor::cat(&reference, 0).unwrap()) < 1e-4);
    }

    #[test]
    fn depthwise_transposed_conv_matches_candle() {
        let weight = randn(&[3, 1, 8]);
        let conv = StreamingConvTranspose1d::new(
            3,
            3,
            8,
            4,
            3,
            false,
            vb(vec![("convtr.weight", weight.clone())]),
        )
        .unwrap();
        let xs = randn(&[2, 3, 5]);
        let fast = conv.transposed_conv(&xs).unwrap();
        let reference = xs.conv_transpose1d(&weight, 0, 0, 4, 1, 3).unwrap();
        assert!(max_diff(&fast, &reference) < 1e-5);
    }

    /// The CPU path of `forward`, against candle's transposed convolution.
    #[test]
    fn transposed_conv_forward_matches_candle() {
        // (in, out, kernel, stride, groups)
        for (inp, out, kernel, stride, groups) in
            [(3, 4, 6, 3, 1), (2, 5, 4, 4, 1), (3, 3, 8, 4, 3)]
        {
            let weight = randn(&[inp, out / groups, kernel]);
            let bias = randn(&[out]);
            let conv = StreamingConvTranspose1d::new(
                inp,
                out,
                kernel,
                stride,
                groups,
                true,
                vb(vec![
                    ("convtr.weight", weight.clone()),
                    ("convtr.bias", bias.clone()),
                ]),
            )
            .unwrap();
            let xs = randn(&[1, inp, 5]);
            let mut state = ConvTranspose1dState::default();
            let fast = conv.forward(&xs, &mut state).unwrap();
            let reference = xs
                .conv_transpose1d(&weight, 0, 0, stride, 1, groups)
                .unwrap()
                .broadcast_add(&bias.reshape((1, (), 1)).unwrap())
                .unwrap();
            assert_eq!(fast.dims(), &[1, out, 5 * stride]);
            assert!(max_diff(&fast, &reference.narrow(2, 0, 5 * stride).unwrap()) < 1e-4);
        }
    }

    #[test]
    fn transposed_conv_streaming_matches_full() {
        let conv = StreamingConvTranspose1d::new(
            2,
            3,
            6,
            3,
            1,
            true,
            vb(vec![
                ("convtr.weight", randn(&[2, 3, 6])),
                ("convtr.bias", randn(&[3])),
            ]),
        )
        .unwrap();
        let xs = randn(&[1, 2, 6]);
        let full = conv.forward(&xs, &mut Default::default()).unwrap();
        let mut state = ConvTranspose1dState::default();
        let chunks: Vec<Tensor> = (0..3)
            .map(|i| {
                conv.forward(&xs.narrow(2, i * 2, 2).unwrap(), &mut state)
                    .unwrap()
            })
            .collect();
        let streamed = Tensor::cat(&chunks, 2).unwrap();
        assert!(max_diff(&full, &streamed) < 1e-5);
    }
}
