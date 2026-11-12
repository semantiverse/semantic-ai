//! The data movement around the matrix multiplications of streaming convolutions, in single
//! passes instead of chains of tensor operations.

use candle_core::{CpuStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};

use super::contiguous;

/// The columns of a convolution as a matrix multiplication: for the input `previous`
/// followed by `xs` (`[batch, channels, time]`, the first optional), the `[batch, channels *
/// kernel, len]` taps, tap `j` of output `l` being input `l * stride + j * dilation`. On CPU in
/// f32.
pub fn im2col(
    previous: Option<&Tensor>,
    xs: &Tensor,
    kernel: usize,
    stride: usize,
    dilation: usize,
) -> Result<Tensor> {
    let (b, c, _) = xs.dims3()?;
    let previous = match previous {
        Some(previous) => previous.contiguous()?,
        None => Tensor::zeros((b, c, 0), xs.dtype(), xs.device())?,
    };
    previous.apply_op2_no_bwd(
        &xs.contiguous()?,
        &Im2Col {
            kernel,
            stride,
            dilation,
        },
    )
}

struct Im2Col {
    kernel: usize,
    stride: usize,
    dilation: usize,
}

impl CustomOp2 for Im2Col {
    fn name(&self) -> &'static str {
        "im2col-f32"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let previous = contiguous(s1, l1, "previous input")?;
        let xs = contiguous(s2, l2, "input")?;
        let (_, _, p) = l1.shape().dims3()?;
        let (b, c, t) = l2.shape().dims3()?;
        let Self {
            kernel,
            stride,
            dilation,
        } = *self;
        let span = (kernel - 1) * dilation + 1;
        if p + t < span {
            candle_core::bail!("im2col: input shorter than the kernel");
        }
        let len = (p + t - span) / stride + 1;
        let mut out = vec![0f32; b * c * kernel * len];
        let mut row = Vec::with_capacity(p + t);
        for (i, out) in out.chunks_exact_mut(kernel * len).enumerate() {
            row.clear();
            row.extend_from_slice(&previous[i * p..(i + 1) * p]);
            row.extend_from_slice(&xs[i * t..(i + 1) * t]);
            for (j, out) in out.chunks_exact_mut(len).enumerate() {
                let taps = &row[j * dilation..];
                if stride == 1 {
                    out.copy_from_slice(&taps[..len]);
                } else {
                    out.iter_mut()
                        .zip(taps.iter().step_by(stride))
                        .for_each(|(o, x)| *o = *x);
                }
            }
        }
        Ok((CpuStorage::F32(out), Shape::from((b, c * kernel, len))))
    }
}

/// The output of a transposed convolution from the products of its inputs with each tap:
/// `products` is `[batch, channels * kernel, time]`, tap `j` of step `t` landing on output
/// `t * stride + j`. Adds `bias` to the first `time * stride` outputs (the complete ones) and
/// `partial` (`[batch, channels, overlap]`, the incomplete outputs of the previous chunk) to
/// the first ones. Returns `[batch, channels, time * stride + overlap]`. On CPU in f32.
pub fn overlap_add(
    products: &Tensor,
    stride: usize,
    bias: Option<&Tensor>,
    partial: &Tensor,
) -> Result<Tensor> {
    let (b, channels, overlap) = partial.dims3()?;
    let bias = match bias {
        Some(bias) => bias.clone(),
        None => Tensor::zeros(channels, products.dtype(), products.device())?,
    };
    products.contiguous()?.apply_op3_no_bwd(
        &bias.contiguous()?,
        &partial.contiguous()?,
        &OverlapAdd {
            batch: b,
            channels,
            stride,
            overlap,
        },
    )
}

struct OverlapAdd {
    batch: usize,
    channels: usize,
    stride: usize,
    overlap: usize,
}

impl CustomOp3 for OverlapAdd {
    fn name(&self) -> &'static str {
        "overlap-add-f32"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let products = contiguous(s1, l1, "products")?;
        let bias = contiguous(s2, l2, "bias")?;
        let partial = contiguous(s3, l3, "partial")?;
        let Self {
            batch,
            channels,
            stride,
            overlap,
        } = *self;
        let (_, rows, time) = l1.shape().dims3()?;
        let kernel = rows / channels;
        if kernel != stride + overlap || bias.len() != channels {
            candle_core::bail!("overlap-add: inconsistent shapes");
        }
        let complete = time * stride;
        let len = complete + overlap;
        let mut out = vec![0f32; batch * channels * len];
        for (row, out) in out.chunks_exact_mut(len).enumerate() {
            out[..complete].fill(bias[row % channels]);
            let partial = &partial[row * overlap..(row + 1) * overlap];
            out.iter_mut().zip(partial).for_each(|(o, p)| *o += p);
            let products = &products[row * kernel * time..(row + 1) * kernel * time];
            for (tap, products) in products.chunks_exact(time).enumerate() {
                for (step, &value) in products.iter().enumerate() {
                    out[step * stride + tap] += value;
                }
            }
        }
        Ok((CpuStorage::F32(out), Shape::from((batch, channels, len))))
    }
}
