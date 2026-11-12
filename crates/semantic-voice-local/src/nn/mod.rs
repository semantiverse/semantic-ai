//! Model building blocks shared by text-to-speech and speech-to-text.

pub mod conv;
pub mod kernels;
pub mod mimi;
pub mod seanet;
pub mod transformer;

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

/// Memory-maps a safetensors file. Weights are converted to `dtype` when read.
pub(crate) fn mmap_weights(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    // SAFETY: model files are not expected to be modified while loaded, the same assumption
    // every safetensors consumer makes.
    unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device) }
}

/// Loads a Mimi checkpoint, renaming moshi's tensor names (`conv.conv`, `convtr.convtr`) to
/// the layout of [`mimi`].
pub(crate) fn load_mimi_weights(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let tensors = candle_core::safetensors::load(path, device)?;
    let has_prefix = tensors.keys().any(|k| k.starts_with("mimi."));
    let tensors: HashMap<String, Tensor> = tensors
        .into_iter()
        .filter_map(|(name, tensor)| {
            // Pocket TTS bundles prefix the codec with `mimi.`.
            let name = if has_prefix {
                name.strip_prefix("mimi.")?.to_string()
            } else {
                name
            };
            let name = name
                .strip_prefix("model.")
                .unwrap_or(&name)
                .replace(".conv.conv.", ".conv.")
                .replace(".convtr.convtr.", ".convtr.");
            Some((name, tensor))
        })
        .collect();
    Ok(VarBuilder::from_tensors(tensors, dtype, device))
}

/// A linear layer, optionally with 8-bit (Q8_0) weights: on CPU, inference of the larger
/// models is bound by memory bandwidth, which quantization divides by almost four.
///
/// Single rows (the steps of streaming inference) go through [`kernels::matvec`].
#[derive(Debug, Clone)]
pub(crate) enum Linear {
    Dense(candle_nn::Linear),
    Quantized(candle_core::quantized::QMatMul),
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self::Dense(candle_nn::Linear::new(weight, bias))
    }

    /// Loads `weight` (`[out, inp]`) under `vb`.
    pub fn load(vb: VarBuilder, out: usize, inp: usize, quantize: bool) -> Result<Self> {
        Self::from_weight(vb.get((out, inp), "weight")?, quantize)
    }

    /// Quantization applies on CPU to layers whose input width is a multiple of the block size.
    pub fn from_weight(weight: Tensor, quantize: bool) -> Result<Self> {
        use candle_core::quantized::{GgmlDType, QMatMul, QTensor};

        let dtype = GgmlDType::Q8_0;
        if quantize && weight.device().is_cpu() && weight.dim(1)? % dtype.block_size() == 0 {
            Ok(Self::Quantized(QMatMul::from_qtensor(QTensor::quantize(
                &weight, dtype,
            )?)?))
        } else {
            Ok(Self::Dense(candle_nn::Linear::new(weight, None)))
        }
    }
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(linear) => match kernels::matvec(xs, linear.weight(), linear.bias())? {
                Some(out) => Ok(out),
                None => linear.forward(xs),
            },
            Self::Quantized(matmul) => matmul.forward(xs),
        }
    }
}
