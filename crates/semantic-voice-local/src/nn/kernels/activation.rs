use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor};

use super::{contiguous, exp, exp_parts, is_cpu_f32, mul_add};

/// Element-wise activations, vectorized: candle evaluates them with a scalar `exp`, `exp_m1`
/// or `tanh` call per value, which dominated the cost of the codec and of the feed-forward
/// layers. Within a few ulp of the exact functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// ELU with `alpha = 1`.
    Elu,
    /// GELU, tanh approximation.
    GeluTanh,
    Silu,
}

impl Activation {
    pub fn apply(self, xs: &Tensor) -> Result<Tensor> {
        if is_cpu_f32(xs) {
            return xs.contiguous()?.apply_op1_no_bwd(&self);
        }
        match self {
            Self::Elu => xs.elu(1.0),
            Self::GeluTanh => xs.gelu(),
            Self::Silu => xs.silu(),
        }
    }

    fn apply_in_place(self, xs: &mut [f32]) {
        apply_in_place(self, xs)
    }
}

impl CustomOp1 for Activation {
    fn name(&self) -> &'static str {
        match self {
            Self::Elu => "elu-f32",
            Self::GeluTanh => "gelu-f32",
            Self::Silu => "silu-f32",
        }
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let mut out = contiguous(storage, layout, "input")?.to_vec();
        self.apply_in_place(&mut out);
        Ok((CpuStorage::F32(out), layout.shape().clone()))
    }
}

with_fma! {
    fn apply_in_place(activation: Activation, xs: &mut [f32]) {
        match activation {
            Activation::Elu => xs.iter_mut().for_each(|x| *x = elu::<FMA>(*x)),
            Activation::GeluTanh => xs.iter_mut().for_each(|x| *x = gelu_tanh::<FMA>(*x)),
            Activation::Silu => xs.iter_mut().for_each(|x| *x = silu::<FMA>(*x)),
        }
    }
}

/// Within a few ulp of `x.exp_m1()` for negative `x`, about thirty times faster.
#[inline(always)]
fn elu<const FMA: bool>(x: f32) -> f32 {
    // exp(x) - 1 = 2^k expm1(r) + (2^k - 1), without cancellation. It rounds to -1 below -20.
    let (two_k, expm1_r) = exp_parts::<FMA>(x.clamp(-20.0, 0.0));
    let negative = mul_add::<FMA>(two_k, expm1_r, two_k - 1.0);
    if x > 0.0 { x } else { negative }
}

/// `x sigmoid(x)`.
#[inline(always)]
fn silu<const FMA: bool>(x: f32) -> f32 {
    x / (1.0 + exp::<FMA>(-x))
}

/// `0.5 x (1 + tanh(y))`, computed as `x sigmoid(2y)`: the same function, without the
/// cancellation of `1 + tanh(y)` for negative `y`.
#[inline(always)]
fn gelu_tanh<const FMA: bool>(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    let y = SQRT_2_OVER_PI * mul_add::<FMA>(0.044_715 * x, x * x, x);
    x / (1.0 + exp::<FMA>(-2.0 * y))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every activation is within a few ulp of its definition evaluated in f64.
    #[test]
    fn activations_are_accurate() {
        let mut xs: Vec<f32> = (0..400_000).map(|i| -100.0 + i as f32 * 5e-4).collect();
        xs.extend([0.0, -0.0, 1e-30, -1e-30, -1e-7, 1e-3, -1e-3, 1e4, -1e4]);
        xs.extend([f32::INFINITY, f32::NEG_INFINITY]);
        let reference = |activation, x: f64| -> f64 {
            match activation {
                Activation::Elu if x > 0.0 => x,
                Activation::Elu => x.exp_m1(),
                Activation::Silu => x / (1.0 + (-x).exp()),
                Activation::GeluTanh => {
                    let y = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
                    // 0.5 x (1 + tanh(y)), without cancellation.
                    x / (1.0 + (-2.0 * y).exp())
                }
            }
        };
        for activation in [Activation::Elu, Activation::Silu, Activation::GeluTanh] {
            let mut ys = xs.clone();
            activation.apply_in_place(&mut ys);
            for (&x, &y) in xs.iter().zip(&ys) {
                let expected = reference(activation, x as f64) as f32;
                // Relative, except for tiny results, whose f32 evaluation is ill-conditioned
                // (and which are negligible next to the other activations).
                let tolerance = 8.0 * f32::EPSILON * expected.abs().max(1e-3);
                assert!(
                    y == expected
                        || (y - expected).abs() <= tolerance
                        || (y.is_nan() && expected.is_nan()),
                    "{activation:?}({x}) = {y}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn falls_back_to_candle_semantics() {
        let xs = Tensor::new(&[-2f32, -0.5, 0.0, 0.7, 3.0], &candle_core::Device::Cpu).unwrap();
        for (activation, expected) in [
            (Activation::Elu, xs.elu(1.0).unwrap()),
            (Activation::GeluTanh, xs.gelu().unwrap()),
            (Activation::Silu, xs.silu().unwrap()),
        ] {
            let got = activation.apply(&xs).unwrap();
            assert!(crate::nn::kernels::max_diff(&got, &expected) < 1e-6);
        }
    }
}
