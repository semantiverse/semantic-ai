//! CPU kernels for the shapes of streaming inference, where candle's generic paths are
//! dominated by overheads: single-row products on every core, attention over a cache without
//! materializing masks, the data movement of streaming convolutions, and vectorized
//! activations.
//!
//! Every kernel computes in f32 what the candle operations it replaces compute, up to the
//! order of the floating-point operations. Each entry point falls back to candle (or returns
//! `None` for the caller to) off the CPU or for other dtypes.

/// Defines `fn $name`, whose body is compiled twice: with `FMA` false for any CPU, and with
/// `FMA` true and AVX2 and FMA enabled, chosen at run time when the CPU has them.
macro_rules! with_fma {
    ($(#[$attr:meta])* fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)? $body:block) => {
        $(#[$attr])*
        fn $name($($arg: $ty),*) $(-> $ret)? {
            #[inline(always)]
            fn generic<const FMA: bool>($($arg: $ty),*) $(-> $ret)? $body

            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2,fma")]
            fn avx2($($arg: $ty),*) $(-> $ret)? {
                generic::<true>($($arg),*)
            }

            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                // SAFETY: the CPU supports the features enabled for the function.
                return unsafe { avx2($($arg),*) };
            }
            generic::<false>($($arg),*)
        }
    };
}

mod activation;
mod attention;
mod conv;
mod matvec;

pub use activation::Activation;
pub use attention::attention;
pub use conv::{im2col, overlap_add};
pub use matvec::matvec;

use candle_core::{CpuStorage, Layout, Result, Tensor};

fn is_cpu_f32(tensor: &Tensor) -> bool {
    tensor.device().is_cpu() && tensor.dtype() == candle_core::DType::F32
}

fn contiguous<'a>(storage: &'a CpuStorage, layout: &Layout, what: &str) -> Result<&'a [f32]> {
    match (storage, layout.contiguous_offsets()) {
        (CpuStorage::F32(data), Some((start, end))) => Ok(&data[start..end]),
        _ => candle_core::bail!("{what} must be contiguous f32"),
    }
}

/// `a * b + c`, fused when the CPU supports it.
#[inline(always)]
fn mul_add<const FMA: bool>(a: f32, b: f32, c: f32) -> f32 {
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

/// Dot product over independent lanes, which the compiler vectorizes.
#[inline(always)]
fn dot<const FMA: bool>(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 16;
    let (a_chunks, a_rest) = a.as_chunks::<LANES>();
    let (b_chunks, b_rest) = b.as_chunks::<LANES>();
    let mut acc = [0f32; LANES];
    for (a, b) in a_chunks.iter().zip(b_chunks) {
        for i in 0..LANES {
            acc[i] = mul_add::<FMA>(a[i], b[i], acc[i]);
        }
    }
    let mut sum = 0.0;
    for (a, b) in a_rest.iter().zip(b_rest) {
        sum += a * b;
    }
    // Pairwise reduction of the lanes.
    let mut width = LANES;
    while width > 1 {
        width /= 2;
        for i in 0..width {
            acc[i] += acc[i + width];
        }
    }
    acc[0] + sum
}

/// `exp(x) = 2^k (expm1(r) + 1)`: returns `(2^k, expm1(r))`, branch-free so that it vectorizes.
/// `x` must be in the range where `exp` is a normal number, [`exp`] handles the others.
#[inline(always)]
fn exp_parts<const FMA: bool>(x: f32) -> (f32, f32) {
    const LN2_HI: f32 = 0.693_145_75;
    const LN2_LO: f32 = 1.428_606_8e-6;
    // Adding then subtracting 1.5 * 2^23 rounds to the nearest integer.
    const ROUND: f32 = 12_582_912.0;
    // x = k ln2 + r, |r| <= ln2 / 2
    let shifted = mul_add::<FMA>(x, std::f32::consts::LOG2_E, ROUND);
    let k = shifted - ROUND;
    let k_int = (shifted.to_bits() as i32).wrapping_sub(ROUND.to_bits() as i32);
    let r = mul_add::<FMA>(-k, LN2_LO, mul_add::<FMA>(-k, LN2_HI, x));
    // expm1(r) by its Taylor series, exact to f32 precision for |r| <= ln2 / 2.
    let mut p = 1.0 / 40_320.0;
    for c in [
        1.0 / 5040.0,
        1.0 / 720.0,
        1.0 / 120.0,
        1.0 / 24.0,
        1.0 / 6.0,
        0.5,
        1.0,
    ] {
        p = mul_add::<FMA>(p, r, c);
    }
    let two_k = f32::from_bits((k_int.wrapping_add(127) as u32) << 23);
    (two_k, p * r)
}

/// Within a few ulp of `f32::exp`, and vectorizable.
#[inline(always)]
fn exp<const FMA: bool>(x: f32) -> f32 {
    let (two_k, expm1_r) = exp_parts::<FMA>(x.clamp(-87.0, 88.0));
    let y = mul_add::<FMA>(two_k, expm1_r, two_k);
    // Overflow, and underflow (flushed to zero).
    let y = if x > 88.72 { f32::INFINITY } else { y };
    if x < -87.0 { 0.0 } else { y }
}

#[cfg(test)]
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
