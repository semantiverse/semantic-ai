use candle_core::{CpuStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

use super::{contiguous, dot, is_cpu_f32};

/// `x · wᵀ (+ bias)` for a single row `x` (`[.., k]` with one row) and `w` (`[n, k]`): the
/// products of streaming inference. They are bound by memory bandwidth, which candle leaves
/// mostly unused by running them on a single core. `None` when the inputs do not fit, for the
/// caller to fall back to candle.
pub fn matvec(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<Option<Tensor>> {
    let k = x.dim(candle_core::D::Minus1)?;
    if !(is_cpu_f32(x) && is_cpu_f32(w) && bias.is_none_or(is_cpu_f32))
        || x.elem_count() != k
        || w.dims2()?.1 != k
    {
        return Ok(None);
    }
    let x = x.contiguous()?;
    let out = match bias {
        None => x.apply_op2_no_bwd(w, &MatVec)?,
        Some(bias) => x.apply_op3_no_bwd(w, bias, &MatVec)?,
    };
    Ok(Some(out))
}

struct MatVec;

impl MatVec {
    fn run(
        &self,
        x: (&CpuStorage, &Layout),
        w: (&CpuStorage, &Layout),
        bias: Option<(&CpuStorage, &Layout)>,
    ) -> Result<(CpuStorage, Shape)> {
        let x_values = contiguous(x.0, x.1, "matvec input")?;
        let ws = contiguous(w.0, w.1, "matvec weight")?;
        let bias = bias
            .map(|(s, l)| contiguous(s, l, "matvec bias"))
            .transpose()?;
        let (n, k) = w.1.shape().dims2()?;
        let mut out = vec![0f32; n];
        // Enough rows per task to amortize scheduling, enough tasks to balance the cores.
        let rows = (n / (4 * rayon::current_num_threads())).clamp(8, 256);
        out.par_chunks_mut(rows)
            .zip(ws.par_chunks(rows * k))
            .enumerate()
            .for_each(|(block, (out, ws))| {
                dot_rows(x_values, ws, out);
                if let Some(bias) = bias {
                    let bias = &bias[block * rows..];
                    out.iter_mut().zip(bias).for_each(|(o, b)| *o += b);
                }
            });
        let mut dims = x.1.shape().dims().to_vec();
        *dims.last_mut().expect("at least one dimension") = n;
        Ok((CpuStorage::F32(out), Shape::from(dims)))
    }
}

impl CustomOp2 for MatVec {
    fn name(&self) -> &'static str {
        "matvec-f32"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        self.run((s1, l1), (s2, l2), None)
    }
}

impl CustomOp3 for MatVec {
    fn name(&self) -> &'static str {
        "matvec-bias-f32"
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
        self.run((s1, l1), (s2, l2), Some((s3, l3)))
    }
}

with_fma! {
    /// `out[j] = x · ws[j]` for the rows of `ws`.
    fn dot_rows(x: &[f32], ws: &[f32], out: &mut [f32]) {
        for (o, w) in out.iter_mut().zip(ws.chunks_exact(x.len())) {
            *o = dot::<FMA>(x, w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::kernels::max_diff;
    use candle_core::Device;

    #[test]
    fn matvec_matches_candle() {
        let dev = Device::Cpu;
        for (n, k) in [(1, 1), (7, 5), (300, 64), (1024, 1000), (3, 4096)] {
            let w = Tensor::randn(0f32, 1.0, (n, k), &dev).unwrap();
            let b = Tensor::randn(0f32, 1.0, n, &dev).unwrap();
            let x = Tensor::randn(0f32, 1.0, (1, 1, k), &dev).unwrap();
            let expected = x.broadcast_matmul(&w.t().unwrap()).unwrap();
            let tolerance = 1e-5 * k as f32;
            let got = matvec(&x, &w, None).unwrap().unwrap();
            assert_eq!(got.dims(), &[1, 1, n]);
            assert!(max_diff(&got, &expected) < tolerance);
            let with_bias = matvec(&x, &w, Some(&b)).unwrap().unwrap();
            let expected = expected.broadcast_add(&b).unwrap();
            assert!(max_diff(&with_bias, &expected) < tolerance);
        }
        let x = Tensor::zeros((2, 5), candle_core::DType::F32, &dev).unwrap();
        let w = Tensor::zeros((3, 5), candle_core::DType::F32, &dev).unwrap();
        assert!(matvec(&x, &w, None).unwrap().is_none());
    }
}
