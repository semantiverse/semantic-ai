use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

use super::{dot, exp, is_cpu_f32, mul_add};

/// Causal attention of the queries `q` (`[batch, heads, t, head_dim]`, at absolute positions
/// `position..position + t`) over the keys and values `k`, `v` (`[batch, kv_heads, n,
/// head_dim]`, at absolute positions `first_key..first_key + n`, rows contiguous). Query `p`
/// sees key `j` when `j <= p` and `p - j < context`: a contiguous range of keys, so no mask is
/// built. Heads share key-value heads in groups of `heads / kv_heads`.
///
/// Returns `[batch, t, heads, head_dim]`, or `None` when the inputs are not f32 on CPU.
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    position: usize,
    first_key: usize,
    context: Option<usize>,
) -> Result<Option<Tensor>> {
    if !(is_cpu_f32(q) && is_cpu_f32(k) && is_cpu_f32(v)) {
        return Ok(None);
    }
    let op = Attention {
        position,
        first_key,
        context: context.unwrap_or(usize::MAX),
    };
    q.contiguous()?.apply_op3_no_bwd(k, v, &op).map(Some)
}

struct Attention {
    position: usize,
    first_key: usize,
    context: usize,
}

/// A rank-4 tensor whose rows (last dimension) are contiguous.
struct Rows<'a> {
    data: &'a [f32],
    offset: usize,
    strides: [usize; 3],
    dims: [usize; 4],
}

impl<'a> Rows<'a> {
    fn new(storage: &'a CpuStorage, layout: &Layout, what: &str) -> Result<Self> {
        let (CpuStorage::F32(data), &[s0, s1, s2, 1], &[d0, d1, d2, d3]) =
            (storage, layout.stride(), layout.dims())
        else {
            candle_core::bail!("attention: {what} must be rank-4 f32 with contiguous rows");
        };
        Ok(Self {
            data,
            offset: layout.start_offset(),
            strides: [s0, s1, s2],
            dims: [d0, d1, d2, d3],
        })
    }

    /// Rows `rows` of matrix `(i, j)`, as a slice starting at the first one, with the stride
    /// between rows.
    fn rows(&self, i: usize, j: usize, rows: std::ops::Range<usize>) -> (&'a [f32], usize) {
        let start = self.offset + i * self.strides[0] + j * self.strides[1];
        let stride = self.strides[2];
        let end = start + rows.end.saturating_sub(1) * stride + self.dims[3];
        let data = if rows.is_empty() {
            &[][..]
        } else {
            &self.data[start + rows.start * stride..end]
        };
        (data, stride)
    }
}

impl CustomOp3 for Attention {
    fn name(&self) -> &'static str {
        "attention-f32"
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
        let (q, k, v) = (
            Rows::new(s1, l1, "queries")?,
            Rows::new(s2, l2, "keys")?,
            Rows::new(s3, l3, "values")?,
        );
        let [b, h, t, d] = q.dims;
        let [_, kv_heads, n, _] = k.dims;
        if k.dims != v.dims || k.dims[0] != b || k.dims[3] != d || h % kv_heads != 0 {
            candle_core::bail!("attention: shapes {:?} {:?} {:?}", q.dims, k.dims, v.dims);
        }
        let scale = (d as f32).powf(-0.5);
        let mut out = vec![0f32; b * t * h * d];
        // One task per query, in the `[batch, t, heads]` order of the output.
        let run = |index: usize, out: &mut [f32]| {
            let (batch, rest) = (index / (t * h), index % (t * h));
            let (step, head) = (rest / h, rest % h);
            let kv_head = head / (h / kv_heads);
            let query = self.position + step;
            // Visible keys, as indices into `k` and `v`.
            let first = query
                .saturating_sub(self.context.saturating_sub(1))
                .max(self.first_key)
                - self.first_key;
            let end = (query + 1).saturating_sub(self.first_key).min(n);
            let (q, _) = q.rows(batch, head, step..step + 1);
            let (keys, key_stride) = k.rows(batch, kv_head, first..end);
            let (values, value_stride) = v.rows(batch, kv_head, first..end);
            attend(
                q,
                (keys, key_stride),
                (values, value_stride),
                end.saturating_sub(first),
                scale,
                out,
            );
        };
        if b * t * h * n * d >= 1 << 16 {
            out.par_chunks_mut(d)
                .enumerate()
                .for_each(|(i, out)| run(i, out));
        } else {
            out.chunks_mut(d)
                .enumerate()
                .for_each(|(i, out)| run(i, out));
        }
        Ok((CpuStorage::F32(out), Shape::from((b, t, h, d))))
    }
}

with_fma! {
    /// `out = softmax(scale q · keys) · values` over `count` keys and values, given as slices
    /// with the stride between their rows. Leaves `out` at zero when there are none.
    fn attend(
        q: &[f32],
        keys: (&[f32], usize),
        values: (&[f32], usize),
        count: usize,
        scale: f32,
        out: &mut [f32]
    ) {
        fn row((data, stride): (&[f32], usize), j: usize, len: usize) -> &[f32] {
            &data[j * stride..j * stride + len]
        }
        let d = q.len();
        let mut weights: Vec<f32> = (0..count)
            .map(|j| scale * dot::<FMA>(q, row(keys, j, d)))
            .collect();
        let Some(max) = weights.iter().copied().reduce(f32::max) else {
            return;
        };
        weights.iter_mut().for_each(|w| *w = exp::<FMA>(*w - max));
        let sum: f32 = weights.iter().sum();
        for (j, w) in weights.iter().enumerate() {
            let w = w / sum;
            for (o, v) in out.iter_mut().zip(row(values, j, d)) {
                *o = mul_add::<FMA>(w, *v, *o);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::kernels::max_diff;
    use candle_core::Device;

    #[test]
    fn attention_matches_candle() {
        let dev = Device::Cpu;
        let (heads, kv_heads, d) = (4, 2, 8);
        // (queries, keys, position of the first query, position of the first key, context)
        for (t, n, position, first_key, context) in [
            (1, 1, 0, 0, None),
            (1, 9, 8, 0, None),
            (5, 12, 7, 0, None),
            (5, 12, 10, 3, Some(4)),
            (3, 6, 20, 17, Some(6)),
        ] {
            let q = Tensor::randn(0f32, 1.0, (1, heads, t, d), &dev).unwrap();
            // Views into larger buffers, as handed out by the caches.
            let buffer = |_| Tensor::randn(0f32, 1.0, (1, kv_heads, n + 5, d), &dev).unwrap();
            let (k, v) = (buffer(0), buffer(1));
            let (k, v) = (k.narrow(2, 2, n).unwrap(), v.narrow(2, 2, n).unwrap());
            let got = attention(&q, &k, &v, position, first_key, context)
                .unwrap()
                .unwrap();

            let repeat = |x: &Tensor| {
                x.unsqueeze(2)
                    .unwrap()
                    .broadcast_as((1, kv_heads, heads / kv_heads, n, d))
                    .unwrap()
                    .reshape((1, heads, n, d))
                    .unwrap()
            };
            let mask: Vec<f32> = (position..position + t)
                .flat_map(|q| {
                    (first_key..first_key + n).map(move |k| {
                        let visible = k <= q && context.is_none_or(|c| q - k < c);
                        if visible { 0.0 } else { f32::NEG_INFINITY }
                    })
                })
                .collect();
            let mask = Tensor::from_vec(mask, (1, 1, t, n), &dev).unwrap();
            let scores = (q.matmul(&repeat(&k).t().unwrap()).unwrap() / (d as f64).sqrt())
                .unwrap()
                .broadcast_add(&mask)
                .unwrap();
            let weights = candle_nn::ops::softmax_last_dim(&scores).unwrap();
            let expected = weights
                .matmul(&repeat(&v))
                .unwrap()
                .transpose(1, 2)
                .unwrap();
            assert!(max_diff(&got, &expected) < 1e-5, "t {t}, n {n}");
        }
    }
}
