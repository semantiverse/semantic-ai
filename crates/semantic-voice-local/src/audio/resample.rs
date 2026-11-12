//! Polyphase resampling, matching `scipy.signal.resample_poly` (which the reference
//! implementations use) with its default Kaiser window (beta = 5).

/// Resamples `input` from `from_rate` to `to_rate`.
pub fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    let mut resampler = Resampler::new(from_rate, to_rate);
    let mut output = resampler.process(input);
    output.extend(resampler.finish());
    output
}

/// Streaming [`resample`]: the output over consecutive chunks is the same as resampling
/// their concatenation at once, while resampling each chunk separately would distort every
/// chunk boundary. Output lags the input by a few samples (the half-width of the filter,
/// under half a millisecond), released by [`Resampler::finish`].
#[derive(Debug, Clone)]
pub struct Resampler {
    up: usize,
    down: usize,
    half_len: usize,
    /// The lowpass filter, `2 * half_len + 1` taps, scaled by `up`.
    taps: Vec<f64>,
    /// Input samples from index `start` on, those that outputs still need.
    buffer: Vec<f32>,
    start: usize,
    received: usize,
    produced: usize,
}

impl Resampler {
    pub fn new(from_rate: u32, to_rate: u32) -> Self {
        let g = gcd(from_rate, to_rate);
        let up = (to_rate / g) as usize;
        let down = (from_rate / g) as usize;
        let (half_len, taps) = if up == down {
            (0, vec![1.0])
        } else {
            let max_rate = up.max(down);
            let half_len = 10 * max_rate;
            let mut taps = lowpass_kaiser(2 * half_len + 1, 1.0 / max_rate as f64, 5.0);
            taps.iter_mut().for_each(|t| *t *= up as f64);
            (half_len, taps)
        };
        Self {
            up,
            down,
            half_len,
            taps,
            buffer: Vec::new(),
            start: 0,
            received: 0,
            produced: 0,
        }
    }

    /// Feeds `input`; returns the output samples that it completes.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if self.up == self.down {
            return input.to_vec();
        }
        self.buffer.extend_from_slice(input);
        self.received += input.len();
        // Output `m` reads the inputs up to `(m * down + half_len) / up`.
        let complete = (self.received * self.up)
            .checked_sub(self.half_len + 1)
            .map_or(0, |last| last / self.down + 1);
        self.produce(complete)
    }

    /// Ends the input: returns the remaining output, as if the input stopped there. The
    /// resampler then starts over.
    pub fn finish(&mut self) -> Vec<f32> {
        let output = self.produce((self.received * self.up).div_ceil(self.down));
        *self = Self {
            buffer: Vec::new(),
            start: 0,
            received: 0,
            produced: 0,
            taps: std::mem::take(&mut self.taps),
            ..*self
        };
        output
    }

    /// Outputs up to `end` (exclusive), with the inputs not received yet taken as zeros.
    fn produce(&mut self, end: usize) -> Vec<f32> {
        let (up, down, half_len) = (self.up, self.down, self.half_len);
        let output = (self.produced..end.max(self.produced))
            .map(|m| {
                // Input `i` meets tap `m * down + half_len - i * up`.
                let pos = m * down + half_len;
                let first = (m * down).saturating_sub(half_len).div_ceil(up);
                let last = (pos / up).min(self.received.saturating_sub(1));
                let acc: f64 = (first..=last)
                    .map(|i| self.taps[pos - i * up] * self.buffer[i - self.start] as f64)
                    .sum();
                acc as f32
            })
            .collect::<Vec<_>>();
        self.produced = self.produced.max(end);
        // Drop the inputs that no later output reads.
        let needed = (self.produced * down).saturating_sub(half_len).div_ceil(up);
        let drop = needed.saturating_sub(self.start).min(self.buffer.len());
        self.buffer.drain(..drop);
        self.start += drop;
        output
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `scipy.signal.firwin(num_taps, cutoff, window=("kaiser", beta))`, cutoff relative to Nyquist.
fn lowpass_kaiser(num_taps: usize, cutoff: f64, beta: f64) -> Vec<f64> {
    let center = (num_taps - 1) as f64 / 2.0;
    let i0_beta = bessel_i0(beta);
    let mut h: Vec<f64> = (0..num_taps)
        .map(|n| {
            let m = n as f64 - center;
            let ratio = 2.0 * n as f64 / (num_taps - 1) as f64 - 1.0;
            let window = bessel_i0(beta * (1.0 - ratio * ratio).max(0.0).sqrt()) / i0_beta;
            cutoff * sinc(cutoff * m) * window
        })
        .collect();
    let sum: f64 = h.iter().sum();
    h.iter_mut().for_each(|v| *v /= sum);
    h
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Modified Bessel function of the first kind, order zero.
fn bessel_i0(x: f64) -> f64 {
    let half = x / 2.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..64 {
        term *= (half / k as f64).powi(2);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The batch implementation before the streaming one, after `scipy.signal.resample_poly`.
    fn reference(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
        let g = gcd(from_rate, to_rate);
        let (up, down) = ((to_rate / g) as usize, (from_rate / g) as usize);
        let max_rate = up.max(down);
        let half_len = 10 * max_rate;
        let mut taps = lowpass_kaiser(2 * half_len + 1, 1.0 / max_rate as f64, 5.0);
        taps.iter_mut().for_each(|t| *t *= up as f64);
        let n_in = input.len();
        let n_out = (n_in * up).div_ceil(down);
        let n_pre_pad = down - half_len % down;
        let n_pre_remove = (half_len + n_pre_pad) / down;
        let upfirdn_len = |h_len: usize| ((n_in - 1) * up + h_len - 1) / down + 1;
        let mut n_post_pad = 0;
        while upfirdn_len(taps.len() + n_pre_pad + n_post_pad) < n_out + n_pre_remove {
            n_post_pad += 1;
        }
        let mut h = vec![0.0; n_pre_pad];
        h.extend_from_slice(&taps);
        h.resize(h.len() + n_post_pad, 0.0);
        (n_pre_remove..n_pre_remove + n_out)
            .map(|m| {
                let pos = m * down;
                let i_max = (pos / up).min(n_in - 1);
                let i_min = (pos + up).saturating_sub(h.len()) / up;
                let acc: f64 = (i_min..=i_max)
                    .zip(&input[i_min..=i_max])
                    .filter_map(|(i, &x)| h.get(pos - i * up).map(|tap| tap * x as f64))
                    .sum();
                acc as f32
            })
            .collect()
    }

    fn noise(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((i * 7919 + 13) % 1009) as f32 / 504.5 - 1.0)
            .collect()
    }

    #[test]
    fn matches_reference() {
        for (from, to) in [
            (44_100, 24_000),
            (48_000, 24_000),
            (16_000, 24_000),
            (24_000, 48_000),
        ] {
            for len in [1, 2, 7, 100, 4801] {
                let x = noise(len);
                let (got, expected) = (resample(&x, from, to), reference(&x, from, to));
                assert_eq!(got.len(), expected.len(), "{from} -> {to}, {len} samples");
                for (a, b) in got.iter().zip(&expected) {
                    assert!((a - b).abs() <= 1e-6, "{from} -> {to}, {len}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn streaming_matches_batch() {
        let x = noise(20_000);
        for (from, to) in [
            (44_100, 24_000),
            (48_000, 24_000),
            (24_000, 44_100),
            (24_000, 24_000),
        ] {
            let mut resampler = Resampler::new(from, to);
            let mut streamed = Vec::new();
            let mut rest = &x[..];
            for size in [1, 441, 10, 0, 3000, 7].iter().cycle() {
                if rest.is_empty() {
                    break;
                }
                let n = (*size).min(rest.len());
                streamed.extend(resampler.process(&rest[..n]));
                rest = &rest[n..];
            }
            streamed.extend(resampler.finish());
            assert_eq!(streamed, resample(&x, from, to), "{from} -> {to}");
            // Little output is held back until the end.
            let total = (x.len() * to as usize).div_ceil(from as usize);
            let held = total - Resampler::new(from, to).process(&x).len();
            assert!(held <= 25, "{from} -> {to}: {held} samples held back");
        }
    }

    #[test]
    fn identity_rate_is_noop() {
        let x = vec![0.1, -0.2, 0.3];
        assert_eq!(resample(&x, 24_000, 24_000), x);
    }

    #[test]
    fn output_length_and_tone_preserved() {
        let from = 44_100;
        let to = 24_000;
        let freq = 440.0;
        let x: Vec<f32> = (0..from)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / from as f64).sin() as f32)
            .collect();
        let y = resample(&x, from, to);
        assert_eq!(y.len(), to as usize);
        // Away from the edges the resampled signal is the same sine at the new rate.
        for (i, &v) in y.iter().enumerate().skip(1000).take(20_000) {
            let expected = (2.0 * std::f64::consts::PI * freq * i as f64 / to as f64).sin();
            assert!(
                (v as f64 - expected).abs() < 1e-3,
                "sample {i}: {v} vs {expected}"
            );
        }
    }
}
