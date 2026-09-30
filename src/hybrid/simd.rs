use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    L2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdFlavor {
    Avx512,
    Neon,
    Scalar,
}

#[derive(Debug, Clone)]
pub(crate) struct QuantizedVector {
    pub values: Vec<i8>,
    pub scale: f32,
    pub zero_point: f32,
    pub norm_sq: i64,
    pub sum: i64,
}

impl QuantizedVector {
    pub fn encode(vector: &[f32]) -> Result<Self> {
        if vector.is_empty() || vector.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invariant(
                "vectors must be non-empty and contain finite values".to_owned(),
            ));
        }
        let (minimum, maximum) = vector.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(minimum, maximum), value| (minimum.min(*value), maximum.max(*value)),
        );
        let scale = if maximum == minimum {
            1.0
        } else {
            (maximum - minimum) / 254.0
        };
        let zero_point = (maximum + minimum) * 0.5;
        let values = vector
            .iter()
            .map(|value| ((value - zero_point) / scale).round().clamp(-127.0, 127.0) as i8)
            .collect::<Vec<_>>();
        let norm_sq = values
            .iter()
            .map(|value| i64::from(*value) * i64::from(*value))
            .sum();
        let sum = values.iter().map(|value| i64::from(*value)).sum();
        Ok(Self {
            values,
            scale,
            zero_point,
            norm_sq,
            sum,
        })
    }

    pub fn from_parts(values: &[i8], scale: f32, zero_point: f32) -> Result<Self> {
        if values.is_empty() || !scale.is_finite() || scale <= 0.0 || !zero_point.is_finite() {
            return Err(Error::Invariant(
                "invalid quantized vector scale, zero point, or dimension".to_owned(),
            ));
        }
        let norm_sq = values
            .iter()
            .map(|value| i64::from(*value) * i64::from(*value))
            .sum();
        let sum = values.iter().map(|value| i64::from(*value)).sum();
        Ok(Self {
            values: values.to_vec(),
            scale,
            zero_point,
            norm_sq,
            sum,
        })
    }

    pub fn distance(&self, other: &Self, metric: DistanceMetric) -> f32 {
        debug_assert_eq!(self.values.len(), other.values.len());
        let dot = dot_i8(&self.values, &other.values) as f64;
        let dimension = self.values.len() as f64;
        let left_scale = self.scale as f64;
        let right_scale = other.scale as f64;
        let left_zero = self.zero_point as f64;
        let right_zero = other.zero_point as f64;
        let dequantized_dot = left_scale * right_scale * dot
            + left_scale * right_zero * self.sum as f64
            + right_scale * left_zero * other.sum as f64
            + dimension * left_zero * right_zero;
        let left_norm = left_scale * left_scale * self.norm_sq as f64
            + 2.0 * left_scale * left_zero * self.sum as f64
            + dimension * left_zero * left_zero;
        let right_norm = right_scale * right_scale * other.norm_sq as f64
            + 2.0 * right_scale * right_zero * other.sum as f64
            + dimension * right_zero * right_zero;
        match metric {
            DistanceMetric::Cosine => {
                if left_norm <= f64::EPSILON || right_norm <= f64::EPSILON {
                    return 1.0;
                }
                (1.0 - dequantized_dot / (left_norm.sqrt() * right_norm.sqrt())) as f32
            }
            DistanceMetric::L2 => (left_norm + right_norm - 2.0 * dequantized_dot).max(0.0) as f32,
        }
    }
}

pub fn selected_simd_flavor() -> SimdFlavor {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
        return SimdFlavor::Avx512;
    }
    #[cfg(target_arch = "aarch64")]
    {
        return SimdFlavor::Neon;
    }
    SimdFlavor::Scalar
}

pub fn dot_i8(left: &[i8], right: &[i8]) -> i64 {
    assert_eq!(left.len(), right.len(), "dot-product dimensions differ");
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
        // SAFETY: runtime detection proves the required CPU features.
        return unsafe { dot_avx512(left, right) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory in AArch64.
        return unsafe { dot_neon(left, right) };
    }
    dot_scalar(left, right)
}

pub(crate) fn dot_scalar(left: &[i8], right: &[i8]) -> i64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| i64::from(*left) * i64::from(*right))
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dot_avx512(left: &[i8], right: &[i8]) -> i64 {
    use std::arch::x86_64::{
        __m128i, __m512i, _mm512_add_epi32, _mm512_cvtepi8_epi32, _mm512_mullo_epi32,
        _mm512_setzero_si512, _mm512_storeu_si512, _mm_loadu_si128,
    };

    let mut index = 0;
    let mut accumulator = _mm512_setzero_si512();
    while index + 16 <= left.len() {
        let left_bytes = _mm_loadu_si128(left.as_ptr().add(index).cast::<__m128i>());
        let right_bytes = _mm_loadu_si128(right.as_ptr().add(index).cast::<__m128i>());
        let left_i32 = _mm512_cvtepi8_epi32(left_bytes);
        let right_i32 = _mm512_cvtepi8_epi32(right_bytes);
        accumulator = _mm512_add_epi32(accumulator, _mm512_mullo_epi32(left_i32, right_i32));
        index += 16;
    }
    let mut lanes = [0_i32; 16];
    _mm512_storeu_si512(lanes.as_mut_ptr().cast::<__m512i>(), accumulator);
    lanes.iter().map(|value| i64::from(*value)).sum::<i64>()
        + dot_scalar(&left[index..], &right[index..])
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(left: &[i8], right: &[i8]) -> i64 {
    use std::arch::aarch64::{vget_high_s8, vget_low_s8, vld1q_s8, vmull_s8, vst1q_s16};

    let mut index = 0;
    let mut total = 0_i64;
    while index + 16 <= left.len() {
        let left_bytes = vld1q_s8(left.as_ptr().add(index));
        let right_bytes = vld1q_s8(right.as_ptr().add(index));
        let products_low = vmull_s8(vget_low_s8(left_bytes), vget_low_s8(right_bytes));
        let products_high = vmull_s8(vget_high_s8(left_bytes), vget_high_s8(right_bytes));
        let mut lanes = [0_i16; 8];
        vst1q_s16(lanes.as_mut_ptr(), products_low);
        total += lanes.iter().map(|value| i64::from(*value)).sum::<i64>();
        vst1q_s16(lanes.as_mut_ptr(), products_high);
        total += lanes.iter().map(|value| i64::from(*value)).sum::<i64>();
        index += 16;
    }
    total + dot_scalar(&left[index..], &right[index..])
}
