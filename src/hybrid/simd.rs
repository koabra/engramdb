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
    pub norm_sq: i64,
}

impl QuantizedVector {
    pub fn encode(vector: &[f32]) -> Result<Self> {
        if vector.is_empty() || vector.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invariant(
                "vectors must be non-empty and contain finite values".to_owned(),
            ));
        }
        let max_abs = vector
            .iter()
            .fold(0.0_f32, |maximum, value| maximum.max(value.abs()));
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let values = vector
            .iter()
            .map(|value| (value / scale).round().clamp(-127.0, 127.0) as i8)
            .collect::<Vec<_>>();
        let norm_sq = values
            .iter()
            .map(|value| i64::from(*value) * i64::from(*value))
            .sum();
        Ok(Self {
            values,
            scale,
            norm_sq,
        })
    }

    pub fn from_parts(values: &[i8], scale: f32) -> Result<Self> {
        if values.is_empty() || !scale.is_finite() || scale <= 0.0 {
            return Err(Error::Invariant(
                "invalid quantized vector scale or dimension".to_owned(),
            ));
        }
        let norm_sq = values
            .iter()
            .map(|value| i64::from(*value) * i64::from(*value))
            .sum();
        Ok(Self {
            values: values.to_vec(),
            scale,
            norm_sq,
        })
    }

    pub fn distance(&self, other: &Self, metric: DistanceMetric) -> f32 {
        debug_assert_eq!(self.values.len(), other.values.len());
        let dot = dot_i8(&self.values, &other.values) as f64;
        match metric {
            DistanceMetric::Cosine => {
                if self.norm_sq == 0 || other.norm_sq == 0 {
                    return 1.0;
                }
                (1.0 - dot / ((self.norm_sq as f64).sqrt() * (other.norm_sq as f64).sqrt())) as f32
            }
            DistanceMetric::L2 => {
                let left = self.scale as f64;
                let right = other.scale as f64;
                (left * left * self.norm_sq as f64 + right * right * other.norm_sq as f64
                    - 2.0 * left * right * dot)
                    .max(0.0) as f32
            }
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
        __m128i, __m512i, _mm512_add_epi32, _mm512_cvtepi8_epi32, _mm512_loadu_si512,
        _mm512_mullo_epi32, _mm512_setzero_si512, _mm512_storeu_si512, _mm_loadu_si128,
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
