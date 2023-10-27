#![feature(portable_simd)]
#![feature(stdsimd)]
#![feature(int_roundings)]
#![feature(avx512_target_feature)]

pub mod gen;
pub mod v1;

use std::{arch::x86_64::{_mm256_permutevar8x32_epi32, _mm256_storeu_epi32, _mm_shuffle_epi8, _mm_storeu_si128, _mm_load_si128, _mm_prefetch, _MM_HINT_T0, __m128i, _mm_setzero_si128, _mm_set_epi64x, _mm_add_epi8, _mm256_loadu_epi32}, ptr::copy_nonoverlapping};
use arrow2::{bitmap::{Bitmap, utils::BitChunksExact}, buffer::Buffer};

use crate::gen::{MASK_ARRAY_0, MASK_ARRAY_8_LO, MASK_ARRAY_8_HI};

// Bitmap bit 1 means should to be filtered
#[target_feature(enable = "avx512f,avx512vl")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "sse")]
pub unsafe fn filter_epi32(buffer: &Buffer<i32>, filter: &Bitmap) -> Buffer<i32> {
    // len in bit
    let len = buffer.len() - filter.unset_bits();
    if len == 0 {
        return Buffer::<i32>::default();
    }
    if len == buffer.len() {
        return buffer.clone();
    }
    let mut dst = Vec::with_capacity(len + 8);
    unsafe {
        dst.set_len(len);
    }
    let (f, offset, f_len) = filter.as_slice();
    debug_assert!(offset == 0, "offset in bit must eq 0");
    let mut src_ptr = buffer.as_slice().as_ptr();
    let mut dst_ptr: *mut i32 = dst.as_mut_ptr();
    unsafe {
        let exact = BitChunksExact::<u64>::new(f, f_len);
        for mask64 in exact {
            // try to change to stream load
            // 32byte align load
            // let a = _mm256_load_epi32(src_ptr);
            for i in 0..8 {
                // let src = _mm256_loadu_epi32(src_ptr.add(i*8));
                let src = _mm256_loadu_epi32(src_ptr.add(i*8));
                let m: u8 = (mask64 >> i*8) as u8;
                let p = _mm256_permutevar8x32_epi32(
                    src, 
                    MASK_ARRAY_0[m as usize]);
                // change to stream store
                _mm256_storeu_epi32(dst_ptr, p);
                dst_ptr = dst_ptr.add(m.count_ones() as _);
            }
            src_ptr = src_ptr.offset(64);
            // prepare next loop data
            _mm_prefetch(src_ptr.add(256) as _, _MM_HINT_T0);
        }
    }
    let mut out: Buffer<i32> = dst.into();
    unsafe {
        out.set_len(len);
    }
    out 
}

#[target_feature(enable = "sse")]
#[target_feature(enable = "sse2")]
#[target_feature(enable = "ssse3")]
pub unsafe fn filter_epi8(buffer: &Buffer<i8>, filter: &Bitmap) -> Buffer<i8> {
    let len = buffer.len() - filter.unset_bits();
    if len == 0 {
        return Buffer::<i8>::default();
    }
    if len == buffer.len() {
        return buffer.clone();
    }
    let mut dst: Vec<i8> = Vec::with_capacity(len + 8);
    unsafe {
        dst.set_len(len);
    }
    let mut src_ptr = buffer.as_slice().as_ptr();
    let mut dst_ptr = dst.as_mut_ptr();
    let chunks = filter.chunks::<u64>();
    // 128bit mask, f0 use to fast set low u64(8byte) indexes,
    // f1 use to initialized 32 bytes memory aligned
    // f2 use to set high indexes
    union Mask128 {
        f0: (u64, u64),
        f1: __m128i,
        f2: [i8; 16],
    }
    for mask64 in chunks {
        for i in 0..4 {
            let mut mask = Mask128{ f1: _mm_setzero_si128() };
            let mlo = (mask64 >> i*16) as u8;
            let mhi = (mask64 >> (i*16+8)) as u8;
            let offset = mlo.count_ones() as usize;
            // set low indexes
            mask.f0.0 = MASK_ARRAY_8_LO[mlo as usize];
            // set hi indexes
            copy_nonoverlapping::<i8>(
                MASK_ARRAY_8_HI.as_ptr().offset(mhi as isize) as _, 
                mask.f2.as_mut_ptr().add(offset), 
                8);
            let p = _mm_shuffle_epi8(
                _mm_load_si128(src_ptr  as _), 
                mask.f1);
            _mm_storeu_si128(dst_ptr as _, p);
            src_ptr = src_ptr.add(16);
            dst_ptr = dst_ptr.add(offset + mhi.count_ones() as usize);
        }
        _mm_prefetch(src_ptr.add(64), _MM_HINT_T0);
    }

    let mut out: Buffer<i8> = dst.into();
    unsafe {
        out.set_len(len);
    }
    out
}

#[target_feature(enable = "sse")]
#[target_feature(enable = "sse2")]
#[target_feature(enable = "ssse3")]
pub unsafe fn filter_epi8_1(buffer: &Buffer<i8>, filter: &Bitmap) -> Buffer<i8> {
    let len = buffer.len() - filter.unset_bits();
    if len == 0 {
        return Buffer::<i8>::default();
    }
    if len == buffer.len() {
        return buffer.clone();
    }
    let mut dst: Vec<i8> = Vec::with_capacity(len + 8);
    unsafe {
        dst.set_len(len);
    }
    let mut src_ptr = buffer.as_slice().as_ptr();
    let mut dst_ptr = dst.as_mut_ptr();
    let chunks = filter.chunks::<u64>();
    for mask64 in chunks {
        for i in 0..4 {
            let mlo = (mask64 >> i*16) as u8;
            let mhi = (mask64 >> (i*16+8)) as u8;
            // println!("mask: 0b_{:08b}_{:08b}", mhi, mlo);
            // set low indexes
            // eg: [0,2,3,0,0,0,0,0,9,10,0,0,0,0,0,0,0,0]
            // to: [0,2,3,9,10,0,0,0,...]
            union Mask {
                f0: __m128i,
                f1: u128,
            }
            // eg: [0,0,0,0,0,0,0,0,8,9,0,0,0,0,0,0]
            let mut idx_hi = Mask {
                f0: _mm_set_epi64x(MASK_ARRAY_8_HI[mhi as usize] as _, 0)
            };
            // offset = 5, shift to: [0,0,0,8,9,0,0,0,0,0,0,0,0,0,0,0]
            idx_hi.f1 = idx_hi.f1 >> mlo.count_zeros() * 8;
            // eg: [0,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0]
            let idx_lo = _mm_set_epi64x(0, MASK_ARRAY_8_LO[mlo as usize] as _);
            // idx: [0,1,2,8,9,0,0,0,0,0,0,0,0,0,0,0]
            let idx = _mm_add_epi8(idx_hi.f0, idx_lo);
            // shuffle
            let p = _mm_shuffle_epi8(_mm_load_si128(src_ptr as _), idx);
            _mm_storeu_si128(dst_ptr as _, p);
            src_ptr = src_ptr.add(16);
            dst_ptr = dst_ptr.add((mlo.count_ones() + mhi.count_ones()) as usize);
        }
        _mm_prefetch(src_ptr.add(64), _MM_HINT_T0);
    }

    let mut out: Buffer<i8> = dst.into();
    unsafe {
        out.set_len(len);
    }
    out
}

#[cfg(test)]
mod test {

    use crate::{filter_epi32, gen::gen_input, filter_epi8_1};

    #[test]
    fn test_filter_i32() {
        let len = 1024;
        let (buffer, bitmap, expect) = gen_input(len, |i| i as _);
        let dst = unsafe {
            filter_epi32(&buffer, &bitmap)
        };
        assert_eq!(expect.len(), dst.len());
        assert_eq!(expect.as_slice(), dst.as_slice());
    }

    #[test]
    fn test_filter_i8() {
        let len = 1024;
        let (buffer, bitmap, expect) = gen_input(len, |i| i as _);
        let dst = unsafe {
            filter_epi8_1(&buffer, &bitmap)
        };
        assert_eq!(expect.len(), dst.len());
        assert_eq!(expect.as_slice(), dst.as_slice());
    }

}

