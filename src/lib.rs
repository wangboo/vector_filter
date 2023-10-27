#![feature(portable_simd)]
#![feature(stdsimd)]
#![feature(int_roundings)]
#![feature(avx512_target_feature)]

pub mod gen;
pub mod v1;

use std::{arch::x86_64::{_mm256_permutevar8x32_epi32, _mm256_storeu_epi32, _mm256_load_epi32, _mm_shuffle_epi8, _mm_storeu_si128, _mm_load_si128, _mm_prefetch, _mm_set_epi8, _MM_HINT_T0, __m128i, _mm_setzero_si128, _mm_set_epi64x, _mm_add_epi8, _mm_slli_epi64}, ptr::copy_nonoverlapping, mem::transmute};

use arrow2::{bitmap::Bitmap, buffer::Buffer};
use gen::{MASK_ARRAY_8_LO, MASK_ARRAY_8_HI, MASK_ADD};
use std::arch::asm;

use crate::gen::MASK_ARRAY_0;

// Bitmap bit 1 means should to be filtered
#[target_feature(enable = "avx512f,avx512vl")]
#[target_feature(enable = "avx2")]
pub unsafe fn filter_epi32(buffer: &Buffer<i32>, filter: &Bitmap) -> Buffer<i32> {
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
    let (f, offset, _) = filter.as_slice();
    debug_assert!(offset == 0, "offset in bit must eq 0");
    let mut src_ptr = buffer.as_slice().as_ptr();
    let mut dst_ptr: *mut i32 = dst.as_mut_ptr();
    unsafe {
        for &mask in f {
            // let a = _mm256_loadu_epi32(src_ptr);
            // 32byte align load
            let a = _mm256_load_epi32(src_ptr);
            let b = MASK_ARRAY_0[mask as usize];
            let p = _mm256_permutevar8x32_epi32(a, b);
            _mm256_storeu_epi32(dst_ptr, p);
            src_ptr = src_ptr.offset(8);
            dst_ptr = dst_ptr.offset(mask.count_ones() as _);
        }
    }
    let mut out: Buffer<i32> = dst.into();
    unsafe {
        out.set_len(len);
    }
    out 
}

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
            };
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
    use arrow2::bitmap::MutableBitmap;
    use crate::{filter_epi32, gen::gen_input, filter_epi8, filter_epi8_1};
    use std::arch::{asm, x86_64::{_mm_loadu_si128, __m128i, _mm256_slli_epi16}};

    #[test]
    fn test_filter_i32() {
        let mut v = Vec::with_capacity(32);
        let mut filter = MutableBitmap::with_capacity(32);
        for i in 0..32_i32 {
            v.push(i);
            // 1, 3, 5, 7 ... will keeped
            filter.push(i % 2 == 0);
            // filter.set(i as usize, i % 2 == 0);
        }
        let dst = unsafe {
            filter_epi32(&v.into(), &filter.into())
        };
        assert_eq!(16, dst.len());
        assert_eq!(&[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], dst.as_slice());
    }

    #[test]
    fn test_filter_i8() {
        let data_size = 64;
        let input = gen_input(data_size, |i| i as i8);
        let out = unsafe {
            // filter_epi8(&input.0, &input.1)
            filter_epi8_1(&input.0, &input.1)
        };
        // println!("expect: {:?}", &input.2);
        // println!("out   : {:?}", out.as_slice());
        assert_eq!(input.2.len(), out.len());
        assert_eq!(input.2.as_slice(), out.as_slice());
    }

}

