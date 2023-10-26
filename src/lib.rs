#![feature(portable_simd)]
#![feature(stdsimd)]
#![feature(int_roundings)]

pub mod gen;
pub mod v1;

use std::arch::x86_64::{_mm256_loadu_epi32, _mm256_permutevar8x32_epi32, _mm256_storeu_epi32, _mm256_load_epi32, _mm_prefetch, _MM_HINT_T0};

use arrow2::{bitmap::{Bitmap, utils::BitChunksExact}, buffer::Buffer};

use crate::gen::MASK_ARRAY_0;


// Bitmap bit 1 means should to be filtered
pub fn filter_epi32(buffer: &Buffer<i32>, filter: &Bitmap) -> Buffer<i32> {
    // len in bit
    let len = filter.unset_bits();
    if len == 0 {
        return Buffer::<i32>::default();
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
                let src = _mm256_loadu_epi32(src_ptr.add(i*8));
                let m: u8 = (mask64 >> i*8) as u8;
                let p = _mm256_permutevar8x32_epi32(
                    src, 
                    _mm256_loadu_epi32((&MASK_ARRAY_0[m as usize]).as_ptr()));
                // change to stream store
                _mm256_storeu_epi32(dst_ptr, p);
                dst_ptr = dst_ptr.add(m.count_ones() as _);
            }
            src_ptr = src_ptr.offset(64);
            _mm_prefetch(src_ptr.add(64) as _, _MM_HINT_T0);
        }
    }
    let mut out: Buffer<i32> = dst.into();
    unsafe {
        out.set_len(len);
    }
    out 
}


#[cfg(test)]
mod test {

    use arrow2::bitmap::MutableBitmap;

    use crate::filter_epi32;

    #[test]
    fn test_filter_i32() {
        let len = 1024;
        let mut v = Vec::with_capacity(len);
        let mut filter = MutableBitmap::with_capacity(len);
        let mut expect = Vec::new();
        for i in 0..len {
            v.push(i as i32);
            // 1, 3, 5, 7 ... will keeped
            filter.push(i % 2 == 0);
            if i%2 == 0 {
                expect.push(i as i32);
            }
            // filter.set(i as usize, i % 2 == 0);
        }
        let dst = filter_epi32(&v.into(), &filter.into());
        assert_eq!(expect.len(), dst.len());
        assert_eq!(expect.as_slice(), dst.as_slice());
    }

}

