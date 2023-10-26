#![feature(portable_simd)]
#![feature(stdsimd)]
#![feature(int_roundings)]
#![feature(avx512_target_feature)]

pub mod gen;
pub mod v1;

use std::arch::x86_64::{_mm256_loadu_epi32, _mm256_permutevar8x32_epi32, _mm256_storeu_epi32, _mm256_load_epi32};

use arrow2::{bitmap::Bitmap, buffer::Buffer};

use crate::gen::MASK_ARRAY_0;

// Bitmap bit 1 means should to be filtered
#[target_feature(enable = "avx512f,avx512vl")]
#[target_feature(enable = "avx2")]
pub unsafe fn filter_epi32(buffer: &Buffer<i32>, filter: &Bitmap) -> Buffer<i32> {
    // len in bit
    let len = filter.unset_bits();
    if len == 0 {
        return Buffer::<i32>::default();
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
            let m = &MASK_ARRAY_0[mask as usize];
            let p = _mm256_permutevar8x32_epi32(a, _mm256_loadu_epi32(m.as_ptr()));
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


#[cfg(test)]
mod test {
    use std::time::Instant;

    use arrow2::bitmap::MutableBitmap;
    use humanize_bytes::humanize_bytes_binary;

    use crate::{filter_epi32, gen::gen_input};

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
    fn test_filter_i32_1() {
        let data_size = 4 * 1024;
        let i = gen_input(data_size as _);
        let times = 1000_0;
        let start_at = Instant::now();
        for _ in 0..times {
            unsafe {
                let _ = filter_epi32(&i.0, &i.1);
            }
        }
        let cost = start_at.elapsed();
        let speed = (data_size * times * 1000_000_000) / (cost.as_nanos());
        println!("test: {} times, cost: {}ms, {}", times, cost.as_millis(), humanize_bytes_binary!(speed));
    }

}

