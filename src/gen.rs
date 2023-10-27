use std::{arch::x86_64::{__m256i, _mm_set_epi64x, __m128i}, simd::{i32x8, i64x2}, mem::transmute};

use arrow2::{buffer::Buffer, bitmap::{Bitmap, MutableBitmap}};
use rand::{rngs::ThreadRng, Rng};


pub const MASK_ARRAY_0: [__m256i; 256] = [
    gen_init_mm256i([0, 0, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 0, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 0, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([2, 0, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 2, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 0, 0, 0, 0, 0]),
    gen_init_mm256i([3, 0, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 3, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 3, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 3, 0, 0, 0, 0, 0]),
    gen_init_mm256i([2, 3, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 3, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 2, 3, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 3, 0, 0, 0, 0]),
    gen_init_mm256i([4, 0, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 4, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 4, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 4, 0, 0, 0, 0, 0]),
    gen_init_mm256i([2, 4, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 4, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 2, 4, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 4, 0, 0, 0, 0]),
    gen_init_mm256i([3, 4, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 3, 4, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 3, 4, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 3, 4, 0, 0, 0, 0]),
    gen_init_mm256i([2, 3, 4, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 3, 4, 0, 0, 0, 0]),
    gen_init_mm256i([1, 2, 3, 4, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 3, 4, 0, 0, 0]),
    gen_init_mm256i([5, 0, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 5, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 5, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 5, 0, 0, 0, 0, 0]),
    gen_init_mm256i([2, 5, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 5, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 2, 5, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 5, 0, 0, 0, 0]),
    gen_init_mm256i([3, 5, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 3, 5, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 3, 5, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 3, 5, 0, 0, 0, 0]),
    gen_init_mm256i([2, 3, 5, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 3, 5, 0, 0, 0, 0]),
    gen_init_mm256i([1, 2, 3, 5, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 3, 5, 0, 0, 0]),
    gen_init_mm256i([4, 5, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 4, 5, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 4, 5, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 4, 5, 0, 0, 0, 0]),
    gen_init_mm256i([2, 4, 5, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 4, 5, 0, 0, 0, 0]),
    gen_init_mm256i([1, 2, 4, 5, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 4, 5, 0, 0, 0]),
    gen_init_mm256i([3, 4, 5, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 3, 4, 5, 0, 0, 0, 0]),
    gen_init_mm256i([1, 3, 4, 5, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 3, 4, 5, 0, 0, 0]),
    gen_init_mm256i([2, 3, 4, 5, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 3, 4, 5, 0, 0, 0]),
    gen_init_mm256i([1, 2, 3, 4, 5, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 3, 4, 5, 0, 0]),
    gen_init_mm256i([6, 0, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 6, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 6, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 6, 0, 0, 0, 0, 0]),
    gen_init_mm256i([2, 6, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 6, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 2, 6, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 6, 0, 0, 0, 0]),
    gen_init_mm256i([3, 6, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 3, 6, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 3, 6, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 3, 6, 0, 0, 0, 0]),
    gen_init_mm256i([2, 3, 6, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 3, 6, 0, 0, 0, 0]),
    gen_init_mm256i([1, 2, 3, 6, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 3, 6, 0, 0, 0]),
    gen_init_mm256i([4, 6, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 4, 6, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 4, 6, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 4, 6, 0, 0, 0, 0]),
    gen_init_mm256i([2, 4, 6, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 4, 6, 0, 0, 0, 0]),
    gen_init_mm256i([1, 2, 4, 6, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 4, 6, 0, 0, 0]),
    gen_init_mm256i([3, 4, 6, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 3, 4, 6, 0, 0, 0, 0]),
    gen_init_mm256i([1, 3, 4, 6, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 3, 4, 6, 0, 0, 0]),
    gen_init_mm256i([2, 3, 4, 6, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 3, 4, 6, 0, 0, 0]),
    gen_init_mm256i([1, 2, 3, 4, 6, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 3, 4, 6, 0, 0]),
    gen_init_mm256i([5, 6, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 5, 6, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 5, 6, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 5, 6, 0, 0, 0, 0]),
    gen_init_mm256i([2, 5, 6, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 5, 6, 0, 0, 0, 0]),
    gen_init_mm256i([1, 2, 5, 6, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 5, 6, 0, 0, 0]),
    gen_init_mm256i([3, 5, 6, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 3, 5, 6, 0, 0, 0, 0]),
    gen_init_mm256i([1, 3, 5, 6, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 3, 5, 6, 0, 0, 0]),
    gen_init_mm256i([2, 3, 5, 6, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 3, 5, 6, 0, 0, 0]),
    gen_init_mm256i([1, 2, 3, 5, 6, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 3, 5, 6, 0, 0]),
    gen_init_mm256i([4, 5, 6, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 4, 5, 6, 0, 0, 0, 0]),
    gen_init_mm256i([1, 4, 5, 6, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 4, 5, 6, 0, 0, 0]),
    gen_init_mm256i([2, 4, 5, 6, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 4, 5, 6, 0, 0, 0]),
    gen_init_mm256i([1, 2, 4, 5, 6, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 4, 5, 6, 0, 0]),
    gen_init_mm256i([3, 4, 5, 6, 0, 0, 0, 0]),
    gen_init_mm256i([0, 3, 4, 5, 6, 0, 0, 0]),
    gen_init_mm256i([1, 3, 4, 5, 6, 0, 0, 0]),
    gen_init_mm256i([0, 1, 3, 4, 5, 6, 0, 0]),
    gen_init_mm256i([2, 3, 4, 5, 6, 0, 0, 0]),
    gen_init_mm256i([0, 2, 3, 4, 5, 6, 0, 0]),
    gen_init_mm256i([1, 2, 3, 4, 5, 6, 0, 0]),
    gen_init_mm256i([0, 1, 2, 3, 4, 5, 6, 0]),
    gen_init_mm256i([7, 0, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 7, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 7, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([2, 7, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 2, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 7, 0, 0, 0, 0]),
    gen_init_mm256i([3, 7, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 3, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 3, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 3, 7, 0, 0, 0, 0]),
    gen_init_mm256i([2, 3, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 3, 7, 0, 0, 0, 0]),
    gen_init_mm256i([1, 2, 3, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 3, 7, 0, 0, 0]),
    gen_init_mm256i([4, 7, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 4, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 4, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 4, 7, 0, 0, 0, 0]),
    gen_init_mm256i([2, 4, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 4, 7, 0, 0, 0, 0]),
    gen_init_mm256i([1, 2, 4, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 4, 7, 0, 0, 0]),
    gen_init_mm256i([3, 4, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 3, 4, 7, 0, 0, 0, 0]),
    gen_init_mm256i([1, 3, 4, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 3, 4, 7, 0, 0, 0]),
    gen_init_mm256i([2, 3, 4, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 3, 4, 7, 0, 0, 0]),
    gen_init_mm256i([1, 2, 3, 4, 7, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 3, 4, 7, 0, 0]),
    gen_init_mm256i([5, 7, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 5, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 5, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 5, 7, 0, 0, 0, 0]),
    gen_init_mm256i([2, 5, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 5, 7, 0, 0, 0, 0]),
    gen_init_mm256i([1, 2, 5, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 5, 7, 0, 0, 0]),
    gen_init_mm256i([3, 5, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 3, 5, 7, 0, 0, 0, 0]),
    gen_init_mm256i([1, 3, 5, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 3, 5, 7, 0, 0, 0]),
    gen_init_mm256i([2, 3, 5, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 3, 5, 7, 0, 0, 0]),
    gen_init_mm256i([1, 2, 3, 5, 7, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 3, 5, 7, 0, 0]),
    gen_init_mm256i([4, 5, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 4, 5, 7, 0, 0, 0, 0]),
    gen_init_mm256i([1, 4, 5, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 4, 5, 7, 0, 0, 0]),
    gen_init_mm256i([2, 4, 5, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 4, 5, 7, 0, 0, 0]),
    gen_init_mm256i([1, 2, 4, 5, 7, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 4, 5, 7, 0, 0]),
    gen_init_mm256i([3, 4, 5, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 3, 4, 5, 7, 0, 0, 0]),
    gen_init_mm256i([1, 3, 4, 5, 7, 0, 0, 0]),
    gen_init_mm256i([0, 1, 3, 4, 5, 7, 0, 0]),
    gen_init_mm256i([2, 3, 4, 5, 7, 0, 0, 0]),
    gen_init_mm256i([0, 2, 3, 4, 5, 7, 0, 0]),
    gen_init_mm256i([1, 2, 3, 4, 5, 7, 0, 0]),
    gen_init_mm256i([0, 1, 2, 3, 4, 5, 7, 0]),
    gen_init_mm256i([6, 7, 0, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 6, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([1, 6, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 6, 7, 0, 0, 0, 0]),
    gen_init_mm256i([2, 6, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 6, 7, 0, 0, 0, 0]),
    gen_init_mm256i([1, 2, 6, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 6, 7, 0, 0, 0]),
    gen_init_mm256i([3, 6, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 3, 6, 7, 0, 0, 0, 0]),
    gen_init_mm256i([1, 3, 6, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 3, 6, 7, 0, 0, 0]),
    gen_init_mm256i([2, 3, 6, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 3, 6, 7, 0, 0, 0]),
    gen_init_mm256i([1, 2, 3, 6, 7, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 3, 6, 7, 0, 0]),
    gen_init_mm256i([4, 6, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 4, 6, 7, 0, 0, 0, 0]),
    gen_init_mm256i([1, 4, 6, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 4, 6, 7, 0, 0, 0]),
    gen_init_mm256i([2, 4, 6, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 4, 6, 7, 0, 0, 0]),
    gen_init_mm256i([1, 2, 4, 6, 7, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 4, 6, 7, 0, 0]),
    gen_init_mm256i([3, 4, 6, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 3, 4, 6, 7, 0, 0, 0]),
    gen_init_mm256i([1, 3, 4, 6, 7, 0, 0, 0]),
    gen_init_mm256i([0, 1, 3, 4, 6, 7, 0, 0]),
    gen_init_mm256i([2, 3, 4, 6, 7, 0, 0, 0]),
    gen_init_mm256i([0, 2, 3, 4, 6, 7, 0, 0]),
    gen_init_mm256i([1, 2, 3, 4, 6, 7, 0, 0]),
    gen_init_mm256i([0, 1, 2, 3, 4, 6, 7, 0]),
    gen_init_mm256i([5, 6, 7, 0, 0, 0, 0, 0]),
    gen_init_mm256i([0, 5, 6, 7, 0, 0, 0, 0]),
    gen_init_mm256i([1, 5, 6, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 1, 5, 6, 7, 0, 0, 0]),
    gen_init_mm256i([2, 5, 6, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 2, 5, 6, 7, 0, 0, 0]),
    gen_init_mm256i([1, 2, 5, 6, 7, 0, 0, 0]),
    gen_init_mm256i([0, 1, 2, 5, 6, 7, 0, 0]),
    gen_init_mm256i([3, 5, 6, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 3, 5, 6, 7, 0, 0, 0]),
    gen_init_mm256i([1, 3, 5, 6, 7, 0, 0, 0]),
    gen_init_mm256i([0, 1, 3, 5, 6, 7, 0, 0]),
    gen_init_mm256i([2, 3, 5, 6, 7, 0, 0, 0]),
    gen_init_mm256i([0, 2, 3, 5, 6, 7, 0, 0]),
    gen_init_mm256i([1, 2, 3, 5, 6, 7, 0, 0]),
    gen_init_mm256i([0, 1, 2, 3, 5, 6, 7, 0]),
    gen_init_mm256i([4, 5, 6, 7, 0, 0, 0, 0]),
    gen_init_mm256i([0, 4, 5, 6, 7, 0, 0, 0]),
    gen_init_mm256i([1, 4, 5, 6, 7, 0, 0, 0]),
    gen_init_mm256i([0, 1, 4, 5, 6, 7, 0, 0]),
    gen_init_mm256i([2, 4, 5, 6, 7, 0, 0, 0]),
    gen_init_mm256i([0, 2, 4, 5, 6, 7, 0, 0]),
    gen_init_mm256i([1, 2, 4, 5, 6, 7, 0, 0]),
    gen_init_mm256i([0, 1, 2, 4, 5, 6, 7, 0]),
    gen_init_mm256i([3, 4, 5, 6, 7, 0, 0, 0]),
    gen_init_mm256i([0, 3, 4, 5, 6, 7, 0, 0]),
    gen_init_mm256i([1, 3, 4, 5, 6, 7, 0, 0]),
    gen_init_mm256i([0, 1, 3, 4, 5, 6, 7, 0]),
    gen_init_mm256i([2, 3, 4, 5, 6, 7, 0, 0]),
    gen_init_mm256i([0, 2, 3, 4, 5, 6, 7, 0]),
    gen_init_mm256i([1, 2, 3, 4, 5, 6, 7, 0]),
    gen_init_mm256i([0, 1, 2, 3, 4, 5, 6, 7]),
];


pub const MASK_ARRAY_8_LO: [u64; 256] = [
    gen_init_u64([0, 0, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 0, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 0, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([2, 0, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 2, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 0, 0, 0, 0, 0]),
    gen_init_u64([3, 0, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 3, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 3, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 3, 0, 0, 0, 0, 0]),
    gen_init_u64([2, 3, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 3, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 2, 3, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 3, 0, 0, 0, 0]),
    gen_init_u64([4, 0, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 4, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 4, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 4, 0, 0, 0, 0, 0]),
    gen_init_u64([2, 4, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 4, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 2, 4, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 4, 0, 0, 0, 0]),
    gen_init_u64([3, 4, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 3, 4, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 3, 4, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 3, 4, 0, 0, 0, 0]),
    gen_init_u64([2, 3, 4, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 3, 4, 0, 0, 0, 0]),
    gen_init_u64([1, 2, 3, 4, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 3, 4, 0, 0, 0]),
    gen_init_u64([5, 0, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 5, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 5, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 5, 0, 0, 0, 0, 0]),
    gen_init_u64([2, 5, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 5, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 2, 5, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 5, 0, 0, 0, 0]),
    gen_init_u64([3, 5, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 3, 5, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 3, 5, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 3, 5, 0, 0, 0, 0]),
    gen_init_u64([2, 3, 5, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 3, 5, 0, 0, 0, 0]),
    gen_init_u64([1, 2, 3, 5, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 3, 5, 0, 0, 0]),
    gen_init_u64([4, 5, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 4, 5, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 4, 5, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 4, 5, 0, 0, 0, 0]),
    gen_init_u64([2, 4, 5, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 4, 5, 0, 0, 0, 0]),
    gen_init_u64([1, 2, 4, 5, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 4, 5, 0, 0, 0]),
    gen_init_u64([3, 4, 5, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 3, 4, 5, 0, 0, 0, 0]),
    gen_init_u64([1, 3, 4, 5, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 3, 4, 5, 0, 0, 0]),
    gen_init_u64([2, 3, 4, 5, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 3, 4, 5, 0, 0, 0]),
    gen_init_u64([1, 2, 3, 4, 5, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 3, 4, 5, 0, 0]),
    gen_init_u64([6, 0, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 6, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 6, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 6, 0, 0, 0, 0, 0]),
    gen_init_u64([2, 6, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 6, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 2, 6, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 6, 0, 0, 0, 0]),
    gen_init_u64([3, 6, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 3, 6, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 3, 6, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 3, 6, 0, 0, 0, 0]),
    gen_init_u64([2, 3, 6, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 3, 6, 0, 0, 0, 0]),
    gen_init_u64([1, 2, 3, 6, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 3, 6, 0, 0, 0]),
    gen_init_u64([4, 6, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 4, 6, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 4, 6, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 4, 6, 0, 0, 0, 0]),
    gen_init_u64([2, 4, 6, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 4, 6, 0, 0, 0, 0]),
    gen_init_u64([1, 2, 4, 6, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 4, 6, 0, 0, 0]),
    gen_init_u64([3, 4, 6, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 3, 4, 6, 0, 0, 0, 0]),
    gen_init_u64([1, 3, 4, 6, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 3, 4, 6, 0, 0, 0]),
    gen_init_u64([2, 3, 4, 6, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 3, 4, 6, 0, 0, 0]),
    gen_init_u64([1, 2, 3, 4, 6, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 3, 4, 6, 0, 0]),
    gen_init_u64([5, 6, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 5, 6, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 5, 6, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 5, 6, 0, 0, 0, 0]),
    gen_init_u64([2, 5, 6, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 5, 6, 0, 0, 0, 0]),
    gen_init_u64([1, 2, 5, 6, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 5, 6, 0, 0, 0]),
    gen_init_u64([3, 5, 6, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 3, 5, 6, 0, 0, 0, 0]),
    gen_init_u64([1, 3, 5, 6, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 3, 5, 6, 0, 0, 0]),
    gen_init_u64([2, 3, 5, 6, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 3, 5, 6, 0, 0, 0]),
    gen_init_u64([1, 2, 3, 5, 6, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 3, 5, 6, 0, 0]),
    gen_init_u64([4, 5, 6, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 4, 5, 6, 0, 0, 0, 0]),
    gen_init_u64([1, 4, 5, 6, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 4, 5, 6, 0, 0, 0]),
    gen_init_u64([2, 4, 5, 6, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 4, 5, 6, 0, 0, 0]),
    gen_init_u64([1, 2, 4, 5, 6, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 4, 5, 6, 0, 0]),
    gen_init_u64([3, 4, 5, 6, 0, 0, 0, 0]),
    gen_init_u64([0, 3, 4, 5, 6, 0, 0, 0]),
    gen_init_u64([1, 3, 4, 5, 6, 0, 0, 0]),
    gen_init_u64([0, 1, 3, 4, 5, 6, 0, 0]),
    gen_init_u64([2, 3, 4, 5, 6, 0, 0, 0]),
    gen_init_u64([0, 2, 3, 4, 5, 6, 0, 0]),
    gen_init_u64([1, 2, 3, 4, 5, 6, 0, 0]),
    gen_init_u64([0, 1, 2, 3, 4, 5, 6, 0]),
    gen_init_u64([7, 0, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 7, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 7, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([2, 7, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 2, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 7, 0, 0, 0, 0]),
    gen_init_u64([3, 7, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 3, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 3, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 3, 7, 0, 0, 0, 0]),
    gen_init_u64([2, 3, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 3, 7, 0, 0, 0, 0]),
    gen_init_u64([1, 2, 3, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 3, 7, 0, 0, 0]),
    gen_init_u64([4, 7, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 4, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 4, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 4, 7, 0, 0, 0, 0]),
    gen_init_u64([2, 4, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 4, 7, 0, 0, 0, 0]),
    gen_init_u64([1, 2, 4, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 4, 7, 0, 0, 0]),
    gen_init_u64([3, 4, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 3, 4, 7, 0, 0, 0, 0]),
    gen_init_u64([1, 3, 4, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 3, 4, 7, 0, 0, 0]),
    gen_init_u64([2, 3, 4, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 3, 4, 7, 0, 0, 0]),
    gen_init_u64([1, 2, 3, 4, 7, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 3, 4, 7, 0, 0]),
    gen_init_u64([5, 7, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 5, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 5, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 5, 7, 0, 0, 0, 0]),
    gen_init_u64([2, 5, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 5, 7, 0, 0, 0, 0]),
    gen_init_u64([1, 2, 5, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 5, 7, 0, 0, 0]),
    gen_init_u64([3, 5, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 3, 5, 7, 0, 0, 0, 0]),
    gen_init_u64([1, 3, 5, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 3, 5, 7, 0, 0, 0]),
    gen_init_u64([2, 3, 5, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 3, 5, 7, 0, 0, 0]),
    gen_init_u64([1, 2, 3, 5, 7, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 3, 5, 7, 0, 0]),
    gen_init_u64([4, 5, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 4, 5, 7, 0, 0, 0, 0]),
    gen_init_u64([1, 4, 5, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 4, 5, 7, 0, 0, 0]),
    gen_init_u64([2, 4, 5, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 4, 5, 7, 0, 0, 0]),
    gen_init_u64([1, 2, 4, 5, 7, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 4, 5, 7, 0, 0]),
    gen_init_u64([3, 4, 5, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 3, 4, 5, 7, 0, 0, 0]),
    gen_init_u64([1, 3, 4, 5, 7, 0, 0, 0]),
    gen_init_u64([0, 1, 3, 4, 5, 7, 0, 0]),
    gen_init_u64([2, 3, 4, 5, 7, 0, 0, 0]),
    gen_init_u64([0, 2, 3, 4, 5, 7, 0, 0]),
    gen_init_u64([1, 2, 3, 4, 5, 7, 0, 0]),
    gen_init_u64([0, 1, 2, 3, 4, 5, 7, 0]),
    gen_init_u64([6, 7, 0, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 6, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([1, 6, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 6, 7, 0, 0, 0, 0]),
    gen_init_u64([2, 6, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 6, 7, 0, 0, 0, 0]),
    gen_init_u64([1, 2, 6, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 6, 7, 0, 0, 0]),
    gen_init_u64([3, 6, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 3, 6, 7, 0, 0, 0, 0]),
    gen_init_u64([1, 3, 6, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 3, 6, 7, 0, 0, 0]),
    gen_init_u64([2, 3, 6, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 3, 6, 7, 0, 0, 0]),
    gen_init_u64([1, 2, 3, 6, 7, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 3, 6, 7, 0, 0]),
    gen_init_u64([4, 6, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 4, 6, 7, 0, 0, 0, 0]),
    gen_init_u64([1, 4, 6, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 4, 6, 7, 0, 0, 0]),
    gen_init_u64([2, 4, 6, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 4, 6, 7, 0, 0, 0]),
    gen_init_u64([1, 2, 4, 6, 7, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 4, 6, 7, 0, 0]),
    gen_init_u64([3, 4, 6, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 3, 4, 6, 7, 0, 0, 0]),
    gen_init_u64([1, 3, 4, 6, 7, 0, 0, 0]),
    gen_init_u64([0, 1, 3, 4, 6, 7, 0, 0]),
    gen_init_u64([2, 3, 4, 6, 7, 0, 0, 0]),
    gen_init_u64([0, 2, 3, 4, 6, 7, 0, 0]),
    gen_init_u64([1, 2, 3, 4, 6, 7, 0, 0]),
    gen_init_u64([0, 1, 2, 3, 4, 6, 7, 0]),
    gen_init_u64([5, 6, 7, 0, 0, 0, 0, 0]),
    gen_init_u64([0, 5, 6, 7, 0, 0, 0, 0]),
    gen_init_u64([1, 5, 6, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 1, 5, 6, 7, 0, 0, 0]),
    gen_init_u64([2, 5, 6, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 2, 5, 6, 7, 0, 0, 0]),
    gen_init_u64([1, 2, 5, 6, 7, 0, 0, 0]),
    gen_init_u64([0, 1, 2, 5, 6, 7, 0, 0]),
    gen_init_u64([3, 5, 6, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 3, 5, 6, 7, 0, 0, 0]),
    gen_init_u64([1, 3, 5, 6, 7, 0, 0, 0]),
    gen_init_u64([0, 1, 3, 5, 6, 7, 0, 0]),
    gen_init_u64([2, 3, 5, 6, 7, 0, 0, 0]),
    gen_init_u64([0, 2, 3, 5, 6, 7, 0, 0]),
    gen_init_u64([1, 2, 3, 5, 6, 7, 0, 0]),
    gen_init_u64([0, 1, 2, 3, 5, 6, 7, 0]),
    gen_init_u64([4, 5, 6, 7, 0, 0, 0, 0]),
    gen_init_u64([0, 4, 5, 6, 7, 0, 0, 0]),
    gen_init_u64([1, 4, 5, 6, 7, 0, 0, 0]),
    gen_init_u64([0, 1, 4, 5, 6, 7, 0, 0]),
    gen_init_u64([2, 4, 5, 6, 7, 0, 0, 0]),
    gen_init_u64([0, 2, 4, 5, 6, 7, 0, 0]),
    gen_init_u64([1, 2, 4, 5, 6, 7, 0, 0]),
    gen_init_u64([0, 1, 2, 4, 5, 6, 7, 0]),
    gen_init_u64([3, 4, 5, 6, 7, 0, 0, 0]),
    gen_init_u64([0, 3, 4, 5, 6, 7, 0, 0]),
    gen_init_u64([1, 3, 4, 5, 6, 7, 0, 0]),
    gen_init_u64([0, 1, 3, 4, 5, 6, 7, 0]),
    gen_init_u64([2, 3, 4, 5, 6, 7, 0, 0]),
    gen_init_u64([0, 2, 3, 4, 5, 6, 7, 0]),
    gen_init_u64([1, 2, 3, 4, 5, 6, 7, 0]),
    gen_init_u64([0, 1, 2, 3, 4, 5, 6, 7]),
];

pub const MASK_ARRAY_8_HI: [u64; 256] = [
    gen_init_u64([0, 0, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 0, 0, 0, 0, 0, 0, 0]),
gen_init_u64([9, 0, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 0, 0, 0, 0, 0, 0]),
gen_init_u64([10, 0, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 10, 0, 0, 0, 0, 0, 0]),
gen_init_u64([9, 10, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 10, 0, 0, 0, 0, 0]),
gen_init_u64([11, 0, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 11, 0, 0, 0, 0, 0, 0]),
gen_init_u64([9, 11, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 11, 0, 0, 0, 0, 0]),
gen_init_u64([10, 11, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 10, 11, 0, 0, 0, 0, 0]),
gen_init_u64([9, 10, 11, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 10, 11, 0, 0, 0, 0]),
gen_init_u64([12, 0, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 12, 0, 0, 0, 0, 0, 0]),
gen_init_u64([9, 12, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 12, 0, 0, 0, 0, 0]),
gen_init_u64([10, 12, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 10, 12, 0, 0, 0, 0, 0]),
gen_init_u64([9, 10, 12, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 10, 12, 0, 0, 0, 0]),
gen_init_u64([11, 12, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 11, 12, 0, 0, 0, 0, 0]),
gen_init_u64([9, 11, 12, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 11, 12, 0, 0, 0, 0]),
gen_init_u64([10, 11, 12, 0, 0, 0, 0, 0]),
gen_init_u64([8, 10, 11, 12, 0, 0, 0, 0]),
gen_init_u64([9, 10, 11, 12, 0, 0, 0, 0]),
gen_init_u64([8, 9, 10, 11, 12, 0, 0, 0]),
gen_init_u64([13, 0, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 13, 0, 0, 0, 0, 0, 0]),
gen_init_u64([9, 13, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 13, 0, 0, 0, 0, 0]),
gen_init_u64([10, 13, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 10, 13, 0, 0, 0, 0, 0]),
gen_init_u64([9, 10, 13, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 10, 13, 0, 0, 0, 0]),
gen_init_u64([11, 13, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 11, 13, 0, 0, 0, 0, 0]),
gen_init_u64([9, 11, 13, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 11, 13, 0, 0, 0, 0]),
gen_init_u64([10, 11, 13, 0, 0, 0, 0, 0]),
gen_init_u64([8, 10, 11, 13, 0, 0, 0, 0]),
gen_init_u64([9, 10, 11, 13, 0, 0, 0, 0]),
gen_init_u64([8, 9, 10, 11, 13, 0, 0, 0]),
gen_init_u64([12, 13, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 12, 13, 0, 0, 0, 0, 0]),
gen_init_u64([9, 12, 13, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 12, 13, 0, 0, 0, 0]),
gen_init_u64([10, 12, 13, 0, 0, 0, 0, 0]),
gen_init_u64([8, 10, 12, 13, 0, 0, 0, 0]),
gen_init_u64([9, 10, 12, 13, 0, 0, 0, 0]),
gen_init_u64([8, 9, 10, 12, 13, 0, 0, 0]),
gen_init_u64([11, 12, 13, 0, 0, 0, 0, 0]),
gen_init_u64([8, 11, 12, 13, 0, 0, 0, 0]),
gen_init_u64([9, 11, 12, 13, 0, 0, 0, 0]),
gen_init_u64([8, 9, 11, 12, 13, 0, 0, 0]),
gen_init_u64([10, 11, 12, 13, 0, 0, 0, 0]),
gen_init_u64([8, 10, 11, 12, 13, 0, 0, 0]),
gen_init_u64([9, 10, 11, 12, 13, 0, 0, 0]),
gen_init_u64([8, 9, 10, 11, 12, 13, 0, 0]),
gen_init_u64([14, 0, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 14, 0, 0, 0, 0, 0, 0]),
gen_init_u64([9, 14, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 14, 0, 0, 0, 0, 0]),
gen_init_u64([10, 14, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 10, 14, 0, 0, 0, 0, 0]),
gen_init_u64([9, 10, 14, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 10, 14, 0, 0, 0, 0]),
gen_init_u64([11, 14, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 11, 14, 0, 0, 0, 0, 0]),
gen_init_u64([9, 11, 14, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 11, 14, 0, 0, 0, 0]),
gen_init_u64([10, 11, 14, 0, 0, 0, 0, 0]),
gen_init_u64([8, 10, 11, 14, 0, 0, 0, 0]),
gen_init_u64([9, 10, 11, 14, 0, 0, 0, 0]),
gen_init_u64([8, 9, 10, 11, 14, 0, 0, 0]),
gen_init_u64([12, 14, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 12, 14, 0, 0, 0, 0, 0]),
gen_init_u64([9, 12, 14, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 12, 14, 0, 0, 0, 0]),
gen_init_u64([10, 12, 14, 0, 0, 0, 0, 0]),
gen_init_u64([8, 10, 12, 14, 0, 0, 0, 0]),
gen_init_u64([9, 10, 12, 14, 0, 0, 0, 0]),
gen_init_u64([8, 9, 10, 12, 14, 0, 0, 0]),
gen_init_u64([11, 12, 14, 0, 0, 0, 0, 0]),
gen_init_u64([8, 11, 12, 14, 0, 0, 0, 0]),
gen_init_u64([9, 11, 12, 14, 0, 0, 0, 0]),
gen_init_u64([8, 9, 11, 12, 14, 0, 0, 0]),
gen_init_u64([10, 11, 12, 14, 0, 0, 0, 0]),
gen_init_u64([8, 10, 11, 12, 14, 0, 0, 0]),
gen_init_u64([9, 10, 11, 12, 14, 0, 0, 0]),
gen_init_u64([8, 9, 10, 11, 12, 14, 0, 0]),
gen_init_u64([13, 14, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 13, 14, 0, 0, 0, 0, 0]),
gen_init_u64([9, 13, 14, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 13, 14, 0, 0, 0, 0]),
gen_init_u64([10, 13, 14, 0, 0, 0, 0, 0]),
gen_init_u64([8, 10, 13, 14, 0, 0, 0, 0]),
gen_init_u64([9, 10, 13, 14, 0, 0, 0, 0]),
gen_init_u64([8, 9, 10, 13, 14, 0, 0, 0]),
gen_init_u64([11, 13, 14, 0, 0, 0, 0, 0]),
gen_init_u64([8, 11, 13, 14, 0, 0, 0, 0]),
gen_init_u64([9, 11, 13, 14, 0, 0, 0, 0]),
gen_init_u64([8, 9, 11, 13, 14, 0, 0, 0]),
gen_init_u64([10, 11, 13, 14, 0, 0, 0, 0]),
gen_init_u64([8, 10, 11, 13, 14, 0, 0, 0]),
gen_init_u64([9, 10, 11, 13, 14, 0, 0, 0]),
gen_init_u64([8, 9, 10, 11, 13, 14, 0, 0]),
gen_init_u64([12, 13, 14, 0, 0, 0, 0, 0]),
gen_init_u64([8, 12, 13, 14, 0, 0, 0, 0]),
gen_init_u64([9, 12, 13, 14, 0, 0, 0, 0]),
gen_init_u64([8, 9, 12, 13, 14, 0, 0, 0]),
gen_init_u64([10, 12, 13, 14, 0, 0, 0, 0]),
gen_init_u64([8, 10, 12, 13, 14, 0, 0, 0]),
gen_init_u64([9, 10, 12, 13, 14, 0, 0, 0]),
gen_init_u64([8, 9, 10, 12, 13, 14, 0, 0]),
gen_init_u64([11, 12, 13, 14, 0, 0, 0, 0]),
gen_init_u64([8, 11, 12, 13, 14, 0, 0, 0]),
gen_init_u64([9, 11, 12, 13, 14, 0, 0, 0]),
gen_init_u64([8, 9, 11, 12, 13, 14, 0, 0]),
gen_init_u64([10, 11, 12, 13, 14, 0, 0, 0]),
gen_init_u64([8, 10, 11, 12, 13, 14, 0, 0]),
gen_init_u64([9, 10, 11, 12, 13, 14, 0, 0]),
gen_init_u64([8, 9, 10, 11, 12, 13, 14, 0]),
gen_init_u64([15, 0, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 15, 0, 0, 0, 0, 0, 0]),
gen_init_u64([9, 15, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 15, 0, 0, 0, 0, 0]),
gen_init_u64([10, 15, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 10, 15, 0, 0, 0, 0, 0]),
gen_init_u64([9, 10, 15, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 10, 15, 0, 0, 0, 0]),
gen_init_u64([11, 15, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 11, 15, 0, 0, 0, 0, 0]),
gen_init_u64([9, 11, 15, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 11, 15, 0, 0, 0, 0]),
gen_init_u64([10, 11, 15, 0, 0, 0, 0, 0]),
gen_init_u64([8, 10, 11, 15, 0, 0, 0, 0]),
gen_init_u64([9, 10, 11, 15, 0, 0, 0, 0]),
gen_init_u64([8, 9, 10, 11, 15, 0, 0, 0]),
gen_init_u64([12, 15, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 12, 15, 0, 0, 0, 0, 0]),
gen_init_u64([9, 12, 15, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 12, 15, 0, 0, 0, 0]),
gen_init_u64([10, 12, 15, 0, 0, 0, 0, 0]),
gen_init_u64([8, 10, 12, 15, 0, 0, 0, 0]),
gen_init_u64([9, 10, 12, 15, 0, 0, 0, 0]),
gen_init_u64([8, 9, 10, 12, 15, 0, 0, 0]),
gen_init_u64([11, 12, 15, 0, 0, 0, 0, 0]),
gen_init_u64([8, 11, 12, 15, 0, 0, 0, 0]),
gen_init_u64([9, 11, 12, 15, 0, 0, 0, 0]),
gen_init_u64([8, 9, 11, 12, 15, 0, 0, 0]),
gen_init_u64([10, 11, 12, 15, 0, 0, 0, 0]),
gen_init_u64([8, 10, 11, 12, 15, 0, 0, 0]),
gen_init_u64([9, 10, 11, 12, 15, 0, 0, 0]),
gen_init_u64([8, 9, 10, 11, 12, 15, 0, 0]),
gen_init_u64([13, 15, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 13, 15, 0, 0, 0, 0, 0]),
gen_init_u64([9, 13, 15, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 13, 15, 0, 0, 0, 0]),
gen_init_u64([10, 13, 15, 0, 0, 0, 0, 0]),
gen_init_u64([8, 10, 13, 15, 0, 0, 0, 0]),
gen_init_u64([9, 10, 13, 15, 0, 0, 0, 0]),
gen_init_u64([8, 9, 10, 13, 15, 0, 0, 0]),
gen_init_u64([11, 13, 15, 0, 0, 0, 0, 0]),
gen_init_u64([8, 11, 13, 15, 0, 0, 0, 0]),
gen_init_u64([9, 11, 13, 15, 0, 0, 0, 0]),
gen_init_u64([8, 9, 11, 13, 15, 0, 0, 0]),
gen_init_u64([10, 11, 13, 15, 0, 0, 0, 0]),
gen_init_u64([8, 10, 11, 13, 15, 0, 0, 0]),
gen_init_u64([9, 10, 11, 13, 15, 0, 0, 0]),
gen_init_u64([8, 9, 10, 11, 13, 15, 0, 0]),
gen_init_u64([12, 13, 15, 0, 0, 0, 0, 0]),
gen_init_u64([8, 12, 13, 15, 0, 0, 0, 0]),
gen_init_u64([9, 12, 13, 15, 0, 0, 0, 0]),
gen_init_u64([8, 9, 12, 13, 15, 0, 0, 0]),
gen_init_u64([10, 12, 13, 15, 0, 0, 0, 0]),
gen_init_u64([8, 10, 12, 13, 15, 0, 0, 0]),
gen_init_u64([9, 10, 12, 13, 15, 0, 0, 0]),
gen_init_u64([8, 9, 10, 12, 13, 15, 0, 0]),
gen_init_u64([11, 12, 13, 15, 0, 0, 0, 0]),
gen_init_u64([8, 11, 12, 13, 15, 0, 0, 0]),
gen_init_u64([9, 11, 12, 13, 15, 0, 0, 0]),
gen_init_u64([8, 9, 11, 12, 13, 15, 0, 0]),
gen_init_u64([10, 11, 12, 13, 15, 0, 0, 0]),
gen_init_u64([8, 10, 11, 12, 13, 15, 0, 0]),
gen_init_u64([9, 10, 11, 12, 13, 15, 0, 0]),
gen_init_u64([8, 9, 10, 11, 12, 13, 15, 0]),
gen_init_u64([14, 15, 0, 0, 0, 0, 0, 0]),
gen_init_u64([8, 14, 15, 0, 0, 0, 0, 0]),
gen_init_u64([9, 14, 15, 0, 0, 0, 0, 0]),
gen_init_u64([8, 9, 14, 15, 0, 0, 0, 0]),
gen_init_u64([10, 14, 15, 0, 0, 0, 0, 0]),
gen_init_u64([8, 10, 14, 15, 0, 0, 0, 0]),
gen_init_u64([9, 10, 14, 15, 0, 0, 0, 0]),
gen_init_u64([8, 9, 10, 14, 15, 0, 0, 0]),
gen_init_u64([11, 14, 15, 0, 0, 0, 0, 0]),
gen_init_u64([8, 11, 14, 15, 0, 0, 0, 0]),
gen_init_u64([9, 11, 14, 15, 0, 0, 0, 0]),
gen_init_u64([8, 9, 11, 14, 15, 0, 0, 0]),
gen_init_u64([10, 11, 14, 15, 0, 0, 0, 0]),
gen_init_u64([8, 10, 11, 14, 15, 0, 0, 0]),
gen_init_u64([9, 10, 11, 14, 15, 0, 0, 0]),
gen_init_u64([8, 9, 10, 11, 14, 15, 0, 0]),
gen_init_u64([12, 14, 15, 0, 0, 0, 0, 0]),
gen_init_u64([8, 12, 14, 15, 0, 0, 0, 0]),
gen_init_u64([9, 12, 14, 15, 0, 0, 0, 0]),
gen_init_u64([8, 9, 12, 14, 15, 0, 0, 0]),
gen_init_u64([10, 12, 14, 15, 0, 0, 0, 0]),
gen_init_u64([8, 10, 12, 14, 15, 0, 0, 0]),
gen_init_u64([9, 10, 12, 14, 15, 0, 0, 0]),
gen_init_u64([8, 9, 10, 12, 14, 15, 0, 0]),
gen_init_u64([11, 12, 14, 15, 0, 0, 0, 0]),
gen_init_u64([8, 11, 12, 14, 15, 0, 0, 0]),
gen_init_u64([9, 11, 12, 14, 15, 0, 0, 0]),
gen_init_u64([8, 9, 11, 12, 14, 15, 0, 0]),
gen_init_u64([10, 11, 12, 14, 15, 0, 0, 0]),
gen_init_u64([8, 10, 11, 12, 14, 15, 0, 0]),
gen_init_u64([9, 10, 11, 12, 14, 15, 0, 0]),
gen_init_u64([8, 9, 10, 11, 12, 14, 15, 0]),
gen_init_u64([13, 14, 15, 0, 0, 0, 0, 0]),
gen_init_u64([8, 13, 14, 15, 0, 0, 0, 0]),
gen_init_u64([9, 13, 14, 15, 0, 0, 0, 0]),
gen_init_u64([8, 9, 13, 14, 15, 0, 0, 0]),
gen_init_u64([10, 13, 14, 15, 0, 0, 0, 0]),
gen_init_u64([8, 10, 13, 14, 15, 0, 0, 0]),
gen_init_u64([9, 10, 13, 14, 15, 0, 0, 0]),
gen_init_u64([8, 9, 10, 13, 14, 15, 0, 0]),
gen_init_u64([11, 13, 14, 15, 0, 0, 0, 0]),
gen_init_u64([8, 11, 13, 14, 15, 0, 0, 0]),
gen_init_u64([9, 11, 13, 14, 15, 0, 0, 0]),
gen_init_u64([8, 9, 11, 13, 14, 15, 0, 0]),
gen_init_u64([10, 11, 13, 14, 15, 0, 0, 0]),
gen_init_u64([8, 10, 11, 13, 14, 15, 0, 0]),
gen_init_u64([9, 10, 11, 13, 14, 15, 0, 0]),
gen_init_u64([8, 9, 10, 11, 13, 14, 15, 0]),
gen_init_u64([12, 13, 14, 15, 0, 0, 0, 0]),
gen_init_u64([8, 12, 13, 14, 15, 0, 0, 0]),
gen_init_u64([9, 12, 13, 14, 15, 0, 0, 0]),
gen_init_u64([8, 9, 12, 13, 14, 15, 0, 0]),
gen_init_u64([10, 12, 13, 14, 15, 0, 0, 0]),
gen_init_u64([8, 10, 12, 13, 14, 15, 0, 0]),
gen_init_u64([9, 10, 12, 13, 14, 15, 0, 0]),
gen_init_u64([8, 9, 10, 12, 13, 14, 15, 0]),
gen_init_u64([11, 12, 13, 14, 15, 0, 0, 0]),
gen_init_u64([8, 11, 12, 13, 14, 15, 0, 0]),
gen_init_u64([9, 11, 12, 13, 14, 15, 0, 0]),
gen_init_u64([8, 9, 11, 12, 13, 14, 15, 0]),
gen_init_u64([10, 11, 12, 13, 14, 15, 0, 0]),
gen_init_u64([8, 10, 11, 12, 13, 14, 15, 0]),
gen_init_u64([9, 10, 11, 12, 13, 14, 15, 0]),
gen_init_u64([8, 9, 10, 11, 12, 13, 14, 15]),
];

const fn gen_init_mm256i(a: [i32; 8]) -> __m256i {
    unsafe {
        transmute(i32x8::from_array(a))
    }
}

const fn gen_init_u64(a: [i8; 8]) -> u64 {
    unsafe {
        transmute(a)
    }
}

pub fn gen_input<T: Copy, F>(data_size: usize, f: F) -> (Buffer<T>, Bitmap, Vec<T>) 
where F: Fn(usize) -> T,
{
    let len = data_size; // 4 * 1024
    let mut rnd = ThreadRng::default();
    let mut data = Vec::<T>::with_capacity(len);
    let mut filter = MutableBitmap::with_capacity(len);
    let mut expect = Vec::<T>::new();
    let true_prob = 0.8;
    for i in 0..len {
        let e = f(i);
        data.push(e);
        let b = rnd.gen_bool(true_prob);
        filter.push(b);
        if b {
            expect.push(e);
        }
    }
    (data.into(), filter.into(), expect)
}

// const MASK_ARRAY_1: ;

#[cfg(test)]
mod test {

    /// 
    #[test]
    fn gen_mask_array() {
        for i in 0..256 {
            let a = mask_array(i as u8);
            println!("gen_init_u64({:?}),", a);
        }
    }

    fn mask_array(mut mask: u8) -> [u32; 8] {
        let mut arr = [0; 8];
        let mut idx = 0;
        for i in 0..8 {
            if mask & 0b0000_0001 == 0b0000_0001 {
                arr[idx] = i;
                idx += 1;
            }
            mask = mask >> 1;
        }
        arr
    }


}