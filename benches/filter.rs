use std::alloc::{alloc, Layout};

use arrow2::{buffer::Buffer, bitmap::{Bitmap, MutableBitmap}};
use criterion::{Criterion, Throughput};
use vector_filter::{filter_epi32, v1::filter_primitive_types};

#[macro_use]
extern crate criterion;

const DATA_LEN: usize = 4 * 1024;

fn filter(c: &mut Criterion) {
    let mut g = c.benchmark_group("filter");
    g.throughput(Throughput::Bytes(DATA_LEN as _));
    // v2
    g.bench_function("v2-simd", |b| {
        let input = gen_input();
        b.iter(|| {
            filter_epi32(&input.0, &input.1);
        });
    });
    // v1
    g.bench_function("v1-databend", |b| {
        let input = gen_input();
        b.iter(|| {
            filter_primitive_types::<i32>(&input.0, &input.1);
        });
    });
}

fn gen_input() -> (Buffer<i32>, Bitmap) {
    let mut data = unsafe {
        let ptr = alloc(Layout::from_size_align_unchecked(DATA_LEN * 4, 64));
        Vec::<i32>::from_raw_parts(ptr as _, DATA_LEN, DATA_LEN)
    };
    let mut filter = MutableBitmap::with_capacity(DATA_LEN);
    for i in 0..DATA_LEN {
        data.push(i as _);
        filter.push(i%2 == 0);
    }
    (data.into(), filter.into())
}

criterion_group!(bench, filter);
criterion_main!(bench);