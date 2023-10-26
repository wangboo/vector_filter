


use criterion::{Criterion, Throughput};
use vector_filter::{v1::filter_primitive_types, gen::gen_input};

#[macro_use]
extern crate criterion;

const DATA_LEN: usize = 4 * 1024;

fn filter_i32(c: &mut Criterion) {
    let mut g = c.benchmark_group("filter_i32");
    g.throughput(Throughput::Bytes(DATA_LEN as _));
    // v2
    g.bench_function("v2-simd", |b| {
        let (buffer, bitmap, _) = gen_input(DATA_LEN, |i| i as i32);
        b.iter(|| {
            unsafe {
                let _ = vector_filter::filter_epi32(&buffer, &bitmap);
            }
        });
    });
    // v1
    g.bench_function("v1-databend", |b| {
        let (buffer, bitmap, _) = gen_input(DATA_LEN, |i| i as i32);
        b.iter(|| {
            filter_primitive_types::<i32>(&buffer, &bitmap);
        });
    });
}

criterion_group!(bench, filter_i32);
criterion_main!(bench);