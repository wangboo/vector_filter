use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use vector_filter::{gen::gen_input, v1::filter_primitive_types};

const DATA_LEN: usize = 4 * 1024;

fn filter_i8(c: &mut Criterion) {
    let mut g = c.benchmark_group("filter_i8");
    g.throughput(Throughput::Bytes(DATA_LEN as _));
    // v2
    g.bench_function("v2-simd-1", |b| {
        let (buffer, bitmap, _) = gen_input(DATA_LEN, |i| i as i8);
        b.iter(|| {
            unsafe {
                let _ = vector_filter::filter_epi8(&buffer, &bitmap);
            }
        });
    });
    g.bench_function("v2-simd-2", |b| {
        let (buffer, bitmap, _) = gen_input(DATA_LEN, |i| i as i8);
        b.iter(|| {
            unsafe {
                let _ = vector_filter::filter_epi8_1(&buffer, &bitmap);
            }
        });
    });
    // v1
    g.bench_function("v1-databend", |b| {
        let (buffer, bitmap, _) = gen_input(DATA_LEN, |i| i as i8);
        b.iter(|| {
            filter_primitive_types::<i8>(&buffer, &bitmap);
        });
    });
}


criterion_group!(bench, filter_i8);
criterion_main!(bench);