use arrow2::{buffer::Buffer, bitmap::{Bitmap, utils::{BitChunksExact, BitChunkIterExact}}};


pub fn filter_primitive_types<T: Copy>(values: &Buffer<T>, filter: &Bitmap) -> Buffer<T> {
    debug_assert_eq!(values.len(), filter.len());
    let num_rows = filter.len() - filter.unset_bits();
    if num_rows == values.len() {
        return values.clone();
    }

    let mut builder: Vec<T> = Vec::with_capacity(num_rows);
    let mut ptr = builder.as_mut_ptr();
    let mut values_ptr = values.as_slice().as_ptr();
    let (mut slice, offset, mut length) = filter.as_slice();

    unsafe {
        if offset > 0 {
            let mut mask = slice[0];
            while mask != 0 {
                let n = mask.trailing_zeros() as usize;
                if n >= offset {
                    copy_advance_aligned(values_ptr.add(n - offset), &mut ptr, 1);
                }
                mask = mask & (mask - 1);
            }
            length -= 8 - offset;
            slice = &slice[1..];
            values_ptr = values_ptr.add(8 - offset);
        }

        const CHUNK_SIZE: usize = 64;
        let mut mask_chunks = BitChunksExact::<u64>::new(slice, length);
        let mut continuous_selected = 0;
        for mut mask in mask_chunks.by_ref() {
            if mask == u64::MAX {
                continuous_selected += CHUNK_SIZE;
            } else {
                if continuous_selected > 0 {
                    copy_advance_aligned(values_ptr, &mut ptr, continuous_selected);
                    values_ptr = values_ptr.add(continuous_selected);
                    continuous_selected = 0;
                }
                while mask != 0 {
                    let n = mask.trailing_zeros() as usize;
                    copy_advance_aligned(values_ptr.add(n), &mut ptr, 1);
                    mask = mask & (mask - 1);
                }
                values_ptr = values_ptr.add(CHUNK_SIZE);
            }
        }
        if continuous_selected > 0 {
            copy_advance_aligned(values_ptr, &mut ptr, continuous_selected);
            values_ptr = values_ptr.add(continuous_selected);
        }

        for (i, is_selected) in mask_chunks.remainder_iter().enumerate() {
            if is_selected {
                copy_advance_aligned(values_ptr.add(i), &mut ptr, 1);
            }
        }

        set_vec_len_by_ptr(&mut builder, ptr);
    }

    builder.into()
}


#[inline(always)]
unsafe fn copy_advance_aligned<T>(src: *const T, ptr: &mut *mut T, count: usize) {
    unsafe {
        std::ptr::copy_nonoverlapping(src, *ptr, count);
        *ptr = ptr.add(count);
    }
}

#[inline(always)]
unsafe fn set_vec_len_by_ptr<T>(vec: &mut Vec<T>, ptr: *const T) {
    unsafe {
        vec.set_len((ptr as usize - vec.as_ptr() as usize) / std::mem::size_of::<T>());
    }
}

#[cfg(test)]
mod test {
    use crate::gen::gen_input;

    use super::filter_primitive_types;


    #[test]
    fn test_v1() {
        let input = gen_input(32);
        let dst = filter_primitive_types(&input.0, &input.1);
        assert_eq!(16, dst.len());
        assert_eq!(&[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30], &*dst);
    }

}