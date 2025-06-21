mod sys {
    #![allow(
        non_camel_case_types,
        non_snake_case,
        dead_code,
        non_upper_case_globals
    )]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

// void nvte_fused_rope_forward(const NVTETensor input, const NVTETensor cu_seqlens,
//     const NVTETensor freqs, const NVTETensor start_positions,
//     NVTETensor output, const NVTE_QKV_Format qkv_format,
//     const bool interleaved, const int cp_size, const int cp_rank,
//     const int s, const int b, const int h, const int d, const int d2,
//     const int stride_s_or_t, const int stride_b, const int stride_h,
//     const int stride_d, cudaStream_t stream);

pub mod rope;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_nvte_ffi() {
        unsafe {
            // Pick any QKV layout that is always defined in the enum
            let layout = sys::NVTE_QKV_Layout_NVTE_BS3HD;
            let fmt = sys::nvte_get_qkv_format(layout);
            println!("Layout {layout:?} → QKV format {fmt:?}");
            assert_eq!(fmt, sys::NVTE_QKV_Format_NVTE_BSHD);
        }
    }
}
