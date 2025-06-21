#![allow(non_camel_case_types, non_snake_case, dead_code)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_nvte_ffi() {
        unsafe {
            // Pick any QKV layout that is always defined in the enum
            let layout = NVTE_QKV_Layout_NVTE_BS3HD;
            let fmt = nvte_get_qkv_format(layout);
            println!("Layout {layout:?} → QKV format {fmt:?}");
            assert_eq!(fmt, NVTE_QKV_Format_NVTE_BSHD);
        }
    }
}
