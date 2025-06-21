use std::{env, path::PathBuf};

pub fn main() {
    let lib_root = env::var("CARGO_MANIFEST_DIR").unwrap();
    let header_path = PathBuf::from(&lib_root).join("te/wrapper.h");
    // let lib_path = PathBuf::from(&lib_root).join("te/lib/aarch64-linux-gnu");

    println!("cargo:rustc-link-lib=transformer_engine");
    println!("cargo:rustc-link-lib=dylib=nvrtc"); // run-time compiler API
    println!("cargo:rustc-link-lib=dylib=cudart"); // CUDA run-time (required by nvrtc)
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");

    let bindings = bindgen::Builder::default()
        .header(header_path.to_str().unwrap())
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++11")
        .clang_arg("-I/usr/include")
        .clang_arg("-I/usr/include/c++/11")
        .clang_arg("-I/usr/include/aarch64-linux-gnu")
        .clang_arg("-I/usr/include/aarch64-linux-gnu/c++/11")
        // ---- (a) allow-list the public C-API --------------------------------
        .allowlist_function("^nvte_.*") // every exported TE symbol
        .allowlist_type("^NVTE.*") // tensor structs, enums
        .allowlist_var("^NVTE.*")
        // ---- (b) block/opaque everything else --------------------------------
        // .opaque_type("^std::.*") // treat all STL types as opaque
        // .blocklist_type("^std::.*")
        // .blocklist_item("^std::.*")
        // .opaque_type("^__gnu_cxx::.*")
        // .blocklist_type("^__gnu_cxx::.*")
        // .blocklist_item("^__gnu_cxx::.*")
        // ---- (c) silence CUDA FP8 intrinsics clang can’t parse yet ------------
        // .blocklist_type("__nv_fp8_e4m3")
        // .blocklist_type("__nv_fp8_e5m2")
        // ---- (d) skip bindgen’s layout tests (they choke on opaque types) ----
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
