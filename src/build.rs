fn main() {
    // 1. OpenCVのリンク設定 (既存)
    println!(r"cargo:rustc-link-search=native=C:\dev\deps\opencv\opencv\build\x64\vc16\lib");
    println!("cargo:rustc-link-lib=opencv_world4130");

    // 2. Pdfiumのリンク設定
    // ライブラリがあるフォルダを指定
    println!(r"cargo:rustc-link-search=native=C:\dev\deps\pdfium\lib");
    // ファイル名 "pdfium.dll.lib" から拡張子を除いて指定
    println!("cargo:rustc-link-lib=pdfium.dll");

    println!("cargo:rerun-if-changed=build.rs");
}