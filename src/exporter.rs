use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use opencv::{core, imgcodecs, prelude::*};
use chrono::Local;
use crate::text_detector::Detection;
use crate::text_ocr::OcrEngine;
use crate::config::AppConfig;

static SAVED_COUNT: AtomicU64 = AtomicU64::new(0);

pub fn save_evidence(
    page: u16,
    _detections: Vec<Detection>,
    display_mat: &core::Mat,
    bgr_raw_mat: &core::Mat,
    _ocr_engine: &mut Option<OcrEngine>,
    pdf_path: &str,
) -> anyhow::Result<()> {
    // 1. 出力先フォルダの準備
    if !std::path::Path::new(AppConfig::OUTPUT_DIR).exists() {
        fs::create_dir_all(AppConfig::OUTPUT_DIR)?;
    }

    // 2. 日時・ハッシュ・連番の取得
    let now = Local::now();
    let date_str = now.format("%Y%m%d").to_string();
    let time_str = now.format("%H%M%S%.9f").to_string();

    let mut hasher = DefaultHasher::new();
    if let Ok(data_bytes) = bgr_raw_mat.data_bytes() {
        data_bytes.hash(&mut hasher);
    }
    let hash_val = hasher.finish();
    let seq = SAVED_COUNT.fetch_add(1, Ordering::SeqCst);

    let file_stem = std::path::Path::new(pdf_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // 3. ファイルパスの組み立て (output/フォルダ内へ)
    let filename = format!(
        "evidence_{}_P{:03}_{}_{}_{:x}_{:04}.png",
        file_stem, page + 1, date_str, time_str, hash_val, seq
    );
    let out_path = std::path::Path::new(AppConfig::OUTPUT_DIR).join(filename);

    // 4. 画像の保存
    imgcodecs::imwrite(out_path.to_str().unwrap(), display_mat, &core::Vector::new())?;

    // 5. 読み取り専用属性の付与 (ReadOnly)
    let mut perms = fs::metadata(&out_path)?.permissions();
    perms.set_readonly(true);
    fs::set_permissions(&out_path, perms)?;

    // system_audit.log への処理は完全に削除されました
    Ok(())
}