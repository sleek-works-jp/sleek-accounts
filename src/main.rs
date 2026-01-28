//! # AI-Assisted Text Detector & OCR Tool for Accounting
//!
//! This application loads a PDF, detects text regions using ONNX models (PaddleOCR),
//! and allows users to interactively select and save receipts or invoices.
//!
//! ## Key Features
//! - **Text Detection**: Uses a lightweight ONNX detection model to find text blocks.
//! - **OCR Analysis**: Automatically filters detections based on keywords (e.g., "Total", "Invoice", "%").
//! - **Interactive UI**: OpenCV-based window to view, click, and verify detections.
//! - **Evidence Saving**: Exports selected regions as verified evidence images.
//!
//! ## Architecture
//! - **Main Thread**: Handles UI rendering (OpenCV), User Input, and PDF Rendering (Pdfium).
//! - **Worker Thread**: Runs the heavy AI inference (TextDetector) in the background to keep UI responsive.
//! - **State Management**: Uses `Arc<Mutex<AppState>>` to share date between UI and AI threads safely.

mod text_detector;
mod text_ocr;
mod models;
mod exporter;
mod ui;
mod config;

use opencv::{core, highgui, imgproc, prelude::*};
#[cfg(debug_assertions)]
use opencv::imgcodecs;
use pdfium_render::prelude::*;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use anyhow::Context;
use text_detector::{TextDetector, Detection};
use text_ocr::OcrEngine;
use config::AppConfig;

fn main() -> anyhow::Result<()> {
    let pdfium = Pdfium::new(Pdfium::bind_to_system_library().expect("Pdfium DLL not found."));
    let document = pdfium.load_pdf_from_file(AppConfig::PDF_PATH, None).context("PDF読込失敗")?;
    let total_pages = document.pages().len();
    
    // ページ番号をタグとして画像・結果に付与するチャネル
    let (tx_img, rx_img) = mpsc::channel::<(u16, core::Mat)>();
    let (tx_det, rx_det) = mpsc::channel::<(u16, Vec<Detection>)>();
    
    let ocr = OcrEngine::new(AppConfig::OCR_MODEL, AppConfig::OCR_DICT).ok();

    let state = Arc::new(Mutex::new(models::AppState {
        mouse_x: 0, mouse_y: 0,
        offset_x: 0, offset_y: 0,
        is_dragging: false,
        drag_last_x: 0, drag_last_y: 0,
        current_page: 0,
        ocr_engine: ocr,
        detections: Vec::new(),
        save_msg_timer: 0,
    }));

    let mut main_bgr_frame = core::Mat::default();
    let mut main_display_view = core::Mat::default();
    let mut resizing_buf = core::Mat::default();

    let state_for_thread = Arc::clone(&state);
    thread::spawn(move || {
        let mut detector = TextDetector::new(AppConfig::DETECTOR_MODEL).expect("Detector init failed");
        detector.padding_ratio_w = 0.2;
        detector.padding_ratio_h = 0.3;
        
        loop {
            // 新しいリクエストが届くまで待機
            if let Ok((mut last_page_idx, mut last_img)) = rx_img.recv() {
                
                // 【スキップ処理】キューにある「より新しい画像」をすべて吸い出す
                while let Ok((newer_idx, newer_img)) = rx_img.try_recv() {
                    last_page_idx = newer_idx;
                    last_img = newer_img;
                }

                // 【直前チェック】解析を始める直前に、今のページ番号が最新か確認
                let current_actual_page = {
                    let s = state_for_thread.lock().expect("Mutex poisoning");
                    s.current_page
                };
                
                // すでにページがめくられていたら、この解析はスキップして次の受信待ちへ
                if last_page_idx != current_actual_page {
                    continue;
                }

                // 重い計算（AI検出）を実行
                let mut s = state_for_thread.lock().expect("Mutex poisoning");
                if let Some(ref mut engine) = s.ocr_engine {
                    if let Ok(dets) = detector.detect(&last_img, engine, last_page_idx) {
                        // ページ番号を添えてメインスレッドに送り返す
                        let _ = tx_det.send((last_page_idx, dets));
                    }
                }
            }
        }
    });

    highgui::named_window(AppConfig::WINDOW_NAME, highgui::WINDOW_NORMAL)?;
    highgui::resize_window(AppConfig::WINDOW_NAME, AppConfig::DEFAULT_WIN_W, AppConfig::DEFAULT_WIN_H)?;
    
    let s_ptr = Arc::clone(&state);
    highgui::set_mouse_callback(AppConfig::WINDOW_NAME, Some(Box::new(move |event, x, y, _flags| {
        if let Ok(mut s) = s_ptr.lock() {
            s.mouse_x = x; s.mouse_y = y;
            let scale_x = AppConfig::RENDER_WIDTH as f32 / AppConfig::DEFAULT_WIN_W as f32;
            let scale_y = AppConfig::RENDER_WIDTH as f32 / AppConfig::DEFAULT_WIN_W as f32; 
            match event {
                highgui::EVENT_LBUTTONDOWN => {
                    s.is_dragging = true; s.drag_last_x = x; s.drag_last_y = y;
                    let mx = (x as f32 * scale_x) as i32;
                    let my = (y as f32 * scale_y) as i32;
                    let p = core::Point::new(mx, my);
                    for det in &mut s.detections { if det.rect.contains(p) { det.color_state = 1; } }
                }
                highgui::EVENT_RBUTTONDOWN => {
                    let mx = (x as f32 * scale_x) as i32;
                    let my = (y as f32 * scale_y) as i32;
                    let p = core::Point::new(mx, my);
                    for det in &mut s.detections { if det.rect.contains(p) { det.color_state = 2; } }
                }
                highgui::EVENT_MOUSEMOVE => {
                    if s.is_dragging {
                        s.offset_x += x - s.drag_last_x;
                        s.offset_y += y - s.drag_last_y;
                        s.drag_last_x = x; s.drag_last_y = y;
                    }
                }
                highgui::EVENT_LBUTTONUP => s.is_dragging = false,
                _ => {}
            }
        }
    })))?;

    let mut needs_reload = true;
    let mut s_key_locked = false;
    
    // デバッグ保存用トリガー
    #[cfg(debug_assertions)]
    let mut debug_save_trigger = false;

    loop {
        let current_p = { state.lock().expect("Failed to lock state").current_page };

        if needs_reload {
            let page = document.pages().get(current_p)?;
            let render_config = PdfRenderConfig::new().set_target_width(AppConfig::RENDER_WIDTH);
            let bitmap = page.render_with_config(&render_config)?;
            
            let mut frame = core::Mat::new_rows_cols_with_default(
                bitmap.height() as i32, bitmap.width() as i32, core::CV_8UC4, core::Scalar::all(255.0)
            )?;
            frame.data_bytes_mut()?.copy_from_slice(&bitmap.as_raw_bytes());
            
            unsafe {
                imgproc::cvt_color(&frame, &mut main_bgr_frame, imgproc::COLOR_BGRA2BGR, 0, std::mem::transmute(0))?;
            }
            // 画像と一緒にページ番号を送信
            let _ = tx_img.send((current_p, main_bgr_frame.clone()));
            needs_reload = false;
        }

        // AIスレッドからの結果受け取り
        if let Ok((res_page_idx, new_dets)) = rx_det.try_recv() {
            let mut s = state.lock().expect("Failed to lock state");
            // 【照合】今のページ番号と解析結果の番号が一致する場合のみ描画に反映
            if res_page_idx == s.current_page {
                s.detections = new_dets;
                #[cfg(debug_assertions)]
                { debug_save_trigger = true; }
            }
        }

        if !main_bgr_frame.empty() {
            let view_width = AppConfig::DEFAULT_WIN_W;
            let view_height = (main_bgr_frame.rows() as f32 * (view_width as f32 / main_bgr_frame.cols() as f32)) as i32;
            imgproc::resize(&main_bgr_frame, &mut resizing_buf, core::Size::new(view_width, view_height), 0.0, 0.0, imgproc::INTER_LINEAR)?;
            
            // Fix color inversion for display (BGR -> RGB swap)
            imgproc::cvt_color(&resizing_buf, &mut main_display_view, imgproc::COLOR_BGR2RGB, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;

            {
                let mut s = state.lock().expect("Failed to lock state");
                let (mx, my, cp, timer) = (s.mouse_x, s.mouse_y, s.current_page, s.save_msg_timer);
                let detections = s.detections.clone();
                
                ui::Renderer::draw_all(
                    &mut main_display_view, 
                    &main_bgr_frame, 
                    &detections, 
                    mx, my, cp, 
                    total_pages.into(), 
                    timer
                )?;
                
                if s.save_msg_timer > 0 { s.save_msg_timer -= 1; }
            }

            // =========================================================
            // [DEBUG] 最終描画結果の保存 (検出更新時のみ)
            // =========================================================
            #[cfg(debug_assertions)]
            if debug_save_trigger {
                use std::fs;
                let debug_dir = "debug_output";
                if !std::path::Path::new(debug_dir).exists() {
                     let _ = fs::create_dir_all(debug_dir);
                }
                
                let current_p = { state.lock().unwrap().current_page };
                let filename = format!("{}/Page_{}_final_view.png", debug_dir, current_p);
                // main_display_view は RGB なので BGR に戻してから保存しないと色が変になる可能性があるが
                // to_str().unwrap() は main_display_view が BGR2RGB 済みであることを考慮すると
                // imwrite は BGR を期待するので、ここで再度変換が必要。
                
                let mut save_buf = core::Mat::default();
                let _ = imgproc::cvt_color(&main_display_view, &mut save_buf, imgproc::COLOR_RGB2BGR, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT);
                let _ = imgcodecs::imwrite(&filename, &save_buf, &core::Vector::new());
                
                debug_save_trigger = false;
            }
            // =========================================================

            highgui::imshow(AppConfig::WINDOW_NAME, &main_display_view)?;
        }

        // wait_key(1) で入力を超高速化
        let key = highgui::wait_key(1)?;
        if key != -1 {
            let key_code = (key & 0xFF) as u8;
            match key_code {
                27 | 113 => break,
                115 => {
                    if !s_key_locked {
                        let mut s = state.lock().expect("Failed to lock state");
                        let to_process: Vec<Detection> = s.detections.iter().filter(|d| d.color_state > 0).cloned().collect();
                        if !to_process.is_empty() {
                            let page = s.current_page;
                            if let Ok(_) = exporter::save_evidence(page, to_process, &main_display_view, &main_bgr_frame, &mut s.ocr_engine, AppConfig::PDF_PATH) {
                                s.save_msg_timer = 50;
                            }
                        }
                        s_key_locked = true;
                    }
                }
                100 | 50 | 68 => { // d, 2, or D (Next)
                    let mut s = state.lock().expect("Failed to lock state");
                    if s.current_page < (total_pages as u16) - 1 {  
                        s.current_page += 1; 
                        s.detections.clear(); // 即座に古い枠を消してレスポンス向上
                        needs_reload = true; 
                    }
                }
                97 | 49 | 65 => { // a, 1, or A (Prev)
                    let mut s = state.lock().expect("Failed to lock state");
                    if s.current_page > 0 { 
                        s.current_page -= 1; 
                        s.detections.clear(); 
                        needs_reload = true; 
                    }
                }
                _ => {}
            }
        } else {
            s_key_locked = false;
        }
    }
    Ok(())
}