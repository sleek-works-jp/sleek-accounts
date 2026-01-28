use opencv::{core, prelude::*, imgproc};
#[cfg(debug_assertions)]
use opencv::imgcodecs;
use ort::{session::Session, value::Value, inputs};
use ndarray::Array4;
use anyhow::Result;
use crate::text_ocr::OcrEngine;

pub const MODEL_SIZE: i32 = 640;

/// Detection result structure containing the bounding box and color state.
#[derive(Debug, Clone)]
pub struct Detection {
    pub rect: core::Rect,
    /// Color state: 0: Default, 1: Green (Selected), 2: Red (Right-click)
    pub color_state: u8, 
}

use std::sync::OnceLock;

static RE_INVOICE: OnceLock<regex::Regex> = OnceLock::new();
static RE_PHONE: OnceLock<regex::Regex> = OnceLock::new();

/// TextDetector orchestrates the ONNX runtime session for text detection.
/// It holds reusable buffers to minimize memory allocation during video/image frame processing.
pub struct TextDetector {
    session: Session,
    ratio: f32,
    offset_x: f32,
    offset_y: f32,
    pub padding_ratio_w: f32,
    pub padding_ratio_h: f32,
    
    // Model input/output buffers re-used across frames for performance
    resized_buf: core::Mat,
    canvas_buf: core::Mat,
    rgb_buf: core::Mat,
    prob_map: core::Mat,
    roi_buf: core::Mat,
    binary_map_f32: core::Mat,
    binary_map_8u: core::Mat,
}

impl TextDetector {
    /// Initializes a new TextDetector with the given ONNX model path.
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()?.with_intra_threads(4)?.commit_from_file(model_path)?;
        Ok(Self { 
            session, 
            ratio: 1.0, 
            offset_x: 0.0, 
            offset_y: 0.0, 
            padding_ratio_w: 0.2, 
            padding_ratio_h: 0.3,
            resized_buf: core::Mat::default(),
            canvas_buf: core::Mat::default(),
            rgb_buf: core::Mat::default(),

            prob_map: core::Mat::default(),
            roi_buf: core::Mat::default(),
            binary_map_f32: core::Mat::default(),
            binary_map_8u: core::Mat::default(),
        })
    }

    /// Detects text regions in the given frame.
    /// 
    /// # Arguments
    /// * `frame` - The input BGR image.
    /// * `ocr` - Mutable reference to the OCR engine for content filtering.
    /// * `page_num` - Current page number for debug logging.
    /// 
    /// # Returns
    /// A vector of `Detection` results.
    pub fn detect(&mut self, frame: &core::Mat, ocr: &mut OcrEngine, _page_num: u16) -> Result<Vec<Detection>> {
        let input_array = self.preprocess_letterbox(frame)?;
        let input_tensor = Value::from_array(input_array)?;
        let outputs = self.session.run(inputs![&input_tensor])?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        
        let h = shape[2] as i32;
        let w = shape[3] as i32;
        
        if self.prob_map.rows() != h || self.prob_map.cols() != w {
            self.prob_map = core::Mat::new_rows_cols_with_default(h, w, core::CV_32F, core::Scalar::all(0.0))?;
        }
        
        let ptr = self.prob_map.data_mut() as *mut f32;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, (h * w) as usize); }

        // 閾値処理を取っ払う - prob_mapを直接使用
        self.prob_map.convert_to(&mut self.binary_map_f32, core::CV_32F, 255.0, 0.0)?;
        self.binary_map_f32.convert_to(&mut self.binary_map_8u, core::CV_8U, 1.0, 0.0)?;

        let mut contours = core::Vector::<core::Vector<core::Point>>::new();
        imgproc::find_contours(&self.binary_map_8u, &mut contours, imgproc::RETR_LIST, imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0,0))?;

        // Store all detected rectangles before filtering
        let mut all_rects = Vec::new(); // AI検出の生データ

        
        // まず全ての矩形を収集
        for contour in contours.iter() {
            let rect = imgproc::bounding_rect(&contour)?;
            // サイズフィルタを取っ払う

            let x = ((rect.x as f32 - self.offset_x) / self.ratio) as i32;
            let y = ((rect.y as f32 - self.offset_y) / self.ratio) as i32;
            let width = (rect.width as f32 / self.ratio) as i32;
            let height = (rect.height as f32 / self.ratio) as i32;

            let pad_w = (width as f32 * self.padding_ratio_w) as i32;
            let pad_h = (height as f32 * self.padding_ratio_h) as i32;
            
            let safe_rect = core::Rect::new(
                (x - pad_w).max(0), (y - pad_h).max(0),
                (width + pad_w * 2).min(frame.cols() - (x - pad_w).max(0)),
                (height + pad_h * 2).min(frame.rows() - (y - pad_h).max(0))
            );

            all_rects.push(safe_rect);
        }

        // OCRエンジンの行高さ基準を補正 (中央値算出)
        ocr.calibrate_line_height(&all_rects);

        // =========================================================
        // [DEBUG] 全検出矩形 (Raw) の画像出力
        // リリースビルドではコンパイルから除外されます
        // =========================================================
        #[cfg(debug_assertions)]
        {
            use std::fs;
            let debug_dir = "debug_output";
            if !std::path::Path::new(debug_dir).exists() {
                let _ = fs::create_dir_all(debug_dir);
            }

            let mut debug_img = frame.clone();
            for rect in &all_rects {
                // 青色 (BGR: 255, 0, 0) で描画
                let _ = imgproc::rectangle(
                    &mut debug_img,
                    *rect,
                    core::Scalar::new(255.0, 0.0, 0.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    0
                );
            }
            
            let filename = format!("{}/Page_{}_detect_raw.png", debug_dir, _page_num);
            let _ = imgcodecs::imwrite(&filename, &debug_img, &core::Vector::new());
        }
        // =========================================================

        // Perform OCR-based content filtering to remove redundant detections
        let mut content_filtered_rects = Vec::new();
        
        // 正規表現オブジェクトの取得(初回のみコンパイル)
        let re_invoice = RE_INVOICE.get_or_init(|| regex::Regex::new(r"(?i)T[0-9]{13}").unwrap());
        let re_phone = RE_PHONE.get_or_init(|| regex::Regex::new(r"0\d{1,4}-\d{1,4}-\d{3,4}").unwrap());

        // 1. Collect all OCR results first
        struct ScannedRect {
            rect: core::Rect,
            text: String, // raw
        }
        let mut scanned_rects = Vec::new();

        for (_idx, rect) in all_rects.iter().enumerate() {
            if let Ok(roi_ref) = core::Mat::roi(frame, *rect) {
                if roi_ref.copy_to(&mut self.roi_buf).is_ok() {
                    // まず垂直分割を試みる (2行以上のものを分割)
                    if let Ok(segments) = ocr.segment_lines(&self.roi_buf) {
                    for (rel_rect, seg_img) in segments {
                        // 相対座標を絶対座標に変換
                        let abs_rect = core::Rect::new(
                            rect.x + rel_rect.x,
                            rect.y + rel_rect.y,
                            rel_rect.width,
                            rel_rect.height
                        );
                        
                        let mut text = String::new();
                        // 分割された画像チャンクに対して個別に認識を実行
                        if let Ok(res) = ocr.recognize_chunk(&seg_img) {
                            text = res;
                        }
                        
                        scanned_rects.push(ScannedRect { rect: abs_rect, text });
                    }
                }
            }
        }
    }

        // =========================================================
        // [DEBUG] 全矩形のOCR結果をテキストファイル出力
        // =========================================================
        #[cfg(debug_assertions)]
        #[cfg(debug_assertions)]
        {
            use std::fs;
            use std::io::Write;
            
            let debug_dir = "debug_output";
            if !std::path::Path::new(debug_dir).exists() {
                let _ = fs::create_dir_all(debug_dir);
            }

            let txt_filename = format!("{}/Page_{}_detect_raw.txt", debug_dir, _page_num);
            // OpenOptionsではなく単純なFile::createを使用 (上書き作成)
            match fs::File::create(&txt_filename) {
                Ok(mut file) => {
                    for (i, sr) in scanned_rects.iter().enumerate() {
                        let _ = writeln!(
                            file,
                            "Rect[{}]: x={}, y={}, w={}, h={}\nText:\n{}\n----------------------------------------",
                            i, sr.rect.x, sr.rect.y, sr.rect.width, sr.rect.height, sr.text
                        );
                    }
                },
                Err(e) => {
                    println!("[DEBUG ERROR] Failed to create log file {}: {:?}", txt_filename, e);
                }
            }
        }
        // =========================================================

        // 2. Filter based on content (Keyword & Overlap check)

        for i in 0..scanned_rects.len() {
            let current = &scanned_rects[i];
            
            let mut is_keep = false;

            // 行ごとに判定
            for line in current.text.lines() {
                let clean_line = line.replace('\r', "").trim().to_string();
                if clean_line.is_empty() { continue; }

                let has_kei = clean_line.contains("計");
                let is_shokei = clean_line.contains("小計");
                let has_kei_valid = has_kei && !is_shokei;

                let is_invoice = re_invoice.is_match(&clean_line);
                let is_phone = re_phone.is_match(&clean_line);

                // % 判定: 自身に % があり、割・引がない
                let has_percent_self = clean_line.contains('%') 
                                    && !clean_line.contains("割") 
                                    && !clean_line.contains("引");
                
                let mut has_percent_valid = false;
                let mut neighbor_waribiki_found = false;

                if has_percent_self {
                    // 周辺(同じ行)に「割」「引」を持つ矩形がないかチェック
                    for j in 0..scanned_rects.len() {
                        if i == j { continue; }
                        let other = &scanned_rects[j];
                        
                        // Y方向の重なりチェック (閾値0.5)
                        if is_y_overlap_sufficient(&current.rect, &other.rect, 0.5) {
                            let other_clean = other.text.replace('\n', "");
                            if other_clean.contains("割") || other_clean.contains("引") {
                                neighbor_waribiki_found = true;
                                break;
                            }
                        }
                    }
                    if !neighbor_waribiki_found {
                        has_percent_valid = true;
                    }
                }

                if has_kei_valid || has_percent_valid || is_invoice || is_phone {
                    is_keep = true;
                    break;
                }
            }

            if is_keep {
                content_filtered_rects.push(current.rect);
            }
        }
        
        // 元の変数名を復帰・ソート処理へ
        let mut proc_rects = content_filtered_rects; 
        
        // 3. Remove duplicates based on area coverage (Duplicate Removal)
        proc_rects.sort_by(|a, b| {
            let area_a = a.width * a.height;
            let area_b = b.width * b.height;
            area_b.cmp(&area_a) // 大きい順
        });
        
        let mut filtered_rects = Vec::new();
        let threshold = 0.8; // 80%以上カバーされていたら冗長とみなす
        
        for (i, large_rect) in proc_rects.iter().enumerate() {
            // この大きな矩形内に含まれる小さな矩形たちを収集
            let mut inner_rects = Vec::new();
            for (j, small_rect) in proc_rects.iter().enumerate() {
                if i != j && rect_overlaps(large_rect, small_rect) {
                    // large_rect内の重なり部分を計算
                    if let Some(overlap) = rect_intersection(large_rect, small_rect) {
                        inner_rects.push(overlap);
                    }
                }
            }
            
            // inner_rects同士の重複を考慮した合計面積を計算
            let total_inner_area = calculate_union_area(&inner_rects);
            let large_area = (large_rect.width * large_rect.height) as f32;
            
            // 冗長性判定: 内部の矩形が大きな矩形の80%以上をカバーしていたら除外
            let coverage_ratio = total_inner_area / large_area;
            if coverage_ratio < threshold {
                filtered_rects.push(*large_rect);
            }
        }
        
        // 画面表示用には duplicate_removed.png と同じ filtered_rects をそのまま返す
        let final_detections = filtered_rects.iter().map(|r| Detection {
            rect: *r,
            color_state: 0,
        }).collect();


        
        Ok(final_detections)
    }

    fn preprocess_letterbox(&mut self, frame: &core::Mat) -> Result<Array4<f32>> {
        let (w, h) = (frame.cols() as f32, frame.rows() as f32);
        self.ratio = (MODEL_SIZE as f32 / w.max(h)).min(1.0);
        let (new_w, new_h) = ((w * self.ratio) as i32, (h * self.ratio) as i32);

        imgproc::resize(frame, &mut self.resized_buf, core::Size::new(new_w, new_h), 0.0, 0.0, imgproc::INTER_LINEAR)?;
        
        if self.canvas_buf.cols() != MODEL_SIZE || self.canvas_buf.rows() != MODEL_SIZE {
            self.canvas_buf = core::Mat::new_rows_cols_with_default(MODEL_SIZE, MODEL_SIZE, core::CV_8UC3, core::Scalar::all(0.0))?;
        } else {
            self.canvas_buf.set_to(&core::Scalar::all(0.0), &core::no_array())?;
        }

        self.offset_x = (MODEL_SIZE - new_w) as f32 / 2.0;
        self.offset_y = (MODEL_SIZE - new_h) as f32 / 2.0;
        let target_rect = core::Rect::new(self.offset_x as i32, self.offset_y as i32, new_w, new_h);
        
        // 修正ポイント2: 描画先(ROI)へのコピー
        // copy_to は BoxedRefMut に対しても有効なはずですが、
        // エラーを避けるため &mut * を使わず直接メソッドを呼びます
        let mut roi_mut_ref = core::Mat::roi_mut(&mut self.canvas_buf, target_rect)?;
        self.resized_buf.copy_to(&mut roi_mut_ref)?;

        imgproc::cvt_color(&self.canvas_buf, &mut self.rgb_buf, imgproc::COLOR_BGR2RGB, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        
        let mut input_array = Array4::<f32>::zeros((1, 3, MODEL_SIZE as usize, MODEL_SIZE as usize));
        for y in 0..MODEL_SIZE {
            for x in 0..MODEL_SIZE {
                let pixel = self.rgb_buf.at_2d::<core::Vec3b>(y, x)?;
                input_array[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
                input_array[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
                input_array[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
            }
        }
        Ok(input_array)
    }
}


// Helper: Determines if the vertical overlap between two rectangles is sufficient to consider them as being on the same line.
fn is_y_overlap_sufficient(row_bbox: &core::Rect, rect: &core::Rect, threshold_ratio: f32) -> bool {
    let y1 = row_bbox.y.max(rect.y);
    let y2 = (row_bbox.y + row_bbox.height).min(rect.y + rect.height);
    
    if y1 < y2 {
        let overlap_h = y2 - y1;
        let min_h = row_bbox.height.min(rect.height);
        // 重なり部分が、小さい方の高さの threshold_ratio (50%) 以上あれば同じ行とする
        (overlap_h as f32) >= (min_h as f32 * threshold_ratio)
    } else {
        false
    }
}

/// Checks if two rectangles overlap.
fn rect_overlaps(a: &core::Rect, b: &core::Rect) -> bool {
    !(a.x + a.width <= b.x || b.x + b.width <= a.x ||
      a.y + a.height <= b.y || b.y + b.height <= a.y)
}

/// Returns the intersection rectangle of two rectangles, or None if they don't intersect.
fn rect_intersection(a: &core::Rect, b: &core::Rect) -> Option<core::Rect> {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);
    
    if x1 < x2 && y1 < y2 {
        Some(core::Rect::new(x1, y1, x2 - x1, y2 - y1))
    } else {
        None
    }
}

/// Calculates the union area of multiple rectangles using the sweep-line algorithm.
fn calculate_union_area(rects: &[core::Rect]) -> f32 {
    if rects.is_empty() {
        return 0.0;
    }
    
    // スイープライン法で和集合の面積を計算
    // イベントポイント: (x座標, 開始/終了, y範囲)
    let mut events: Vec<(i32, bool, i32, i32)> = Vec::new();
    
    for rect in rects {
        events.push((rect.x, true, rect.y, rect.y + rect.height)); // 開始
        events.push((rect.x + rect.width, false, rect.y, rect.y + rect.height)); // 終了
    }
    
    // x座標でソート
    events.sort_by_key(|e| (e.0, !e.1));
    
    let mut total_area = 0.0;
    let mut active_intervals: Vec<(i32, i32)> = Vec::new();
    let mut prev_x = events[0].0;
    
    for (x, is_start, y1, y2) in events {
        // 前回のx座標から今回のx座標までの面積を加算
        if x != prev_x && !active_intervals.is_empty() {
            let width = (x - prev_x) as f32;
            let height = merge_intervals_height(&active_intervals);
            total_area += width * height;
        }
        
        // アクティブな区間を更新
        if is_start {
            active_intervals.push((y1, y2));
        } else {
            active_intervals.retain(|&(ay1, ay2)| ay1 != y1 || ay2 != y2);
        }
        
        prev_x = x;
    }
    
    total_area
}

/// Helper for sweep-line algorithm to merge vertical intervals.
fn merge_intervals_height(intervals: &[(i32, i32)]) -> f32 {
    if intervals.is_empty() {
        return 0.0;
    }
    
    let mut sorted = intervals.to_vec();
    sorted.sort_by_key(|&(start, _)| start);
    
    let mut merged: Vec<(i32, i32)> = Vec::new();
    let mut current = sorted[0];
    
    for &(start, end) in &sorted[1..] {
        if start <= current.1 {
            // 重なっている場合はマージ
            current.1 = current.1.max(end);
        } else {
            // 重なっていない場合は新しい区間として追加
            merged.push(current);
            current = (start, end);
        }
    }
    merged.push(current);
    
    // 合計の高さを計算
    merged.iter().map(|(start, end)| (end - start) as f32).sum()
}

