use opencv::{core, imgproc, prelude::*};
use ort::{session::Session, value::Value, inputs};
use ndarray::Array4;
use anyhow::{Result, bail};
use std::fs::File;
use std::io::{BufRead, BufReader};

/// OcrEngine handles text recognition using ONNX Runtime.
/// It uses a character dictionary to map model outputs to strings.
pub struct OcrEngine {
    session: Session,
    alphabet: Vec<String>,
    resized_buf: core::Mat,
    rgb_buf: core::Mat,
    gray_buf: core::Mat,
    binary_buf: core::Mat,
    avg_line_height: i32,
}

impl OcrEngine {
    /// Initializes a new OcrEngine.
    /// 
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file.
    /// * `dict_path` - Path to the dictionary file containing supported characters.
    pub fn new(model_path: &str, dict_path: &str) -> Result<Self> {
        let mut alphabet = vec!["blank".to_string()]; 
        let file = File::open(dict_path).map_err(|_| { anyhow::anyhow!("辞書ファイル '{}' が見つかりません。", dict_path) })?;
        for line in BufReader::new(file).lines() { alphabet.push(line?); }
        alphabet.push(" ".to_string());
        let session = Session::builder()?.with_intra_threads(1)?.commit_from_file(model_path)?;
        Ok(Self { 
            session, 
            alphabet,
            resized_buf: core::Mat::default(),
            rgb_buf: core::Mat::default(),
            gray_buf: core::Mat::default(),
            binary_buf: core::Mat::default(),
            avg_line_height: 30, // デフォルト値
        })
    }

    /// Calibrates the expected line height based on detected rectangles.
    /// This helps in handling multi-column or multi-row layout decisions.
    pub fn calibrate_line_height(&mut self, rects: &[core::Rect]) {
        if rects.is_empty() { return; }
        

        let mut heights: Vec<i32> = rects.iter().map(|r| r.height).collect();
        heights.sort();
        
        let mid = heights.len() / 2;
        self.avg_line_height = heights[mid];
    }

    /// Recognizes text within the specified image area.
    /// 
    /// This method preprocesses the image (resize, normalize) and runs inference.
    /// It also handles potential multi-line text by splitting vertically if clear gaps are found.
    pub fn recognize(&mut self, img: &core::Mat) -> Result<String> {
        if img.empty() { bail!("入力画像が空です。"); }

        // 多段組チェック（簡易版）
        // 目安: 中央値の1.5倍以上あれば2行の可能性がある
        let h = img.rows();
        let split_threshold = (self.avg_line_height as f32 * 1.5) as i32;
        
        if h > split_threshold {
            // グレースケール化 & 二値化
            
            // カラーなら変換、そうでなければコピー
            if img.channels() == 3 {
                imgproc::cvt_color(img, &mut self.gray_buf, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
            } else {
                img.copy_to(&mut self.gray_buf)?;
            }
            
            // 白黒反転（文字が白、背景黒になるように調整）
            // 通常OCRは白地に黒文字が多いが、二値化で文字を抽出する
            imgproc::threshold(&self.gray_buf, &mut self.binary_buf, 0.0, 255.0, imgproc::THRESH_BINARY_INV | imgproc::THRESH_OTSU)?;

            // 水平射影（行ごとの画素値合計）
            let w = img.cols();
            let mut min_val = w as i32 * 255;
            let mut split_y = -1;
            
            // 上下20%は無視して、中間領域で分割点を探す
            let start_y = (h as f32 * 0.2) as i32;
            let end_y = (h as f32 * 0.8) as i32;
            
            for y in start_y..end_y {
                let mut row_sum = 0;
                for x in 0..w {
                    let val = *self.binary_buf.at_2d::<u8>(y, x)? as i32;
                    if val > 0 { row_sum += 1; }
                }
                
                // 行に含まれる白い画素（文字部分）が極端に少ない場所を探す
                if row_sum < min_val {
                    min_val = row_sum;
                    split_y = y;
                }
            }

            // 分割点が見つかり、かつそこが十分に空白（幅の5%以下しか文字がない）なら分割
            if split_y != -1 && (min_val as f32) < (w as f32 * 0.05) {
                // 上下分割
                let rect_top = core::Rect::new(0, 0, w, split_y);
                let rect_bottom = core::Rect::new(0, split_y, w, h - split_y);
                
                let img_top = core::Mat::roi(img, rect_top)?;
                let img_bottom = core::Mat::roi(img, rect_bottom)?; // Note: roi creates reference, safe to pass to recognize? Yes if properly cloned or handled.
                
                // 再帰呼び出しのためにMatをコピー（recognizeは&mut selfをとるがimgはimmutなのでOKだが、roi参照問題を避けるためコピー推奨）
                let mut top_clone = core::Mat::default();
                let mut bot_clone = core::Mat::default();
                img_top.copy_to(&mut top_clone)?;
                img_bottom.copy_to(&mut bot_clone)?;
                
                let res_top = self.recognize(&top_clone)?;
                let res_bottom = self.recognize(&bot_clone)?;
                
                return Ok(format!("{}\n{}", res_top, res_bottom));
            }
        }

        let target_h = 48;
        let aspect_ratio = img.cols() as f32 / img.rows() as f32;
        let target_w = (target_h as f32 * aspect_ratio) as i32;

        imgproc::resize(img, &mut self.resized_buf, core::Size::new(target_w, target_h), 0.0, 0.0, imgproc::INTER_LINEAR)?;
        imgproc::cvt_color(&self.resized_buf, &mut self.rgb_buf, imgproc::COLOR_BGR2RGB, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;

        let mut input_array = Array4::<f32>::zeros((1, 3, target_h as usize, target_w as usize));
        for y in 0..target_h {
            for x in 0..target_w {
                let pixel = self.rgb_buf.at_2d::<core::Vec3b>(y, x)?;
                input_array[[0, 0, y as usize, x as usize]] = (pixel[0] as f32 / 255.0 - 0.5) / 0.5;
                input_array[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 / 255.0 - 0.5) / 0.5;
                input_array[[0, 2, y as usize, x as usize]] = (pixel[2] as f32 / 255.0 - 0.5) / 0.5;
            }
        }

        let input_tensor = Value::from_array(input_array)?;
        let outputs = self.session.run(inputs![&input_tensor])?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        
        let steps = shape[1] as usize;
        let voc_size = shape[2] as usize;
        let mut recognized_text = String::new();
        let mut last_index = 0;

        for i in 0..steps {
            let step_data = &data[i * voc_size .. (i + 1) * voc_size];
            let mut max_idx = 0;
            let mut max_val = step_data[0];
            for (idx, &val) in step_data.iter().enumerate() {
                if val > max_val { max_val = val; max_idx = idx; }
            }
            if max_idx > 0 && max_idx < self.alphabet.len() && max_idx != last_index {
                recognized_text.push_str(&self.alphabet[max_idx]);
            }
            last_index = max_idx;
        }
        

        
        Ok(recognized_text)
    }
}