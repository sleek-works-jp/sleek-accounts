use crate::text_detector::Detection;
use crate::text_ocr::OcrEngine;

pub struct AppState {
    pub mouse_x: i32,
    pub mouse_y: i32,
    pub offset_x: i32,
    pub offset_y: i32,
    pub is_dragging: bool,
    pub drag_last_x: i32,
    pub drag_last_y: i32,
    pub current_page: u16,
    pub ocr_engine: Option<OcrEngine>,
    pub detections: Vec<Detection>,
    pub save_msg_timer: i32,
}