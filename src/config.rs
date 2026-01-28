pub struct AppConfig;

impl AppConfig {
    // モデル関連
    pub const DETECTOR_MODEL: &'static str = "text_detection_model.onnx";
    pub const OCR_MODEL: &'static str = "ch_PP-OCRv3_rec_infer.onnx";
    pub const OCR_DICT: &'static str = "ppocr_keys_v1.txt";

    // 入出力
    // 実行ディレクトリからの相対パス
    pub const PDF_PATH: &'static str = "sample.pdf"; 
    pub const OUTPUT_DIR: &'static str = "output";

    // UI/表示設定
    pub const WINDOW_NAME: &'static str = "Sleek Accounts";
    pub const DEFAULT_WIN_W: i32 = 600;
    pub const DEFAULT_WIN_H: i32 = 800;
    pub const RENDER_WIDTH: i32 = 1200; // main.rsの型エラー防止のためi32
    
    // 判定ロジック
    pub const ZOOM_FACTOR: f32 = 2.5;
}