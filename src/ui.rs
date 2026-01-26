use opencv::{core, imgproc, prelude::*};
use crate::text_detector::Detection;
use crate::config::AppConfig;

pub struct Renderer;

impl Renderer {
    pub fn draw_all(
        display_view: &mut core::Mat,
        bgr_frame: &core::Mat,
        detections: &[Detection],
        mouse_x: i32,
        mouse_y: i32,
        current_page: u16,
        total_pages: usize,
        save_msg_timer: i32,
    ) -> anyhow::Result<()> {
        let scale_x = display_view.cols() as f32 / bgr_frame.cols().max(1) as f32;
        let scale_y = display_view.rows() as f32 / bgr_frame.rows().max(1) as f32;

        let mut active_det = None;

        for det in detections {
            let d_rect = core::Rect::new(
                (det.rect.x as f32 * scale_x) as i32,
                (det.rect.y as f32 * scale_y) as i32,
                (det.rect.width as f32 * scale_x) as i32,
                (det.rect.height as f32 * scale_y) as i32,
            );



            let base_color = match det.color_state {
                1 => core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                2 => core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                _ => core::Scalar::new(0.0, 165.0, 255.0, 0.0),
            };

            if d_rect.contains(core::Point::new(mouse_x, mouse_y)) {
                active_det = Some((det.rect, d_rect));
                imgproc::rectangle(display_view, d_rect, core::Scalar::new(255.0, 200.0, 0.0, 0.0), 2, 8, 0)?;
            } else {
                imgproc::rectangle(display_view, d_rect, base_color, 1, 8, 0)?;
            }
        }

        if let Some((orig_rect, d_rect)) = active_det {
            let zoom_factor = AppConfig::ZOOM_FACTOR;
            let (zw, zh) = ((d_rect.width as f32 * zoom_factor) as i32, (d_rect.height as f32 * zoom_factor) as i32);
            let (zx, zy) = (d_rect.x + (d_rect.width - zw) / 2, d_rect.y + d_rect.height + 10);

            if zx >= 0 && zy >= 0 && (zx + zw) <= display_view.cols() && (zy + zh) <= display_view.rows() {
                let safe_orig = core::Rect::new(
                    orig_rect.x.max(0),
                    orig_rect.y.max(0),
                    orig_rect.width.min(bgr_frame.cols() - orig_rect.x.max(0)),
                    orig_rect.height.min(bgr_frame.rows() - orig_rect.y.max(0)),
                );

                if let Ok(roi) = core::Mat::roi(bgr_frame, safe_orig) {
                    let mut zoomed_roi = core::Mat::default();
                    imgproc::resize(&roi, &mut zoomed_roi, core::Size::new(zw, zh), 0.0, 0.0, imgproc::INTER_LINEAR)?;
                    let mut dest_roi = core::Mat::roi_mut(display_view, core::Rect::new(zx, zy, zw, zh))?;
                    zoomed_roi.copy_to(&mut dest_roi)?;
                    imgproc::rectangle(display_view, core::Rect::new(zx, zy, zw, zh), core::Scalar::new(0.0, 0.0, 255.0, 0.0), 2, 8, 0)?;
                }
            }
        }

        let page_text = format!("{}/{}", current_page + 1, total_pages);
        imgproc::put_text(display_view, &page_text, core::Point::new(10, 40), imgproc::FONT_HERSHEY_SIMPLEX, 0.7, core::Scalar::all(0.0), 2, imgproc::LINE_AA, false)?;

        if save_msg_timer > 0 {
            imgproc::put_text(display_view, "SAVE DATA!", core::Point::new(200, 400), imgproc::FONT_HERSHEY_SIMPLEX, 1.5, core::Scalar::new(0.0, 0.0, 255.0, 0.0), 3, imgproc::LINE_AA, false)?;
        }

        Ok(())
    }
}