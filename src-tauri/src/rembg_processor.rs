use std::path::PathBuf;

use image::{DynamicImage, Rgba, RgbaImage};
use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

use crate::BoundingBox;
use crate::ProcessingConfig;

pub struct RembgProcessor {
    session: Option<Session>,
    current_model: String,
    input_h: u32,
    input_w: u32,
}

fn to_err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

impl RembgProcessor {
    pub fn new() -> Self {
        Self {
            session: None,
            current_model: String::new(),
            input_h: 320,
            input_w: 320,
        }
    }

    pub fn ensure_model(&mut self, model_name: &str) -> Result<(), String> {
        if self.current_model == model_name && self.session.is_some() {
            return Ok(());
        }

        let model_path = models_dir().join(model_name);
        if !model_path.exists() {
            return Err(format!(
                "模型文件不存在: {}。请将 ONNX 模型放入 models 目录 (u2net.onnx / u2net_human_seg.onnx / silueta.onnx)。",
                model_path.display()
            ));
        }

        let session = Session::builder()
            .map_err(to_err)?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(to_err)?
            .with_intra_threads(4)
            .map_err(to_err)?
            .commit_from_file(model_path.as_path())
            .map_err(|e| format!("加载 rembg 模型失败 {}: {}", model_name, e))?;

        let (input_h, input_w) = Self::detect_input_hw(&session);

        self.input_h = input_h;
        self.input_w = input_w;
        self.session = Some(session);
        self.current_model = model_name.to_string();

        eprintln!(
            "[rembg] 已加载模型 {} (输入尺寸 {}x{})",
            model_name, self.input_w, self.input_h
        );
        Ok(())
    }

    fn detect_input_hw(session: &Session) -> (u32, u32) {
        if let Some(outlet) = session.inputs().first() {
            if let Some(shape) = outlet.dtype().tensor_shape() {
                let dims: Vec<i64> = shape.iter().map(|d| *d).collect();
                if dims.len() >= 4 {
                    let h = dims[2] as u32;
                    let w = dims[3] as u32;
                    if h > 0 && w > 0 {
                        return (h, w);
                    }
                }
            }
        }
        (320, 320)
    }

    pub fn mask_for_bbox(
        &mut self,
        image: &DynamicImage,
        config: &ProcessingConfig,
        bbox: &BoundingBox,
    ) -> Result<Vec<bool>, String> {
        if !config.enable_rembg {
            return Ok(Vec::new());
        }

        let (img_w, img_h) = (image.width(), image.height());
        let x1 = bbox.x.min(img_w);
        let y1 = bbox.y.min(img_h);
        let x2 = (bbox.x + bbox.width).min(img_w);
        let y2 = (bbox.y + bbox.height).min(img_h);

        if x2 <= x1 || y2 <= y1 {
            return Ok(Vec::new());
        }

        let crop = image.crop_imm(x1, y1, x2 - x1, y2 - y1);
        let crop_w = crop.width();
        let crop_h = crop.height();

        let model_name = model_filename(&config.rembg_model);
        self.ensure_model(&model_name)?;

        let session = self
            .session
            .as_mut()
            .ok_or_else(|| "rembg 会话未初始化".to_string())?;

        let target_w = self.input_w;
        let target_h = self.input_h;

        let resized = crop.resize_exact(target_w, target_h, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();

        let mut input = Array4::<f32>::zeros((1, 3, target_h as usize, target_w as usize));
        for y in 0..target_h as usize {
            for x in 0..target_w as usize {
                let px = rgb.get_pixel(x as u32, y as u32);
                input[[0, 0, y, x]] = px[0] as f32 / 255.0;
                input[[0, 1, y, x]] = px[1] as f32 / 255.0;
                input[[0, 2, y, x]] = px[2] as f32 / 255.0;
            }
        }

        let input_tensor = Tensor::from_array(input).map_err(to_err)?;
        let input_value: SessionInputValue<'_> = input_tensor.into();

        let outputs = session
            .run([input_value])
            .map_err(|e| format!("rembg 推理失败: {}", e))?;

        let (shape, slice): (_, &[f32]) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("输出类型错误: {}", e))?;

        let shape_dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();

        eprintln!(
            "[rembg] 输出 shape={:?}, len={}, target={}x{}",
            shape_dims,
            slice.len(),
            target_w,
            target_h
        );

        let mask_img = Self::process_output(
            slice,
            &shape_dims,
            target_w,
            target_h,
            config,
            crop_w,
            crop_h,
        )?;
        let mask_raw = mask_img.into_raw();
        let mask_len = (crop_w as usize) * (crop_h as usize);
        let mut out = vec![false; mask_len];
        for y in 0..crop_h as usize {
            for x in 0..crop_w as usize {
                let pixel_idx = (y * crop_w as usize + x) * 4;
                if pixel_idx + 3 < mask_raw.len() {
                    let alpha = mask_raw[pixel_idx + 3];
                    out[y * crop_w as usize + x] = alpha > 128;
                }
            }
        }
        Ok(out)
    }

    fn process_output(
        slice: &[f32],
        shape: &[usize],
        target_w: u32,
        target_h: u32,
        config: &ProcessingConfig,
        orig_w: u32,
        orig_h: u32,
    ) -> Result<RgbaImage, String> {
        eprintln!("[rembg] 处理输出: shape={:?}, slice_len={}", shape, slice.len());

        let per_channel = (target_w as usize) * (target_h as usize);

        // 根据 shape 推断输出维度
        // 可能的格式:
        //   [1, 1, H, W]       - 单通道 alpha，取第一个通道
        //   [1, 3, H, W]       - RGB 三通道，取第一个通道
        //   [1, H, W]          - 单通道，直接用
        //   [N, H, W]          - 多通道，取第一个
        //   [H, W]             - 2D，直接用
        let data: &[f32] = if shape.len() == 4 {
            // [N, C, H, W]
            let c = shape[1];
            let h = shape[2];
            let w = shape[3];
            let hw = h * w;
            if hw == per_channel {
                // 取第一个通道
                &slice[..per_channel]
            } else if c == 3 && hw == per_channel {
                // RGB 格式，取第一个通道
                &slice[..per_channel]
            } else {
                // 尝试用实际的 HxW
                eprintln!("[rembg] shape mismatch: expected {}x{}, got {}x{}", target_w, target_h, w, h);
                let actual_len = hw.min(slice.len());
                &slice[..actual_len]
            }
        } else if shape.len() == 3 {
            // [N, H, W] 或 [N, C, L]
            let last = shape[2];
            let mid = shape[1];
            if mid * last >= per_channel {
                // 取第一个 "channel"
                let total = (shape[0] * shape[1] * shape[2]).min(slice.len());
                let usable = per_channel.min(total);
                &slice[..usable]
            } else {
                // H,W 在后
                let hw = mid * last;
                let usable = hw.min(slice.len());
                &slice[..usable]
            }
        } else if shape.len() == 2 {
            let hw = shape[0] * shape[1];
            let usable = hw.min(slice.len());
            &slice[..usable]
        } else {
            // 回退: 直接用前 per_channel 个
            let usable = per_channel.min(slice.len());
            &slice[..usable]
        };

        eprintln!("[rembg] data_len={}, per_channel={}", data.len(), per_channel);

        let threshold = (config.rembg_threshold as f32).clamp(0.0, 1.0);
        let binary = config.rembg_binary_mode;

        let out_h = target_h as usize;
        let out_w = target_w as usize;
        let mut mask_img = RgbaImage::from_pixel(target_w, target_h, Rgba([0u8, 0u8, 0u8, 0u8]));

        for y in 0..out_h {
            for x in 0..out_w {
                let idx = y * out_w + x;
                let val = if idx < data.len() {
                    data[idx]
                } else {
                    0.0f32
                };
                let alpha = if binary {
                    if val > threshold {
                        255u8
                    } else {
                        0u8
                    }
                } else {
                    let v = (val * 255.0).clamp(0.0, 255.0) as u8;
                    if val > threshold {
                        v.max(128)
                    } else {
                        v
                    }
                };
                mask_img.put_pixel(x as u32, y as u32, Rgba([255, 255, 255, alpha]));
            }
        }

        if target_w != orig_w || target_h != orig_h {
            use image::imageops::FilterType;
            eprintln!(
                "[rembg] 上采样 mask: {}x{} -> {}x{}",
                target_w, target_h, orig_w, orig_h
            );
            let start = std::time::Instant::now();
            let resized = image::imageops::resize(&mask_img, orig_w, orig_h, FilterType::Triangle);
            eprintln!(
                "[rembg] 上采样完成 ({:.2}s), 开始生成布尔mask",
                start.elapsed().as_secs_f64()
            );
            Ok(resized)
        } else {
            Ok(mask_img)
        }
    }
}

impl Default for RembgProcessor {
    fn default() -> Self {
        Self::new()
    }
}

pub fn models_dir() -> PathBuf {
    use std::env;
    use std::path::PathBuf;

    let base = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let mut candidates: Vec<PathBuf> = Vec::new();

    // 1. current_dir/models (dev: target/debug/_up_/models)
    candidates.push({
        let mut p = base.clone();
        p.push("models");
        p
    });

    // 2. 向上查找祖级目录的 models
    let mut cur = base.clone();
    for _ in 0..5 {
        if let Some(parent) = cur.parent() {
            let mut p = parent.to_path_buf();
            p.push("models");
            candidates.push(p.clone());
            cur = parent.to_path_buf();
        }
    }

    // 3. src-tauri 子目录的 models
    {
        let mut p = base.clone();
        p.push("src-tauri");
        p.push("models");
        candidates.push(p);
    }

    // 优先选择包含 rembg 模型的目录
    for cand in &candidates {
        if cand.exists() && has_rembg_models(cand) {
            return cand.clone();
        }
    }

    // 回退：选择第一个存在的目录
    for cand in &candidates {
        if cand.exists() {
            return cand.clone();
        }
    }

    candidates.first().cloned().unwrap_or(base)
}

fn has_rembg_models(dir: &std::path::Path) -> bool {
    for (_, file) in available_models() {
        if dir.join(file).exists() {
            return true;
        }
    }
    false
}

pub fn available_models() -> Vec<(&'static str, &'static str)> {
    vec![
        ("u2net", "u2net.onnx"),
        ("u2net_human_seg", "u2net_human_seg.onnx"),
        ("silueta", "silueta.onnx"),
    ]
}

fn model_filename(model_key: &str) -> String {
    for (key, file) in available_models() {
        if key == model_key {
            return file.to_string();
        }
    }
    "u2net.onnx".to_string()
}

pub fn list_available_models() -> Vec<String> {
    let dir = models_dir();
    available_models()
        .iter()
        .filter_map(|(_, file)| {
            let path = dir.join(file);
            if path.exists() {
                Some(file.to_string())
            } else {
                None
            }
        })
        .collect()
}
