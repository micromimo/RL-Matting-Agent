use image::{DynamicImage, GrayImage, ImageFormat, Luma, Rgba, RgbaImage};
use imageproc::{
    edges::canny,
    morphology::{close, dilate, erode, open},
    distance_transform::{distance_transform, Norm},
    region_labelling::{connected_components, Connectivity},
    contrast::adaptive_threshold,
    filter::bilateral_filter,
};
use rayon::prelude::*;

use crate::{BoundingBox, ProcessingConfig};

pub fn load_image(path: &str) -> Result<DynamicImage, String> {
    image::open(path).map_err(|e| e.to_string())
}

pub fn crop_resize_patch(
    img: &DynamicImage,
    bbox: &BoundingBox,
    target_size: u32,
) -> Result<Vec<f32>, String> {
    let (w, h) = (img.width(), img.height());

    let x = bbox.x.min(w.saturating_sub(1));
    let y = bbox.y.min(h.saturating_sub(1));
    let bw = bbox.width.min(w.saturating_sub(x));
    let bh = bbox.height.min(h.saturating_sub(y));

    let crop = img.crop_imm(x, y, bw, bh);
    let resized = crop.resize_exact(target_size, target_size, image::imageops::FilterType::Triangle);

    let rgba = resized.to_rgba8();
    let pixels: Vec<f32> = rgba
        .pixels()
        .par_bridge()
        .flat_map_iter(|p| {
            let [r, g, b, _a] = p.0;
            [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
        })
        .collect();

    Ok(pixels)
}

pub fn generate_mask_with_feature(
    img: &DynamicImage,
    bbox: &BoundingBox,
    _target_feature: &[f32],
    target_confidence: f32,
    feature_grid: &[Vec<Option<Vec<f32>>>],
    confidence_grid: &[Vec<f32>],
    step: usize,
    config: &ProcessingConfig,
) -> Result<String, String> {
    use base64::Engine;
    let enable_traditional = config.enable_traditional;
    let (w, h) = (img.width(), img.height());
    let base = img.to_rgba8();

    let x1 = bbox.x.min(w);
    let y1 = bbox.y.min(h);
    let x2 = (bbox.x + bbox.width).min(w);
    let y2 = (bbox.y + bbox.height).min(h);

    let crop_w = (x2 - x1) as usize;
    let crop_h = (y2 - y1) as usize;

    if crop_w == 0 || crop_h == 0 {
        return Err("empty bbox".to_string());
    }

    let mut mask = RgbaImage::from_pixel(w, h, Rgba([0u8, 0u8, 0u8, 0u8]));

    let crop_rgba_img = crop_rgba_as_image(&base, x1, y1, x2, y2);
    let crop_gray_img = crop_gray_as_image(&base, x1, y1, x2, y2);

    let bg_color = estimate_background_color_from_image(&crop_rgba_img);
    eprintln!("Estimated background: RGB({}, {}, {})", bg_color.0, bg_color.1, bg_color.2);

    let grid_cols = feature_grid.len();
    let grid_rows = if !feature_grid.is_empty() { feature_grid[0].len() } else { 0 };
    let conf_cols = confidence_grid.len();
    let conf_rows = if !confidence_grid.is_empty() { confidence_grid[0].len() } else { 0 };

    eprintln!("Mask gen: crop={}x{}, grid={}x{}, conf_grid={}x{}, target_conf={:.3}, traditional={}",
        crop_w, crop_h, grid_cols, grid_rows, conf_cols, conf_rows, target_confidence, enable_traditional);

    let score_map = compute_score_map(
        &crop_rgba_img,
        &crop_gray_img,
        feature_grid,
        confidence_grid,
        target_confidence,
        step,
        bg_color,
    );

    let mut working_gray = crop_gray_img.clone();

    if enable_traditional && config.trad_bilateral_filter {
        working_gray = bilateral_filter(
            &working_gray,
            3u32,
            config.trad_bilateral_sigma_color as f32,
            config.trad_bilateral_sigma_space as f32,
        );
    }

    let edges = if enable_traditional {
        canny(&working_gray, config.trad_canny_low as f32, config.trad_canny_high as f32)
    } else {
        GrayImage::from_pixel(crop_w as u32, crop_h as u32, Luma([0u8]))
    };

    let min_score = score_map.iter().flatten().cloned().fold(f32::MAX, |min, x| min.min(x));
    let max_score = score_map.iter().flatten().cloned().fold(f32::MIN, |max, x| max.max(x));
    let contrast = (max_score - min_score).max(0.001);

    let mut normalized_scores: Vec<Vec<f32>> = vec![vec![0.0f32; crop_h]; crop_w];
    for cy in 0..crop_h {
        for cx in 0..crop_w {
            let s = (score_map[cx][cy] - min_score) / contrast;
            normalized_scores[cx][cy] = s.max(0.0).min(1.0);
        }
    }

    let threshold = compute_adaptive_threshold(&normalized_scores);
    let threshold_norm = (threshold - min_score) / contrast;

    eprintln!("Threshold: {:.4} (normalized: {:.4})", threshold, threshold_norm);

    let mut binary_mask = GrayImage::from_pixel(crop_w as u32, crop_h as u32, Luma([0u8]));
    for cy in 0..crop_h {
        for cx in 0..crop_w {
            let edge_boost = edges.get_pixel(cx as u32, cy as u32)[0] as f32 / 255.0;
            let combined_score = if enable_traditional {
                normalized_scores[cx][cy] + config.trad_edge_weight as f32 * edge_boost
            } else {
                normalized_scores[cx][cy]
            };
            let adjusted_threshold = if enable_traditional && edge_boost > 0.1 {
                threshold_norm * 0.7
            } else {
                threshold_norm
            };
            if combined_score > adjusted_threshold {
                binary_mask.put_pixel(cx as u32, cy as u32, Luma([255u8]));
            }
        }
    }

    if enable_traditional && config.trad_use_adaptive_threshold {
        let mut score_gray = GrayImage::from_pixel(crop_w as u32, crop_h as u32, Luma([0u8]));
        for cy in 0..crop_h {
            for cx in 0..crop_w {
                let v = (normalized_scores[cx][cy] * 255.0) as u8;
                score_gray.put_pixel(cx as u32, cy as u32, Luma([v]));
            }
        }
        let block = config.trad_adaptive_threshold_block.max(3) as u32;
        let block = if block % 2 == 0 { block + 1 } else { block };
        let adaptive_mask = adaptive_threshold(&score_gray, block);
        for cy in 0..crop_h {
            for cx in 0..crop_w {
                if adaptive_mask.get_pixel(cx as u32, cy as u32)[0] == 255 {
                    let new_val = (normalized_scores[cx][cy] * 255.0) as u8;
                    if new_val > 128 {
                        binary_mask.put_pixel(cx as u32, cy as u32, Luma([255u8]));
                    }
                }
            }
        }
    }

    let binary_mask = if enable_traditional {
        apply_traditional_postprocessing(&binary_mask, config.trad_morphology_radius)
    } else {
        binary_mask
    };

    let mut final_mask_raw = extract_foreground_mask(&binary_mask, config.trad_min_component_ratio);

    if enable_traditional && config.trad_use_distance_transform {
        let mut binary_u8 = GrayImage::from_pixel(crop_w as u32, crop_h as u32, Luma([0u8]));
        for cy in 0..crop_h {
            for cx in 0..crop_w {
                if final_mask_raw[cx][cy] {
                    binary_u8.put_pixel(cx as u32, cy as u32, Luma([255u8]));
                }
            }
        }
        let bg_dist = distance_transform(&binary_u8, Norm::L2);
        for cy in 0..crop_h {
            for cx in 0..crop_w {
                if final_mask_raw[cx][cy] {
                    let d = bg_dist.get_pixel(cx as u32, cy as u32)[0] as f32;
                    if d < 2.0 && normalized_scores[cx][cy] < threshold_norm * 1.2 {
                        final_mask_raw[cx][cy] = false;
                    }
                }
            }
        }
    }

    let final_mask = final_mask_raw;

    for cy in 0..crop_h {
        for cx in 0..crop_w {
            if final_mask[cx][cy] {
                let y = y1 + cy as u32;
                let x = x1 + cx as u32;
                if y < h && x < w {
                    let pixel = crop_rgba_img.get_pixel(cx as u32, cy as u32);
                    mask.put_pixel(x, y, Rgba([pixel[0], pixel[1], pixel[2], 255]));
                }
            }
        }
    }

    let final_count = final_mask.iter().flatten().filter(|&&b| b).count();
    eprintln!("Final mask: {} pixels ({:.1}% of crop)", final_count, 100.0 * final_count as f64 / (crop_w * crop_h) as f64);

    let mut out: Vec<u8> = Vec::new();
    {
        let mut cursor = std::io::Cursor::new(&mut out);
        mask.write_to(&mut cursor, ImageFormat::Png)
            .map_err(|e| e.to_string())?;
    }

    Ok(base64::engine::general_purpose::STANDARD.encode(&out))
}

fn crop_rgba_as_image(base: &RgbaImage, x1: u32, y1: u32, x2: u32, y2: u32) -> RgbaImage {
    let crop = image::DynamicImage::ImageRgba8(base.clone())
        .crop_imm(x1, y1, x2 - x1, y2 - y1);
    crop.to_rgba8()
}

fn crop_gray_as_image(base: &RgbaImage, x1: u32, y1: u32, x2: u32, y2: u32) -> GrayImage {
    let crop = image::DynamicImage::ImageRgba8(base.clone())
        .crop_imm(x1, y1, x2 - x1, y2 - y1);
    crop.to_luma8()
}

fn compute_score_map(
    crop_rgba: &RgbaImage,
    crop_gray: &GrayImage,
    _feature_grid: &[Vec<Option<Vec<f32>>>],
    confidence_grid: &[Vec<f32>],
    target_confidence: f32,
    step: usize,
    bg_color: (u8, u8, u8),
) -> Vec<Vec<f32>> {
    let crop_w = crop_rgba.width() as usize;
    let crop_h = crop_rgba.height() as usize;

    let conf_cols = confidence_grid.len();
    let conf_rows = if !confidence_grid.is_empty() { confidence_grid[0].len() } else { 0 };

    let gray_values: Vec<f32> = crop_gray.pixels().map(|p| p[0] as f32).collect();
    let center_gray = if !gray_values.is_empty() {
        gray_values[gray_values.len() / 2]
    } else {
        128.0
    };

    let bg_hsv = rgb_to_hsv(bg_color.0 as f32, bg_color.1 as f32, bg_color.2 as f32);

    let mut score_map: Vec<Vec<f32>> = vec![vec![0.0f32; crop_h]; crop_w];

    for cy in 0..crop_h {
        for cx in 0..crop_w {
            let mut score = 0.0f32;
            let mut weights = 0.0f32;

            if conf_cols > 0 && conf_rows > 0 {
                let gx_idx = (cx / step).min(conf_cols.saturating_sub(1));
                let gy_idx = (cy / step).min(conf_rows.saturating_sub(1));
                if gx_idx < conf_cols && gy_idx < conf_rows {
                    let conf = confidence_grid[gx_idx][gy_idx];
                    let conf_score = if conf > target_confidence {
                        (conf - target_confidence).min(1.0)
                    } else {
                        0.0
                    };
                    score += conf_score * 2.0;
                    weights += 2.0;
                }
            }

            let pixel = crop_rgba.get_pixel(cx as u32, cy as u32);
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;

            let (h_val, s_val, v_val) = rgb_to_hsv(r, g, b);
            let h_diff = (h_val - bg_hsv.0).abs().min(360.0 - (h_val - bg_hsv.0).abs());
            let s_diff = (s_val - bg_hsv.1).abs();
            let v_diff = (v_val - bg_hsv.2).abs();

            let h_score = 1.0 - (h_diff / 60.0).min(1.0);
            let s_score = (s_diff / 1.0).min(1.0);
            let v_score = (v_diff / 255.0).min(1.0);

            let combined_bg_score = if bg_hsv.1 < 0.15 {
                v_score
            } else {
                (h_score * 0.4 + s_score * 0.35 + v_score * 0.25).min(1.0)
            };

            score += combined_bg_score * 3.0;
            weights += 3.0;

            let euclidean_dist = ((r - bg_color.0 as f32).powi(2) +
                          (g - bg_color.1 as f32).powi(2) +
                          (b - bg_color.2 as f32).powi(2)).sqrt() / 441.67_f32;
            score += euclidean_dist.min(1.0) * 1.0;
            weights += 1.0;

            let gray_val = pixel[0] as f32 / 3.0 + pixel[1] as f32 / 3.0 + pixel[2] as f32 / 3.0;
            let color_diff = (gray_val - center_gray).abs() / 255.0;
            score += color_diff * 0.3;
            weights += 0.3;

            let dx = cx as f32 / crop_w as f32 - 0.5;
            let dy = cy as f32 / crop_h as f32 - 0.5;
            let dist_from_center = (dx * dx + dy * dy).sqrt() * 2.0;
            let center_prior = 1.0 - dist_from_center.min(1.0);
            score += center_prior * 0.2;
            weights += 0.2;

            let final_score = if weights > 0.0 { score / weights } else { 0.5 };

            score_map[cx][cy] = final_score;
        }
    }

    score_map
}

fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let rf = r / 255.0;
    let gf = g / 255.0;
    let bf = b / 255.0;
    
    let max_val = rf.max(gf).max(bf);
    let min_val = rf.min(gf).min(bf);
    let diff = max_val - min_val;
    
    let h = if diff < 0.001 {
        0.0
    } else if max_val == rf {
        60.0 * (((gf - bf) / diff) % 6.0)
    } else if max_val == gf {
        60.0 * (((bf - rf) / diff) + 2.0)
    } else {
        60.0 * (((rf - gf) / diff) + 4.0)
    };
    
    let s = if max_val < 0.001 { 0.0 } else { diff / max_val };
    let v = max_val;
    
    (h.abs(), s, v * 255.0)
}

fn compute_adaptive_threshold(scores: &[Vec<f32>]) -> f32 {
    let mut values: Vec<f32> = scores.iter().flatten().cloned().collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = values.len();
    if n == 0 {
        return 0.5;
    }

    let q10_idx = ((n as f64 * 0.10) as usize).min(n.saturating_sub(1));
    let q25_idx = ((n as f64 * 0.25) as usize).min(n.saturating_sub(1));
    let q50_idx = ((n as f64 * 0.50) as usize).min(n.saturating_sub(1));
    let q75_idx = ((n as f64 * 0.75) as usize).min(n.saturating_sub(1));

    let q10 = values[q10_idx];
    let q25 = values[q25_idx];
    let q50 = values[q50_idx];
    let q75 = values[q75_idx];

    let scores_std = {
        let mean = values.iter().sum::<f32>() / n as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
        variance.sqrt()
    };

    eprintln!("Percentiles: q10={:.4}, q25={:.4}, q50={:.4}, q75={:.4}, std={:.4}",
        q10, q25, q50, q75, scores_std);

    if scores_std > 0.05 {
        q50
    } else if scores_std > 0.02 {
        (q25 + q50) / 2.0
    } else {
        (q10 + q25) / 2.0
    }
}

fn apply_traditional_postprocessing(mask: &GrayImage, radius: u32) -> GrayImage {
    let r: u8 = radius.clamp(1, 8) as u8;
    let mask = open(mask, Norm::L1, r);
    let mask = close(&mask, Norm::L1, r);

    let eroded = erode(&mask, Norm::L1, r.max(2) - 1);
    let dilated = dilate(&eroded, Norm::L1, r);

    let mask = close(&dilated, Norm::L1, r.max(2) - 1);

    dilate(&mask, Norm::L1, 1)
}

fn extract_foreground_mask(mask: &GrayImage, min_component_ratio: f64) -> Vec<Vec<bool>> {
    let w = mask.width() as usize;
    let h = mask.height() as usize;

    let labeled = connected_components(mask, Connectivity::Eight, Luma([0u8]));
    let max_label = labeled.pixels().map(|p| p[0]).max().unwrap_or(0);

    let mut component_sizes: Vec<u64> = vec![0; max_label as usize + 1];
    for pixel in labeled.pixels() {
        let label = pixel[0] as usize;
        if label > 0 && label < component_sizes.len() {
            component_sizes[label] += 1;
        }
    }

    let mut largest_label = 0u32;
    let mut second_largest_label = 0u32;
    let mut largest_size = 0u64;
    let mut second_size = 0u64;

    let bg_size_threshold = (w as u64 * h as u64 * 3 / 10) as u64;
    let min_component_size = ((w as f64 * h as f64) * min_component_ratio) as u64;

    for (label, &size) in component_sizes.iter().enumerate() {
        if label == 0 {
            continue;
        }
        if size > largest_size {
            second_largest_label = largest_label;
            second_size = largest_size;
            largest_label = label as u32;
            largest_size = size;
        } else if size > second_size {
            second_largest_label = label as u32;
            second_size = size;
        }
    }

    eprintln!("Largest component: label={}, size={}px, 2nd: label={}, size={}px",
        largest_label, largest_size, second_largest_label, second_size);

    let is_largest_bg = largest_size > bg_size_threshold;

    let mut foreground_labels: Vec<u32> = Vec::new();

    if is_largest_bg {
        eprintln!("Largest component appears to be background (size={} > 30% of crop), excluding it", largest_size);
        for (label, &size) in component_sizes.iter().enumerate() {
            if label == 0 || label as u32 == largest_label {
                continue;
            }
            if size >= min_component_size.max(1) {
                foreground_labels.push(label as u32);
            }
        }
    } else {
        let min_size_threshold = (largest_size as f64 * min_component_ratio) as u64;
        foreground_labels.push(largest_label);
        if second_size >= min_size_threshold.max(1) {
            foreground_labels.push(second_largest_label);
        }
        for (label, &size) in component_sizes.iter().enumerate() {
            if label == 0 || label as u32 == largest_label || label as u32 == second_largest_label {
                continue;
            }
            if size >= min_size_threshold.max(1) {
                foreground_labels.push(label as u32);
            }
        }
    }

    eprintln!("Foreground labels to keep: {:?}", foreground_labels);

    if foreground_labels.is_empty() {
        eprintln!("No foreground labels found, keeping largest component as fallback");
        foreground_labels.push(largest_label);
    }

    let mut result: Vec<Vec<bool>> = vec![vec![false; h]; w];

    for cy in 0..h {
        for cx in 0..w {
            let label = labeled.get_pixel(cx as u32, cy as u32)[0] as u32;
            if label == 0 {
                continue;
            }
            if foreground_labels.contains(&label) {
                result[cx][cy] = true;
            }
        }
    }

    result
}

fn estimate_background_color_from_image(img: &RgbaImage) -> (u8, u8, u8) {
    let w = img.width() as usize;
    let h = img.height() as usize;
    let border_size = ((w.min(h) as f64) * 0.05) as usize;
    let border_size = border_size.max(3).min(10);

    let mut r_values: Vec<u8> = Vec::new();
    let mut g_values: Vec<u8> = Vec::new();
    let mut b_values: Vec<u8> = Vec::new();

    for cx in 0..w {
        for by in 0..border_size.min(h) {
            let p = img.get_pixel(cx as u32, by as u32);
            r_values.push(p[0]);
            g_values.push(p[1]);
            b_values.push(p[2]);
        }
        for by in (h - border_size).min(h)..h {
            let p = img.get_pixel(cx as u32, by as u32);
            r_values.push(p[0]);
            g_values.push(p[1]);
            b_values.push(p[2]);
        }
    }

    for cy in 0..h {
        for bx in 0..border_size.min(w) {
            let p = img.get_pixel(bx as u32, cy as u32);
            r_values.push(p[0]);
            g_values.push(p[1]);
            b_values.push(p[2]);
        }
        for bx in (w - border_size).min(w)..w {
            let p = img.get_pixel(bx as u32, cy as u32);
            r_values.push(p[0]);
            g_values.push(p[1]);
            b_values.push(p[2]);
        }
    }

    if r_values.is_empty() {
        return (255, 255, 255);
    }

    let median = |mut v: Vec<u8>| -> u8 {
        v.sort();
        let n = v.len();
        if n % 2 == 0 {
            ((v[n / 2 - 1] as u16 + v[n / 2] as u16) / 2) as u8
        } else {
            v[n / 2]
        }
    };

    let r_median = median(r_values.clone());
    let g_median = median(g_values.clone());
    let b_median = median(b_values.clone());

    let r_var: f32 = r_values.iter().map(|x| {
        let diff = *x as f32 - r_median as f32;
        diff * diff
    }).sum::<f32>() / r_values.len() as f32;
    
    let g_var: f32 = g_values.iter().map(|x| {
        let diff = *x as f32 - g_median as f32;
        diff * diff
    }).sum::<f32>() / g_values.len() as f32;
    
    let b_var: f32 = b_values.iter().map(|x| {
        let diff = *x as f32 - b_median as f32;
        diff * diff
    }).sum::<f32>() / b_values.len() as f32;

    let color_var = (r_var.sqrt() + g_var.sqrt() + b_var.sqrt()) / 3.0;

    eprintln!("Background color estimate: RGB({}, {}, {}), variance: {:.1}", r_median, g_median, b_median, color_var);

    (r_median, g_median, b_median)
}

pub fn compute_feature_and_confidence_grids<F>(
    crop_rgba: &[Vec<(u8, u8, u8)>],
    crop_gray: &[Vec<f32>],
    crop_w: usize,
    crop_h: usize,
    step: usize,
    mut model_fn: F,
) -> (Vec<Vec<Option<Vec<f32>>>>, Vec<Vec<f32>>)
where
    F: FnMut(&[f32], &[f32]) -> Result<(u32, f32, Vec<f32>), String>,
{
    let grid_cols = (crop_w + step - 1) / step;
    let grid_rows = (crop_h + step - 1) / step;
    let mut feature_grid: Vec<Vec<Option<Vec<f32>>>> = vec![vec![None; grid_rows]; grid_cols];
    let mut confidence_grid: Vec<Vec<f32>> = vec![vec![0.5f32; grid_rows]; grid_cols];

    let patch_size = 224usize;

    let mut filled = 0u32;
    for gy_idx in 0..grid_rows {
        for gx_idx in 0..grid_cols {
            let cx_start = gx_idx * step;
            let cy_start = gy_idx * step;
            let patch = extract_patch_at(crop_rgba, crop_gray, cx_start, cy_start, patch_size, crop_w, crop_h);

            let bbox_state = [
                (cx_start + step / 2) as f32 / crop_w as f32,
                (cy_start + step / 2) as f32 / crop_h as f32,
                step as f32 / crop_w as f32,
                step as f32 / crop_h as f32,
            ];

            if let Ok((_, conf, feat)) = model_fn(&patch, &bbox_state) {
                feature_grid[gx_idx][gy_idx] = Some(feat);
                confidence_grid[gx_idx][gy_idx] = conf;
                filled += 1;
            }
        }
    }
    eprintln!("Feature+Confidence grid computed: {}/{} cells filled", filled, grid_cols * grid_rows);
    (feature_grid, confidence_grid)
}

pub fn compute_feature_grid<F>(
    crop_rgba: &[Vec<(u8, u8, u8)>],
    crop_gray: &[Vec<f32>],
    crop_w: usize,
    crop_h: usize,
    step: usize,
    model_fn: F,
) -> Vec<Vec<Option<Vec<f32>>>>
where
    F: FnMut(&[f32], &[f32]) -> Result<(u32, f32, Vec<f32>), String>,
{
    let (feature_grid, _) = compute_feature_and_confidence_grids(
        crop_rgba, crop_gray, crop_w, crop_h, step, model_fn
    );
    feature_grid
}

fn extract_patch_at(
    crop_rgba: &[Vec<(u8, u8, u8)>],
    _crop_gray: &[Vec<f32>],
    px: usize,
    py: usize,
    patch_size: usize,
    max_w: usize,
    max_h: usize,
) -> Vec<f32> {
    let mut patch = vec![0.0f32; 3 * patch_size * patch_size];

    let src_size = step_min(max_w, max_h, 32);

    for dy in 0..patch_size {
        for dx in 0..patch_size {
            let src_x_idx = (dx * src_size / patch_size).min(src_size.saturating_sub(1));
            let src_y_idx = (dy * src_size / patch_size).min(src_size.saturating_sub(1));

            let abs_x = (px + src_x_idx).min(max_w.saturating_sub(1));
            let abs_y = (py + src_y_idx).min(max_h.saturating_sub(1));

            if abs_x < crop_rgba.len() && abs_y < crop_rgba[0].len() {
                let (r, g, b) = crop_rgba[abs_x][abs_y];
                let idx = (dy * patch_size + dx) * 3;
                patch[idx] = r as f32 / 255.0;
                patch[idx + 1] = g as f32 / 255.0;
                patch[idx + 2] = b as f32 / 255.0;
            }
        }
    }

    patch
}

fn step_min(a: usize, b: usize, min: usize) -> usize {
    a.min(b).max(min)
}

pub fn prepare_crop_data(
    img: &DynamicImage,
    bbox: &BoundingBox,
) -> (Vec<Vec<(u8, u8, u8)>>, Vec<Vec<f32>>, usize, usize) {
    let (w, h) = (img.width(), img.height());
    let base = img.to_rgba8();

    let x1 = bbox.x.min(w);
    let y1 = bbox.y.min(h);
    let x2 = (bbox.x + bbox.width).min(w);
    let y2 = (bbox.y + bbox.height).min(h);

    let crop_w = (x2 - x1) as usize;
    let crop_h = (y2 - y1) as usize;

    let mut crop_rgba: Vec<Vec<(u8, u8, u8)>> = vec![vec![(0u8, 0u8, 0u8); crop_h]; crop_w];
    let mut crop_gray: Vec<Vec<f32>> = vec![vec![0.0f32; crop_h]; crop_w];

    for cy in 0..crop_h {
        for cx in 0..crop_w {
            let gy = y1 + cy as u32;
            let gx = x1 + cx as u32;
            if gy < h && gx < w {
                let p = base.get_pixel(gx, gy);
                crop_rgba[cx][cy] = (p[0], p[1], p[2]);
                crop_gray[cx][cy] = (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0;
            }
        }
    }

    (crop_rgba, crop_gray, crop_w, crop_h)
}

pub fn generate_mask(img: &DynamicImage, bbox: &BoundingBox, config: &ProcessingConfig) -> Result<String, String> {
    if config.enable_rembg && !config.enable_traditional && !config.enable_rl_model {
        eprintln!("[rembg-only] 正在运行仅rembg模式");
        return generate_rembg_only_mask(img, bbox, config);
    }

    let (_, _, crop_w, crop_h) = prepare_crop_data(img, bbox);
    let step = ((crop_w.min(crop_h) as f32) / 24.0).max(4.0) as usize;
    let empty_grid: Vec<Vec<Option<Vec<f32>>>> = Vec::new();
    let empty_conf_grid: Vec<Vec<f32>> = Vec::new();
    let traditional_mask = generate_mask_with_feature(img, bbox, &[], 0.5, &empty_grid, &empty_conf_grid, step, config)?;

    if !config.enable_rembg {
        return Ok(traditional_mask);
    }

    combine_with_rembg(img, bbox, config, &traditional_mask)
}

fn generate_rembg_only_mask(
    img: &DynamicImage,
    bbox: &BoundingBox,
    config: &ProcessingConfig,
) -> Result<String, String> {
    use base64::Engine;
    use crate::rembg_processor::RembgProcessor;

    let mut processor = RembgProcessor::new();

    let model_file = match config.rembg_model.as_str() {
        "u2net" | "" => "u2net.onnx",
        "u2net_human_seg" => "u2net_human_seg.onnx",
        "silueta" => "silueta.onnx",
        _ => "u2net.onnx",
    };

    let model_path = crate::rembg_processor::models_dir().join(model_file);
    if !model_path.exists() {
        return Err(format!(
            "[rembg] 模型文件不存在: {}",
            model_path.display()
        ));
    }

    processor
        .ensure_model(model_file)
        .map_err(|e| format!("[rembg] 模型加载失败: {}", e))?;

    let rembg_bool = processor.mask_for_bbox(img, config, bbox)?;
    if rembg_bool.is_empty() {
        return Err("[rembg] 推理返回空结果".to_string());
    }

    let (img_w, img_h) = (img.width(), img.height());
    let x1 = bbox.x.min(img_w);
    let y1 = bbox.y.min(img_h);
    let x2 = (bbox.x + bbox.width).min(img_w);
    let y2 = (bbox.y + bbox.height).min(img_h);
    let crop_w = (x2 - x1) as usize;
    let crop_h = (y2 - y1) as usize;

    let mut result_img = RgbaImage::from_pixel(img_w, img_h, Rgba([0u8, 0u8, 0u8, 0u8]));
    let source_rgba = img.to_rgba8();

    for cy in 0..crop_h {
        for cx in 0..crop_w {
            let px = x1 + cx as u32;
            let py = y1 + cy as u32;
            if px >= img_w || py >= img_h {
                continue;
            }
            let idx = cy * crop_w + cx;
            if idx < rembg_bool.len() && rembg_bool[idx] {
                let s = source_rgba.get_pixel(px, py);
                result_img.put_pixel(px, py, Rgba([s[0], s[1], s[2], 255]));
            } else {
                result_img.put_pixel(px, py, Rgba([0u8, 0u8, 0u8, 0u8]));
            }
        }
    }

    let mut out: Vec<u8> = Vec::new();
    {
        let mut cursor = std::io::Cursor::new(&mut out);
        result_img
            .write_to(&mut cursor, image::ImageFormat::Png)
            .map_err(|e| e.to_string())?;
    }

    Ok(base64::engine::general_purpose::STANDARD.encode(&out))
}

pub fn generate_mask_with_rembg_first(
    img: &DynamicImage,
    bbox: &BoundingBox,
    config: &ProcessingConfig,
    rl_feature: &[f32],
    rl_conf: f32,
    feature_grid: &[Vec<Option<Vec<f32>>>],
    confidence_grid: &[Vec<f32>],
    step: usize,
) -> Result<String, String> {
    let traditional_mask = generate_mask_with_feature(
        img, bbox, rl_feature, rl_conf, feature_grid, confidence_grid, step, config,
    )?;

    if !config.enable_rembg {
        return Ok(traditional_mask);
    }

    combine_with_rembg(img, bbox, config, &traditional_mask)
}

fn combine_with_rembg(
    img: &DynamicImage,
    bbox: &BoundingBox,
    config: &ProcessingConfig,
    traditional_mask_b64: &str,
) -> Result<String, String> {
    use base64::Engine;
    use crate::rembg_processor::RembgProcessor;

    let mut processor = RembgProcessor::new();

    let model_name = match config.rembg_model.as_str() {
        "u2net" | "" => "u2net",
        "u2net_human_seg" => "u2net_human_seg",
        "silueta" => "silueta",
        other => other,
    };
    let model_file = match model_name {
        "u2net" => "u2net.onnx",
        "u2net_human_seg" => "u2net_human_seg.onnx",
        "silueta" => "silueta.onnx",
        _ => "u2net.onnx",
    };

    let model_path = crate::rembg_processor::models_dir().join(model_file);
    if !model_path.exists() {
        eprintln!("[rembg] 模型文件不存在，跳过 rembg 合成：{}", model_path.display());
        return Ok(traditional_mask_b64.to_string());
    }

    processor.ensure_model(model_file).map_err(|e| {
        eprintln!("[rembg] 模型加载失败: {}", e);
        e
    })?;

    let rembg_bool = match processor.mask_for_bbox(img, config, bbox) {
        Ok(m) if !m.is_empty() => {
            eprintln!("[rembg] mask_for_bbox 成功, 布尔mask长度={}", m.len());
            m
        }
        Ok(_) => {
            eprintln!("[rembg] 空结果，保留传统 mask");
            return Ok(traditional_mask_b64.to_string());
        }
        Err(e) => {
            eprintln!("[rembg] 推理失败: {}，保留传统 mask", e);
            return Ok(traditional_mask_b64.to_string());
        }
    };

    let (img_w, img_h) = (img.width(), img.height());
    let x1 = bbox.x.min(img_w);
    let y1 = bbox.y.min(img_h);
    let x2 = (bbox.x + bbox.width).min(img_w);
    let y2 = (bbox.y + bbox.height).min(img_h);
    let crop_w = (x2 - x1) as usize;
    let crop_h = (y2 - y1) as usize;

    if crop_w == 0 || crop_h == 0 {
        eprintln!("[combine] crop 无效, 保留传统 mask");
        return Ok(traditional_mask_b64.to_string());
    }

    eprintln!(
        "[combine] 解码传统mask (crop={}x{}, rembg_bool_len={})",
        crop_w, crop_h, rembg_bool.len()
    );
    let base_bytes = base64::engine::general_purpose::STANDARD
        .decode(traditional_mask_b64)
        .map_err(|e| e.to_string())?;
    let base_mask = image::load_from_memory(&base_bytes).map_err(|e| e.to_string())?;
    let base_rgba = base_mask.to_rgba8();
    eprintln!(
        "[combine] 传统mask已解码 ({}x{})",
        base_rgba.width(),
        base_rgba.height()
    );

    let mut combined = RgbaImage::from_pixel(img_w, img_h, Rgba([0u8, 0u8, 0u8, 0u8]));
    let source_rgba = img.to_rgba8();

    let rembg_len = rembg_bool.len();
    let rembg_h = if crop_w > 0 && rembg_len >= crop_w {
        rembg_len / crop_w
    } else {
        0
    };

    let start = std::time::Instant::now();
    eprintln!(
        "[combine] 开始合成像素 (crop={}x{}, rembg_bool_len={}, rembg_h={})",
        crop_w, crop_h, rembg_len, rembg_h
    );
    let mut fg_in_rembg = 0usize;
    let mut fg_in_both = 0usize;
    for cy in 0..crop_h {
        let row_offset = cy * crop_w;
        for cx in 0..crop_w {
            let px = x1 + cx as u32;
            let py = y1 + cy as u32;
            if px >= img_w || py >= img_h {
                continue;
            }

            let trad_pixel = base_rgba.get_pixel(px, py);
            let trad_alpha = trad_pixel[3];

            let rembg_idx = row_offset + cx;
            let rembg_fg = rembg_idx < rembg_len && rembg_bool[rembg_idx];

            let final_pixel: Rgba<u8> = if rembg_fg && trad_alpha > 0 {
                fg_in_both += 1;
                *trad_pixel
            } else if rembg_fg {
                fg_in_rembg += 1;
                let source_pixel = source_rgba.get_pixel(px, py);
                Rgba([source_pixel[0], source_pixel[1], source_pixel[2], 255])
            } else {
                Rgba([0u8, 0u8, 0u8, 0u8])
            };

            combined.put_pixel(px, py, final_pixel);
        }
    }
    eprintln!(
        "[combine] 像素合成完成 ({:.2}s), fg_in_both={}, fg_only_rembg={}, 开始编码PNG",
        start.elapsed().as_secs_f64(),
        fg_in_both,
        fg_in_rembg
    );

    let enc_start = std::time::Instant::now();
    let mut out: Vec<u8> = Vec::new();
    {
        let mut cursor = std::io::Cursor::new(&mut out);
        combined.write_to(&mut cursor, image::ImageFormat::Png).map_err(|e| e.to_string())?;
    }
    eprintln!(
        "[combine] PNG编码完成: {} 字节 ({:.2}s)",
        out.len(),
        enc_start.elapsed().as_secs_f64()
    );

    Ok(base64::engine::general_purpose::STANDARD.encode(&out))
}
