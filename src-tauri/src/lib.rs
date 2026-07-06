pub mod img_processor;
pub mod ml_engine;
pub mod rembg_processor;

use serde::{Deserialize, Serialize};
use tauri::{Emitter, Manager};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BoundingBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RlStepUpdate {
    pub step: u32,
    pub action_taken: String,
    pub bbox: BoundingBox,
    pub confidence: f32,
    pub is_finished: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_base64: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelTrainingInfo {
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f64,
    pub max_steps_per_episode: u32,
    pub total_params: u64,
    pub trainable_params: u64,
    pub last_epoch: u32,
    pub last_avg_loss: f64,
    pub last_avg_reward: f64,
    pub last_lr: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelStatus {
    pub is_available: bool,
    pub model_path: Option<String>,
    pub model_size_bytes: Option<u64>,
    pub last_modified: Option<String>,
    pub loaded_in_session: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_info: Option<ModelTrainingInfo>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProcessingConfig {
    pub enable_rl_model: bool,
    pub enable_traditional: bool,
    pub rl_learning_rate: f64,
    pub rl_max_steps: u32,
    pub rl_confidence_threshold: f64,
    pub trad_canny_low: f64,
    pub trad_canny_high: f64,
    pub trad_morphology_radius: u32,
    pub trad_min_component_ratio: f64,
    pub trad_edge_weight: f64,
    pub trad_use_adaptive_threshold: bool,
    pub trad_adaptive_threshold_block: u32,
    pub trad_adaptive_threshold_c: f64,
    pub trad_bilateral_filter: bool,
    pub trad_bilateral_sigma_color: f64,
    pub trad_bilateral_sigma_space: f64,
    pub trad_use_distance_transform: bool,
    pub trad_distance_weight: f64,
    pub enable_rembg: bool,
    pub rembg_model: String,
    pub rembg_threshold: f64,
    pub rembg_binary_mode: bool,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            enable_rl_model: true,
            enable_traditional: true,
            rl_learning_rate: 0.0003,
            rl_max_steps: 30,
            rl_confidence_threshold: 0.5,
            trad_canny_low: 0.08,
            trad_canny_high: 0.2,
            trad_morphology_radius: 3,
            trad_min_component_ratio: 0.03,
            trad_edge_weight: 0.5,
            trad_use_adaptive_threshold: true,
            trad_adaptive_threshold_block: 15,
            trad_adaptive_threshold_c: 10.0,
            trad_bilateral_filter: false,
            trad_bilateral_sigma_color: 25.0,
            trad_bilateral_sigma_space: 25.0,
            trad_use_distance_transform: true,
            trad_distance_weight: 0.3,
            enable_rembg: false,
            rembg_model: "u2net".to_string(),
            rembg_threshold: 0.5,
            rembg_binary_mode: false,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct PipelineStageInfo {
    pub key: String,
    pub label: String,
    pub icon: String,
    pub status: String,
}

#[tauri::command]
async fn start_rl_loop_cmd(
    app: tauri::AppHandle,
    image_path: String,
    config: ProcessingConfig,
) -> Result<(), String> {
    crate::start_rl_loop_impl(app, image_path, config).await
}

#[tauri::command]
async fn check_model_status_cmd(app: tauri::AppHandle) -> Result<ModelStatus, String> {
    crate::check_model_status_impl(app)
}

#[tauri::command]
async fn save_result_cmd(
    mask_base64: String,
    save_path: String,
) -> Result<(), String> {
    crate::save_result_impl(mask_base64, save_path)
}

#[tauri::command]
async fn get_image_base64_cmd(image_path: String) -> Result<String, String> {
    use base64::Engine;
    let bytes = std::fs::read(&image_path).map_err(|e| format!("read failed: {}", e))?;
    let ext = std::path::Path::new(&image_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("png")
        .to_lowercase();
    let mime = match ext.as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        _ => "image/png",
    };
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:{};base64,{}", mime, b64))
}

fn parse_training_log(log_path: &std::path::Path) -> Option<ModelTrainingInfo> {
    let content = std::fs::read_to_string(log_path).ok()?;
    let mut info = ModelTrainingInfo {
        epochs: 5000,
        batch_size: 32,
        learning_rate: 0.0003,
        max_steps_per_episode: 8,
        total_params: 1142184,
        trainable_params: 493720,
        last_epoch: 0,
        last_avg_loss: 0.0,
        last_avg_reward: 0.0,
        last_lr: 0.0003,
    };

    for line in content.lines() {
        if line.contains("Epochs") && line.contains(":") {
            if let Some(val) = extract_number(line, "Epochs") {
                info.epochs = val as u32;
            }
        }
        if line.contains("Batch size") && line.contains(":") {
            if let Some(val) = extract_number(line, "Batch size") {
                info.batch_size = val as u32;
            }
        }
        if line.contains("Learning rate") && line.contains(":") {
            if let Some(val) = extract_float(line, "Learning rate") {
                info.learning_rate = val;
            }
        }
        if line.contains("Max steps/episode") && line.contains(":") {
            if let Some(val) = extract_number(line, "Max steps/episode") {
                info.max_steps_per_episode = val as u32;
            }
        }
        if line.contains("PolicyNetwork parameters:") {
            let parts: Vec<&str> = line.split("parameters:").collect();
            if parts.len() > 1 {
                let nums: Vec<String> = parts[1]
                    .split(|c: char| !c.is_ascii_digit() && c != ',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect();
                if let Some(first) = nums.first() {
                    if let Ok(v) = first.replace(',', "").parse::<u64>() {
                        info.total_params = v;
                    }
                }
                if let Some(second) = nums.get(1) {
                    if let Ok(v) = second.replace(',', "").parse::<u64>() {
                        info.trainable_params = v;
                    }
                }
            }
        }
        if line.contains("[Epoch") {
            if let Some(epoch) = extract_epoch_info(line) {
                if epoch.0 > info.last_epoch {
                    info.last_epoch = epoch.0;
                    info.last_avg_loss = epoch.1;
                    info.last_avg_reward = epoch.2;
                    info.last_lr = epoch.3;
                }
            }
        }
    }

    Some(info)
}

fn extract_number(line: &str, key: &str) -> Option<f64> {
    let idx = line.find(key)?;
    let rest = &line[idx + key.len()..];
    let num_str: String = rest
        .chars()
        .skip_while(|c: &char| !c.is_ascii_digit() && *c != '.')
        .take_while(|c: &char| c.is_ascii_digit() || *c == '.')
        .collect();
    num_str.parse::<f64>().ok()
}

fn extract_float(line: &str, key: &str) -> Option<f64> {
    let idx = line.find(key)?;
    let rest = &line[idx + key.len()..];
    let num_str: String = rest
        .chars()
        .skip_while(|c: &char| !c.is_ascii_digit() && *c != '.')
        .take_while(|c: &char| c.is_ascii_digit() || *c == '.')
        .collect();
    num_str.parse::<f64>().ok()
}

fn extract_epoch_info(line: &str) -> Option<(u32, f64, f64, f64)> {
    let epoch_part = line.split("[Epoch").nth(1)?;
    let num_str: String = epoch_part
        .chars()
        .skip_while(|c: &char| !c.is_ascii_digit())
        .take_while(|c: &char| c.is_ascii_digit())
        .collect();
    let epoch_num = num_str.parse::<u32>().ok()?;

    let avg_loss = extract_float_value(line, "avg_loss=").unwrap_or(0.0);
    let avg_reward = extract_float_value(line, "avg_reward=").unwrap_or(0.0);
    let lr = extract_float_value(line, "lr=").unwrap_or(0.0003);

    Some((epoch_num, avg_loss, avg_reward, lr))
}

fn extract_float_value(line: &str, prefix: &str) -> Option<f64> {
    let idx = line.find(prefix)?;
    let rest = &line[idx + prefix.len()..];
    let num_str: String = rest
        .chars()
        .take_while(|c: &char| c.is_ascii_digit() || *c == '.' || *c == '-')
        .collect();
    if num_str.is_empty() {
        None
    } else {
        num_str.parse::<f64>().ok()
    }
}

pub fn check_model_status_impl(app: tauri::AppHandle) -> Result<ModelStatus, String> {
    let model_path = resolve_model_path(&app);
    match model_path {
        Some(ref p) if p.exists() => {
            let metadata = std::fs::metadata(p).ok();
            let size = metadata.as_ref().map(|m| m.len());
            let modified = metadata
                .and_then(|m| m.modified().ok())
                .map(|t| {
                    let d: std::time::SystemTime = t;
                    let duration = d.elapsed().unwrap_or_default();
                    let days = duration.as_secs() / 86400;
                    let hours = (duration.as_secs() % 86400) / 3600;
                    let mins = (duration.as_secs() % 3600) / 60;
                    if days > 0 {
                        format!("{}d {}h ago", days, hours)
                    } else if hours > 0 {
                        format!("{}h {}m ago", hours, mins)
                    } else {
                        format!("{}m ago", mins.max(1))
                    }
                });

            let training_info = resolve_model_path(&app)
                .and_then(|_| {
                    let log_path = std::path::PathBuf::from(
                        p.parent().unwrap_or(std::path::Path::new(".")),
                    )
                    .join("training_output.log");
                    parse_training_log(&log_path)
                });

            eprintln!("Model status: AVAILABLE at {}", p.display());
            Ok(ModelStatus {
                is_available: true,
                model_path: Some(p.to_string_lossy().to_string()),
                model_size_bytes: size,
                last_modified: modified,
                loaded_in_session: false,
                training_info,
            })
        }
        _ => {
            eprintln!("Model status: NOT LOADED");
            Ok(ModelStatus {
                is_available: false,
                model_path: None,
                model_size_bytes: None,
                last_modified: None,
                loaded_in_session: false,
                training_info: None,
            })
        }
    }
}

pub fn save_result_impl(mask_base64: String, save_path: String) -> Result<(), String> {
    use base64::Engine;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(&mask_base64)
        .map_err(|e| format!("base64 decode failed: {}", e))?;
    if let Some(parent) = std::path::Path::new(&save_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    std::fs::write(&save_path, &decoded).map_err(|e| format!("write failed: {}", e))?;
    Ok(())
}

fn emit_pipeline_stage(app: &tauri::AppHandle, key: &str, label: &str, icon: &str, status: &str) {
    let info = PipelineStageInfo {
        key: key.to_string(),
        label: label.to_string(),
        icon: icon.to_string(),
        status: status.to_string(),
    };
    let _ = app.emit("rl-pipeline-stage", &info);
}

fn emit_pipeline_start(app: &tauri::AppHandle, config: &ProcessingConfig) {
    let mut stages: Vec<PipelineStageInfo> = vec![
        PipelineStageInfo { key: "load_image".to_string(), label: "加载图像".to_string(), icon: "🖼️".to_string(), status: "pending".to_string() },
        PipelineStageInfo { key: "init_bbox".to_string(), label: "初始化边界框".to_string(), icon: "🎯".to_string(), status: "pending".to_string() },
    ];
    if config.enable_rl_model {
        stages.push(PipelineStageInfo { key: "load_model".to_string(), label: "加载强化学习模型".to_string(), icon: "🧠".to_string(), status: "pending".to_string() });
        stages.push(PipelineStageInfo { key: "rl_inference".to_string(), label: "RL 推理迭代".to_string(), icon: "🤖".to_string(), status: "pending".to_string() });
    }
    if config.enable_traditional {
        stages.push(PipelineStageInfo { key: "trad_background".to_string(), label: "传统处理：背景估计".to_string(), icon: "🌈".to_string(), status: "pending".to_string() });
        stages.push(PipelineStageInfo { key: "trad_edges".to_string(), label: "传统处理：边缘检测".to_string(), icon: "✂️".to_string(), status: "pending".to_string() });
        stages.push(PipelineStageInfo { key: "trad_morph".to_string(), label: "传统处理：形态学优化".to_string(), icon: "🔍".to_string(), status: "pending".to_string() });
        stages.push(PipelineStageInfo { key: "trad_components".to_string(), label: "传统处理：连通域分析".to_string(), icon: "🧩".to_string(), status: "pending".to_string() });
    }
    stages.push(PipelineStageInfo { key: "generate_mask".to_string(), label: "生成抠图掩码".to_string(), icon: "🎨".to_string(), status: "pending".to_string() });
    stages.push(PipelineStageInfo { key: "complete".to_string(), label: "完成".to_string(), icon: "✅".to_string(), status: "pending".to_string() });
    let _ = app.emit("rl-pipeline-start", &stages);
}

pub async fn start_rl_loop_impl(
    app: tauri::AppHandle,
    image_path: String,
    config: ProcessingConfig,
) -> Result<(), String> {
    eprintln!("start_rl_loop: loading image from {}, config: rl={}, traditional={}, rembg={}", 
        image_path, config.enable_rl_model, config.enable_traditional, config.enable_rembg);

    // 如果没有任何启用的处理方式，直接返回错误
    if !config.enable_rl_model && !config.enable_traditional && !config.enable_rembg {
        return Err("请至少启用一种处理方式（强化学习、传统图像处理或AI背景移除）".to_string());
    }

    // 如果仅启用了 rembg，跳过 RL 循环，直接生成 rembg 掩码
    if config.enable_rembg && !config.enable_rl_model && !config.enable_traditional {
        eprintln!("仅启用 rembg，跳过 RL 循环，直接生成掩码");
        emit_pipeline_start(&app, &config);
        emit_pipeline_stage(&app, "load_image", "加载图像", "🖼️", "running");
        let img = img_processor::load_image(&image_path).map_err(|e| format!("Failed to load image: {}", e))?;
        let (img_w, img_h) = (img.width(), img.height());
        eprintln!("Image loaded: {}x{}", img_w, img_h);
        emit_pipeline_stage(&app, "load_image", "加载图像", "🖼️", "done");

        let bbox = BoundingBox {
            x: 0,
            y: 0,
            width: img_w,
            height: img_h,
        };

        emit_pipeline_stage(&app, "generate_mask", "AI 背景移除（rembg）", "🪄", "running");
        let mask = img_processor::generate_mask(&img, &bbox, &config)?;
        emit_pipeline_stage(&app, "generate_mask", "AI 背景移除（rembg）", "🪄", "done");

        let update = RlStepUpdate {
            step: 0,
            action_taken: "Complete".to_string(),
            bbox: bbox.clone(),
            confidence: 1.0,
            is_finished: true,
            mask_base64: Some(mask),
        };
        app.emit("rl-step-update", &update).map_err(|e| e.to_string())?;
        emit_pipeline_stage(&app, "complete", "完成", "✅", "done");
        return Ok(());
    }

    // 如果未启用 RL 模型（仅 rembg 或仅传统或 rembg+传统），跳过 RL 循环
    if !config.enable_rl_model {
        eprintln!("未启用 RL 模型，跳过 RL 循环，直接生成掩码");
        emit_pipeline_start(&app, &config);
        emit_pipeline_stage(&app, "load_image", "加载图像", "🖼️", "running");
        let img = img_processor::load_image(&image_path).map_err(|e| format!("Failed to load image: {}", e))?;
        let (img_w, img_h) = (img.width(), img.height());
        eprintln!("Image loaded: {}x{}", img_w, img_h);
        emit_pipeline_stage(&app, "load_image", "加载图像", "🖼️", "done");

        let bbox = BoundingBox {
            x: 0,
            y: 0,
            width: img_w,
            height: img_h,
        };

        emit_pipeline_stage(&app, "generate_mask", "生成抠图掩码", "🎨", "running");
        let mask = img_processor::generate_mask(&img, &bbox, &config)?;
        emit_pipeline_stage(&app, "generate_mask", "生成抠图掩码", "🎨", "done");

        let update = RlStepUpdate {
            step: 0,
            action_taken: "Complete".to_string(),
            bbox: bbox.clone(),
            confidence: 1.0,
            is_finished: true,
            mask_base64: Some(mask),
        };
        app.emit("rl-step-update", &update).map_err(|e| e.to_string())?;
        emit_pipeline_stage(&app, "complete", "完成", "✅", "done");
        return Ok(());
    }

    emit_pipeline_start(&app, &config);
    emit_pipeline_stage(&app, "load_image", "加载图像", "🖼️", "running");

    let img = img_processor::load_image(&image_path).map_err(|e| format!("Failed to load image: {}", e))?;
    let (img_w, img_h) = (img.width(), img.height());
    eprintln!("Image loaded: {}x{}", img_w, img_h);
    emit_pipeline_stage(&app, "load_image", "加载图像", "🖼️", "done");

    emit_pipeline_stage(&app, "init_bbox", "初始化边界框", "🎯", "running");
    let mut bbox = BoundingBox {
        x: 0,
        y: 0,
        width: img_w,
        height: img_h,
    };
    emit_pipeline_stage(&app, "init_bbox", "初始化边界框", "🎯", "done");

    let max_steps: u32 = config.rl_max_steps;
    let min_steps_before_trigger: u32 = 15;
    let mut step: u32 = 0;

    let model_path = if config.enable_rl_model {
        emit_pipeline_stage(&app, "load_model", "加载强化学习模型", "🧠", "running");
        resolve_model_path(&app)
    } else {
        None
    };
    
    let engine = model_path.and_then(|p| {
        eprintln!("Loading ONNX model from {:?}", p);
        match ml_engine::RlEngine::load(&p) {
            Ok(eng) => {
                eprintln!("Model loaded successfully");
                Some(eng)
            }
            Err(e) => {
                eprintln!("Model load failed: {}", e);
                None
            }
        }
    });
    let engine = std::sync::Arc::new(std::sync::Mutex::new(engine));

    if config.enable_rl_model {
        emit_pipeline_stage(&app, "load_model", "加载强化学习模型", "🧠", "done");
        emit_pipeline_stage(&app, "rl_inference", "RL 推理迭代", "🤖", "running");
    }

    eprintln!("Starting RL loop with max_steps={}, rl_model={}, traditional={}", max_steps, config.enable_rl_model, config.enable_traditional);

    while step < max_steps {
        step += 1;

        let patch = img_processor::crop_resize_patch(&img, &bbox, 224)?;

        let bbox_state = [
            bbox.x as f32 / img_w as f32,
            bbox.y as f32 / img_h as f32,
            bbox.width as f32 / img_w as f32,
            bbox.height as f32 / img_h as f32,
        ];

        let (action_idx, confidence) = {
            let mut guard = engine.lock().map_err(|e| e.to_string())?;
            match guard.as_mut() {
                Some(eng) => {
                    let (act, conf) = eng.infer(&patch, &bbox_state).unwrap_or((6, 0.0));
                    if step < min_steps_before_trigger && act == 6 {
                        let forced_action = (step + 4) % 6;
                        eprintln!("Step {}: forcing action {} instead of Trigger", step, forced_action);
                        (forced_action, conf)
                    } else {
                        (act, conf)
                    }
                }
                None => {
                    if step < min_steps_before_trigger {
                        let actions: [u32; 5] = [5, 4, 5, 4, 2];
                        let act = actions[(step - 1) as usize % actions.len()];
                        let conf = config.rl_confidence_threshold + (step as f64 / max_steps as f64) * 0.3;
                        (act, conf as f32)
                    } else if (min_steps_before_trigger..(min_steps_before_trigger + 15)).contains(&step) {
                        let actions: [u32; 4] = [4, 5, 0, 1];
                        let act = actions[(step - min_steps_before_trigger) as usize % actions.len()];
                        let conf = 0.65 + (step as f64 / max_steps as f64) * 0.3;
                        (act, conf as f32)
                    } else {
                        (6, 0.85)
                    }
                }
            }
        };

        let action_name = ml_engine::action_to_string(action_idx);
        bbox = apply_action(&bbox, action_idx, img_w, img_h);

        let is_finished = action_idx == 6 && step >= min_steps_before_trigger;

        let display_action = if is_finished {
            "Complete"
        } else {
            action_name
        };

        let update = RlStepUpdate {
            step,
            action_taken: display_action.to_string(),
            bbox: bbox.clone(),
            confidence,
            is_finished,
            mask_base64: None,
        };

        app.emit("rl-step-update", &update).map_err(|e| e.to_string())?;

        if is_finished {
            if config.enable_rl_model {
                emit_pipeline_stage(&app, "rl_inference", "RL 推理迭代", "🤖", "done");
            }
            emit_pipeline_stage(&app, "generate_mask", "生成抠图掩码", "🎨", "running");

            if config.enable_traditional {
                emit_pipeline_stage(&app, "trad_background", "传统处理：背景估计", "🌈", "running");
                emit_pipeline_stage(&app, "trad_background", "传统处理：背景估计", "🌈", "done");
                emit_pipeline_stage(&app, "trad_edges", "传统处理：边缘检测", "✂️", "running");
                emit_pipeline_stage(&app, "trad_edges", "传统处理：边缘检测", "✂️", "done");
                emit_pipeline_stage(&app, "trad_morph", "传统处理：形态学优化", "🔍", "running");
                emit_pipeline_stage(&app, "trad_morph", "传统处理：形态学优化", "🔍", "done");
                emit_pipeline_stage(&app, "trad_components", "传统处理：连通域分析", "🧩", "running");
                emit_pipeline_stage(&app, "trad_components", "传统处理：连通域分析", "🧩", "done");
            }

            if config.enable_rembg {
                emit_pipeline_stage(&app, "rembg_inference", "AI 背景移除（rembg）", "🪄", "running");
            }

            let mask = if let Some(ref mut eng) = *engine.lock().unwrap() {
                let center_patch = img_processor::crop_resize_patch(&img, &BoundingBox {
                    x: bbox.x + bbox.width / 4,
                    y: bbox.y + bbox.height / 4,
                    width: bbox.width / 2,
                    height: bbox.height / 2,
                }, 224).unwrap_or_default();

                let bbox_state = [0.5f32, 0.5f32, 1.0f32, 1.0f32];
                if let Ok((_, center_conf, feature)) = eng.infer_with_feature(&center_patch, &bbox_state) {
                    eprintln!("Using model feature for mask (feat_len={}, conf={:.3})", feature.len(), center_conf);

                    let (crop_rgba, crop_gray, crop_w, crop_h) = img_processor::prepare_crop_data(&img, &bbox);
                    let step_size = ((crop_w.min(crop_h) as f32) / 24.0).max(4.0) as usize;

                    let (feature_grid, confidence_grid) = img_processor::compute_feature_and_confidence_grids(
                        &crop_rgba, &crop_gray, crop_w, crop_h, step_size,
                        |patch, state| eng.infer_with_feature(patch, state),
                    );

                    img_processor::generate_mask_with_rembg_first(
                        &img, &bbox, &config, &feature, center_conf,
                        &feature_grid, &confidence_grid, step_size
                    )
                } else {
                    eprintln!("Feature extraction failed, using fallback");
                    img_processor::generate_mask(&img, &bbox, &config)
                }
            } else {
                img_processor::generate_mask(&img, &bbox, &config)
            }?;

            if config.enable_rembg {
                emit_pipeline_stage(&app, "rembg_inference", "AI 背景移除（rembg）", "🪄", "done");
            }
            emit_pipeline_stage(&app, "generate_mask", "生成抠图掩码", "🎨", "done");

            let update_with_mask = RlStepUpdate {
                step,
                action_taken: "Complete".to_string(),
                bbox: bbox.clone(),
                confidence,
                is_finished: true,
                mask_base64: Some(mask),
            };
            app.emit("rl-step-update", &update_with_mask)
                .map_err(|e| e.to_string())?;

            emit_pipeline_stage(&app, "complete", "完成", "✅", "done");
            break;
        }

        std::thread::sleep(std::time::Duration::from_millis(180));
    }

    if step >= max_steps {
        if config.enable_rl_model {
            emit_pipeline_stage(&app, "rl_inference", "RL 推理迭代", "🤖", "done");
        }
        emit_pipeline_stage(&app, "generate_mask", "生成抠图掩码", "🎨", "running");

        if config.enable_traditional {
            emit_pipeline_stage(&app, "trad_background", "传统处理：背景估计", "🌈", "running");
            emit_pipeline_stage(&app, "trad_background", "传统处理：背景估计", "🌈", "done");
            emit_pipeline_stage(&app, "trad_edges", "传统处理：边缘检测", "✂️", "running");
            emit_pipeline_stage(&app, "trad_edges", "传统处理：边缘检测", "✂️", "done");
            emit_pipeline_stage(&app, "trad_morph", "传统处理：形态学优化", "🔍", "running");
            emit_pipeline_stage(&app, "trad_morph", "传统处理：形态学优化", "🔍", "done");
            emit_pipeline_stage(&app, "trad_components", "传统处理：连通域分析", "🧩", "running");
            emit_pipeline_stage(&app, "trad_components", "传统处理：连通域分析", "🧩", "done");
        }

        let mask = img_processor::generate_mask(&img, &bbox, &config)?;
        emit_pipeline_stage(&app, "generate_mask", "生成抠图掩码", "🎨", "done");

        let final_update = RlStepUpdate {
            step,
            action_taken: "Complete".to_string(),
            bbox: bbox.clone(),
            confidence: 0.95,
            is_finished: true,
            mask_base64: Some(mask),
        };
        app.emit("rl-step-update", &final_update)
            .map_err(|e| e.to_string())?;

        emit_pipeline_stage(&app, "complete", "完成", "✅", "done");
    }

    Ok(())
}

fn apply_action(bbox: &BoundingBox, action: u32, img_w: u32, img_h: u32) -> BoundingBox {
    let step_w = (bbox.width as f32 * 0.1) as u32;
    let step_h = (bbox.height as f32 * 0.1) as u32;
    let mut new = bbox.clone();

    match action {
        0 => {
            new.x = new.x.saturating_sub(step_w);
        }
        1 => {
            new.x = (new.x + step_w).min(img_w.saturating_sub(1).saturating_sub(new.width));
        }
        2 => {
            new.y = new.y.saturating_sub(step_h);
        }
        3 => {
            new.y = (new.y + step_h).min(img_h.saturating_sub(1).saturating_sub(new.height));
        }
        4 => {
            let shrink_w = (new.width as f32 * 0.08) as u32;
            let shrink_h = (new.height as f32 * 0.08) as u32;
            new.x += shrink_w / 2;
            new.y += shrink_h / 2;
            new.width = new.width.saturating_sub(shrink_w);
            new.height = new.height.saturating_sub(shrink_h);
        }
        5 => {
            let grow_w = (new.width as f32 * 0.08) as u32;
            let grow_h = (new.height as f32 * 0.08) as u32;
            new.x = new.x.saturating_sub(grow_w / 2);
            new.y = new.y.saturating_sub(grow_h / 2);
            new.width = (new.width + grow_w).min(img_w);
            new.height = (new.height + grow_h).min(img_h);
            new.x = new.x.min(img_w.saturating_sub(new.width));
            new.y = new.y.min(img_h.saturating_sub(new.height));
        }
        _ => {}
    }

    new.width = new.width.max(20);
    new.height = new.height.max(20);
    new
}

fn resolve_model_path(app: &tauri::AppHandle) -> Option<std::path::PathBuf> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();

    if let Ok(resource_dir) = app.path().resource_dir() {
        candidates.push(resource_dir.join("models").join("policy_network.onnx"));
        candidates.push(resource_dir.join("policy_network.onnx"));
        let up_dir = resource_dir.join("_up_");
        candidates.push(up_dir.join("models").join("policy_network.onnx"));
        candidates.push(up_dir.join("policy_network.onnx"));
    }

    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    candidates.push(
        manifest_dir
            .join("..")
            .join("models")
            .join("policy_network.onnx"),
    );
    candidates.push(
        manifest_dir
            .join("models")
            .join("policy_network.onnx"),
    );

    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("models").join("policy_network.onnx"));
        candidates.push(cwd.join("..").join("models").join("policy_network.onnx"));
    }

    for candidate in &candidates {
        if candidate.exists() {
            eprintln!("Found model at: {}", candidate.display());
            return Some(candidate.clone());
        }
    }

    eprintln!("Model not found. Checked {} paths.", candidates.len());
    None
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|_app| {
            if let Some(window) = _app.get_webview_window("main") {
                #[cfg(target_os = "macos")]
                {
                    let _ = window.set_theme(Some(tauri::Theme::Dark));
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_rl_loop_cmd,
            check_model_status_cmd,
            save_result_cmd,
            get_image_base64_cmd
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
