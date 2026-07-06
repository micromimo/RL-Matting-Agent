use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::path::Path;

const ACTION_NAMES: [&str; 7] = [
    "Move Left",
    "Move Right",
    "Move Up",
    "Move Down",
    "Zoom In",
    "Zoom Out",
    "Trigger",
];

pub fn action_to_string(action: u32) -> &'static str {
    ACTION_NAMES[action as usize % ACTION_NAMES.len()]
}

pub struct RlEngine {
    policy_session: Session,
}

fn to_err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

impl RlEngine {
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self, String> {
        let model_path = model_path.as_ref();
        if !model_path.exists() {
            return Err(format!("model not found at {}", model_path.display()));
        }

        let _env = ort::init()
            .with_name("rl-matting")
            .commit();

        let policy_session = Session::builder()
            .map_err(to_err)?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(to_err)?
            .with_intra_threads(4)
            .map_err(to_err)?
            .commit_from_file(model_path)
            .map_err(to_err)?;

        Ok(Self { policy_session })
    }

    pub fn infer(&mut self, patch: &[f32], bbox_state: &[f32]) -> Result<(u32, f32), String> {
        use ort::session::SessionInputValue;
        use ort::value::Tensor;

        let patch_array = ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, h, w)| {
            let idx = c * 224 * 224 + h * 224 + w;
            patch[idx % patch.len()]
        });

        let bbox_array = ndarray::Array2::from_shape_vec((1, 4), bbox_state.to_vec())
            .map_err(|e| to_err(e))?;

        let patch_tensor = Tensor::from_array(patch_array).map_err(to_err)?;
        let bbox_tensor = Tensor::from_array(bbox_array).map_err(to_err)?;

        let patch_input: SessionInputValue<'_> = patch_tensor.into();
        let bbox_input: SessionInputValue<'_> = bbox_tensor.into();

        let outputs = self
            .policy_session
            .run([patch_input, bbox_input])
            .map_err(to_err)?;

        let (_, logits_slice): (_, &[f32]) = outputs[0].try_extract_tensor::<f32>().map_err(to_err)?;
        let (_, conf_slice): (_, &[f32]) = outputs[1].try_extract_tensor::<f32>().map_err(to_err)?;

        let logits: Vec<f32> = logits_slice.to_vec();
        let confidence = conf_slice.first().copied().unwrap_or(0.5).clamp(0.0, 1.0);

        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(6);

        Ok((argmax, confidence))
    }

    pub fn infer_with_feature(&mut self, patch: &[f32], bbox_state: &[f32]) -> Result<(u32, f32, Vec<f32>), String> {
        use ort::session::SessionInputValue;
        use ort::value::Tensor;

        let patch_array = ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, h, w)| {
            let idx = c * 224 * 224 + h * 224 + w;
            patch[idx % patch.len()]
        });

        let bbox_array = ndarray::Array2::from_shape_vec((1, 4), bbox_state.to_vec())
            .map_err(|e| to_err(e))?;

        let patch_tensor = Tensor::from_array(patch_array).map_err(to_err)?;
        let bbox_tensor = Tensor::from_array(bbox_array).map_err(to_err)?;

        let patch_input: SessionInputValue<'_> = patch_tensor.into();
        let bbox_input: SessionInputValue<'_> = bbox_tensor.into();

        let outputs = self
            .policy_session
            .run([patch_input, bbox_input])
            .map_err(to_err)?;

        let (_, logits_slice): (_, &[f32]) = outputs[0].try_extract_tensor::<f32>().map_err(to_err)?;
        let (_, conf_slice): (_, &[f32]) = outputs[1].try_extract_tensor::<f32>().map_err(to_err)?;
        let (_, feature_slice): (_, &[f32]) = outputs[2].try_extract_tensor::<f32>().map_err(to_err)?;

        let logits: Vec<f32> = logits_slice.to_vec();
        let confidence = conf_slice.first().copied().unwrap_or(0.5).clamp(0.0, 1.0);
        let feature: Vec<f32> = feature_slice.to_vec();

        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(6);

        Ok((argmax, confidence, feature))
    }
}
