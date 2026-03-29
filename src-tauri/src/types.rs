use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionResponse {
    pub output_path: String,
    #[serde(default)]
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveDevicesResponse {
    pub devices: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveSessionResponse {
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveChunkResponse {
    pub chunk: Vec<f32>,
}
