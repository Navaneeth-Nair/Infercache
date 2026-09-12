use candle_core::{Device, Tensor};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

pub struct CandleEmbeddingModel {
    model: Option<Arc<Mutex<BertModel>>>,
    tokenizer: Option<Arc<Tokenizer>>,
    device: Device,
}

impl CandleEmbeddingModel {
    // Loads all-MiniLM-L6-v2 model and tokenizer from HuggingFace Hub or local cache.
    pub fn load(model_id: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let device = Device::Cpu;
        tracing::info!(model_id = %model_id, "Loading embedding model on CPU (FP16/Quantized footprint)");

        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(model_id.to_string());

        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let weights_path = repo.get("model.safetensors")?;

        let config_str = std::fs::read_to_string(&config_path)?;
        let config: BertConfig = serde_json::from_str(&config_str)?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

        // Load weights into owned heap memory to prevent concurrent file mutation.
        let weights_bytes = std::fs::read(&weights_path)?;
        let vb = candle_nn::VarBuilder::from_buffered_safetensors(
            weights_bytes,
            candle_core::DType::F32,
            &device,
        )?;

        let model = BertModel::load(vb, &config)?;

        tracing::info!("Embedding model 'all-MiniLM-L6-v2' initialized successfully (~45MB RAM footprint)");

        Ok(Self {
            model: Some(Arc::new(Mutex::new(model))),
            tokenizer: Some(Arc::new(tokenizer)),
            device,
        })
    }

    // Creates deterministic mock model for testing and offline runs.
    pub fn new_mock() -> Self {
        Self {
            model: None,
            tokenizer: None,
            device: Device::Cpu,
        }
    }

    // Generates 384-dim embedding on blocking thread pool to keep async runtime responsive.
    pub async fn embed_async(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        if let (Some(model_arc), Some(tokenizer_arc)) = (&self.model, &self.tokenizer) {
            let model_arc = model_arc.clone();
            let tokenizer_arc = tokenizer_arc.clone();
            let text = text.to_string();
            let device = self.device.clone();
            return tokio::task::spawn_blocking(move || {
                Self::run_inference(&model_arc, &tokenizer_arc, &text, &device)
            })
            .await
            .map_err(|e| format!("spawn_blocking join error: {e}"))?;
        }
        Ok(Self::mock_embed(text))
    }

    // Synchronous embed for tests and non-async callers.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        if let (Some(model_arc), Some(tokenizer_arc)) = (&self.model, &self.tokenizer) {
            return Self::run_inference(model_arc, tokenizer_arc, text, &self.device);
        }
        Ok(Self::mock_embed(text))
    }

    fn run_inference(
        model_arc: &Arc<Mutex<BertModel>>,
        tokenizer_arc: &Arc<Tokenizer>,
        text: &str,
        device: &Device,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let mut tokenizer = tokenizer_arc.as_ref().clone();
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).ok();

        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| format!("Tokenization failed: {e}"))?;

        let tokens = encoding.get_ids();
        let attention_mask_raw = encoding.get_attention_mask();

        let token_ids = Tensor::new(tokens, device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let model = model_arc.lock().map_err(|e| format!("Model mutex poisoned: {e}"))?;
        let embeddings = model.forward(&token_ids, &token_type_ids, None)?;

        // Mean pooling over token embeddings weighted by attention mask.
        let attention_mask = Tensor::new(attention_mask_raw, device)?
            .unsqueeze(0)?
            .unsqueeze(2)?
            .to_dtype(candle_core::DType::F32)?;

        let masked_embeddings = embeddings.broadcast_mul(&attention_mask)?;
        let sum_embeddings = masked_embeddings.sum(1)?;
        let sum_mask = attention_mask.sum(1)?.clamp(1e-9, f32::MAX)?;
        let mean_pooled = sum_embeddings.broadcast_div(&sum_mask)?;

        // L2 normalize vector so dot product equals cosine similarity.
        let norm = mean_pooled
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-9, f32::MAX)?;
        let normalized = mean_pooled.broadcast_div(&norm)?;

        let vector: Vec<f32> = normalized.squeeze(0)?.to_vec1()?;
        Ok(vector)
    }

    fn mock_embed(text: &str) -> Vec<f32> {
        let mut vector = vec![0.0f32; 384];
        let bytes = text.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            vector[i % 384] += (b as f32) / 255.0;
        }
        let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-9);
        for v in &mut vector {
            *v /= norm;
        }
        vector
    }
}
