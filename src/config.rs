use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub upstream_url: String,
    pub upstream_api_key: Option<String>,
    pub similarity_threshold: f32,
    pub vector_backend: VectorBackend,
    pub qdrant_url: String,
    pub qdrant_collection: String,
    pub model_id: String,
    // Comma-separated list of allowed CORS origins. Empty means cross-origin denied.
    pub cors_allowed_origins: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VectorBackend {
    Memory,
    Qdrant,
}

impl Config {
    pub fn from_env() -> Self {
        let host = env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
        let port = env::var("PORT")
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(8080);

        let upstream_url = env::var("UPSTREAM_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1/chat/completions".to_string());
        let upstream_api_key = env::var("UPSTREAM_API_KEY").ok();

        let similarity_threshold = env::var("SIMILARITY_THRESHOLD")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.95);

        let vector_backend = match env::var("VECTOR_BACKEND")
            .unwrap_or_else(|_| "memory".to_string())
            .to_lowercase()
            .as_str()
        {
            "qdrant" => VectorBackend::Qdrant,
            _ => VectorBackend::Memory,
        };

        let qdrant_url = env::var("QDRANT_URL")
            .unwrap_or_else(|_| "http://localhost:6334".to_string());
        let qdrant_collection = env::var("QDRANT_COLLECTION")
            .unwrap_or_else(|_| "infercache".to_string());

        let model_id = env::var("MODEL_ID")
            .unwrap_or_else(|_| "sentence-transformers/all-MiniLM-L6-v2".to_string());

        // Parse comma-separated allowed origins e.g. "http://localhost:3000".
        let cors_allowed_origins = env::var("CORS_ALLOWED_ORIGINS")
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect::<Vec<_>>();

        Self {
            host,
            port,
            upstream_url,
            upstream_api_key,
            similarity_threshold,
            vector_backend,
            qdrant_url,
            qdrant_collection,
            model_id,
            cors_allowed_origins,
        }
    }
}
