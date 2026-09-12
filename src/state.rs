use crate::coalescing::{CoalescingEngine, InFlightRequests};
use crate::config::Config;
use crate::embeddings::CandleEmbeddingModel;
use crate::vector_store::VectorStore;
use dashmap::DashMap;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

pub struct Metrics {
    pub total_requests: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub coalesced_requests: AtomicU64,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            coalesced_requests: AtomicU64::new(0),
        }
    }
}

pub struct AppState {
    pub config: Config,
    pub embedding_model: Arc<CandleEmbeddingModel>,
    pub vector_store: Arc<dyn VectorStore>,
    #[allow(dead_code)]
    pub in_flight_requests: Arc<InFlightRequests>,
    pub coalescing_engine: Arc<CoalescingEngine>,
    pub http_client: reqwest::Client,
    pub metrics: Arc<Metrics>,
}

impl AppState {
    pub fn new(
        config: Config,
        embedding_model: Arc<CandleEmbeddingModel>,
        vector_store: Arc<dyn VectorStore>,
    ) -> Self {
        let in_flight_requests = Arc::new(DashMap::new());
        let coalescing_engine = Arc::new(CoalescingEngine::new(in_flight_requests.clone()));

        let http_client = reqwest::Client::builder()
            .tcp_nodelay(true)
            .pool_max_idle_per_host(100)
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .expect("Failed to initialize HTTP client");

        Self {
            config,
            embedding_model,
            vector_store,
            in_flight_requests,
            coalescing_engine,
            http_client,
            metrics: Arc::new(Metrics::new()),
        }
    }
}
