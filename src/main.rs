use axum::extract::State;
use axum::http::{HeaderValue, Method, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use infercache::config::{Config, VectorBackend};
use infercache::embeddings::CandleEmbeddingModel;
use infercache::proxy;
use infercache::state::AppState;
use infercache::vector_store::{InMemoryVectorStore, QdrantVectorStore, VectorStore};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::trace::TraceLayer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize structured tracing.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "infercache=info,tower_http=info".into()),
        )
        .init();

    print_banner();

    let config = Config::from_env();
    tracing::info!(
        host = %config.host,
        port = %config.port,
        upstream = %config.upstream_url,
        threshold = %config.similarity_threshold,
        backend = ?config.vector_backend,
        "Starting InferCache reverse proxy"
    );

    // 1. Initialize local Candle FP16 embedding model (~45MB).
    let embedding_model = Arc::new(CandleEmbeddingModel::load(&config.model_id)?);

    // 2. Initialize vector store (Qdrant or In-Memory).
    let vector_store: Arc<dyn VectorStore> = match config.vector_backend {
        VectorBackend::Qdrant => {
            match QdrantVectorStore::connect(&config.qdrant_url, &config.qdrant_collection).await {
                Ok(qdrant) => Arc::new(qdrant),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Failed to connect to Qdrant instance; falling back to high-performance In-Memory vector store"
                    );
                    Arc::new(InMemoryVectorStore::new())
                }
            }
        }
        VectorBackend::Memory => Arc::new(InMemoryVectorStore::new()),
    };

    let state = Arc::new(AppState::new(config.clone(), embedding_model, vector_store));

    // 3. Configure CORS policy from explicit origin allowlist.
    let cors_layer = if config.cors_allowed_origins.is_empty() {
        tracing::info!("CORS: no CORS_ALLOWED_ORIGINS set - browser cross-origin requests denied");
        CorsLayer::new()
            .allow_methods([Method::POST, Method::GET, Method::OPTIONS])
            .allow_headers([
                axum::http::header::CONTENT_TYPE,
                axum::http::header::AUTHORIZATION,
            ])
            .allow_origin(AllowOrigin::list([]))
    } else {
        let parsed: Vec<HeaderValue> = config
            .cors_allowed_origins
            .iter()
            .filter_map(|o| o.parse::<HeaderValue>().ok())
            .collect();
        tracing::info!(allowed_origins = ?config.cors_allowed_origins, "CORS: explicit origin allowlist active");
        CorsLayer::new()
            .allow_methods([Method::POST, Method::GET, Method::OPTIONS])
            .allow_headers([
                axum::http::header::CONTENT_TYPE,
                axum::http::header::AUTHORIZATION,
            ])
            .allow_origin(AllowOrigin::list(parsed))
    };

    // 4. Initialize Prometheus metrics registry.
    infercache::metrics::init_metrics();

    // 5. Build Axum routing engine.
    let app = Router::new()
        .route("/v1/chat/completions", post(proxy::chat_completions_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/stats", get(stats_handler))
        .layer(TraceLayer::new_for_http())
        .layer(cors_layer)
        .with_state(state);

    let bind_addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    tracing::info!("InferCache listening on http://{}", bind_addr);

    axum::serve(listener, app).await?;
    Ok(())
}

async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let cached_items = state.vector_store.len().await;
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "healthy",
            "service": "infercache",
            "version": env!("CARGO_PKG_VERSION"),
            "cached_embeddings_count": cached_items,
            "target_ram_budget": "< 300MB"
        })),
    )
}

// Prometheus scraping endpoint.
async fn metrics_handler() -> impl IntoResponse {
    let body = infercache::metrics::get_metrics();
    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

// JSON statistics endpoint.
async fn stats_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let total = state.metrics.total_requests.load(Ordering::Relaxed);
    let hits = state.metrics.cache_hits.load(Ordering::Relaxed);
    let misses = state.metrics.cache_misses.load(Ordering::Relaxed);
    let coalesced = state.metrics.coalesced_requests.load(Ordering::Relaxed);
    let hit_rate = if total > 0 {
        (hits as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "total_requests": total,
            "cache_hits": hits,
            "cache_misses": misses,
            "coalesced_requests": coalesced,
            "cache_hit_rate_pct": format!("{:.2}%", hit_rate),
            "cached_vectors_in_store": state.vector_store.len().await,
        })),
    )
}

fn print_banner() {
    println!(
        r#"
===================================================================
  ___        __              ____           _          
 |_ _|_ __  / _| ___ _ __   / ___|__ _  ___| |__   ___ 
  | || '_ \| |_ / _ \ '__| | |   / _` |/ __| '_ \ / _ \
  | || | | |  _|  __/ |    | |__| (_| | (__| | | |  __/
 |___|_| |_|_|  \___|_|     \____\__,_|\___|_| |_|\___|
 
  Ultra-Low-RAM Semantic Caching LLM Reverse Proxy
  Target RAM Budget: < 300MB | Axum + Tokio + Candle + Qdrant
===================================================================
"#
    );
}
