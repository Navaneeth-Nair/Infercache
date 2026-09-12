use axum::body::Body;
use axum::http::{header, StatusCode};
use axum::response::Response;
use axum::routing::post;
use axum::{Json, Router};
use infercache::config::{Config, VectorBackend};
use infercache::embeddings::CandleEmbeddingModel;
use infercache::models::{ChatCompletionRequest, ChatMessage};
use infercache::proxy;
use infercache::state::AppState;
use infercache::vector_store::InMemoryVectorStore;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::time::{sleep, Duration};

static MOCK_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

async fn mock_openai_handler(Json(_req): Json<serde_json::Value>) -> Response {
    MOCK_CALL_COUNT.fetch_add(1, Ordering::SeqCst);

    let chunks = vec![
        "data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"InferCache \"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-mock\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"is blistering fast.\"},\"finish_reason\":null}]}\n\n",
        "data: [DONE]\n\n",
    ];

    let stream = futures_util::stream::iter(chunks.into_iter().map(Ok::<_, std::io::Error>));

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap()
}

#[tokio::test]
async fn test_proxy_cache_miss_then_hit_e2e() {
    // Reset shared static counter to avoid order-dependent test flakiness
    MOCK_CALL_COUNT.store(0, Ordering::SeqCst);

    // 1. Start Mock OpenAI Upstream Server
    let mock_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mock_port = mock_listener.local_addr().unwrap().port();
    let mock_app = Router::new().route("/v1/chat/completions", post(mock_openai_handler));

    tokio::spawn(async move {
        axum::serve(mock_listener, mock_app).await.unwrap();
    });

    // 2. Start InferCache Server pointing to Mock Upstream
    let config = Config {
        host: "127.0.0.1".to_string(),
        port: 0,
        upstream_url: format!("http://127.0.0.1:{}/v1/chat/completions", mock_port),
        upstream_api_key: None,
        similarity_threshold: 0.95,
        vector_backend: VectorBackend::Memory,
        qdrant_url: "http://localhost:6334".to_string(),
        qdrant_collection: "infercache".to_string(),
        model_id: "test".to_string(),
        cors_allowed_origins: vec![],
    };

    let embedding_model = Arc::new(CandleEmbeddingModel::new_mock());
    let vector_store = Arc::new(InMemoryVectorStore::new());
    let state = Arc::new(AppState::new(config.clone(), embedding_model, vector_store));

    let app = Router::new()
        .route("/v1/chat/completions", post(proxy::chat_completions_handler))
        .with_state(state.clone());

    let proxy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_port = proxy_listener.local_addr().unwrap().port();

    tokio::spawn(async move {
        axum::serve(proxy_listener, app).await.unwrap();
    });

    // Wait 50ms for servers to bind
    sleep(Duration::from_millis(50)).await;

    let client = reqwest::Client::new();
    let test_req = ChatCompletionRequest {
        model: "gpt-4o-mini".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Explain memory budget of InferCache".to_string(),
            name: None,
        }],
        stream: Some(true),
        temperature: None,
        max_tokens: None,
        extra: serde_json::Map::new(),
    };

    // 3. FIRST REQUEST -> CACHE MISS (Hits Mock Upstream)
    let res1 = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", proxy_port))
        .json(&test_req)
        .send()
        .await
        .unwrap();

    assert_eq!(res1.status(), StatusCode::OK);
    assert_eq!(
        res1.headers().get("x-infercache-status").unwrap(),
        "MISS"
    );

    let body1 = res1.text().await.unwrap();
    assert!(body1.contains("InferCache"));
    assert!(body1.contains("is blistering fast."));
    assert_eq!(MOCK_CALL_COUNT.load(Ordering::SeqCst), 1);

    // Wait 100ms for async background cache insertion to complete
    sleep(Duration::from_millis(100)).await;

    // 4. SECOND REQUEST -> CACHE HIT (Zero upstream calls!)
    let res2 = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", proxy_port))
        .json(&test_req)
        .send()
        .await
        .unwrap();

    assert_eq!(res2.status(), StatusCode::OK);
    assert_eq!(
        res2.headers().get("x-infercache-status").unwrap(),
        "HIT"
    );

    let body2 = res2.text().await.unwrap();
    assert!(body2.contains("InferCache"));
    assert!(body2.contains("is blistering fast."));

    // Verify upstream was NOT called a second time!
    assert_eq!(MOCK_CALL_COUNT.load(Ordering::SeqCst), 1);

    // Verify metrics in state
    assert_eq!(state.metrics.total_requests.load(Ordering::Relaxed), 2);
    assert_eq!(state.metrics.cache_hits.load(Ordering::Relaxed), 1);
    assert_eq!(state.metrics.cache_misses.load(Ordering::Relaxed), 1);
}
