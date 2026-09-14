use crate::coalescing::{CoalescingGuard, CoalescingSlot};
use crate::models::{CachePayload, ChatCompletionRequest};
use crate::state::AppState;
use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::StreamExt;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio_stream::wrappers::BroadcastStream;

// Handler for POST /v1/chat/completions.
pub async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    state.metrics.total_requests.fetch_add(1, Ordering::Relaxed);
    let req_start = std::time::Instant::now();

    let is_streaming = request.stream.unwrap_or(false);
    let prompt = request.extract_prompt();
    let prompt_hash = ChatCompletionRequest::prompt_hash_from_prompt(&prompt);
    let model = request.model.clone();
    let coalescing_key = format!("{}:{}", model, prompt_hash);

    tracing::debug!(
        model = %model,
        prompt_hash = %prompt_hash,
        streaming = is_streaming,
        "Received chat completion request"
    );

    // 1. Generate embedding vector on blocking thread pool.
    let embed_start = std::time::Instant::now();
    let embedding_result = state.embedding_model.embed_async(&prompt).await;
    crate::metrics::EMBEDDING_DURATION.observe(embed_start.elapsed().as_secs_f64());

    let vector = match embedding_result {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(error = %e, "Failed to compute embedding; bypassing cache");
            return forward_direct_upstream(&state, headers, &request).await;
        }
    };

    // 2. Query vector store for similarity above threshold.
    let search_start = std::time::Instant::now();
    let cache_lookup = state
        .vector_store
        .search(&vector, &model, state.config.similarity_threshold)
        .await;
    crate::metrics::VECTOR_SEARCH_DURATION.observe(search_start.elapsed().as_secs_f64());

    // 3. Cache hit: serve cached SSE or JSON response immediately.
    if let Ok(Some(cached_payload)) = cache_lookup {
        state.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
        let elapsed = req_start.elapsed().as_secs_f64();
        crate::metrics::record_cache_hit(elapsed);
        tracing::info!(prompt_hash = %cached_payload.prompt_hash, latency_ms = %(elapsed * 1000.0), "Cache Hit! Serving cached response");

        if is_streaming {
            return serve_cached_sse_stream(cached_payload.sse_chunks);
        } else {
            return (
                StatusCode::OK,
                [
                    (header::CONTENT_TYPE, "application/json"),
                    (header::HeaderName::from_static("x-infercache-status"), "HIT"),
                ],
                Json(cached_payload.to_non_streaming_json()),
            )
                .into_response();
        }
    }

    // 4. Cache miss: check coalescing engine or proxy upstream.
    state.metrics.cache_misses.fetch_add(1, Ordering::Relaxed);
    tracing::info!(prompt_hash = %prompt_hash, "Cache Miss. Checking coalescing engine");

    match state.coalescing_engine.acquire(coalescing_key.clone()) {
        CoalescingSlot::Subscriber { receiver, key: _ } => {
            state.metrics.coalesced_requests.fetch_add(1, Ordering::Relaxed);
            let elapsed = req_start.elapsed().as_secs_f64();
            crate::metrics::record_coalesced(&model, elapsed);
            tracing::info!("Subscribing to active in-flight stream (Thundering Herd protected)");

            let stream = BroadcastStream::new(receiver).filter_map(|res| async move {
                match res {
                    Ok(chunk) => Some(Ok::<_, std::io::Error>(chunk)),
                    Err(tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(n)) => {
                        tracing::warn!(skipped = n, "Coalesced subscriber lagged; chunks dropped by broadcast channel");
                        None
                    }
                }
            });

            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .header(header::CACHE_CONTROL, "no-cache")
                .header(header::CONNECTION, "keep-alive")
                .header("X-InferCache-Status", "COALESCED")
                .body(Body::from_stream(stream))
                .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
        }
        CoalescingSlot::Leader { sender, key } => {
            let elapsed = req_start.elapsed().as_secs_f64();
            crate::metrics::record_cache_miss(elapsed);
            tracing::info!("Leader assigned; dispatching request to upstream LLM");
            handle_leader_upstream(state, headers, request, prompt, prompt_hash, vector, key, sender).await
        }
    }
}

// Stream cached raw SSE strings directly to client.
fn serve_cached_sse_stream(chunks: Vec<String>) -> Response {
    let stream = futures_util::stream::iter(chunks.into_iter().map(Ok::<_, std::io::Error>));

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .header("X-InferCache-Status", "HIT")
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

// Primary request handler: streams upstream, broadcasts, and saves to cache.
#[allow(clippy::too_many_arguments)]
async fn handle_leader_upstream(
    state: Arc<AppState>,
    headers: HeaderMap,
    request: ChatCompletionRequest,
    prompt: String,
    prompt_hash: String,
    vector: Vec<f32>,
    coalesce_key: String,
    sender: tokio::sync::broadcast::Sender<String>,
) -> Response {
    let upstream_url = state.config.upstream_url.clone();
    let mut req_builder = state.http_client.post(&upstream_url).json(&request);

    // Forward client Authorization header or use configured upstream key.
    if let Some(auth_val) = headers.get(header::AUTHORIZATION) {
        req_builder = req_builder.header(header::AUTHORIZATION, auth_val);
    } else if let Some(ref api_key) = state.config.upstream_api_key {
        req_builder = req_builder.header(header::AUTHORIZATION, format!("Bearer {api_key}"));
    }

    let coalescing_guard = CoalescingGuard::new(state.coalescing_engine.clone(), coalesce_key);

    let upstream_res = match req_builder.send().await {
        Ok(res) => res,
        Err(e) => {
            tracing::error!(error = %e, "Failed to connect to upstream LLM");
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "error": {
                        "message": "Failed to connect to upstream LLM. Please retry.",
                        "type": "infercache_gateway_error"
                    }
                })),
            )
                .into_response();
        }
    };

    if !upstream_res.status().is_success() {
        let status = upstream_res.status();
        let body = upstream_res.text().await.unwrap_or_default();
        return (status, body).into_response();
    }

    // Set up channel for SSE streaming.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::io::Error>>(64);

    let state_clone = state.clone();
    let model = request.model.clone();

    tokio::spawn(async move {
        // Coalescing guard is moved here and automatically releases leader key on scope exit or panic.
        let _guard = coalescing_guard;
        let mut byte_stream = upstream_res.bytes_stream();
        let mut accumulated_chunks: Vec<String> = Vec::new();
        let mut stream_complete = false;
        let mut primary_active = true;

        while let Some(chunk_res) = byte_stream.next().await {
            match chunk_res {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes).to_string();
                    accumulated_chunks.push(text.clone());

                    // Mark stream complete when [DONE] chunk is received.
                    if text.contains("[DONE]") {
                        stream_complete = true;
                    }

                    // Broadcast chunk to secondary subscribers.
                    let _ = sender.send(text.clone());

                    // Send chunk to primary client if still connected.
                    if primary_active {
                        if tx.send(Ok(text)).await.is_err() {
                            tracing::warn!("Primary client disconnected during stream");
                            primary_active = false;
                            // If there are no secondary subscribers, abort upstream fetch.
                            if sender.receiver_count() == 0 {
                                break;
                            }
                        }
                    } else if sender.receiver_count() == 0 {
                        // All secondary subscribers have also disconnected.
                        break;
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "Error reading upstream byte chunk");
                    if primary_active {
                        let _ = tx
                            .send(Err(std::io::Error::other(e)))
                            .await;
                    }
                    break;
                }
            }
        }

        // Cache response only if stream completed successfully.
        if stream_complete && !accumulated_chunks.is_empty() {
            let payload = CachePayload::new(prompt_hash, prompt, model, accumulated_chunks);
            let vs = state_clone.vector_store.clone();
            tokio::spawn(async move {
                if let Err(e) = vs.insert(payload, &vector).await {
                    tracing::error!(error = %e, "Failed to persist payload into vector store");
                } else {
                    tracing::info!("Successfully cached LLM stream into vector database");
                }
            });
        } else if !stream_complete {
            tracing::warn!("Upstream stream ended without [DONE] marker; skipping cache insert to avoid partial response caching");
        }
    });

    let body_stream = tokio_stream::wrappers::ReceiverStream::new(rx);

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .header("X-InferCache-Status", "MISS")
        .body(Body::from_stream(body_stream))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

// Direct upstream forwarding bypass if embeddings or cache fail.
async fn forward_direct_upstream(
    state: &AppState,
    headers: HeaderMap,
    request: &ChatCompletionRequest,
) -> Response {
    let mut req_builder = state.http_client.post(&state.config.upstream_url).json(request);

    if let Some(auth_val) = headers.get(header::AUTHORIZATION) {
        req_builder = req_builder.header(header::AUTHORIZATION, auth_val);
    } else if let Some(ref api_key) = state.config.upstream_api_key {
        req_builder = req_builder.header(header::AUTHORIZATION, format!("Bearer {api_key}"));
    }

    match req_builder.send().await {
        Ok(res) => {
            let status = res.status();
            let body_stream = res.bytes_stream();
            Response::builder()
                .status(status)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .header("X-InferCache-Status", "BYPASS")
                .body(Body::from_stream(body_stream))
                .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to connect to upstream LLM during bypass");
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "error": { "message": "Failed to connect to upstream LLM. Please retry." }
                })),
            )
                .into_response()
        }
    }
}
