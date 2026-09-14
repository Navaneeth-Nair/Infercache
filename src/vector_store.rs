use crate::models::CachePayload;
use async_trait::async_trait;
use parking_lot::RwLock;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointStruct, SearchPointsBuilder, Value, VectorParamsBuilder,
};
use qdrant_client::Qdrant;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub type DynError = Box<dyn std::error::Error + Send + Sync>;

#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn search(&self, vector: &[f32], model: &str, threshold: f32) -> Result<Option<CachePayload>, DynError>;
    async fn insert(&self, payload: CachePayload, vector: &[f32]) -> Result<(), DynError>;
    async fn len(&self) -> usize;
    async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}

struct InMemoryInner {
    exact_hashes: HashSet<String>,
    vectors: Vec<(Vec<f32>, Arc<CachePayload>)>,
}

// In-memory vector store (<50MB RAM, zero external dependencies).
pub struct InMemoryVectorStore {
    inner: RwLock<InMemoryInner>,
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(InMemoryInner {
                exact_hashes: HashSet::new(),
                vectors: Vec::new(),
            }),
        }
    }
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn search(&self, vector: &[f32], model: &str, threshold: f32) -> Result<Option<CachePayload>, DynError> {
        // Search vectors using cosine similarity.
        let inner = self.inner.read();
        let mut best_match: Option<(f32, Arc<CachePayload>)> = None;

        for (stored_vec, payload) in inner.vectors.iter() {
            if payload.model != model {
                continue;
            }

            // Dot product equals cosine similarity for L2-normalized vectors.
            let similarity: f32 = vector
                .iter()
                .zip(stored_vec.iter())
                .map(|(a, b)| a * b)
                .sum();

            if similarity >= threshold {
                match &best_match {
                    Some((best_score, _)) if similarity > *best_score => {
                        best_match = Some((similarity, payload.clone()));
                    }
                    None => {
                        best_match = Some((similarity, payload.clone()));
                    }
                    _ => {}
                }
            }
        }

        if let Some((score, payload)) = best_match {
            tracing::info!(score = %score, prompt_hash = %payload.prompt_hash, "In-Memory Semantic Cache Hit");
            return Ok(Some((*payload).clone()));
        }

        Ok(None)
    }

    async fn insert(&self, payload: CachePayload, vector: &[f32]) -> Result<(), DynError> {
        const MAX_CACHE_ENTRIES: usize = 10_000;
        let mut inner = self.inner.write();

        // Atomic deduplication by prompt hash: single lock prevents any TOCTOU race.
        if inner.exact_hashes.contains(&payload.prompt_hash) {
            tracing::debug!(prompt_hash = %payload.prompt_hash, "Exact-hash duplicate; skipping insert");
            return Ok(());
        }

        // Enforce 10,000 entry cap by FIFO eviction of oldest entry.
        if inner.vectors.len() >= MAX_CACHE_ENTRIES {
            let evicted = inner.vectors.remove(0);
            inner.exact_hashes.remove(&evicted.1.prompt_hash);
            tracing::warn!(
                evicted_hash = %evicted.1.prompt_hash,
                "InMemoryVectorStore at capacity ({}); evicted oldest entry (FIFO)",
                MAX_CACHE_ENTRIES
            );
        }

        let prompt_hash = payload.prompt_hash.clone();
        inner.exact_hashes.insert(prompt_hash);
        inner.vectors.push((vector.to_vec(), Arc::new(payload)));
        Ok(())
    }

    async fn len(&self) -> usize {
        self.inner.read().vectors.len()
    }
}

// Qdrant vector store backend.
pub struct QdrantVectorStore {
    client: Arc<Qdrant>,
    collection_name: String,
}

impl QdrantVectorStore {
    pub async fn connect(url: &str, collection_name: &str) -> Result<Self, DynError> {
        tracing::info!(url = %url, collection = %collection_name, "Connecting to Qdrant vector database");
        let client = Arc::new(Qdrant::from_url(url).build()?);

        // Ensure collection exists with 384 dimensions and Cosine distance.
        let collections = client.list_collections().await?;
        let exists = collections.collections.iter().any(|c| c.name == collection_name);

        if !exists {
            tracing::info!(collection = %collection_name, "Creating Qdrant collection with 384-dim Cosine config");
            client
                .create_collection(
                    CreateCollectionBuilder::new(collection_name).vectors_config(
                        VectorParamsBuilder::new(384, Distance::Cosine),
                    ),
                )
                .await?;
        }

        Ok(Self {
            client,
            collection_name: collection_name.to_string(),
        })
    }
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn search(&self, vector: &[f32], model: &str, threshold: f32) -> Result<Option<CachePayload>, DynError> {
        let search_result = self
            .client
            .search_points(
                SearchPointsBuilder::new(&self.collection_name, vector.to_vec(), 1)
                    .score_threshold(threshold)
                    .with_payload(true),
            )
            .await?;

        if let Some(point) = search_result.result.into_iter().next() {
            if point.score < threshold {
                return Ok(None);
            }

            // Extract cache payload from search result.
            let prompt_hash = point
                .payload
                .get("prompt_hash")
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_default();

            let prompt = point
                .payload
                .get("prompt")
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_default();

            let stored_model = point
                .payload
                .get("model")
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_default();

            if stored_model != model {
                return Ok(None);
            }

            let sse_chunks: Vec<String> = point
                .payload
                .get("sse_chunks")
                .and_then(|v| v.as_list())
                .map(|list| {
                    list.iter()
                        .filter_map(|item| item.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            let created_at = point
                .payload
                .get("created_at")
                .and_then(|v| v.as_integer())
                .unwrap_or_default();

            tracing::info!(score = %point.score, prompt_hash = %prompt_hash, "Qdrant Vector Cache Hit");

            return Ok(Some(CachePayload {
                prompt_hash,
                prompt,
                model: stored_model,
                sse_chunks,
                created_at,
            }));
        }

        Ok(None)
    }

    async fn insert(&self, payload: CachePayload, vector: &[f32]) -> Result<(), DynError> {
        // Use prompt_hash as deterministic point ID for deduplication.
        let point_id = payload.prompt_hash.clone();
        let mut qdrant_payload: HashMap<String, Value> = HashMap::new();

        qdrant_payload.insert("prompt_hash".into(), payload.prompt_hash.into());
        qdrant_payload.insert("prompt".into(), payload.prompt.into());
        qdrant_payload.insert("model".into(), payload.model.into());
        qdrant_payload.insert("created_at".into(), payload.created_at.into());

        let chunks_values: Vec<Value> = payload
            .sse_chunks
            .into_iter()
            .map(|chunk| chunk.into())
            .collect();
        qdrant_payload.insert("sse_chunks".into(), chunks_values.into());

        let point = PointStruct::new(point_id, vector.to_vec(), qdrant_payload);

        self.client
            .upsert_points(qdrant_client::qdrant::UpsertPointsBuilder::new(
                &self.collection_name,
                vec![point],
            ))
            .await?;

        Ok(())
    }

    async fn len(&self) -> usize {
        let info = self.client.collection_info(&self.collection_name).await;
        info.map(|i| i.result.and_then(|r| r.points_count).unwrap_or(0) as usize)
            .unwrap_or(0)
    }
}
