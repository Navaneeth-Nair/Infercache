use infercache::models::CachePayload;
use infercache::vector_store::{InMemoryVectorStore, VectorStore};

fn make_normalized_vector(dim: usize, primary_idx: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[primary_idx] = 1.0f32;
    v
}

#[tokio::test]
async fn test_in_memory_vector_store_cosine_search() {
    let store = InMemoryVectorStore::new();

    let vec_384 = make_normalized_vector(384, 0);
    let payload = CachePayload::new(
        "hash_exact".to_string(),
        "Tell me about Rust memory optimization".to_string(),
        "gpt-4o-mini".to_string(),
        vec!["data: {\"test\":1}\n\n".to_string()],
    );

    store.insert(payload.clone(), &vec_384).await.unwrap();
    assert_eq!(store.len().await, 1);

    // 1. Exact match (cosine similarity = 1.0 >= 0.95)
    let hit = store.search(&vec_384, "gpt-4o-mini", 0.95).await.unwrap();
    assert!(hit.is_some());
    assert_eq!(hit.unwrap().prompt_hash, "hash_exact");

    // 2. High similarity vector (cosine ~0.99 >= 0.95)
    let mut close_vec = vec_384.clone();
    close_vec[0] = 0.995;
    close_vec[1] = 0.09987; // sqrt(0.995^2 + 0.09987^2) ~ 1.0 (dot product ~0.995)
    let hit_close = store.search(&close_vec, "gpt-4o-mini", 0.95).await.unwrap();
    assert!(hit_close.is_some());

    // 3. Low similarity vector (orthogonal: primary_idx 1, dot product = 0.0 < 0.95)
    let orthogonal_vec = make_normalized_vector(384, 1);
    let miss = store.search(&orthogonal_vec, "gpt-4o-mini", 0.95).await.unwrap();
    assert!(miss.is_none());

    // 4. Different model name should not match even if vector is identical
    let diff_model = store.search(&vec_384, "claude-3-5-sonnet", 0.95).await.unwrap();
    assert!(diff_model.is_none());
}

#[tokio::test]
async fn test_in_memory_vector_store_deduplication() {
    let store = InMemoryVectorStore::new();
    let vec_384 = make_normalized_vector(384, 0);

    let payload1 = CachePayload::new(
        "dup_hash".to_string(),
        "Hello duplicate".to_string(),
        "gpt-4o-mini".to_string(),
        vec!["chunk1".to_string()],
    );
    let payload2 = CachePayload::new(
        "dup_hash".to_string(),
        "Hello duplicate".to_string(),
        "gpt-4o-mini".to_string(),
        vec!["chunk2".to_string()],
    );

    store.insert(payload1, &vec_384).await.unwrap();
    assert_eq!(store.len().await, 1);

    // Second insert with same prompt_hash must be deduplicated
    store.insert(payload2, &vec_384).await.unwrap();
    assert_eq!(store.len().await, 1);
}
