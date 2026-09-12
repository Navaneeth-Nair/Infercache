use infercache::models::{CachePayload, ChatCompletionRequest, ChatMessage};

#[test]
fn test_prompt_extraction_and_hashing() {
    let req = ChatCompletionRequest {
        model: "gpt-4o-mini".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are an ultra-fast caching proxy.".to_string(),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "What is the memory budget of InferCache?".to_string(),
                name: None,
            },
        ],
        stream: Some(true),
        temperature: None,
        max_tokens: None,
        extra: serde_json::Map::new(),
    };

    let prompt = req.extract_prompt();
    assert!(prompt.contains("system: You are an ultra-fast caching proxy."));
    assert!(prompt.contains("user: What is the memory budget of InferCache?"));

    let hash = req.prompt_hash();
    assert_eq!(hash.len(), 64); // SHA-256 hex string

    let key = req.coalescing_key();
    assert_eq!(key, format!("gpt-4o-mini:{}", hash));
}

#[test]
fn test_cache_payload_to_non_streaming_json() {
    let chunks = vec![
        "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"InferCache \"},\"finish_reason\":null}]}\n\n".to_string(),
        "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"runs in <300MB RAM.\"},\"finish_reason\":null}]}\n\n".to_string(),
        "data: [DONE]\n\n".to_string(),
    ];

    let payload = CachePayload::new(
        "abc123hash".to_string(),
        "sample prompt".to_string(),
        "gpt-4o-mini".to_string(),
        chunks,
    );

    let json_resp = payload.to_non_streaming_json();
    assert_eq!(json_resp["model"], "gpt-4o-mini");
    assert_eq!(json_resp["cached"], true);
    assert_eq!(
        json_resp["choices"][0]["message"]["content"],
        "InferCache runs in <300MB RAM."
    );
}
