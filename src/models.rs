use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl ChatCompletionRequest {
    // Extracts canonical prompt representation from messages.
    pub fn extract_prompt(&self) -> String {
        let mut full_prompt = String::new();
        for msg in &self.messages {
            full_prompt.push_str(&msg.role);
            full_prompt.push_str(": ");
            full_prompt.push_str(&msg.content);
            full_prompt.push('\n');
        }
        full_prompt.trim_end().to_string()
    }

    // Computes SHA-256 hash of normalized prompt.
    pub fn prompt_hash(&self) -> String {
        let prompt = self.extract_prompt();
        let mut hasher = Sha256::new();
        hasher.update(prompt.as_bytes());
        hex::encode(hasher.finalize())
    }

    // Unique request key for in-flight request coalescing.
    pub fn coalescing_key(&self) -> String {
        format!("{}:{}", self.model, self.prompt_hash())
    }
}

// Stored vector cache entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePayload {
    pub prompt_hash: String,     // SHA-256 hash of prompt for exact match fallback
    pub prompt: String,          // Normalized prompt text
    pub model: String,           // Model identifier e.g. gpt-4o-mini
    pub sse_chunks: Vec<String>, // Raw SSE strings ["data: ...\n\n"]
    pub created_at: i64,         // Unix timestamp for TTL tracking
}

impl CachePayload {
    pub fn new(prompt_hash: String, prompt: String, model: String, sse_chunks: Vec<String>) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        Self {
            prompt_hash,
            prompt,
            model,
            sse_chunks,
            created_at,
        }
    }

    // Assembles standard non-streaming JSON from cached SSE chunks.
    pub fn to_non_streaming_json(&self) -> serde_json::Value {
        let mut full_content = String::new();
        let id_suffix = self.prompt_hash.get(..8).unwrap_or(&self.prompt_hash);
        let mut id = format!("chatcmpl-cache-{}", id_suffix);
        let mut created = self.created_at;

        for chunk in &self.sse_chunks {
            for line in chunk.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    if data.trim() == "[DONE]" {
                        continue;
                    }
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(i) = val.get("id").and_then(|v| v.as_str()) {
                            id = i.to_string();
                        }
                        if let Some(c) = val.get("created").and_then(|v| v.as_i64()) {
                            created = c;
                        }
                        if let Some(delta) = val
                            .get("choices")
                            .and_then(|choices| choices.get(0))
                            .and_then(|choice| choice.get("delta"))
                            .and_then(|delta| delta.get("content"))
                            .and_then(|content| content.as_str())
                        {
                            full_content.push_str(delta);
                        }
                    }
                }
            }
        }

        serde_json::json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            "cached": true
        })
    }
}
