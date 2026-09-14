use lazy_static::lazy_static;
use prometheus::{
    opts, Encoder, Gauge, Histogram, HistogramOpts, IntCounter, IntCounterVec, Registry,
    TextEncoder,
};
use std::sync::Mutex;
use sysinfo::{Pid, System};

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();

    // Estimated USD saved based on gpt-4o-mini token pricing.
    pub static ref ESTIMATED_USD_SAVED: Gauge = Gauge::new(
        "infercache_estimated_usd_saved_usd",
        "Estimated USD saved by serving cached responses (based on gpt-4o-mini pricing)"
    ).expect("Failed to create ESTIMATED_USD_SAVED metric");

    // Cache operations categorized by status: hit, miss, or coalesced.
    pub static ref CACHE_OPERATIONS: IntCounterVec = IntCounterVec::new(
        opts!("infercache_cache_operations_total", "Total cache operations (hit, miss, coalesced)"),
        &["status"]
    ).expect("Failed to create CACHE_OPERATIONS metric");

    // Total request latency histogram.
    pub static ref REQUEST_DURATION: Histogram = Histogram::with_opts(
        HistogramOpts::new("infercache_request_duration_seconds", "Total end-to-end request latency")
            .buckets(vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5])
    ).expect("Failed to create REQUEST_DURATION metric");

    // Local Candle embedding latency histogram.
    pub static ref EMBEDDING_DURATION: Histogram = Histogram::with_opts(
        HistogramOpts::new("infercache_embedding_duration_seconds", "Local Candle embedding inference latency")
            .buckets(vec![0.001, 0.002, 0.003, 0.005, 0.01, 0.025])
    ).expect("Failed to create EMBEDDING_DURATION metric");

    // Vector store search latency histogram.
    pub static ref VECTOR_SEARCH_DURATION: Histogram = Histogram::with_opts(
        HistogramOpts::new("infercache_vector_search_duration_seconds", "Vector store search latency")
            .buckets(vec![0.0005, 0.001, 0.002, 0.005, 0.01, 0.025])
    ).expect("Failed to create VECTOR_SEARCH_DURATION metric");

    // Current process RAM usage in bytes.
    pub static ref MEMORY_USAGE_BYTES: Gauge = Gauge::new(
        "infercache_process_memory_bytes",
        "Current RAM usage of the InferCache process in bytes"
    ).expect("Failed to create MEMORY_USAGE_BYTES metric");

    // Coalesced requests served via broadcast channel.
    pub static ref COALESCED_REQUESTS_TOTAL: IntCounterVec = IntCounterVec::new(
        opts!("infercache_coalesced_requests_total", "Requests served via broadcast fanout"),
        &["model"]
    ).expect("Failed to create COALESCED_REQUESTS_TOTAL metric");

    // Total cached prompts in vector store.
    pub static ref VECTOR_STORE_SIZE: Gauge = Gauge::new(
        "infercache_vector_store_size",
        "Total cached prompts stored in vector database"
    ).expect("Failed to create VECTOR_STORE_SIZE metric");

    // Total encrypted payloads served.
    pub static ref ENCRYPTED_PAYLOADS_SERVED: IntCounter = IntCounter::new(
        "infercache_encrypted_payloads_served_total",
        "Total secure/encrypted cache payloads served"
    ).expect("Failed to create ENCRYPTED_PAYLOADS_SERVED metric");

    static ref SYSTEM_MONITOR: Mutex<System> = Mutex::new(System::new());
    static ref LAST_MEMORY_UPDATE: Mutex<Option<std::time::Instant>> = Mutex::new(None);
}

// Register all metrics in the Prometheus registry.
pub fn init_metrics() {
    REGISTRY.register(Box::new(ESTIMATED_USD_SAVED.clone())).ok();
    REGISTRY.register(Box::new(CACHE_OPERATIONS.clone())).ok();
    REGISTRY.register(Box::new(REQUEST_DURATION.clone())).ok();
    REGISTRY.register(Box::new(EMBEDDING_DURATION.clone())).ok();
    REGISTRY.register(Box::new(VECTOR_SEARCH_DURATION.clone())).ok();
    REGISTRY.register(Box::new(MEMORY_USAGE_BYTES.clone())).ok();
    REGISTRY.register(Box::new(COALESCED_REQUESTS_TOTAL.clone())).ok();
    REGISTRY.register(Box::new(VECTOR_STORE_SIZE.clone())).ok();
    REGISTRY.register(Box::new(ENCRYPTED_PAYLOADS_SERVED.clone())).ok();

    update_memory_usage();
}

// Refresh current process memory usage via sysinfo. Throttled to at most once every 2s.
pub fn update_memory_usage() {
    if let Ok(mut last_update) = LAST_MEMORY_UPDATE.lock() {
        if last_update.is_some_and(|last| last.elapsed().as_secs() < 2) {
            return;
        }

        if let Ok(mut sys) = SYSTEM_MONITOR.lock() {
            let pid = Pid::from_u32(std::process::id());
            sys.refresh_process(pid);
            if let Some(proc) = sys.process(pid) {
                let mem_bytes = proc.memory();
                MEMORY_USAGE_BYTES.set(mem_bytes as f64);
                *last_update = Some(std::time::Instant::now());
            }
        }
    }
}

// Record cache hit and increment estimated cost savings.
pub fn record_cache_hit(duration_secs: f64) {
    CACHE_OPERATIONS.with_label_values(&["hit"]).inc();
    REQUEST_DURATION.observe(duration_secs);
    ESTIMATED_USD_SAVED.add(0.00035);
    ENCRYPTED_PAYLOADS_SERVED.inc();
}

// Record cache miss and track request duration.
pub fn record_cache_miss(duration_secs: f64) {
    CACHE_OPERATIONS.with_label_values(&["miss"]).inc();
    REQUEST_DURATION.observe(duration_secs);
}

// Record coalesced request deduplicated via broadcast channel.
pub fn record_coalesced(model: &str, duration_secs: f64) {
    CACHE_OPERATIONS.with_label_values(&["coalesced"]).inc();
    COALESCED_REQUESTS_TOTAL.with_label_values(&[model]).inc();
    REQUEST_DURATION.observe(duration_secs);
    ESTIMATED_USD_SAVED.add(0.00035);
}

// Return metrics in Prometheus text exposition format.
pub fn get_metrics() -> String {
    update_memory_usage();
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();
    if encoder.encode(&metric_families, &mut buffer).is_ok() {
        String::from_utf8(buffer).unwrap_or_default()
    } else {
        String::new()
    }
}
