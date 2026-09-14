use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::broadcast;

// Maps request_key to a broadcast channel sender with 128-chunk capacity.
pub type InFlightRequests = DashMap<String, broadcast::Sender<String>>;

pub enum CoalescingSlot {
    // Primary request responsible for fetching from upstream.
    Leader {
        sender: broadcast::Sender<String>,
        key: String,
    },
    // Duplicate request that subscribes to the active stream.
    Subscriber {
        receiver: broadcast::Receiver<String>,
        #[allow(dead_code)]
        key: String,
    },
}

// RAII guard ensuring in-flight key release even upon task cancellation or panic.
pub struct CoalescingGuard {
    engine: Arc<CoalescingEngine>,
    key: Option<String>,
}

impl CoalescingGuard {
    pub fn new(engine: Arc<CoalescingEngine>, key: String) -> Self {
        Self {
            engine,
            key: Some(key),
        }
    }

    pub fn disarm(&mut self) {
        self.key = None;
    }
}

impl Drop for CoalescingGuard {
    fn drop(&mut self) {
        if let Some(ref key) = self.key {
            self.engine.release(key);
        }
    }
}

pub struct CoalescingEngine {
    in_flight: Arc<InFlightRequests>,
}

impl CoalescingEngine {
    pub fn new(in_flight: Arc<InFlightRequests>) -> Self {
        Self { in_flight }
    }

    // Acquire leader slot for new request, or subscribe if already in flight.
    // Atomically checks and inserts via DashMap Entry API to prevent TOCTOU race conditions.
    pub fn acquire(&self, key: String) -> CoalescingSlot {
        match self.in_flight.entry(key.clone()) {
            Entry::Occupied(entry) => {
                let rx = entry.get().subscribe();
                tracing::info!(key = %key, "Request coalesced with active in-flight stream");
                CoalescingSlot::Subscriber {
                    receiver: rx,
                    key,
                }
            }
            Entry::Vacant(entry) => {
                let (tx, _) = broadcast::channel(128);
                entry.insert(tx.clone());
                tracing::debug!(key = %key, "Created new leader stream for coalescing");
                CoalescingSlot::Leader { sender: tx, key }
            }
        }
    }

    // Release leader key after upstream stream finishes.
    pub fn release(&self, key: &str) {
        self.in_flight.remove(key);
        tracing::debug!(key = %key, "Released coalescing leader key");
    }
}
