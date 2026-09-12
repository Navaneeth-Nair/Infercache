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

pub struct CoalescingEngine {
    in_flight: Arc<InFlightRequests>,
}

impl CoalescingEngine {
    pub fn new(in_flight: Arc<InFlightRequests>) -> Self {
        Self { in_flight }
    }

    // Acquire leader slot for new request, or subscribe if already in flight.
    pub fn acquire(&self, key: String) -> CoalescingSlot {
        if let Some(existing_tx) = self.in_flight.get(&key) {
            let rx = existing_tx.subscribe();
            tracing::info!(key = %key, "Request coalesced with active in-flight stream");
            return CoalescingSlot::Subscriber {
                receiver: rx,
                key,
            };
        }

        let (tx, _) = broadcast::channel(128);
        self.in_flight.insert(key.clone(), tx.clone());
        tracing::debug!(key = %key, "Created new leader stream for coalescing");

        CoalescingSlot::Leader { sender: tx, key }
    }

    // Release leader key after upstream stream finishes.
    pub fn release(&self, key: &str) {
        self.in_flight.remove(key);
        tracing::debug!(key = %key, "Released coalescing leader key");
    }
}
