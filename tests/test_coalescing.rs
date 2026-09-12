use dashmap::DashMap;
use infercache::coalescing::{CoalescingEngine, CoalescingSlot};
use std::sync::Arc;

#[tokio::test]
async fn test_request_coalescing_thundering_herd() {
    let in_flight = Arc::new(DashMap::new());
    let engine = Arc::new(CoalescingEngine::new(in_flight));

    let key = "gpt-4o-mini:sample_prompt_hash".to_string();

    // Request 1: should acquire Leader
    let slot1 = engine.acquire(key.clone());
    let (tx, leader_key) = match slot1 {
        CoalescingSlot::Leader { sender, key } => (sender, key),
        _ => panic!("Expected Leader slot for primary request"),
    };

    // Request 2: duplicate request for same key should acquire Subscriber
    let slot2 = engine.acquire(key.clone());
    let mut rx = match slot2 {
        CoalescingSlot::Subscriber { receiver, .. } => receiver,
        _ => panic!("Expected Subscriber slot for secondary request"),
    };

    // Request 3: third duplicate request should also acquire Subscriber
    let slot3 = engine.acquire(key.clone());
    let mut rx2 = match slot3 {
        CoalescingSlot::Subscriber { receiver, .. } => receiver,
        _ => panic!("Expected Subscriber slot for tertiary request"),
    };

    // Send chunks through leader
    tx.send("data: chunk 1\n\n".to_string()).unwrap();
    tx.send("data: chunk 2\n\n".to_string()).unwrap();

    // Verify subscriber 1 received both chunks
    assert_eq!(rx.recv().await.unwrap(), "data: chunk 1\n\n");
    assert_eq!(rx.recv().await.unwrap(), "data: chunk 2\n\n");

    // Verify subscriber 2 received both chunks
    assert_eq!(rx2.recv().await.unwrap(), "data: chunk 1\n\n");
    assert_eq!(rx2.recv().await.unwrap(), "data: chunk 2\n\n");

    // Release key
    engine.release(&leader_key);

    // After release, new request should acquire a new Leader slot
    let slot_new = engine.acquire(key.clone());
    match slot_new {
        CoalescingSlot::Leader { .. } => {}
        _ => panic!("Expected new Leader slot after release"),
    }
}
