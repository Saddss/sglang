/*
    Truncation-Aware Routing (sticky pool / truncation pool split)

    Motivation: the business layer truncates conversation history with a
    per-turn sliding window. A truncated turn's prefix shifts position, so its
    KV cache is not reusable on ANY worker (only the system segment survives).
    Such requests gain nothing from session affinity, and they actively hurt
    it: a ~context-limit-sized fresh prefill evicts hot prefixes of live
    sessions from the target worker's radix cache.

    Strategy: virtually partition the (homogeneous) worker set into
    - a sticky pool routed by the embedded CacheAwarePolicy (unchanged), and
    - a truncation pool of size K routed by min in-flight load, never inserted
      into the cache-aware tree.

    Requests carry the split signal in body `enable_kv_evict` (plumbed here as
    `SelectWorkerInfo::truncated`).

    Pool sizing is share tracking, not load feedback: K tracks the EWMA share
    of truncation-flagged requests so that per-worker QPS stays comparable
    across pools. Truncated requests are allowed to be slower per request
    (zero cache hit); the controller intentionally never steals extra sticky
    workers for the truncation pool — capacity is a k8s/HPA decision. A
    pressure gauge alerts when the truncation pool runs disproportionately hot.

    Membership is rendezvous-hashed over worker URLs: minimal churn on worker
    add/remove, and multiple router replicas independently converge to the
    same partition for the same (worker set, K) without coordination.
    `labels["pool"] = "truncation" | "sticky"` pins a worker unconditionally.

    Moving a worker sticky→truncation evicts it from the cache-aware tree
    (its sessions rebalance and pay one re-prefill); truncation→sticky is
    free (its radix cache was churn anyway). The controller therefore steps
    K by ±1 with a cooldown between changes.
*/

use std::{
    collections::HashSet,
    hash::{Hash, Hasher},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex, RwLock,
    },
    time::Instant,
};

use async_trait::async_trait;
use smg_mesh::OptionalMeshSyncManager;
use tracing::{debug, info};

use super::{
    utils::PeriodicTask, CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::core::Worker;
use crate::observability::metrics::{metrics_labels, Metrics};

/// Worker label key/values for pinning pool membership.
const POOL_LABEL: &str = "pool";
const POOL_LABEL_TRUNCATION: &str = "truncation";
const POOL_LABEL_STICKY: &str = "sticky";

/// Configuration for the truncation-aware policy.
#[derive(Debug, Clone)]
pub struct TruncationAwareConfig {
    /// EWMA window for the truncated-request share (seconds).
    pub ewma_window_secs: u64,
    /// Controller tick interval (seconds).
    pub tick_secs: u64,
    /// Minimum time between two K adjustments (seconds).
    pub cooldown_secs: u64,
    /// Minimum number of workers that must stay in the sticky pool.
    pub sticky_min: usize,
    /// |K* - K| must exceed this to trigger a resize (0 = any difference).
    pub deadband: usize,
    /// Pressure alert threshold: truncation per-worker inflight >
    /// pressure_ratio x sticky per-worker inflight.
    pub pressure_ratio: f32,
    /// Config for the embedded sticky CacheAwarePolicy.
    pub cache_aware: CacheAwareConfig,
}

impl Default for TruncationAwareConfig {
    fn default() -> Self {
        Self {
            ewma_window_secs: 60,
            tick_secs: 30,
            cooldown_secs: 300,
            sticky_min: 1,
            deadband: 0,
            pressure_ratio: 4.0,
            cache_aware: CacheAwareConfig::default(),
        }
    }
}

/// Shared mutable state between the policy (request path) and the controller
/// (background tick).
#[derive(Debug)]
struct ControllerState {
    total_requests: AtomicU64,
    truncated_requests: AtomicU64,
    /// Counter snapshots from the previous tick (delta = this window's traffic).
    prev_total: AtomicU64,
    prev_truncated: AtomicU64,
    /// EWMA of the truncated share, updated each tick.
    ewma_share: Mutex<f64>,
    /// Current truncation pool size target.
    k: AtomicUsize,
    /// Worker count observed on the most recent selection (controller input).
    last_worker_count: AtomicUsize,
    last_resize_at: Mutex<Option<Instant>>,
    /// Model id observed on the most recent selection (metrics label only).
    model_for_metrics: Mutex<String>,
}

impl ControllerState {
    fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            truncated_requests: AtomicU64::new(0),
            prev_total: AtomicU64::new(0),
            prev_truncated: AtomicU64::new(0),
            ewma_share: Mutex::new(0.0),
            k: AtomicUsize::new(0),
            last_worker_count: AtomicUsize::new(0),
            last_resize_at: Mutex::new(None),
            model_for_metrics: Mutex::new(String::new()),
        }
    }
}

/// Deterministic rendezvous score for a worker URL.
///
/// `DefaultHasher::new()` is SipHash with fixed keys, so every router replica
/// computes the same score for the same URL — replicas converge to the same
/// partition without coordination.
fn rendezvous_score(url: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    url.hash(&mut hasher);
    hasher.finish()
}

fn pool_pin(worker: &dyn Worker) -> Option<&str> {
    worker
        .metadata()
        .labels
        .get(POOL_LABEL)
        .map(|s| s.as_str())
}

/// Partition workers into (sticky_indices, truncation_indices) for a target
/// truncation pool size `k`.
///
/// Pinned workers (`labels["pool"]`) go to their pool unconditionally; the
/// remaining floaters fill the truncation pool up to `k` in descending
/// rendezvous-score order (stable top-K subset under worker churn).
fn partition_workers(workers: &[Arc<dyn Worker>], k: usize) -> (Vec<usize>, Vec<usize>) {
    let mut sticky = Vec::with_capacity(workers.len());
    let mut truncation = Vec::new();
    let mut floaters: Vec<usize> = Vec::with_capacity(workers.len());

    for (idx, worker) in workers.iter().enumerate() {
        match pool_pin(worker.as_ref()) {
            Some(POOL_LABEL_TRUNCATION) => truncation.push(idx),
            Some(POOL_LABEL_STICKY) => sticky.push(idx),
            _ => floaters.push(idx),
        }
    }

    let float_slots = k.saturating_sub(truncation.len()).min(floaters.len());
    floaters.sort_by_key(|&idx| std::cmp::Reverse(rendezvous_score(workers[idx].url())));
    for (pos, idx) in floaters.into_iter().enumerate() {
        if pos < float_slots {
            truncation.push(idx);
        } else {
            sticky.push(idx);
        }
    }

    (sticky, truncation)
}

/// Compute the next K given the current K and the EWMA share.
///
/// K* = clamp(round(n * share), 0, n - sticky_min); step toward it by at most
/// 1, and only when the difference exceeds the deadband.
fn next_k(current_k: usize, share: f64, n: usize, config: &TruncationAwareConfig) -> usize {
    if n == 0 {
        return current_k;
    }
    let upper = n.saturating_sub(config.sticky_min);
    let k_star = (((n as f64) * share).round() as usize).min(upper);
    if k_star.abs_diff(current_k) <= config.deadband && k_star != current_k {
        return current_k;
    }
    match k_star.cmp(&current_k) {
        std::cmp::Ordering::Greater => current_k + 1,
        std::cmp::Ordering::Less => current_k - 1,
        std::cmp::Ordering::Equal => current_k,
    }
}

/// Truncation-aware routing policy (see module docs).
#[derive(Debug)]
pub struct TruncationAwarePolicy {
    config: TruncationAwareConfig,
    sticky: Arc<CacheAwarePolicy>,
    state: Arc<ControllerState>,
    /// URLs currently in the truncation pool; used to diff partition changes
    /// and evict newly-demoted workers from the sticky tree.
    trunc_urls: RwLock<HashSet<String>>,
    _controller: Option<PeriodicTask>,
}

impl TruncationAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(TruncationAwareConfig::default())
    }

    pub fn with_config(config: TruncationAwareConfig) -> Self {
        let sticky = Arc::new(CacheAwarePolicy::with_config(config.cache_aware.clone()));
        let state = Arc::new(ControllerState::new());

        let controller = if config.tick_secs > 0 {
            let state_clone = Arc::clone(&state);
            let config_clone = config.clone();
            Some(PeriodicTask::spawn(
                config.tick_secs,
                "TruncationPoolController",
                move || controller_tick(&state_clone, &config_clone),
            ))
        } else {
            None
        };

        Self {
            config,
            sticky,
            state,
            trunc_urls: RwLock::new(HashSet::new()),
            _controller: controller,
        }
    }

    /// Access the embedded sticky policy (registry init/removal paths).
    pub fn sticky_policy(&self) -> &CacheAwarePolicy {
        &self.sticky
    }

    /// Current truncation pool size target (tests/observability).
    pub fn current_k(&self) -> usize {
        self.state.k.load(Ordering::Relaxed)
    }

    /// Force the truncation pool size (tests only).
    #[cfg(test)]
    fn set_k(&self, k: usize) {
        self.state.k.store(k, Ordering::Relaxed);
    }

    /// Diff the new truncation membership against the previous one, evicting
    /// workers that just moved sticky→truncation from the cache-aware tree so
    /// their sessions rebalance immediately, and re-seeding workers that moved
    /// truncation→sticky into the tree so smallest-tenant routing can pick them.
    fn apply_partition_changes(&self, workers: &[Arc<dyn Worker>], truncation: &[usize]) {
        let new_urls: HashSet<String> = truncation
            .iter()
            .map(|&idx| workers[idx].url().to_string())
            .collect();

        {
            let current = self.trunc_urls.read().unwrap();
            if *current == new_urls {
                return;
            }
        }

        let mut current = self.trunc_urls.write().unwrap();
        // Recheck under the write lock (another request may have raced us).
        if *current == new_urls {
            return;
        }

        for url in new_urls.difference(&current) {
            self.sticky.remove_worker_by_url(url);
            Metrics::record_truncation_pool_evicted_tenant();
            info!("worker {} moved to truncation pool; evicted from sticky tree", url);
        }
        for worker in workers {
            let url = worker.url();
            if current.contains(url) && !new_urls.contains(url) {
                self.sticky.add_worker(worker.as_ref());
                info!("worker {} returned to sticky pool", url);
            }
        }

        *current = new_urls;
    }

    /// Min in-flight load selection within a subset of indices.
    fn select_min_load(workers: &[Arc<dyn Worker>], subset: &[usize]) -> Option<usize> {
        subset
            .iter()
            .copied()
            .min_by_key(|&idx| workers[idx].load())
    }

    fn update_pressure_gauge(
        &self,
        workers: &[Arc<dyn Worker>],
        sticky: &[usize],
        truncation: &[usize],
        model: &str,
    ) {
        if sticky.is_empty() || truncation.is_empty() {
            Metrics::set_truncation_pool_pressure(model, false);
            return;
        }
        let avg = |subset: &[usize]| -> f64 {
            subset.iter().map(|&i| workers[i].load()).sum::<usize>() as f64 / subset.len() as f64
        };
        let pressured = avg(truncation) > (self.config.pressure_ratio as f64) * avg(sticky).max(1.0);
        Metrics::set_truncation_pool_pressure(model, pressured);
    }
}

impl Default for TruncationAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// One controller tick: fold this window's traffic into the EWMA share and
/// step K toward round(N * share) under cooldown/deadband constraints.
fn controller_tick(state: &ControllerState, config: &TruncationAwareConfig) {
    let total = state.total_requests.load(Ordering::Relaxed);
    let truncated = state.truncated_requests.load(Ordering::Relaxed);
    let delta_total = total.saturating_sub(state.prev_total.swap(total, Ordering::Relaxed));
    let delta_truncated =
        truncated.saturating_sub(state.prev_truncated.swap(truncated, Ordering::Relaxed));

    let model = state.model_for_metrics.lock().unwrap().clone();

    let share = {
        let mut ewma = state.ewma_share.lock().unwrap();
        if delta_total > 0 {
            let window_share = delta_truncated as f64 / delta_total as f64;
            // alpha = tick/window: the EWMA time constant tracks ewma_window_secs.
            let alpha =
                (config.tick_secs as f64 / config.ewma_window_secs.max(1) as f64).clamp(0.0, 1.0);
            *ewma = alpha * window_share + (1.0 - alpha) * *ewma;
        }
        *ewma
    };
    if !model.is_empty() {
        Metrics::set_truncated_share(&model, share);
    }

    let n = state.last_worker_count.load(Ordering::Relaxed);
    if n == 0 {
        return;
    }

    let current_k = state.k.load(Ordering::Relaxed);
    let new_k = next_k(current_k, share, n, config);
    if new_k == current_k {
        return;
    }

    {
        let mut last = state.last_resize_at.lock().unwrap();
        if let Some(at) = *last {
            if at.elapsed().as_secs() < config.cooldown_secs {
                return;
            }
        }
        *last = Some(Instant::now());
    }

    state.k.store(new_k, Ordering::Relaxed);
    let direction = if new_k > current_k {
        metrics_labels::POOL_RESIZE_GROW
    } else {
        metrics_labels::POOL_RESIZE_SHRINK
    };
    if !model.is_empty() {
        Metrics::record_truncation_pool_resize(&model, direction);
    }
    info!(
        "truncation pool resized {} -> {} (share {:.4}, n {})",
        current_k, new_k, share, n
    );
}

#[async_trait]
impl LoadBalancingPolicy for TruncationAwarePolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        if workers.is_empty() {
            return None;
        }

        // Feed the controller.
        self.state.total_requests.fetch_add(1, Ordering::Relaxed);
        if info.truncated {
            self.state
                .truncated_requests
                .fetch_add(1, Ordering::Relaxed);
        }
        self.state
            .last_worker_count
            .store(workers.len(), Ordering::Relaxed);
        let model = workers[0].model_id().to_string();
        {
            let mut m = self.state.model_for_metrics.lock().unwrap();
            if *m != model {
                *m = model.clone();
            }
        }

        // Partition under the current K (clamped to this worker set).
        let n = workers.len();
        let k = self
            .state
            .k
            .load(Ordering::Relaxed)
            .min(n.saturating_sub(self.config.sticky_min));
        let (sticky, truncation) = partition_workers(workers, k);
        self.apply_partition_changes(workers, &truncation);
        Metrics::set_truncation_pool_sizes(&model, sticky.len(), truncation.len());
        self.update_pressure_gauge(workers, &sticky, &truncation, &model);

        if info.truncated {
            // Truncated turn: full re-prefill anywhere, so never insert it into
            // the sticky tree; spread by min load inside the truncation pool.
            if !truncation.is_empty() {
                Metrics::record_truncation_route(metrics_labels::TRUNCATION_TRUNC_POOL);
                return Self::select_min_load(workers, &truncation);
            }
            // K=0 (or nothing partitioned): degrade to min load over everyone,
            // still without a tree insert.
            Metrics::record_truncation_route(metrics_labels::TRUNCATION_FALLBACK_ALL);
            let all: Vec<usize> = (0..n).collect();
            return Self::select_min_load(workers, &all);
        }

        if sticky.is_empty() {
            // Availability first: spill unflagged traffic into the truncation
            // pool without a tree insert; it re-sticks once sticky recovers.
            Metrics::record_truncation_route(metrics_labels::TRUNCATION_STICKY_SPILL);
            return Self::select_min_load(workers, &truncation);
        }

        // Delegate to the embedded cache-aware policy on the sticky subset and
        // map the sub-index back to the caller's index space.
        let sticky_workers: Vec<Arc<dyn Worker>> =
            sticky.iter().map(|&idx| Arc::clone(&workers[idx])).collect();
        let sub_idx = self.sticky.select_worker(&sticky_workers, info).await?;
        Metrics::record_truncation_route(metrics_labels::TRUNCATION_STICKY);
        debug!(
            "truncation_aware sticky route: {} (pool {}/{})",
            sticky_workers[sub_idx].url(),
            sticky.len(),
            n
        );
        Some(sticky[sub_idx])
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        self.sticky.on_request_complete(worker_url, success);
    }

    fn name(&self) -> &'static str {
        "truncation_aware"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn update_loads(&self, loads: &std::collections::HashMap<String, isize>) {
        self.sticky.update_loads(loads);
    }

    fn set_mesh_sync(&mut self, mesh_sync: OptionalMeshSyncManager) {
        if let Some(sticky) = Arc::get_mut(&mut self.sticky) {
            sticky.set_mesh_sync(mesh_sync);
        }
    }

    fn reset(&self) {
        self.sticky.reset();
        self.state.total_requests.store(0, Ordering::Relaxed);
        self.state.truncated_requests.store(0, Ordering::Relaxed);
        self.state.prev_total.store(0, Ordering::Relaxed);
        self.state.prev_truncated.store(0, Ordering::Relaxed);
        *self.state.ewma_share.lock().unwrap() = 0.0;
        self.state.k.store(0, Ordering::Relaxed);
        *self.state.last_resize_at.lock().unwrap() = None;
        self.trunc_urls.write().unwrap().clear();
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    fn make_worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .build(),
        )
    }

    fn make_pinned_worker(url: &str, pool: &str) -> Arc<dyn Worker> {
        let mut labels = HashMap::new();
        labels.insert(POOL_LABEL.to_string(), pool.to_string());
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .labels(labels)
                .build(),
        )
    }

    fn make_workers(n: usize) -> Vec<Arc<dyn Worker>> {
        (0..n)
            .map(|i| make_worker(&format!("http://w{}:8000", i)))
            .collect()
    }

    fn test_config() -> TruncationAwareConfig {
        TruncationAwareConfig {
            tick_secs: 0, // no background controller in tests; tick manually
            ..Default::default()
        }
    }

    // ---- partition_workers ----

    #[test]
    fn test_partition_deterministic_and_sized() {
        let workers = make_workers(8);
        let (s1, t1) = partition_workers(&workers, 3);
        let (s2, t2) = partition_workers(&workers, 3);
        assert_eq!(s1, s2);
        assert_eq!(t1, t2);
        assert_eq!(t1.len(), 3);
        assert_eq!(s1.len(), 5);
    }

    #[test]
    fn test_partition_minimal_churn_on_k_change() {
        let workers = make_workers(8);
        let (_, t3) = partition_workers(&workers, 3);
        let (_, t4) = partition_workers(&workers, 4);
        // Growing K by 1 keeps the previous members and adds exactly one.
        let set3: HashSet<_> = t3.iter().collect();
        let set4: HashSet<_> = t4.iter().collect();
        assert!(set3.is_subset(&set4));
        assert_eq!(set4.len(), set3.len() + 1);
    }

    #[test]
    fn test_partition_minimal_churn_on_worker_removal() {
        let workers = make_workers(8);
        let (_, t_before) = partition_workers(&workers, 3);
        let removed_url = workers[0].url().to_string();
        let survivors: Vec<Arc<dyn Worker>> = workers[1..].to_vec();
        let (_, t_after) = partition_workers(&survivors, 3);

        let before_urls: HashSet<String> = t_before
            .iter()
            .map(|&i| workers[i].url().to_string())
            .collect();
        let after_urls: HashSet<String> = t_after
            .iter()
            .map(|&i| survivors[i].url().to_string())
            .collect();
        // Members that survive the removal keep their assignment.
        let stable: HashSet<_> = before_urls
            .iter()
            .filter(|u| **u != removed_url)
            .collect();
        assert!(stable.iter().all(|u| after_urls.contains(*u)));
    }

    #[test]
    fn test_partition_respects_pins() {
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_pinned_worker("http://pin-t:8000", POOL_LABEL_TRUNCATION),
            make_pinned_worker("http://pin-s:8000", POOL_LABEL_STICKY),
            make_worker("http://float-1:8000"),
            make_worker("http://float-2:8000"),
        ];
        // k=1 is already satisfied by the pinned truncation worker: floaters stay sticky.
        let (sticky, truncation) = partition_workers(&workers, 1);
        assert_eq!(truncation, vec![0]);
        assert_eq!(sticky.len(), 3);

        // k=2: one floater joins the pinned truncation worker; pinned sticky never moves.
        let (sticky, truncation) = partition_workers(&workers, 2);
        assert_eq!(truncation.len(), 2);
        assert!(truncation.contains(&0));
        assert!(sticky.contains(&1));
    }

    // ---- next_k ----

    #[test]
    fn test_next_k_steps_toward_target_by_one() {
        let cfg = test_config();
        // share 0.5 over 8 workers → K* = 4; step from 0 is 1.
        assert_eq!(next_k(0, 0.5, 8, &cfg), 1);
        assert_eq!(next_k(3, 0.5, 8, &cfg), 4);
        assert_eq!(next_k(4, 0.5, 8, &cfg), 4);
        // shrink
        assert_eq!(next_k(4, 0.0, 8, &cfg), 3);
    }

    #[test]
    fn test_next_k_respects_sticky_min() {
        let cfg = test_config();
        // share 1.0 would want K = n, but sticky_min=1 caps at n-1.
        assert_eq!(next_k(6, 1.0, 8, &cfg), 7);
        assert_eq!(next_k(7, 1.0, 8, &cfg), 7);
    }

    #[test]
    fn test_next_k_deadband() {
        let cfg = TruncationAwareConfig {
            deadband: 1,
            ..test_config()
        };
        // K* = 4, current 3 → |diff| = 1 <= deadband → hold.
        assert_eq!(next_k(3, 0.5, 8, &cfg), 3);
        // K* = 4, current 2 → |diff| = 2 > deadband → step.
        assert_eq!(next_k(2, 0.5, 8, &cfg), 3);
    }

    #[test]
    fn test_controller_tick_converges_and_cooldown() {
        let cfg = TruncationAwareConfig {
            ewma_window_secs: 30,
            tick_secs: 30, // alpha = 1: window share adopted immediately
            cooldown_secs: 3600,
            ..test_config()
        };
        let state = ControllerState::new();
        state.last_worker_count.store(8, Ordering::Relaxed);

        // Window 1: 50% truncated → K* = 4, K steps 0 → 1.
        state.total_requests.store(100, Ordering::Relaxed);
        state.truncated_requests.store(50, Ordering::Relaxed);
        controller_tick(&state, &cfg);
        assert_eq!(state.k.load(Ordering::Relaxed), 1);

        // Window 2: same share, but cooldown (1h) blocks the next step.
        state.total_requests.store(200, Ordering::Relaxed);
        state.truncated_requests.store(100, Ordering::Relaxed);
        controller_tick(&state, &cfg);
        assert_eq!(state.k.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_controller_tick_no_traffic_keeps_share() {
        let cfg = TruncationAwareConfig {
            cooldown_secs: 0,
            ..test_config()
        };
        let state = ControllerState::new();
        state.last_worker_count.store(8, Ordering::Relaxed);
        *state.ewma_share.lock().unwrap() = 0.5;

        // No new requests this window: share must not decay toward 0.
        controller_tick(&state, &cfg);
        assert!((*state.ewma_share.lock().unwrap() - 0.5).abs() < 1e-9);
    }

    // ---- routing branches ----

    #[tokio::test]
    async fn test_truncated_routes_to_trunc_pool_min_load() {
        let policy = TruncationAwarePolicy::with_config(test_config());
        let workers = make_workers(4);
        policy.set_k(2);

        let (_, truncation) = partition_workers(&workers, 2);
        // Load the first trunc member so min-load picks the other.
        workers[truncation[0]].increment_load();

        let info = SelectWorkerInfo {
            request_text: Some("hello"),
            truncated: true,
            ..Default::default()
        };
        let idx = policy.select_worker(&workers, &info).await.unwrap();
        assert_eq!(idx, truncation[1]);
    }

    #[tokio::test]
    async fn test_truncated_never_inserted_into_sticky_tree() {
        let policy = TruncationAwarePolicy::with_config(test_config());
        let workers = make_workers(4);
        policy.set_k(1);
        policy.sticky_policy().init_workers(&workers);

        let text = "truncated conversation prefix that must not stick";
        let info = SelectWorkerInfo {
            request_text: Some(text),
            truncated: true,
            ..Default::default()
        };
        let trunc_idx = policy.select_worker(&workers, &info).await.unwrap();

        // The same text as a NON-truncated request must not show affinity to
        // the truncation worker: the sticky tree never saw it.
        let info_sticky = SelectWorkerInfo {
            request_text: Some(text),
            truncated: false,
            ..Default::default()
        };
        let sticky_idx = policy.select_worker(&workers, &info_sticky).await.unwrap();
        assert_ne!(sticky_idx, trunc_idx);
    }

    #[tokio::test]
    async fn test_truncated_with_k0_falls_back_to_all() {
        let policy = TruncationAwarePolicy::with_config(test_config());
        let workers = make_workers(4);
        // K stays 0: truncated requests spread over everyone by min load.
        for w in &workers[..3] {
            w.increment_load();
        }
        let info = SelectWorkerInfo {
            request_text: Some("hi"),
            truncated: true,
            ..Default::default()
        };
        let idx = policy.select_worker(&workers, &info).await.unwrap();
        assert_eq!(idx, 3);
    }

    #[tokio::test]
    async fn test_sticky_traffic_uses_cache_aware_and_sticks() {
        let policy = TruncationAwarePolicy::with_config(test_config());
        let workers = make_workers(4);
        policy.set_k(1);
        policy.sticky_policy().init_workers(&workers);

        let info = SelectWorkerInfo {
            request_text: Some("session prefix for stickiness test"),
            truncated: false,
            ..Default::default()
        };
        let first = policy.select_worker(&workers, &info).await.unwrap();
        let second = policy.select_worker(&workers, &info).await.unwrap();
        assert_eq!(first, second, "same prefix must stick to the same worker");
    }

    #[tokio::test]
    async fn test_sticky_spill_when_all_pinned_truncation() {
        let policy = TruncationAwarePolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_pinned_worker("http://t1:8000", POOL_LABEL_TRUNCATION),
            make_pinned_worker("http://t2:8000", POOL_LABEL_TRUNCATION),
        ];
        workers[0].increment_load();
        let info = SelectWorkerInfo {
            request_text: Some("hi"),
            truncated: false,
            ..Default::default()
        };
        // Sticky pool is empty → spill into truncation pool by min load.
        let idx = policy.select_worker(&workers, &info).await.unwrap();
        assert_eq!(idx, 1);
    }

    #[tokio::test]
    async fn test_partition_change_evicts_from_sticky_tree() {
        let policy = TruncationAwarePolicy::with_config(test_config());
        let workers = make_workers(4);
        policy.sticky_policy().init_workers(&workers);

        // Establish stickiness with K=0 (all workers sticky).
        let text = "conversation whose tenant will move pools";
        let info = SelectWorkerInfo {
            request_text: Some(text),
            truncated: false,
            ..Default::default()
        };
        let tenant = policy.select_worker(&workers, &info).await.unwrap();

        // Pin K so that the tenant's worker lands in the truncation pool:
        // grow K until the tenant is a member.
        let mut k = 1;
        loop {
            let (_, truncation) = partition_workers(&workers, k);
            if truncation.contains(&tenant) {
                break;
            }
            k += 1;
            assert!(k < workers.len(), "tenant must eventually join the pool");
        }
        policy.set_k(k);

        // Next sticky request with the same prefix must NOT land on the old
        // tenant anymore: it was evicted from the tree and removed from the
        // sticky subset.
        let new_idx = policy.select_worker(&workers, &info).await.unwrap();
        assert_ne!(new_idx, tenant);
    }

    #[tokio::test]
    async fn test_empty_workers_returns_none() {
        let policy = TruncationAwarePolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![];
        let info = SelectWorkerInfo {
            truncated: true,
            ..Default::default()
        };
        assert!(policy.select_worker(&workers, &info).await.is_none());
    }
}
