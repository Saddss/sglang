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

    Membership is authoritative per controller tick, not per request: the
    request path only records which workers it sees and routes within the
    current membership snapshot, while the tick recomputes the partition from
    the known worker set. The known set is synced authoritatively by the
    registry via `init_workers` (worker add/remove) and refreshed by the
    request path; the TTL prune is a backstop that only runs in windows that
    carried traffic — an idle window refreshes no stamps and must not
    collapse the partition. A transient health flap therefore narrows the
    routable subset but cannot reshuffle membership or evict sticky-tree
    tenants mid-flap; tree evictions happen only on tick, after the flap
    either heals or outlives the TTL.

    Moving a worker sticky→truncation evicts it from the cache-aware tree
    (its sessions rebalance and pay one re-prefill); truncation→sticky is
    free (its radix cache was churn anyway). The controller therefore steps
    K by ±1 with a cooldown between changes.

    All controller state (counters, EWMA share, K, seen workers, membership)
    is keyed by model id, so one policy instance serves multi-model
    deployments with independent pools per model.
*/

use std::{
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex, OnceLock, RwLock,
    },
    time::{Duration, Instant},
};

use async_trait::async_trait;
use dashmap::DashMap;
use smg_mesh::OptionalMeshSyncManager;
use tracing::{info, warn};

use super::{
    get_healthy_worker_indices, normalize_model_key, utils::PeriodicTask, CacheAwareConfig,
    CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::core::Worker;
use crate::observability::metrics::{metrics_labels, Metrics};

/// Worker label key/values for pinning pool membership.
const POOL_LABEL: &str = "pool";
const POOL_LABEL_TRUNCATION: &str = "truncation";
const POOL_LABEL_STICKY: &str = "sticky";

/// How often (at most) the request path refreshes a worker's last-seen stamp.
const KNOWN_TOUCH_INTERVAL: Duration = Duration::from_secs(1);

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

impl TruncationAwareConfig {
    /// TTL after which an unseen worker is dropped from the known set
    /// (approximates registry removal without registry access).
    fn known_ttl(&self) -> Duration {
        Duration::from_secs((3 * self.tick_secs).max(90))
    }
}

/// A worker observed on the request path; kept so the tick can re-add it to
/// the sticky tree (needs the object, not just the URL) and read its load.
struct KnownWorker {
    worker: Arc<dyn Worker>,
    last_seen: Instant,
}

/// Per-model controller state. Pools, counters, and K are independent across
/// models served by the same policy instance.
struct ModelState {
    total_requests: AtomicU64,
    truncated_requests: AtomicU64,
    /// Counter snapshots from the previous tick (delta = this window's traffic).
    prev_total: AtomicU64,
    prev_truncated: AtomicU64,
    /// EWMA of the truncated share, updated each tick. `None` until the
    /// first window with traffic (warm start: the first observation is used
    /// as-is instead of blending from zero).
    ewma_share: Mutex<Option<f64>>,
    /// Current truncation pool size target.
    k: AtomicUsize,
    last_resize_at: Mutex<Option<Instant>>,
    /// Workers recently seen on the request path (URL → object + stamp).
    known: DashMap<String, KnownWorker>,
    /// Authoritative truncation membership, recomputed on tick only.
    trunc_urls: RwLock<HashSet<String>>,
}

impl ModelState {
    fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            truncated_requests: AtomicU64::new(0),
            prev_total: AtomicU64::new(0),
            prev_truncated: AtomicU64::new(0),
            ewma_share: Mutex::new(None),
            k: AtomicUsize::new(0),
            last_resize_at: Mutex::new(None),
            known: DashMap::new(),
            trunc_urls: RwLock::new(HashSet::new()),
        }
    }
}

/// Deterministic rendezvous score for a worker URL.
///
/// `DefaultHasher::new()` is SipHash with fixed keys, so every router replica
/// computes the same score for the same URL.
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

/// Compute the authoritative truncation membership for a target size `k`.
///
/// Pinned entries go to their pool unconditionally; floaters fill the
/// remaining slots in descending rendezvous-score order (stable top-K subset
/// under worker churn).
fn compute_trunc_set(entries: &[(String, Option<String>)], k: usize) -> HashSet<String> {
    let mut trunc: HashSet<String> = HashSet::new();
    let mut floaters: Vec<&String> = Vec::with_capacity(entries.len());

    for (url, pin) in entries {
        match pin.as_deref() {
            Some(POOL_LABEL_TRUNCATION) => {
                trunc.insert(url.clone());
            }
            Some(POOL_LABEL_STICKY) => {}
            _ => floaters.push(url),
        }
    }

    let float_slots = k.saturating_sub(trunc.len()).min(floaters.len());
    floaters.sort_by_key(|url| std::cmp::Reverse(rendezvous_score(url)));
    for url in floaters.into_iter().take(float_slots) {
        trunc.insert(url.clone());
    }

    trunc
}

/// Whether a worker belongs to the truncation pool: pins win over the
/// tick-computed membership snapshot (covers workers seen before their first
/// tick).
fn is_trunc_member(worker: &dyn Worker, trunc_urls: &HashSet<String>) -> bool {
    match pool_pin(worker) {
        Some(POOL_LABEL_TRUNCATION) => true,
        Some(POOL_LABEL_STICKY) => false,
        _ => trunc_urls.contains(worker.url()),
    }
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

/// Fold this window's traffic into the EWMA share and step K under
/// cooldown/deadband constraints. Membership is applied separately by
/// `repartition`.
fn update_share_and_k(model: &str, state: &ModelState, config: &TruncationAwareConfig) {
    let total = state.total_requests.load(Ordering::Relaxed);
    let truncated = state.truncated_requests.load(Ordering::Relaxed);
    let delta_total = total.saturating_sub(state.prev_total.swap(total, Ordering::Relaxed));
    let delta_truncated =
        truncated.saturating_sub(state.prev_truncated.swap(truncated, Ordering::Relaxed));

    let share = {
        let mut ewma = state.ewma_share.lock().unwrap();
        if delta_total > 0 {
            let window_share = delta_truncated as f64 / delta_total as f64;
            // alpha = tick/window: the EWMA time constant tracks ewma_window_secs.
            let alpha =
                (config.tick_secs as f64 / config.ewma_window_secs.max(1) as f64).clamp(0.0, 1.0);
            *ewma = Some(match *ewma {
                // Warm start: the first traffic window is the best available
                // estimate. Blending from an implicit 0 undersized the pool
                // for several minutes after startup (observed live at high
                // truncation share).
                None => window_share,
                Some(prev) => alpha * window_share + (1.0 - alpha) * prev,
            });
        }
        ewma.unwrap_or(0.0)
    };
    Metrics::set_truncated_share(model, share);

    let n = state.known.len();
    if n == 0 {
        return;
    }

    // Capacity-forced clamp when the worker set shrinks: not a controller
    // step (no cooldown), otherwise a stale large K would re-apply in one
    // shot when capacity recovers, evicting several workers in a single tick.
    let upper = n.saturating_sub(config.sticky_min);
    let mut current_k = state.k.load(Ordering::Relaxed);
    if current_k > upper {
        state.k.store(upper, Ordering::Relaxed);
        Metrics::record_truncation_pool_resize(
            model,
            metrics_labels::POOL_RESIZE_SHRINK,
            metrics_labels::POOL_RESIZE_REASON_CLAMP,
        );
        info!(
            "truncation pool ({}) clamped {} -> {} (n {}, sticky_min {})",
            model, current_k, upper, n, config.sticky_min
        );
        current_k = upper;
    }

    let new_k = next_k(current_k, share, n, config);
    if new_k == current_k {
        return;
    }

    // Cooldown damps membership churn (a grow step evicts a sticky worker's
    // sessions). Exception: while the truncation pool is measurably
    // overloaded (pressure bit), grow steps are allowed every tick so an
    // undersized pool converges within ticks instead of cooldown periods.
    let growing = new_k > current_k;
    let mut pressure_bypass = false;
    {
        let mut last = state.last_resize_at.lock().unwrap();
        let in_cooldown = last
            .map(|at| at.elapsed().as_secs() < config.cooldown_secs)
            .unwrap_or(false);
        if in_cooldown {
            if !(growing && pool_pressured(state, config)) {
                return;
            }
            pressure_bypass = true;
        }
        *last = Some(Instant::now());
    }

    state.k.store(new_k, Ordering::Relaxed);
    let direction = if growing {
        metrics_labels::POOL_RESIZE_GROW
    } else {
        metrics_labels::POOL_RESIZE_SHRINK
    };
    let reason = if pressure_bypass {
        metrics_labels::POOL_RESIZE_REASON_PRESSURE
    } else {
        metrics_labels::POOL_RESIZE_REASON_SHARE
    };
    Metrics::record_truncation_pool_resize(model, direction, reason);
    info!(
        "truncation pool ({}) resized {} -> {} (share {:.4}, n {})",
        model, current_k, new_k, share, n
    );
}

/// Recompute the authoritative membership from the known worker set and apply
/// the diff to the sticky tree: newly-promoted workers are evicted (their
/// sessions rebalance), returning workers are re-seeded.
fn repartition(state: &ModelState, sticky: &CacheAwarePolicy, config: &TruncationAwareConfig) {
    let entries: Vec<(String, Option<String>)> = state
        .known
        .iter()
        .map(|e| {
            (
                e.key().clone(),
                pool_pin(e.value().worker.as_ref()).map(str::to_string),
            )
        })
        .collect();
    let n = entries.len();
    let k = state
        .k
        .load(Ordering::Relaxed)
        .min(n.saturating_sub(config.sticky_min));
    let new_set = compute_trunc_set(&entries, k);

    let mut current = state.trunc_urls.write().unwrap();
    if *current == new_set {
        return;
    }

    for url in new_set.difference(&current) {
        sticky.remove_worker_by_url(url);
        Metrics::record_truncation_pool_evicted_tenant();
        info!("worker {} moved to truncation pool; evicted from sticky tree", url);
    }
    for entry in state.known.iter() {
        let url = entry.key();
        if current.contains(url) && !new_set.contains(url) {
            sticky.add_worker(entry.value().worker.as_ref());
            info!("worker {} returned to sticky pool", url);
        }
    }

    *current = new_set;
}

/// Pool sizes and average per-worker in-flight load (sticky, truncation)
/// under the current membership snapshot.
fn pool_inflight_averages(state: &ModelState) -> (usize, f64, usize, f64) {
    let trunc_urls = state.trunc_urls.read().unwrap();
    let (mut s_count, mut s_load, mut t_count, mut t_load) = (0usize, 0usize, 0usize, 0usize);
    for entry in state.known.iter() {
        let worker = entry.value().worker.as_ref();
        if is_trunc_member(worker, &trunc_urls) {
            t_count += 1;
            t_load += worker.load();
        } else {
            s_count += 1;
            s_load += worker.load();
        }
    }
    let s_avg = if s_count > 0 { s_load as f64 / s_count as f64 } else { 0.0 };
    let t_avg = if t_count > 0 { t_load as f64 / t_count as f64 } else { 0.0 };
    (s_count, s_avg, t_count, t_avg)
}

/// Truncation pool overload bit: per-worker inflight beyond
/// `pressure_ratio` x the sticky pool's per-worker inflight.
fn pool_pressured(state: &ModelState, config: &TruncationAwareConfig) -> bool {
    let (s_count, s_avg, t_count, t_avg) = pool_inflight_averages(state);
    s_count > 0 && t_count > 0 && t_avg > (config.pressure_ratio as f64) * s_avg
}

/// Publish per-pool sizes, per-worker inflight averages, and the pressure bit.
fn publish_pool_gauges(model: &str, state: &ModelState, config: &TruncationAwareConfig) {
    let (s_count, s_avg, t_count, t_avg) = pool_inflight_averages(state);
    Metrics::set_truncation_pool_sizes(model, s_count, t_count);
    Metrics::set_pool_inflight_per_worker(model, metrics_labels::POOL_STICKY, s_avg);
    Metrics::set_pool_inflight_per_worker(model, metrics_labels::POOL_TRUNCATION, t_avg);

    let pressured =
        s_count > 0 && t_count > 0 && t_avg > (config.pressure_ratio as f64) * s_avg;
    Metrics::set_truncation_pool_pressure(model, pressured);
}

fn tick_model(model: &str, state: &ModelState, sticky: &CacheAwarePolicy, config: &TruncationAwareConfig) {
    // TTL-prune only when the window carried traffic: the request path
    // touches every routable worker, so under traffic a stale stamp means
    // the worker left the registry. An idle window refreshes nothing and
    // must not collapse the partition (observed live: pools dropped to 0/0
    // after an idle gap and truncated traffic then spilled into the sticky
    // pool until the next tick). Registry changes also sync the known set
    // directly via `init_workers`.
    let had_traffic = state.total_requests.load(Ordering::Relaxed)
        != state.prev_total.load(Ordering::Relaxed);
    if had_traffic {
        let ttl = config.known_ttl();
        state.known.retain(|_, e| e.last_seen.elapsed() < ttl);
    }

    update_share_and_k(model, state, config);
    repartition(state, sticky, config);
    publish_pool_gauges(model, state, config);
}

/// Truncation-aware routing policy (see module docs).
pub struct TruncationAwarePolicy {
    config: TruncationAwareConfig,
    sticky: Arc<CacheAwarePolicy>,
    models: Arc<DashMap<String, Arc<ModelState>>>,
    /// Spawned lazily on first selection so `set_mesh_sync` (startup path,
    /// needs exclusive access to `sticky`) still works after construction.
    controller: OnceLock<PeriodicTask>,
}

impl std::fmt::Debug for TruncationAwarePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TruncationAwarePolicy")
            .field("config", &self.config)
            .field("models", &self.models.len())
            .finish()
    }
}

impl TruncationAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(TruncationAwareConfig::default())
    }

    pub fn with_config(config: TruncationAwareConfig) -> Self {
        let sticky = Arc::new(CacheAwarePolicy::with_config(config.cache_aware.clone()));
        Self {
            config,
            sticky,
            models: Arc::new(DashMap::new()),
            controller: OnceLock::new(),
        }
    }

    /// Access the embedded sticky policy (registry init/removal paths).
    pub fn sticky_policy(&self) -> &CacheAwarePolicy {
        &self.sticky
    }

    /// Seed/re-seed the sticky tree from the registry worker set, then
    /// re-apply the current partition: `CacheAwarePolicy::init_workers` adds
    /// every worker to the tree, which must not resurrect truncation members.
    pub fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        self.sticky.init_workers(workers);
        for worker in workers {
            if pool_pin(worker.as_ref()) == Some(POOL_LABEL_TRUNCATION) {
                self.sticky.remove_worker_by_url(worker.url());
            }
        }
        for entry in self.models.iter() {
            for url in entry.value().trunc_urls.read().unwrap().iter() {
                self.sticky.remove_worker_by_url(url);
            }
        }

        // Authoritative known-set sync: the registry calls this on worker
        // add/remove with the model's full worker list, so the controller
        // learns about registry changes directly instead of relying on
        // request-path touches plus the TTL backstop. (A concurrent request
        // may transiently re-insert a just-removed worker via touch_known;
        // the TTL prune clears it on the next traffic-carrying tick.)
        let mut by_model: HashMap<String, Vec<&Arc<dyn Worker>>> = HashMap::new();
        for worker in workers {
            by_model
                .entry(normalize_model_key(worker.model_id()).to_string())
                .or_default()
                .push(worker);
        }
        for (model_key, group) in by_model {
            let state = self.model_state(&model_key);
            let urls: HashSet<&str> = group.iter().map(|w| w.url()).collect();
            state.known.retain(|url, _| urls.contains(url.as_str()));
            for worker in group {
                state.known.insert(
                    worker.url().to_string(),
                    KnownWorker {
                        worker: Arc::clone(worker),
                        last_seen: Instant::now(),
                    },
                );
            }
        }
    }

    fn ensure_controller(&self) {
        if self.config.tick_secs == 0 {
            return;
        }
        self.controller.get_or_init(|| {
            let models = Arc::clone(&self.models);
            let sticky = Arc::clone(&self.sticky);
            let config = self.config.clone();
            PeriodicTask::spawn(config.tick_secs, "TruncationPoolController", move || {
                for entry in models.iter() {
                    tick_model(entry.key(), entry.value(), &sticky, &config);
                }
            })
        });
    }

    fn model_state(&self, model_key: &str) -> Arc<ModelState> {
        // Fast path avoids the owned-key allocation on every request.
        if let Some(state) = self.models.get(model_key) {
            return Arc::clone(&state);
        }
        self.models
            .entry(model_key.to_string())
            .or_insert_with(|| Arc::new(ModelState::new()))
            .clone()
    }

    /// Refresh last-seen stamps for the workers observed on this request
    /// (throttled to once per KNOWN_TOUCH_INTERVAL per worker).
    fn touch_known(state: &ModelState, workers: &[Arc<dyn Worker>]) {
        for worker in workers {
            let fresh = state
                .known
                .get(worker.url())
                .map(|e| e.last_seen.elapsed() < KNOWN_TOUCH_INTERVAL)
                .unwrap_or(false);
            if !fresh {
                state.known.insert(
                    worker.url().to_string(),
                    KnownWorker {
                        worker: Arc::clone(worker),
                        last_seen: Instant::now(),
                    },
                );
            }
        }
    }

    /// Pool label of a worker under the current membership snapshot
    /// (used by the HTTP router for per-pool QPS/latency metrics).
    pub fn pool_label_for(&self, worker: &dyn Worker) -> &'static str {
        let state = self.model_state(normalize_model_key(worker.model_id()));
        let is_trunc = {
            let trunc_urls = state.trunc_urls.read().unwrap();
            is_trunc_member(worker, &trunc_urls)
        };
        if is_trunc {
            metrics_labels::POOL_TRUNCATION
        } else {
            metrics_labels::POOL_STICKY
        }
    }

    /// Min in-flight load selection within a subset of indices.
    fn select_min_load(workers: &[Arc<dyn Worker>], subset: &[usize]) -> Option<usize> {
        subset
            .iter()
            .copied()
            .min_by_key(|&idx| workers[idx].load())
    }

    #[cfg(test)]
    fn test_state(&self, workers: &[Arc<dyn Worker>]) -> Arc<ModelState> {
        let state = self.model_state(normalize_model_key(workers[0].model_id()));
        for worker in workers {
            state.known.insert(
                worker.url().to_string(),
                KnownWorker {
                    worker: Arc::clone(worker),
                    last_seen: Instant::now(),
                },
            );
        }
        state
    }

    #[cfg(test)]
    fn test_force_partition(&self, workers: &[Arc<dyn Worker>], k: usize) {
        let state = self.test_state(workers);
        state.k.store(k, Ordering::Relaxed);
        repartition(&state, &self.sticky, &self.config);
    }
}

impl Default for TruncationAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LoadBalancingPolicy for TruncationAwarePolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }
        self.ensure_controller();

        let state = self.model_state(normalize_model_key(workers[0].model_id()));
        state.total_requests.fetch_add(1, Ordering::Relaxed);
        if info.truncated {
            state.truncated_requests.fetch_add(1, Ordering::Relaxed);
        }
        Self::touch_known(&state, workers);

        // Split within the tick-computed membership snapshot; the request path
        // never mutates membership (see module docs on health flaps).
        let (sticky_idx, trunc_idx): (Vec<usize>, Vec<usize>) = {
            let trunc_urls = state.trunc_urls.read().unwrap();
            healthy
                .iter()
                .partition(|&&idx| !is_trunc_member(workers[idx].as_ref(), &trunc_urls))
        };

        if info.truncated {
            // Truncated turn: full re-prefill anywhere, so never insert it into
            // the sticky tree; spread by min load inside the truncation pool.
            if !trunc_idx.is_empty() {
                Metrics::record_truncation_route(metrics_labels::TRUNCATION_TRUNC_POOL);
                return Self::select_min_load(workers, &trunc_idx);
            }
            // K=0 or no healthy truncation member present: degrade to min load
            // over the healthy set, still without a tree insert.
            Metrics::record_truncation_route(metrics_labels::TRUNCATION_FALLBACK_ALL);
            return Self::select_min_load(workers, &healthy);
        }

        if sticky_idx.is_empty() {
            // Availability first: spill unflagged traffic into the truncation
            // pool without a tree insert; it re-sticks once sticky recovers.
            Metrics::record_truncation_route(metrics_labels::TRUNCATION_STICKY_SPILL);
            return Self::select_min_load(workers, &trunc_idx);
        }

        // Delegate to the embedded cache-aware policy on the sticky subset and
        // map the sub-index back to the caller's index space.
        let sticky_workers: Vec<Arc<dyn Worker>> = sticky_idx
            .iter()
            .map(|&idx| Arc::clone(&workers[idx]))
            .collect();
        let sub_idx = self.sticky.select_worker(&sticky_workers, info).await?;
        Metrics::record_truncation_route(metrics_labels::TRUNCATION_STICKY);
        Some(sticky_idx[sub_idx])
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

    fn update_loads(&self, loads: &HashMap<String, isize>) {
        self.sticky.update_loads(loads);
    }

    fn set_mesh_sync(&mut self, mesh_sync: OptionalMeshSyncManager) {
        match Arc::get_mut(&mut self.sticky) {
            Some(sticky) => sticky.set_mesh_sync(mesh_sync),
            // Only reachable if called after the controller captured its clone
            // (i.e. after traffic started); mesh is wired at startup in practice.
            None => warn!("set_mesh_sync ignored: truncation_aware controller already running"),
        }
    }

    fn reset(&self) {
        self.sticky.reset();
        self.models.clear();
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

    fn entries(workers: &[Arc<dyn Worker>]) -> Vec<(String, Option<String>)> {
        workers
            .iter()
            .map(|w| {
                (
                    w.url().to_string(),
                    pool_pin(w.as_ref()).map(str::to_string),
                )
            })
            .collect()
    }

    fn test_config() -> TruncationAwareConfig {
        TruncationAwareConfig {
            tick_secs: 0, // no background controller in tests; tick manually
            ..Default::default()
        }
    }

    fn seeded_state(n_workers: usize) -> (ModelState, Vec<Arc<dyn Worker>>) {
        let state = ModelState::new();
        let workers = make_workers(n_workers);
        for w in &workers {
            state.known.insert(
                w.url().to_string(),
                KnownWorker {
                    worker: Arc::clone(w),
                    last_seen: Instant::now(),
                },
            );
        }
        (state, workers)
    }

    // ---- compute_trunc_set ----

    #[test]
    fn test_partition_deterministic_and_sized() {
        let workers = make_workers(8);
        let t1 = compute_trunc_set(&entries(&workers), 3);
        let t2 = compute_trunc_set(&entries(&workers), 3);
        assert_eq!(t1, t2);
        assert_eq!(t1.len(), 3);
    }

    #[test]
    fn test_partition_minimal_churn_on_k_change() {
        let workers = make_workers(8);
        let t3 = compute_trunc_set(&entries(&workers), 3);
        let t4 = compute_trunc_set(&entries(&workers), 4);
        // Growing K by 1 keeps the previous members and adds exactly one.
        assert!(t3.is_subset(&t4));
        assert_eq!(t4.len(), t3.len() + 1);
    }

    #[test]
    fn test_partition_minimal_churn_on_worker_removal() {
        let workers = make_workers(8);
        let before = compute_trunc_set(&entries(&workers), 3);
        let removed_url = workers[0].url().to_string();
        let after = compute_trunc_set(&entries(&workers[1..]), 3);

        // Members that survive the removal keep their assignment.
        for url in before.iter().filter(|u| **u != removed_url) {
            assert!(after.contains(url));
        }
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
        let t = compute_trunc_set(&entries(&workers), 1);
        assert_eq!(t.len(), 1);
        assert!(t.contains("http://pin-t:8000"));

        // k=2: one floater joins; the pinned sticky worker never moves.
        let t = compute_trunc_set(&entries(&workers), 2);
        assert_eq!(t.len(), 2);
        assert!(t.contains("http://pin-t:8000"));
        assert!(!t.contains("http://pin-s:8000"));
    }

    // ---- next_k / controller ----

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
    fn test_controller_converges_and_cooldown() {
        let cfg = TruncationAwareConfig {
            ewma_window_secs: 30,
            tick_secs: 30, // alpha = 1: window share adopted immediately
            cooldown_secs: 3600,
            ..test_config()
        };
        let (state, _workers) = seeded_state(8);

        // Window 1: 50% truncated → K* = 4, K steps 0 → 1.
        state.total_requests.store(100, Ordering::Relaxed);
        state.truncated_requests.store(50, Ordering::Relaxed);
        update_share_and_k("m", &state, &cfg);
        assert_eq!(state.k.load(Ordering::Relaxed), 1);

        // Window 2: same share, but cooldown (1h) blocks the next step.
        state.total_requests.store(200, Ordering::Relaxed);
        state.truncated_requests.store(100, Ordering::Relaxed);
        update_share_and_k("m", &state, &cfg);
        assert_eq!(state.k.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_controller_no_traffic_keeps_share() {
        let cfg = TruncationAwareConfig {
            cooldown_secs: 0,
            ..test_config()
        };
        let (state, _workers) = seeded_state(8);
        *state.ewma_share.lock().unwrap() = Some(0.5);

        // No new requests this window: share must not decay toward 0.
        update_share_and_k("m", &state, &cfg);
        assert!((state.ewma_share.lock().unwrap().unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_controller_ewma_warm_start_on_first_window() {
        let cfg = TruncationAwareConfig {
            ewma_window_secs: 60,
            tick_secs: 10, // alpha = 1/6: blending from 0 would give 0.1
            cooldown_secs: 0,
            ..test_config()
        };
        let (state, _workers) = seeded_state(8);

        // First window with traffic: 60% truncated is adopted as-is.
        state.total_requests.store(100, Ordering::Relaxed);
        state.truncated_requests.store(60, Ordering::Relaxed);
        update_share_and_k("m", &state, &cfg);
        assert!((state.ewma_share.lock().unwrap().unwrap() - 0.6).abs() < 1e-9);

        // Subsequent windows blend normally (0.6 → toward 0.0 by alpha=1/6).
        state.total_requests.store(200, Ordering::Relaxed);
        update_share_and_k("m", &state, &cfg);
        let blended = state.ewma_share.lock().unwrap().unwrap();
        assert!((blended - 0.5).abs() < 1e-9, "got {}", blended);
    }

    #[test]
    fn test_pressured_growth_bypasses_cooldown() {
        let cfg = TruncationAwareConfig {
            ewma_window_secs: 30,
            tick_secs: 30, // alpha = 1
            cooldown_secs: 3600,
            ..test_config()
        };
        let policy = TruncationAwarePolicy::with_config(cfg.clone());
        let workers = make_workers(8);
        policy.test_force_partition(&workers, 1);
        let state = policy.test_state(&workers);
        *state.last_resize_at.lock().unwrap() = Some(Instant::now());

        // 50% truncated → K* = 4 > 1, but the pool is not pressured
        // (no in-flight load anywhere): cooldown must hold K.
        state.total_requests.store(100, Ordering::Relaxed);
        state.truncated_requests.store(50, Ordering::Relaxed);
        update_share_and_k("m", &state, &cfg);
        assert_eq!(state.k.load(Ordering::Relaxed), 1);

        // Overload the truncation member (sticky stays idle): the pressure
        // bit must let the grow step through the cooldown.
        let trunc_set = compute_trunc_set(&entries(&workers), 1);
        for w in &workers {
            if trunc_set.contains(w.url()) {
                for _ in 0..5 {
                    w.increment_load();
                }
            }
        }
        state.total_requests.store(200, Ordering::Relaxed);
        state.truncated_requests.store(100, Ordering::Relaxed);
        update_share_and_k("m", &state, &cfg);
        assert_eq!(state.k.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_idle_tick_preserves_known_set_and_membership() {
        let cfg = test_config();
        let policy = TruncationAwarePolicy::with_config(cfg.clone());
        let workers = make_workers(8);
        policy.test_force_partition(&workers, 1);
        let state = policy.test_state(&workers);
        // Prior traffic established a share matching K=1 (round(8*0.125)),
        // so the idle window must hold both share and K.
        *state.ewma_share.lock().unwrap() = Some(0.125);
        let membership_before = state.trunc_urls.read().unwrap().clone();
        assert_eq!(membership_before.len(), 1);

        // Age every stamp beyond the TTL (assumes host uptime > 2 minutes).
        let stale = Instant::now()
            .checked_sub(Duration::from_secs(cfg.known_ttl().as_secs() + 30))
            .expect("test host uptime must exceed the known TTL");
        let urls: Vec<String> = state.known.iter().map(|e| e.key().clone()).collect();
        for url in &urls {
            if let Some(mut entry) = state.known.get_mut(url) {
                entry.last_seen = stale;
            }
        }

        // Idle window (no traffic): known set and membership must survive.
        tick_model("m", &state, policy.sticky_policy(), &cfg);
        assert_eq!(state.known.len(), 8);
        assert_eq!(*state.trunc_urls.read().unwrap(), membership_before);

        // Window with traffic: stale entries are now evidence of removal.
        state.total_requests.store(10, Ordering::Relaxed);
        tick_model("m", &state, policy.sticky_policy(), &cfg);
        assert_eq!(state.known.len(), 0);
    }

    #[test]
    fn test_init_workers_syncs_known_set() {
        let policy = TruncationAwarePolicy::with_config(test_config());
        let workers = make_workers(8);
        policy.init_workers(&workers);
        let state = policy.model_state(normalize_model_key(workers[0].model_id()));
        assert_eq!(state.known.len(), 8);

        // Registry shrinks to 5 workers: the known set follows immediately,
        // without waiting for request-path touches or the TTL backstop.
        policy.init_workers(&workers[..5]);
        assert_eq!(state.known.len(), 5);
        for w in &workers[..5] {
            assert!(state.known.contains_key(w.url()));
        }
    }

    #[test]
    fn test_controller_clamps_k_when_worker_set_shrinks() {
        let cfg = TruncationAwareConfig {
            ewma_window_secs: 30,
            tick_secs: 30,
            cooldown_secs: 3600, // cooldown must NOT block the capacity clamp
            ..test_config()
        };
        let (state, _workers) = seeded_state(8);
        state.k.store(5, Ordering::Relaxed);
        *state.last_resize_at.lock().unwrap() = Some(Instant::now());

        // Worker set shrinks to 3 (upper = 3 - sticky_min = 2).
        let urls: Vec<String> = state.known.iter().map(|e| e.key().clone()).collect();
        for url in urls.iter().take(5) {
            state.known.remove(url);
        }
        update_share_and_k("m", &state, &cfg);
        assert_eq!(state.k.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_controller_jitter_held_by_deadband() {
        let cfg = TruncationAwareConfig {
            ewma_window_secs: 30,
            tick_secs: 30, // alpha = 1: each window's share is adopted directly
            cooldown_secs: 0,
            deadband: 1,
            ..test_config()
        };
        let (state, _workers) = seeded_state(8);
        state.k.store(4, Ordering::Relaxed);

        // Share jitters between 0.4 (K*=3) and 0.6 (K*=5): |K*-K| = 1 stays
        // inside the deadband, so K must not flap.
        let mut total = 0u64;
        let mut truncated = 0u64;
        for window in 0..6 {
            let share = if window % 2 == 0 { 0.4 } else { 0.6 };
            total += 100;
            truncated += (100.0 * share) as u64;
            state.total_requests.store(total, Ordering::Relaxed);
            state.truncated_requests.store(truncated, Ordering::Relaxed);
            update_share_and_k("m", &state, &cfg);
            assert_eq!(state.k.load(Ordering::Relaxed), 4, "window {}", window);
        }
    }

    // ---- routing branches ----

    #[tokio::test]
    async fn test_truncated_routes_to_trunc_pool_min_load() {
        let policy = TruncationAwarePolicy::with_config(test_config());
        let workers = make_workers(4);
        policy.test_force_partition(&workers, 2);

        let trunc_set = compute_trunc_set(&entries(&workers), 2);
        let trunc_idx: Vec<usize> = (0..workers.len())
            .filter(|&i| trunc_set.contains(workers[i].url()))
            .collect();
        // Load the first trunc member so min-load picks the other.
        workers[trunc_idx[0]].increment_load();

        let info = SelectWorkerInfo {
            request_text: Some("hello"),
            truncated: true,
            ..Default::default()
        };
        let idx = policy.select_worker(&workers, &info).await.unwrap();
        assert_eq!(idx, trunc_idx[1]);
    }

    #[tokio::test]
    async fn test_truncated_never_inserted_into_sticky_tree() {
        let policy = TruncationAwarePolicy::with_config(test_config());
        let workers = make_workers(4);
        policy.sticky_policy().init_workers(&workers);
        policy.test_force_partition(&workers, 1);

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
    async fn test_unhealthy_workers_never_selected() {
        let policy = TruncationAwarePolicy::with_config(test_config());
        let workers = make_workers(4);
        policy.test_force_partition(&workers, 2);

        let trunc_set = compute_trunc_set(&entries(&workers), 2);
        // Mark all truncation members unhealthy: truncated traffic must fall
        // back to healthy (sticky) workers instead of a dead member.
        for w in &workers {
            if trunc_set.contains(w.url()) {
                w.set_healthy(false);
            }
        }
        let info = SelectWorkerInfo {
            request_text: Some("hi"),
            truncated: true,
            ..Default::default()
        };
        let idx = policy.select_worker(&workers, &info).await.unwrap();
        assert!(workers[idx].is_healthy());
        assert!(!trunc_set.contains(workers[idx].url()));
    }

    #[tokio::test]
    async fn test_sticky_traffic_uses_cache_aware_and_sticks() {
        let policy = TruncationAwarePolicy::with_config(test_config());
        let workers = make_workers(4);
        policy.sticky_policy().init_workers(&workers);
        policy.test_force_partition(&workers, 1);

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
        policy.test_force_partition(&workers, 0);

        // Establish stickiness with K=0 (all workers sticky).
        let text = "conversation whose tenant will move pools";
        let info = SelectWorkerInfo {
            request_text: Some(text),
            truncated: false,
            ..Default::default()
        };
        let tenant = policy.select_worker(&workers, &info).await.unwrap();

        // Grow K until the tenant's worker lands in the truncation pool.
        let mut k = 1;
        loop {
            let set = compute_trunc_set(&entries(&workers), k);
            if set.contains(workers[tenant].url()) {
                break;
            }
            k += 1;
            assert!(k < workers.len(), "tenant must eventually join the pool");
        }
        policy.test_force_partition(&workers, k);

        // Next sticky request with the same prefix must NOT land on the old
        // tenant anymore: it was evicted from the tree and removed from the
        // sticky subset.
        let new_idx = policy.select_worker(&workers, &info).await.unwrap();
        assert_ne!(new_idx, tenant);
    }

    #[tokio::test]
    async fn test_health_flap_does_not_reshuffle_membership() {
        let policy = TruncationAwarePolicy::with_config(test_config());
        let workers = make_workers(4);
        policy.sticky_policy().init_workers(&workers);
        policy.test_force_partition(&workers, 1);

        let trunc_set = compute_trunc_set(&entries(&workers), 1);

        // Establish a sticky tenant.
        let info = SelectWorkerInfo {
            request_text: Some("session that must survive a health flap"),
            truncated: false,
            ..Default::default()
        };
        let tenant = policy.select_worker(&workers, &info).await.unwrap();

        // The truncation member goes unhealthy: routing sees a narrower set,
        // but membership must NOT reshuffle and the tenant must keep sticking.
        for w in &workers {
            if trunc_set.contains(w.url()) {
                w.set_healthy(false);
            }
        }
        let during_flap = policy.select_worker(&workers, &info).await.unwrap();
        assert_eq!(during_flap, tenant, "health flap must not evict sticky tenants");

        let snapshot = policy
            .model_state(normalize_model_key(workers[0].model_id()))
            .trunc_urls
            .read()
            .unwrap()
            .clone();
        assert_eq!(snapshot, trunc_set, "membership must not change on a flap");
    }

    #[tokio::test]
    async fn test_init_workers_does_not_resurrect_trunc_members() {
        let policy = TruncationAwarePolicy::with_config(test_config());
        let workers = make_workers(4);
        policy.init_workers(&workers);
        policy.test_force_partition(&workers, 1);

        // Registry re-init (worker update path) re-seeds the tree; truncation
        // members must be removed again afterwards.
        policy.init_workers(&workers);

        let trunc_set = compute_trunc_set(&entries(&workers), 1);
        let info = SelectWorkerInfo {
            request_text: Some("post-reinit stickiness probe"),
            truncated: false,
            ..Default::default()
        };
        for _ in 0..8 {
            let idx = policy.select_worker(&workers, &info).await.unwrap();
            assert!(
                !trunc_set.contains(workers[idx].url()),
                "sticky traffic must not land on a truncation member after re-init"
            );
        }
    }

    #[tokio::test]
    async fn test_per_model_state_is_isolated() {
        let policy = TruncationAwarePolicy::with_config(test_config());
        let workers = make_workers(2);

        let info = SelectWorkerInfo {
            request_text: Some("hi"),
            truncated: true,
            ..Default::default()
        };
        policy.select_worker(&workers, &info).await.unwrap();

        // Counters land in the state keyed by this model, and only there.
        let key = normalize_model_key(workers[0].model_id()).to_string();
        assert_eq!(policy.models.len(), 1);
        let state = policy.models.get(&key).unwrap().clone();
        assert_eq!(state.total_requests.load(Ordering::Relaxed), 1);
        assert_eq!(state.truncated_requests.load(Ordering::Relaxed), 1);
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
