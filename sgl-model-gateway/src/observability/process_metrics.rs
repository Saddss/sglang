//! Process self-resource metrics (CPU, RSS, open FDs, threads).
//!
//! These are sampled by a lightweight background task on a fixed interval and
//! published as gauges/counters. Sampling happens entirely off the request
//! path (a separate tokio task reading `/proc/self`), so it adds **zero**
//! overhead to request handling.
//!
//! Emitted metrics:
//! - `smg_process_cpu_ms_total`          (counter) — cores = rate(...[1m]) / 1000
//! - `smg_process_resident_memory_bytes` (gauge)
//! - `smg_process_open_fds`              (gauge)
//! - `smg_process_threads`              (gauge)
//!
//! Only Linux exposes `/proc/self`; on other platforms the reads fail and the
//! sampler simply publishes nothing (no panics, no cost).

use std::time::Duration;

use tracing::debug;

use crate::observability::metrics::Metrics;

/// Userspace clock ticks per second (`USER_HZ`, from `sysconf(_SC_CLK_TCK)`) is
/// 100 on the Linux distros/containers we ship, so 1 tick = 10 ms. `/proc`
/// CPU fields are always reported in these ticks regardless of kernel `CONFIG_HZ`.
const MS_PER_CLOCK_TICK: u64 = 10;

/// Spawn the background process-metrics sampler.
///
/// `interval_secs` controls how often `/proc/self` is read (a few reads of tiny
/// virtual files). Returns immediately; the task runs for the process lifetime.
pub fn spawn(interval_secs: u64) {
    let interval_secs = interval_secs.max(1);
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs));
        // Cumulative CPU time (ms) already reported, so we can increment by delta.
        let mut last_cpu_ms: u64 = 0;

        loop {
            ticker.tick().await;

            if let Some(cpu_ms) = read_cpu_ms() {
                // Monotonic; guard against any non-monotonic read.
                if cpu_ms >= last_cpu_ms {
                    Metrics::add_process_cpu_ms(cpu_ms - last_cpu_ms);
                    last_cpu_ms = cpu_ms;
                }
            }

            if let Some(rss) = read_rss_bytes() {
                Metrics::set_process_resident_memory_bytes(rss);
            }

            if let Some(fds) = read_open_fds() {
                Metrics::set_process_open_fds(fds);
            }

            if let Some(threads) = read_threads() {
                Metrics::set_process_threads(threads);
            }
        }
    });
    debug!(
        "Started process-metrics sampler ({}s interval)",
        interval_secs
    );
}

/// Total CPU time (user + system) in milliseconds from `/proc/self/stat`.
fn read_cpu_ms() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // The `comm` field (2nd) is wrapped in parentheses and may itself contain
    // spaces or ')', so split on the LAST ')' and index from `state` (field 3).
    let after = stat.rsplit_once(')')?.1;
    let fields: Vec<&str> = after.split_whitespace().collect();
    // fields[0] == field 3 (state); utime = field 14, stime = field 15.
    let utime: u64 = fields.get(14 - 3)?.parse().ok()?;
    let stime: u64 = fields.get(15 - 3)?.parse().ok()?;
    Some((utime + stime) * MS_PER_CLOCK_TICK)
}

/// Resident set size in bytes from `/proc/self/status` (`VmRSS: N kB`).
fn read_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// Number of OS threads from `/proc/self/stat` (field 20).
fn read_threads() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let after = stat.rsplit_once(')')?.1;
    let fields: Vec<&str> = after.split_whitespace().collect();
    fields.get(20 - 3)?.parse().ok()
}

/// Number of open file descriptors (entries in `/proc/self/fd`).
///
/// `read_dir` itself holds one fd open while iterating, so subtract it to
/// report the steady-state count (matches Prometheus `process_open_fds`).
fn read_open_fds() -> Option<u64> {
    let count = std::fs::read_dir("/proc/self/fd").ok()?.count() as u64;
    Some(count.saturating_sub(1))
}
