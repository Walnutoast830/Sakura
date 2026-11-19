// SPDX-FileCopyrightText: Copyright 2026 suyu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Diagnostics for pipelines whose driver compile never finishes. Every build logs when a worker
// starts it and when the driver returns, with the thread id that `ps -T` shows, so a build that
// logged "begin" but never "end" names the pipeline the driver is stuck on. A command-stream wait
// on an unbuilt pipeline logs every few seconds instead of waiting silently.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>

#if defined(__linux__) || defined(__ANDROID__)
#include <unistd.h>
#endif

#include "common/common_types.h"
#include "common/logging.h"

namespace Vulkan::PipelineBuildTrace {

/// Builds slower than this are logged as warnings.
constexpr auto SLOW_BUILD = std::chrono::milliseconds{1000};
/// How often a wait on an unbuilt pipeline reports that it is still waiting.
constexpr auto WAIT_REPORT_INTERVAL = std::chrono::seconds{5};
/// How long a draw or dispatch that cannot be skipped waits for its pipeline before giving up.
/// Normal builds finish in well under a second; some Mali drivers never return from
/// vkCreate*Pipelines for certain shaders, and waiting on those freezes emulation for good.
constexpr auto BUILD_TIMEOUT = std::chrono::milliseconds{2000};

/// Tracks a pipeline whose build has outlived BUILD_TIMEOUT, so later draws skip it at once
/// instead of waiting again. Cleared implicitly once the build finishes.
class BuildGate {
public:
    /// Waits up to `timeout` for `is_built()`. Returns whether it holds.
    template <typename Predicate>
    static bool WaitFor(std::condition_variable& condvar, std::mutex& mutex,
                        Predicate&& is_built, std::chrono::milliseconds timeout) {
        std::unique_lock lock{mutex};
        return condvar.wait_for(lock, timeout, is_built);
    }

    /// True when an earlier wait timed out and the build has still not finished.
    [[nodiscard]] bool GaveUp(bool built) const noexcept {
        return !built && gave_up.load(std::memory_order_relaxed);
    }

    /// Records the timeout. Returns true only the first time, so callers log it once.
    bool MarkGaveUp() noexcept {
        return !gave_up.exchange(true, std::memory_order_relaxed);
    }

    /// Returns true once, the first time it is called after the pipeline has gone unbuilt for at
    /// least `min_age` since it was created. Used to replace the worker a stuck build holds.
    bool ReportStuckOnce(bool built, std::chrono::milliseconds min_age) noexcept {
        if (built || reported_stuck.load(std::memory_order_relaxed)) {
            return false;
        }
        if (std::chrono::steady_clock::now() - created < min_age) {
            return false;
        }
        return !reported_stuck.exchange(true, std::memory_order_relaxed);
    }

private:
    std::chrono::steady_clock::time_point created{std::chrono::steady_clock::now()};
    std::atomic_bool gave_up{false};
    std::atomic_bool reported_stuck{false};
};

/// A build still unfinished this long after its pipeline was requested is treated as stuck in the
/// driver, even if no draw had to wait on it. Longer than BUILD_TIMEOUT because it also counts
/// time spent queued behind other builds.
constexpr auto STUCK_BUILD_AGE = std::chrono::milliseconds{10000};

/// Kernel thread id on Linux/Android, so it matches the TID column of `ps -T`.
inline u64 ThreadId() {
#if defined(__linux__) || defined(__ANDROID__)
    return static_cast<u64>(gettid());
#else
    return static_cast<u64>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
#endif
}

/// Logs the start of a build on construction and its duration on destruction.
class BuildScope {
public:
    explicit BuildScope(const char* kind_, u64 hash_)
        : kind{kind_}, hash{hash_}, tid{ThreadId()}, start{std::chrono::steady_clock::now()} {
        LOG_INFO(Render_Vulkan, "[PipelineBuild] begin {} 0x{:016x} tid={}", kind, hash, tid);
    }

    ~BuildScope() {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        if (elapsed >= SLOW_BUILD) {
            LOG_WARNING(Render_Vulkan, "[PipelineBuild] end {} 0x{:016x} tid={} SLOW {} ms", kind,
                        hash, tid, elapsed.count());
        } else {
            LOG_INFO(Render_Vulkan, "[PipelineBuild] end {} 0x{:016x} tid={} {} ms", kind, hash,
                     tid, elapsed.count());
        }
    }

    BuildScope(const BuildScope&) = delete;
    BuildScope& operator=(const BuildScope&) = delete;

private:
    const char* kind;
    u64 hash;
    u64 tid;
    std::chrono::steady_clock::time_point start;
};

/// Waits until `is_built()` holds, logging every WAIT_REPORT_INTERVAL while it does not.
template <typename Predicate>
void WaitForBuild(std::condition_variable& condvar, std::unique_lock<std::mutex>& lock,
                  Predicate&& is_built, const char* kind, u64 hash) {
    const auto start = std::chrono::steady_clock::now();
    while (!condvar.wait_for(lock, WAIT_REPORT_INTERVAL, is_built)) {
        const auto waited = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start);
        LOG_WARNING(Render_Vulkan,
                    "[PipelineWait] command stream blocked on unbuilt {} 0x{:016x} for {} s "
                    "(waiter tid={})",
                    kind, hash, waited.count(), ThreadId());
    }
    const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    if (waited >= WAIT_REPORT_INTERVAL) {
        LOG_WARNING(Render_Vulkan, "[PipelineWait] {} 0x{:016x} finished after {} ms of waiting",
                    kind, hash, waited.count());
    }
}

} // namespace Vulkan::PipelineBuildTrace
