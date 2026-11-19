// SPDX-FileCopyrightText: Copyright 2026 suyu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <shared_mutex>
#include <mutex>

#include "common/common_types.h"
#include "common/thread_worker.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_vulkan/pipeline_build_trace.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCore {
class ShaderNotify;
}

namespace Vulkan {

class Device;
class PipelineStatistics;
class Scheduler;

class ComputePipeline {
public:
    explicit ComputePipeline(const Device& device, Scheduler& scheduler, vk::PipelineCache& pipeline_cache,
                             DescriptorPool& descriptor_pool,
                             GuestDescriptorQueue& guest_descriptor_queue,
                             Common::ThreadWorker* thread_worker,
                             PipelineStatistics* pipeline_statistics,
                             VideoCore::ShaderNotify* shader_notify, const Shader::Info& info,
                             vk::ShaderModule spv_module, u64 shader_hash,
                             std::shared_mutex* cache_lock = nullptr);

    ComputePipeline& operator=(ComputePipeline&&) noexcept = delete;
    ComputePipeline(ComputePipeline&&) noexcept = delete;

    ComputePipeline& operator=(const ComputePipeline&) = delete;
    ComputePipeline(const ComputePipeline&) = delete;

    void Configure(Tegra::Engines::KeplerCompute& kepler_compute, Tegra::MemoryManager& gpu_memory,
                   Scheduler& scheduler, BufferCache& buffer_cache, TextureCache& texture_cache);

    [[nodiscard]] bool IsBuilt() const noexcept {
        return is_built.load(std::memory_order::relaxed);
    }

    /// Waits up to `timeout` for the build to finish. Returns whether the pipeline is built.
    bool WaitUntilBuilt(std::chrono::milliseconds timeout) {
        return PipelineBuildTrace::BuildGate::WaitFor(
            build_condvar, build_mutex, [this] { return IsBuilt(); }, timeout);
    }

    /// True when an earlier wait gave up on this pipeline and it has still not been built.
    [[nodiscard]] bool BuildGaveUp() const noexcept {
        return build_gate.GaveUp(IsBuilt());
    }

    /// Records that a wait gave up. Returns true only the first time.
    bool MarkBuildGaveUp() noexcept {
        return build_gate.MarkGaveUp();
    }

    /// Returns true once, when the build has been unfinished for at least `min_age`.
    bool ReportStuckOnce(std::chrono::milliseconds min_age) noexcept {
        return build_gate.ReportStuckOnce(IsBuilt(), min_age);
    }

    [[nodiscard]] u64 ShaderHash() const noexcept {
        return shader_hash;
    }

private:
    const Device& device;
    vk::PipelineCache& pipeline_cache;
    GuestDescriptorQueue& guest_descriptor_queue;
    std::shared_mutex* cache_lock{};
    Shader::Info info;
    u64 shader_hash{};
    u32 num_descriptor_entries{};

    VideoCommon::ComputeUniformBufferSizes uniform_buffer_sizes{};

    vk::ShaderModule spv_module;
    vk::DescriptorSetLayout descriptor_set_layout;
    bool uses_push_descriptor{false};
    DescriptorAllocator descriptor_allocator;
    vk::PipelineLayout pipeline_layout;
    vk::DescriptorUpdateTemplate descriptor_update_template;
    vk::Pipeline pipeline;

    std::condition_variable build_condvar;
    std::mutex build_mutex;
    std::atomic_bool is_built{false};
    PipelineBuildTrace::BuildGate build_gate;
};

} // namespace Vulkan
