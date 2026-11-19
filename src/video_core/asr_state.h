// SPDX-FileCopyrightText: Copyright 2026 suyu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// vk_rasterizer.cpp includes this header since ab7d74f, but the file was never committed, so the
// branch does not build. This restores the state it reads. Nothing in the published tree writes
// these values, so they stay zero and the projection jitter in UpdateViewportsState stays off,
// which is what the published code would do if the header existed.

#pragma once

#include <atomic>

#include "common/common_types.h"

namespace VideoCore {

struct AsrState {
    /// Non-zero enables sub-pixel projection jitter.
    std::atomic<u32> jon{0};
    /// Render target size (after resolution scaling) the jitter applies to.
    std::atomic<u32> rw{0};
    std::atomic<u32> rh{0};
    /// Jitter generation; a change forces the viewports to be rebuilt.
    std::atomic<u32> jg{0};
    /// Jitter offset in pixels.
    std::atomic<f32> jx{0.0f};
    std::atomic<f32> jy{0.0f};
};

inline AsrState asr_st{};

} // namespace VideoCore
