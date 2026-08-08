// Copyright (c) 2025 atomicmilkshake and contributors
// Copyright (c) 2026 The llama-wackMall-hybrid contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "common.cuh"

void ggml_cuda_turbo4_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
