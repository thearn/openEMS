/* Copyright (C) 2026 Sean Mollet (sean@malmoset.com) */
#ifndef CUDA_PALETTE_BENCHMARK_H
#define CUDA_PALETTE_BENCHMARK_H

#include <cuda_runtime.h>
#include <string>
#include "FDTD/gpu_coeff_sets.h"

struct CUDA_PaletteBenchmarkInput
{
	std::string path;
	unsigned int nx, ny, nz, xc;
	const void* global_index;
	const float* global_table;
	GPU_CoeffSets coefficients;
	cudaStream_t stream;
};

//! Build, exhaustively verify and time an exact tile-local coefficient encoding.
void CUDA_RunPaletteBenchmark(const CUDA_PaletteBenchmarkInput& input);

#endif
