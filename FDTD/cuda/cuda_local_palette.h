/* Copyright (C) 2026 Sean Mollet (sean@malmoset.com) */
#ifndef CUDA_LOCAL_PALETTE_H
#define CUDA_LOCAL_PALETTE_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "FDTD/gpu_coeff_sets.h"

// One fused tile owns xc by seven by 31 nodes and carries the positive halo
// used while fusing voltage and current updates. Keep this POD device-safe.
struct alignas(16) CUDA_PaletteTile
{
	uint32_t index_offset, dictionary_offset;
	uint16_t nx, ny, nz;
	uint8_t width, pad;
};
static_assert(sizeof(CUDA_PaletteTile)==16,"palette tile metadata must be fully costed");

struct CUDA_LocalPalette
{
	unsigned int nx=0, ny=0, nz=0, xc=0, gx=0, gy=0, gz=0;
	std::vector<CUDA_PaletteTile> meta;
	std::vector<unsigned char> index;
	std::vector<float> table;
	size_t one_byte_tiles=0, two_byte_tiles=0, shared_dictionaries=0;
	size_t logical_lookups=0, local_index_reads=0, index_padding=0;
	size_t dictionary_id_entries=0;
	double encoding_s=0;

	size_t bytes() const
	{
		return index.size()+table.size()*sizeof(float)+meta.size()*sizeof(CUDA_PaletteTile);
	}
};

//! Deterministically encode the exact coefficient footprints of fused tiles.
CUDA_LocalPalette CUDA_BuildLocalPalette(const GPU_CoeffSets& coefficients,
	unsigned int nx, unsigned int ny, unsigned int nz, unsigned int xc);

//! Exhaustively decode every clipped footprint entry and compare all 12 float bits.
void CUDA_ValidateLocalPalette(const CUDA_LocalPalette& palette,
	const GPU_CoeffSets& coefficients);

//! Focused deterministic host checks, including corrupted-index rejection.
void CUDA_SelfTestLocalPalette();

#endif
