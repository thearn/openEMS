/*
* Copyright (C) 2026 Sean Mollet (sean@malmoset.com)
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

#ifndef CUDA_OPPORTUNITY_REPORT_H
#define CUDA_OPPORTUNITY_REPORT_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "FDTD/gpu_coeff_sets.h"

struct GPU_OpportunityDim
{
	unsigned int x, y, z;
};

struct GPU_OpportunityBox
{
	GPU_OpportunityDim start, length;
};

struct GPU_OpportunityADE
{
	unsigned int orders;
	bool lorentz;
	std::vector<unsigned int> flat;
	std::vector<unsigned char> mask;
};

struct GPU_OpportunityModified
{
	std::string source;
	std::vector<unsigned int> flat;
};

struct GPU_OpportunityFixup
{
	unsigned int flat, mask;
};

struct GPU_OpportunityDevice
{
	std::string name;
	int compute_major, compute_minor;
	uint64_t l2_bytes, persisting_l2_max_bytes;
	uint64_t free_bytes, total_bytes, fused_field_bytes, fused_extension_bytes, reserve_bytes;
};

struct GPU_OpportunityInput
{
	std::string path;
	GPU_OpportunityDim dim, begin, end;
	unsigned int xc, useful_y, useful_z;
	std::vector<GPU_OpportunityBox> upml;
	std::vector<GPU_OpportunityADE> ade;
	int folded_ade;
	std::vector<GPU_OpportunityModified> modified;
	std::vector<GPU_OpportunityFixup> fixup;
	GPU_CoeffSets coefficients;
	GPU_OpportunityDevice device;
};

//! Write the setup-only fused-kernel opportunity census. Throws on invalid input or I/O failure.
void GPU_WriteOpportunityReport(const GPU_OpportunityInput& input);

#endif
