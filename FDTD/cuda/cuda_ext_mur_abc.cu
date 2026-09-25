/*
*	Copyright (C) 2026 Sean Mollet (sean@malmoset.com)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "cuda_internal.h"
#include "FDTD/engine.h"
#include "FDTD/extensions/operator_ext_mur_abc.h"
#include "FDTD/extensions/engine_ext_mur_abc.h"

// First order Mur ABC on one boundary plane, see Engine_Ext_Mur_ABC.
// One thread per (i,j) of the plane, i along m_nyP, j along m_nyPP.
struct MurParam { int ny, nyP, nyPP; unsigned int line, line_shift; unsigned int ni, nj; };

__device__ __forceinline__ unsigned int mur_index(const CUDA_GridDim& N, const MurParam& P, unsigned int n, unsigned int line, unsigned int i, unsigned int j)
{
	unsigned int pos[3];
	pos[P.ny] = line;
	pos[P.nyP] = i;
	pos[P.nyPP] = j;
	return nijk(N, n, pos[0], pos[1], pos[2]);
}

// thread index: j along x-threads, i along y-threads
#define MUR_THREAD \
	const unsigned int j = blockIdx.x*blockDim.x + threadIdx.x; \
	const unsigned int i = blockIdx.y*blockDim.y + threadIdx.y; \
	if (i>=P.ni || j>=P.nj) \
		return; \
	const unsigned int ij = i*P.nj + j;

__global__ void mur_pre(const float* volt, float* v_nyP, float* v_nyPP, const float* c_nyP, const float* c_nyPP, CUDA_GridDim N, MurParam P)
{
	MUR_THREAD
	v_nyP[ij]  = volt[mur_index(N, P, P.nyP,  P.line_shift, i, j)] - c_nyP[ij]  * volt[mur_index(N, P, P.nyP,  P.line, i, j)];
	v_nyPP[ij] = volt[mur_index(N, P, P.nyPP, P.line_shift, i, j)] - c_nyPP[ij] * volt[mur_index(N, P, P.nyPP, P.line, i, j)];
}

__device__ __forceinline__ void mur_pre_at(unsigned int i, unsigned int j, const float* volt, float* v_nyP, float* v_nyPP,
                                           const float* c_nyP, const float* c_nyPP, CUDA_GridDim N, MurParam P)
{
	if (i>=P.ni || j>=P.nj)
		return;
	const unsigned int ij = i*P.nj + j;
	v_nyP[ij]  = volt[mur_index(N, P, P.nyP, P.line_shift, i, j)] - c_nyP[ij]  * volt[mur_index(N, P, P.nyP, P.line, i, j)];
	v_nyPP[ij] = volt[mur_index(N, P, P.nyPP, P.line_shift, i, j)] - c_nyPP[ij] * volt[mur_index(N, P, P.nyPP, P.line, i, j)];
}

__global__ void mur_pre_pair(const float* volt,
	float* a_nyP, float* a_nyPP, const float* ac_nyP, const float* ac_nyPP, MurParam A, bool active_a,
	float* b_nyP, float* b_nyPP, const float* bc_nyP, const float* bc_nyPP, MurParam B, bool active_b,
	CUDA_GridDim N)
{
	const unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	if (blockIdx.z==0)
	{
		if (active_a)
			mur_pre_at(i, j, volt, a_nyP, a_nyPP, ac_nyP, ac_nyPP, N, A);
	}
	else if (active_b)
		mur_pre_at(i, j, volt, b_nyP, b_nyPP, bc_nyP, bc_nyPP, N, B);
}

__global__ void mur_post(const float* volt, float* v_nyP, float* v_nyPP, const float* c_nyP, const float* c_nyPP, CUDA_GridDim N, MurParam P)
{
	MUR_THREAD
	v_nyP[ij]  = v_nyP[ij]  + c_nyP[ij]  * volt[mur_index(N, P, P.nyP,  P.line_shift, i, j)];
	v_nyPP[ij] = v_nyPP[ij] + c_nyPP[ij] * volt[mur_index(N, P, P.nyPP, P.line_shift, i, j)];
}

__global__ void mur_apply(float* volt, const float* v_nyP, const float* v_nyPP, CUDA_GridDim N, MurParam P)
{
	MUR_THREAD
	volt[mur_index(N, P, P.nyP,  P.line, i, j)] = v_nyP[ij];
	volt[mur_index(N, P, P.nyPP, P.line, i, j)] = v_nyPP[ij];
}

__device__ __forceinline__ void mur_post_at(unsigned int i, unsigned int j, const float* volt, float* v_nyP, float* v_nyPP,
                                            const float* c_nyP, const float* c_nyPP, CUDA_GridDim N, MurParam P)
{
	if (i>=P.ni || j>=P.nj)
		return;
	const unsigned int ij = i*P.nj + j;
	v_nyP[ij]  = v_nyP[ij]  + c_nyP[ij]  * volt[mur_index(N, P, P.nyP,  P.line_shift, i, j)];
	v_nyPP[ij] = v_nyPP[ij] + c_nyPP[ij] * volt[mur_index(N, P, P.nyPP, P.line_shift, i, j)];
}

__global__ void mur_post_pair(const float* volt,
	float* a_nyP, float* a_nyPP, const float* ac_nyP, const float* ac_nyPP, MurParam A, bool active_a,
	float* b_nyP, float* b_nyPP, const float* bc_nyP, const float* bc_nyPP, MurParam B, bool active_b,
	CUDA_GridDim N)
{
	const unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	if (blockIdx.z==0)
	{
		if (active_a)
			mur_post_at(i, j, volt, a_nyP, a_nyPP, ac_nyP, ac_nyPP, N, A);
	}
	else if (active_b)
		mur_post_at(i, j, volt, b_nyP, b_nyPP, bc_nyP, bc_nyPP, N, B);
}

__device__ __forceinline__ void mur_apply_at(unsigned int i, unsigned int j, float* volt,
                                             const float* v_nyP, const float* v_nyPP, CUDA_GridDim N, MurParam P)
{
	if (i>=P.ni || j>=P.nj)
		return;
	const unsigned int ij = i*P.nj + j;
	volt[mur_index(N, P, P.nyP,  P.line, i, j)] = v_nyP[ij];
	volt[mur_index(N, P, P.nyPP, P.line, i, j)] = v_nyPP[ij];
}

__global__ void mur_apply_pair(float* volt,
	const float* a_nyP, const float* a_nyPP, MurParam A, bool active_a,
	const float* b_nyP, const float* b_nyPP, MurParam B, bool active_b,
	CUDA_GridDim N)
{
	const unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
	if (blockIdx.z==0)
	{
		if (active_a)
			mur_apply_at(i, j, volt, a_nyP, a_nyPP, N, A);
	}
	else if (active_b)
		mur_apply_at(i, j, volt, b_nyP, b_nyPP, N, B);
}

class CUDA_Ext_Mur_ABC : public GPU_Extension
{
public:
	CUDA_Ext_Mur_ABC(GPU_Backend_CUDA::Impl* impl, Operator_Ext_Mur_ABC* op_ext, Engine_Ext_Mur_ABC* eng_ext, Engine* eng);

	virtual void DoPreVoltageUpdates()
	{
		if (m_Partner && m_PairLauncher)
			LaunchPrePair();
		else if (!m_Partner && IsActive())
			CUDA_Launch(d, "mur_pre", mur_pre, m_Param.nj, m_Param.ni, 1, (const float*)d->volt, m_Volt_nyP, m_Volt_nyPP, (const float*)m_Coeff_nyP, (const float*)m_Coeff_nyPP, d->dim, m_Param);
	}
	virtual void DoPostVoltageUpdates()
	{
		if (m_Partner && m_PairLauncher)
			LaunchPostPair();
		else if (!m_Partner && IsActive())
			CUDA_Launch(d, "mur_post", mur_post, m_Param.nj, m_Param.ni, 1, (const float*)d->volt, m_Volt_nyP, m_Volt_nyPP, (const float*)m_Coeff_nyP, (const float*)m_Coeff_nyPP, d->dim, m_Param);
	}
	virtual void Apply2Voltages()
	{
		if (m_Partner && m_PairLauncher)
			LaunchApplyPair();
		else if (!m_Partner && IsActive())
			CUDA_Launch(d, "mur_apply", mur_apply, m_Param.nj, m_Param.ni, 1, d->volt, (const float*)m_Volt_nyP, (const float*)m_Volt_nyPP, d->dim, m_Param);
	}

protected:
	//! the ABC is off until an excitation on its plane is done
	bool IsActive() {return m_Eng->GetNumberOfTimesteps()>=m_StartTS;}
	void LaunchPrePair();
	void LaunchPostPair();
	void LaunchApplyPair();

	GPU_Backend_CUDA::Impl* d;
	Engine* m_Eng;
	unsigned int m_StartTS;
	bool m_Pairs;
	bool m_PairLauncher;
	CUDA_Ext_Mur_ABC* m_Partner;
	MurParam m_Param;

	float *m_Volt_nyP, *m_Volt_nyPP;
	float *m_Coeff_nyP, *m_Coeff_nyPP;
};

CUDA_Ext_Mur_ABC::CUDA_Ext_Mur_ABC(GPU_Backend_CUDA::Impl* impl, Operator_Ext_Mur_ABC* op_ext, Engine_Ext_Mur_ABC* eng_ext, Engine* eng)
{
	d = impl;
	m_Eng = eng;
	m_StartTS = eng_ext->GetStartTimestep();
	m_Pairs = !getenv("OPENEMS_CUDA_MUR_PAIRS") || (atoi(getenv("OPENEMS_CUDA_MUR_PAIRS"))!=0);
	m_PairLauncher = true;
	m_Partner = NULL;
	m_Param.ny = op_ext->m_ny;
	m_Param.nyP = op_ext->m_nyP;
	m_Param.nyPP = op_ext->m_nyPP;
	m_Param.line = op_ext->m_LineNr;
	m_Param.line_shift = op_ext->m_LineNr_Shift;
	m_Param.ni = op_ext->m_numLines[0];
	m_Param.nj = op_ext->m_numLines[1];

	const size_t count = (size_t)m_Param.ni*m_Param.nj;
	// MUR overwrites both tangential boundary voltages between the voltage and
	// current half-steps. The fused kernel computes currents before that
	// overwrite, so register those voltage edges for the existing current
	// fixup pass.
	d->volt_modified_from.push_back(std::make_pair(d->volt_modified.size(), "Mur ABC"));
	for (unsigned int i=0; i<m_Param.ni; ++i)
		for (unsigned int j=0; j<m_Param.nj; ++j)
		{
			unsigned int pos[3];
			pos[m_Param.ny] = m_Param.line;
			pos[m_Param.nyP] = i;
			pos[m_Param.nyPP] = j;
			d->volt_modified.push_back(((m_Param.nyP*d->dim.nx + pos[0])*d->dim.ny + pos[1])*d->dim.nz + pos[2]);
			d->volt_modified.push_back(((m_Param.nyPP*d->dim.nx + pos[0])*d->dim.ny + pos[1])*d->dim.nz + pos[2]);
		}
	m_Volt_nyP  = d->Alloc<float>(count);
	m_Volt_nyPP = d->Alloc<float>(count);
	m_Coeff_nyP  = d->Alloc<float>(count, op_ext->m_Mur_Coeff_nyP.data());
	m_Coeff_nyPP = d->Alloc<float>(count, op_ext->m_Mur_Coeff_nyPP.data());
	if (m_Pairs)
		for (size_t n=0; n<d->mur_extensions.size(); ++n)
			if (d->mur_extensions[n]->m_Param.ny==m_Param.ny)
			{
				m_Partner = d->mur_extensions[n];
				m_Partner->m_Partner = this;
				m_Partner->m_PairLauncher = false;
				break;
			}
	d->mur_extensions.push_back(this);
}

void CUDA_Ext_Mur_ABC::LaunchPrePair()
{
	const bool active_a = IsActive();
	const bool active_b = m_Partner->IsActive();
	if (!active_a && !active_b)
		return;
	CUDA_Launch(d, "mur_pre_pair", mur_pre_pair,
	            std::max(m_Param.nj, m_Partner->m_Param.nj), std::max(m_Param.ni, m_Partner->m_Param.ni), 2,
	            (const float*)d->volt,
	            m_Volt_nyP, m_Volt_nyPP, (const float*)m_Coeff_nyP, (const float*)m_Coeff_nyPP, m_Param, active_a,
	            m_Partner->m_Volt_nyP, m_Partner->m_Volt_nyPP,
	            (const float*)m_Partner->m_Coeff_nyP, (const float*)m_Partner->m_Coeff_nyPP, m_Partner->m_Param, active_b,
	            d->dim);
}

void CUDA_Ext_Mur_ABC::LaunchPostPair()
{
	const bool active_a = IsActive();
	const bool active_b = m_Partner->IsActive();
	if (!active_a && !active_b)
		return;
	CUDA_Launch(d, "mur_post_pair", mur_post_pair,
	            std::max(m_Param.nj, m_Partner->m_Param.nj), std::max(m_Param.ni, m_Partner->m_Param.ni), 2,
	            (const float*)d->volt,
	            m_Volt_nyP, m_Volt_nyPP, (const float*)m_Coeff_nyP, (const float*)m_Coeff_nyPP, m_Param, active_a,
	            m_Partner->m_Volt_nyP, m_Partner->m_Volt_nyPP,
	            (const float*)m_Partner->m_Coeff_nyP, (const float*)m_Partner->m_Coeff_nyPP, m_Partner->m_Param, active_b,
	            d->dim);
}

void CUDA_Ext_Mur_ABC::LaunchApplyPair()
{
	const bool active_a = IsActive();
	const bool active_b = m_Partner->IsActive();
	if (!active_a && !active_b)
		return;
	CUDA_Launch(d, "mur_apply_pair", mur_apply_pair,
	            std::max(m_Param.nj, m_Partner->m_Param.nj), std::max(m_Param.ni, m_Partner->m_Param.ni), 2,
	            d->volt,
	            (const float*)m_Volt_nyP, (const float*)m_Volt_nyPP, m_Param, active_a,
	            (const float*)m_Partner->m_Volt_nyP, (const float*)m_Partner->m_Volt_nyPP, m_Partner->m_Param, active_b,
	            d->dim);
}

GPU_Extension* CUDA_CreateExt_Mur_ABC(GPU_Backend_CUDA::Impl* d, Engine_Extension* eng_ext, Engine* eng)
{
	Engine_Ext_Mur_ABC* mur_ext = dynamic_cast<Engine_Ext_Mur_ABC*>(eng_ext);
	if (!mur_ext)
		return NULL;
	Operator_Ext_Mur_ABC* op_ext = dynamic_cast<Operator_Ext_Mur_ABC*>(eng_ext->GetOperatorExtension());
	if (!op_ext)
		return NULL;
	return new CUDA_Ext_Mur_ABC(d, op_ext, mur_ext, eng);
}
