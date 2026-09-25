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
#include "FDTD/extensions/operator_ext_lorentzmaterial.h"
#include "FDTD/extensions/engine_ext_lorentzmaterial.h"

// Drude/Lorentz/Debye materials and conducting sheets, see Engine_Ext_LorentzMaterial
// and Engine_Ext_Dispersive. One thread per mesh position of a group of orders. The ADE
// state and coefficients are stored per order as [direction][position].

// Orders that share their mesh positions (e.g. the four branches of a conducting sheet) in one
// launch: the field is read (pre) or written (apply) once per active component, the orders are
// processed in their order with the same operations as lorentz_pre/dispersive_apply. A component
// whose c_ext is zero in every order keeps a zero ADE state and is skipped (mask).
#define ADE_GROUP_MAX 8
struct ADEGroup
{
	unsigned int count, orders, lorentz, sn;
	const unsigned int* pos;
	const unsigned char* mask;
	float* ade[ADE_GROUP_MAX];
	float* lor[ADE_GROUP_MAX];
	const float* c_int[ADE_GROUP_MAX];
	const float* c_ext[ADE_GROUP_MAX];
	const float* c_lor[ADE_GROUP_MAX];
};

__global__ void lorentz_pre_group(const float* field, ADEGroup G)
{
	const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i>=G.count)
		return;
	const unsigned int m = G.mask[i];
	for (unsigned int n=0; n<3; ++n)
	{
		if (!((m>>n)&1))
			continue;
		const unsigned int k = n*G.count + i;
		const float f = field[n*G.sn + G.pos[i]];
		for (unsigned int o=0; o<G.orders; ++o)
		{
			float* ade = G.ade[o];
			if (G.lorentz)
			{
				float* lor = G.lor[o];
				lor[k] = lor[k] + G.c_lor[o][k]*ade[k];
				ade[k] = ade[k] * G.c_int[o][k];
				ade[k] = ade[k] + G.c_ext[o][k]*(f - lor[k]);
			}
			else
			{
				ade[k] = ade[k] * G.c_int[o][k];
				ade[k] = ade[k] + G.c_ext[o][k]*f;
			}
		}
	}
}

__global__ void dispersive_apply_group(float* field, ADEGroup G)
{
	const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i>=G.count)
		return;
	const unsigned int m = G.mask[i];
	for (unsigned int n=0; n<3; ++n)
	{
		if (!((m>>n)&1))
			continue;
		const unsigned int g = n*G.sn + G.pos[i];
		float v = field[g];
		for (unsigned int o=0; o<G.orders; ++o)
			v = v - G.ade[o][n*G.count + i];
		field[g] = v;
	}
}

class CUDA_Ext_LorentzMaterial : public GPU_Extension
{
public:
	CUDA_Ext_LorentzMaterial(GPU_Backend_CUDA::Impl* impl, Operator_Ext_LorentzMaterial* op_ext);

	virtual void DoPreVoltageUpdates() {for (size_t g=0; g<m_VoltGroups.size(); ++g) Pre(m_VoltGroups[g], d->volt);}
	virtual void Apply2Voltages()
	{
		for (size_t g=0; g<m_VoltGroups.size(); ++g)
			if (!m_VoltFolded[g])   // else applied in the fused kernel (see CUDA_FusedADE)
				Apply(m_VoltGroups[g], d->volt);
	}
	virtual void DoPreCurrentUpdates() {for (size_t g=0; g<m_CurrGroups.size(); ++g) Pre(m_CurrGroups[g], d->curr);}
	virtual void Apply2Current()       {for (size_t g=0; g<m_CurrGroups.size(); ++g) Apply(m_CurrGroups[g], d->curr);}

protected:
	//! the host description of an order, grouped with the orders at the same positions in Build()
	struct HostOrder
	{
		unsigned int count;
		bool lorentz;
		std::vector<unsigned int> flat;
		FDTD_FLOAT **c_int, **c_ext, **c_lor;
	};
	void Build(std::vector<ADEGroup>& groups, const std::vector<HostOrder>& orders, bool voltage);
	float* Coefficients(unsigned int count, FDTD_FLOAT** c);
	void Pre(ADEGroup& g, float* field)
	{
		CUDA_Launch(d, "lorentz_pre_group", lorentz_pre_group, g.count, 1, 1, (const float*)field, g);
	}
	void Apply(ADEGroup& g, float* field)
	{
		CUDA_Launch(d, "dispersive_apply_group", dispersive_apply_group, g.count, 1, 1, field, g);
	}

	GPU_Backend_CUDA::Impl* d;
	std::vector<ADEGroup> m_VoltGroups;
	std::vector<char> m_VoltFolded;   //!< per voltage group: applied in the fused kernel
	std::vector<ADEGroup> m_CurrGroups;
};

CUDA_Ext_LorentzMaterial::CUDA_Ext_LorentzMaterial(GPU_Backend_CUDA::Impl* impl, Operator_Ext_LorentzMaterial* op_ext)
{
	d = impl;
	std::vector<HostOrder> volt, curr;
	for (int o=0; o<op_ext->m_Order; ++o)
	{
		const unsigned int count = op_ext->m_LM_Count.at(o);
		if (count==0)
			continue;
		std::vector<unsigned int> flat(count);
		unsigned int** pos = op_ext->m_LM_pos[o];
		for (unsigned int i=0; i<count; ++i)
			flat[i] = (pos[0][i]*d->dim.ny + pos[1][i])*d->dim.nz + pos[2][i];
		if (op_ext->m_volt_ADE_On[o])
			volt.push_back({count, op_ext->m_volt_Lor_ADE_On[o], flat, op_ext->v_int_ADE[o], op_ext->v_ext_ADE[o],
			                op_ext->m_volt_Lor_ADE_On[o] ? op_ext->v_Lor_ADE[o] : NULL});
		if (op_ext->m_curr_ADE_On[o])
			curr.push_back({count, op_ext->m_curr_Lor_ADE_On[o], flat, op_ext->i_int_ADE[o], op_ext->i_ext_ADE[o],
			                op_ext->m_curr_Lor_ADE_On[o] ? op_ext->i_Lor_ADE[o] : NULL});
	}
	Build(m_VoltGroups, volt, true);
	Build(m_CurrGroups, curr, false);
}

float* CUDA_Ext_LorentzMaterial::Coefficients(unsigned int count, FDTD_FLOAT** c)
{
	std::vector<float> data(3*(size_t)count, 0);
	if (c)
		for (int n=0; n<3; ++n)
			for (unsigned int i=0; i<count; ++i)
				data[n*count + i] = c[n][i];
	return d->Alloc<float>(data.size(), data.data());
}

void CUDA_Ext_LorentzMaterial::Build(std::vector<ADEGroup>& groups, const std::vector<HostOrder>& orders, bool voltage)
{
	for (size_t first=0; first<orders.size(); )
	{
		// consecutive orders at identical positions with the same kind share a launch
		size_t last = first+1;
		while ((last<orders.size()) && (last-first<ADE_GROUP_MAX) && (orders[last].flat==orders[first].flat)
		       && (orders[last].lorentz==orders[first].lorentz))
			++last;
		const unsigned int count = orders[first].count;
		ADEGroup g;
		g.count = count;
		g.orders = last-first;
		g.lorentz = orders[first].lorentz;
		g.sn = d->numCells;
		std::vector<unsigned char> mask(count, 0);
		for (size_t o=first; o<last; ++o)
			for (int n=0; n<3; ++n)
				for (unsigned int i=0; i<count; ++i)
					if (orders[o].c_ext[n][i]!=0)
						mask[i] |= (unsigned char)(1<<n);
		for (size_t o=first; o<last; ++o)
		{
			const size_t k = o-first;
			g.ade[k] = d->Alloc<float>(3*(size_t)count);
			g.lor[k] = d->Alloc<float>(3*(size_t)count);
			g.c_int[k] = Coefficients(count, orders[o].c_int);
			g.c_ext[k] = Coefficients(count, orders[o].c_ext);
			g.c_lor[k] = Coefficients(count, orders[o].c_lor);
		}
		g.pos = d->Alloc<unsigned int>(count, orders[first].flat.data());
		g.mask = d->Alloc<unsigned char>(count, mask.data());
		groups.push_back(g);
		// fused step: the voltages this group changes (only its active components) are either
		// corrected in the fused kernel or registered for the current fix-up (DecideFusedStep)
		if (voltage)
		{
			const size_t index = groups.size()-1;
			m_VoltFolded.push_back(0);
			CUDA_ADECandidate C;
			C.count = count;
			C.orders = g.orders;
			C.flat = orders[first].flat;
			C.mask = mask;
			for (unsigned int o=0; o<g.orders; ++o)
				C.ade[o] = g.ade[o];
			C.fold = [this, index]() {m_VoltFolded[index] = 1;};
			d->ade_candidates.push_back(C);
		}
		first = last;
	}
}

GPU_Extension* CUDA_CreateExt_LorentzMaterial(GPU_Backend_CUDA::Impl* d, Engine_Extension* eng_ext, Engine* eng)
{
	UNUSED(eng);
	if (!dynamic_cast<Engine_Ext_LorentzMaterial*>(eng_ext))
		return NULL;
	Operator_Ext_LorentzMaterial* op_ext = dynamic_cast<Operator_Ext_LorentzMaterial*>(eng_ext->GetOperatorExtension());
	if (!op_ext)
		return NULL;
	return new CUDA_Ext_LorentzMaterial(d, op_ext);
}
