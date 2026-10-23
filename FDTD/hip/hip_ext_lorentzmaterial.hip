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

#include "hip/hip_runtime.h"
#include <algorithm>
#include <map>
#include <vector>
#include "hip_internal.h"
#include "FDTD/engine.h"
#include "FDTD/extensions/operator_ext_lorentzmaterial.h"
#include "FDTD/extensions/engine_ext_lorentzmaterial.h"

// Drude/Lorentz/Debye materials and conducting sheets, see Engine_Ext_LorentzMaterial
// and Engine_Ext_Dispersive. One thread per mesh position of a group of orders. The ADE
// state and coefficients are stored per order as [direction][position].
//
// Orders that share their mesh positions (e.g. the four branches of a conducting sheet) are one
// group with one launch per phase: the field is read (pre) or written (apply) once per active
// component, the orders are processed in their order with the operations of the engine. A
// component whose c_ext is zero in every order keeps a zero ADE state and never changes its
// field: it is skipped (mask).
struct ADEGroup
{
	size_t count, sn;
	unsigned int orders, lorentz;
	const size_t* pos;
	const unsigned char* mask;
	float* ade[ADE_GROUP_MAX];
	float* lor[ADE_GROUP_MAX];
	const float* c_int[ADE_GROUP_MAX];
	const float* c_ext[ADE_GROUP_MAX];
	const float* c_lor[ADE_GROUP_MAX];
};

// ADE update before the main update, reads the field
__global__ void lorentz_pre_group(const float* field, ADEGroup G)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
	if (i>=G.count)
		return;
	const unsigned int m = G.mask[i];
	for (unsigned int n=0; n<3; ++n)
	{
		if (!((m>>n)&1))
			continue;
		const size_t k = n*G.count + i;
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

// subtract the ADE currents of the orders from the field, in their order
__global__ void dispersive_apply_group(float* field, ADEGroup G)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
	if (i>=G.count)
		return;
	const unsigned int m = G.mask[i];
	for (unsigned int n=0; n<3; ++n)
	{
		if (!((m>>n)&1))
			continue;
		const size_t g = n*G.sn + G.pos[i];
		float v = field[g];
		for (unsigned int o=0; o<G.orders; ++o)
			v = v - G.ade[o][n*G.count + i];
		field[g] = v;
	}
}

class HIP_Ext_LorentzMaterial : public GPU_Extension
{
public:
	HIP_Ext_LorentzMaterial(GPU_Backend_HIP::Impl* impl, Operator_Ext_LorentzMaterial* op_ext);

	virtual void DoPreVoltageUpdates() {for (size_t g=0; g<m_Volt.size(); ++g) Pre(m_Volt[g].G, d->volt);}
	virtual void Apply2Voltages()      {for (size_t g=0; g<m_Volt.size(); ++g) if (!m_Volt[g].folded) Apply(m_Volt[g].G, d->volt);}
	virtual void DoPreCurrentUpdates() {for (size_t g=0; g<m_Curr.size(); ++g) Pre(m_Curr[g].G, d->curr);}
	virtual void Apply2Current()       {for (size_t g=0; g<m_Curr.size(); ++g) Apply(m_Curr[g].G, d->curr);}

protected:
	//! the host description of an order, grouped with the orders at the same positions in Build()
	struct HostOrder
	{
		size_t count;
		bool lorentz;
		std::vector<size_t> flat;
		FDTD_FLOAT **c_int, **c_ext, **c_lor;
	};
	struct Group
	{
		ADEGroup G;
		bool folded;   //!< voltage group applied in the fused kernel (see HIP_ADECandidate)
	};
	void Build(std::vector<Group>& groups, const std::vector<HostOrder>& orders, bool voltage);
	float* Coefficients(size_t count, FDTD_FLOAT** c);
	void Pre(ADEGroup& g, float* field)
	{
		HIP_Launch(d, "lorentz_pre_group", lorentz_pre_group, g.count, 1, 1, (const float*)field, g);
	}
	void Apply(ADEGroup& g, float* field)
	{
		HIP_Launch(d, "dispersive_apply_group", dispersive_apply_group, g.count, 1, 1, field, g);
	}

	GPU_Backend_HIP::Impl* d;
	std::vector<Group> m_Volt;
	std::vector<Group> m_Curr;
};

HIP_Ext_LorentzMaterial::HIP_Ext_LorentzMaterial(GPU_Backend_HIP::Impl* impl, Operator_Ext_LorentzMaterial* op_ext)
{
	d = impl;
	std::vector<HostOrder> volt, curr;
	for (int o=0; o<op_ext->m_Order; ++o)
	{
		const size_t count = op_ext->m_LM_Count.at(o);
		if (count==0)
			continue;
		std::vector<size_t> flat(count);
		unsigned int** pos = op_ext->m_LM_pos[o];
		for (size_t i=0; i<count; ++i)
			flat[i] = ((size_t)pos[0][i]*d->dim.ny + pos[1][i])*d->dim.nz + pos[2][i];
		if (op_ext->m_volt_ADE_On[o])
			volt.push_back({count, op_ext->m_volt_Lor_ADE_On[o], flat, op_ext->v_int_ADE[o], op_ext->v_ext_ADE[o],
			                op_ext->m_volt_Lor_ADE_On[o] ? op_ext->v_Lor_ADE[o] : NULL});
		if (op_ext->m_curr_ADE_On[o])
			curr.push_back({count, op_ext->m_curr_Lor_ADE_On[o], flat, op_ext->i_int_ADE[o], op_ext->i_ext_ADE[o],
			                op_ext->m_curr_Lor_ADE_On[o] ? op_ext->i_Lor_ADE[o] : NULL});
	}
	Build(m_Volt, volt, true);
	Build(m_Curr, curr, false);
}

float* HIP_Ext_LorentzMaterial::Coefficients(size_t count, FDTD_FLOAT** c)
{
	std::vector<float> data(3*(size_t)count, 0);
	if (c)
		for (int n=0; n<3; ++n)
			for (size_t i=0; i<count; ++i)
				data[n*count + i] = c[n][i];
	return d->Alloc<float>(data.size(), data.data());
}

void HIP_Ext_LorentzMaterial::Build(std::vector<Group>& groups, const std::vector<HostOrder>& orders, bool voltage)
{
	// consecutive runs of orders with the same positions and the same kind (Lorentz or not), at
	// most ADE_GROUP_MAX each: the engine processes the orders in order, so a group keeps it
	std::vector<size_t> starts;
	for (size_t o=0; o<orders.size(); ++o)
		if (starts.empty() || (o-starts.back()>=ADE_GROUP_MAX) || (orders[o].flat!=orders[starts.back()].flat)
		    || (orders[o].lorentz!=orders[starts.back()].lorentz))
			starts.push_back(o);
	starts.push_back(orders.size());
	for (size_t s=0; s+1<starts.size(); ++s)
	{
		const HostOrder& first = orders[starts[s]];
		Group grp;
		grp.folded = false;
		ADEGroup& G = grp.G;
		G.count = first.count;
		G.sn = d->numCells;
		G.orders = (unsigned int)(starts[s+1]-starts[s]);
		G.lorentz = first.lorentz;
		std::vector<unsigned char> mask(G.count, 0);
		for (size_t o=starts[s]; o<starts[s+1]; ++o)
			for (unsigned int n=0; n<3; ++n)
				for (size_t i=0; i<G.count; ++i)
					if (orders[o].c_ext && orders[o].c_ext[n][i]!=0)
						mask[i] |= 1u<<n;
		G.pos = d->Alloc<size_t>(G.count, first.flat.data());
		G.mask = d->Alloc<unsigned char>(G.count, mask.data());
		for (unsigned int k=0; k<G.orders; ++k)
		{
			const HostOrder& o = orders[starts[s]+k];
			G.ade[k] = d->Alloc<float>(3*G.count);
			G.lor[k] = d->Alloc<float>(3*G.count);
			G.c_int[k] = Coefficients(G.count, o.c_int);
			G.c_ext[k] = Coefficients(G.count, o.c_ext);
			G.c_lor[k] = Coefficients(G.count, o.c_lor);
		}
		groups.push_back(grp);
		if (voltage)
		{
			// the voltage ADE changes voltages between the half-steps: the fused step applies the
			// group in its kernel, or recomputes the currents they feed (see DecideFusedStep())
			HIP_ADECandidate C;
			C.count = (unsigned int)G.count;
			C.orders = G.orders;
			C.flat = first.flat;
			C.mask = mask;
			for (unsigned int k=0; k<G.orders; ++k)
				C.ade[k] = G.ade[k];
			const size_t index = groups.size()-1;
			C.fold = [this, index]() {m_Volt[index].folded = true;};
			d->ade_candidates.push_back(C);
		}
	}
}

GPU_Extension* HIP_CreateExt_LorentzMaterial(GPU_Backend_HIP::Impl* d, Engine_Extension* eng_ext, Engine* eng)
{
	UNUSED(eng);
	if (!dynamic_cast<Engine_Ext_LorentzMaterial*>(eng_ext))
		return NULL;
	Operator_Ext_LorentzMaterial* op_ext = dynamic_cast<Operator_Ext_LorentzMaterial*>(eng_ext->GetOperatorExtension());
	if (!op_ext)
		return NULL;
	return new HIP_Ext_LorentzMaterial(d, op_ext);
}
