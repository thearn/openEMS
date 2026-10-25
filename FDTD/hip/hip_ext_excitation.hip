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
#include <set>
#include <algorithm>
#include <vector>

#include "hip_internal.h"
#include "FDTD/engine.h"
#include "FDTD/excitation.h"
#include "FDTD/extensions/operator_ext_excitation.h"
#include "FDTD/extensions/engine_ext_excitation.h"

// Soft voltage/current excitation, see Engine_Ext_Excitation.
// One thread per excitation point, or one thread per edge if points share an edge.
struct ExcitationParam { int numTS; int period; int length; unsigned int count; int zero_start; };

__device__ __forceinline__ void excite(float* field, const size_t* index, const float* amp,
                                       const unsigned int* delay, const float* signal, const ExcitationParam& P, unsigned int n)
{
	int exc_pos = P.numTS - (int)delay[n];
	exc_pos *= (exc_pos>0);
	exc_pos %= P.period;
	exc_pos *= (exc_pos<P.length);
	// outside its window an entry adds amp*signal[0], and signal[0] is 0 (every built-in signal starts at 0; checked at
	// setup, P.zero_start): skip the index, amplitude and field traffic (bit-identical; the adjoint's delayed sources
	// are mostly outside their windows)
	if (exc_pos==0 && P.zero_start)
		return;
	field[index[n]] = field[index[n]] + amp[n]*signal[exc_pos];
}

__global__ void excitation_apply(float* field, const size_t* index, const float* amp,
                                 const unsigned int* delay, const float* signal, ExcitationParam P)
{
	const unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;
	if (n<P.count)
		excite(field, index, amp, delay, signal, P, n);
}

// entries that share an edge, sorted by edge (stable): one thread per edge applies its entries in their order, so each
// edge sees the additions of the host in the same order (bit-identical), and edges run in parallel
__global__ void excitation_apply_segments(float* field, const size_t* index, const float* amp,
                                          const unsigned int* delay, const float* signal, ExcitationParam P,
                                          const unsigned int* seg_start, unsigned int num_segments)
{
	const unsigned int s = blockIdx.x*blockDim.x + threadIdx.x;
	if (s>=num_segments)
		return;
	for (unsigned int n=seg_start[s]; n<seg_start[s+1]; ++n)
		excite(field, index, amp, delay, signal, P, n);
}

class HIP_Ext_Excitation : public GPU_Extension
{
public:
	HIP_Ext_Excitation(GPU_Backend_HIP::Impl* impl, Operator_Ext_Excitation* op_ext, Engine* eng);

	virtual void Apply2Voltages() {Apply(m_Volt, d->volt);}
	virtual void Apply2Current() {Apply(m_Curr, d->curr);}

protected:
	struct Sources
	{
		unsigned int count;
		bool serial;            //!< points share an edge: sorted by edge, applied per edge (excitation_apply_segments)
		unsigned int num_segments; //!< distinct edges when serial
		unsigned int* seg_start;   //!< first entry of each distinct edge (num_segments+1)
		size_t* index;          //!< flat NIJK index of the excited edge
		float* amp;
		unsigned int* delay;
		float* signal;
		unsigned int min_delay;   //!< no entry is in its window before this step ...
		unsigned int max_delay;   //!< ... nor after max_delay + length (a non-periodic signal starting at 0)
		bool zero_start;          //!< signal[0] == 0
	};

	void Setup(Sources& s, unsigned int count, unsigned int* const index[3], const unsigned short* dir,
	           const FDTD_FLOAT* amp, const unsigned int* delay, const FDTD_FLOAT* signal);
	void Apply(Sources& s, float* field);

	GPU_Backend_HIP::Impl* d;
	Operator_Ext_Excitation* m_Op_Exc;
	Engine* m_Eng;

	Sources m_Volt;
	Sources m_Curr;
};

HIP_Ext_Excitation::HIP_Ext_Excitation(GPU_Backend_HIP::Impl* impl, Operator_Ext_Excitation* op_ext, Engine* eng)
{
	d = impl;
	m_Op_Exc = op_ext;
	m_Eng = eng;
	Setup(m_Volt, op_ext->Volt_Count, op_ext->Volt_index, op_ext->Volt_dir, op_ext->Volt_amp, op_ext->Volt_delay, op_ext->m_Exc->GetVoltageSignal());
	Setup(m_Curr, op_ext->Curr_Count, op_ext->Curr_index, op_ext->Curr_dir, op_ext->Curr_amp, op_ext->Curr_delay, op_ext->m_Exc->GetCurrentSignal());

	// fused step (see update_fused): the currents next to the excited voltages are recomputed;
	// excited currents on the last mesh lines (never updated) would differ between its buffers
	for (unsigned int n=0; n<op_ext->Volt_Count; ++n)
		d->volt_modified.push_back((((size_t)op_ext->Volt_dir[n]*d->dim.nx + op_ext->Volt_index[0][n])*d->dim.ny + op_ext->Volt_index[1][n])*d->dim.nz + op_ext->Volt_index[2][n]);
	for (unsigned int n=0; n<op_ext->Curr_Count; ++n)
		if ((op_ext->Curr_index[0][n]+1>=d->dim.nx) || (op_ext->Curr_index[1][n]+1>=d->dim.ny) || (op_ext->Curr_index[2][n]+1>=d->dim.nz))
			d->fused_blockers++;
}

void HIP_Ext_Excitation::Setup(Sources& s, unsigned int count, unsigned int* const index[3], const unsigned short* dir,
                                const FDTD_FLOAT* amp, const unsigned int* delay, const FDTD_FLOAT* signal)
{
	s.count = count;
	s.serial = false;
	if (count==0)
		return;
	s.min_delay = *std::min_element(delay, delay+count);
	s.max_delay = *std::max_element(delay, delay+count);
	s.zero_start = (signal[0]==0);

	std::vector<size_t> flat(count);
	std::set<size_t> edges;
	for (unsigned int n=0; n<count; ++n)
	{
		flat[n] = (((size_t)dir[n]*d->dim.nx + index[0][n])*d->dim.ny + index[1][n])*d->dim.nz + index[2][n];
		if (!edges.insert(flat[n]).second)
			s.serial = true;
	}
	s.num_segments = 0;
	s.seg_start = NULL;
	if (!s.serial)
	{
		s.index = d->Alloc<size_t>(count, flat.data());
		s.amp = d->Alloc<float>(count, amp);
		s.delay = d->Alloc<unsigned int>(count, delay);
	}
	else
	{
		std::vector<unsigned int> order(count);
		for (unsigned int n=0; n<count; ++n)
			order[n] = n;
		std::stable_sort(order.begin(), order.end(), [&flat](unsigned int a, unsigned int b) {return flat[a]<flat[b];});
		std::vector<size_t> f(count);
		std::vector<unsigned int> dl(count), starts;
		std::vector<float> am(count);
		for (unsigned int n=0; n<count; ++n)
		{
			f[n] = flat[order[n]]; am[n] = amp[order[n]]; dl[n] = delay[order[n]];
			if (n==0 || f[n]!=f[n-1])
				starts.push_back(n);
		}
		s.num_segments = starts.size();
		starts.push_back(count);
		s.index = d->Alloc<size_t>(count, f.data());
		s.amp = d->Alloc<float>(count, am.data());
		s.delay = d->Alloc<unsigned int>(count, dl.data());
		s.seg_start = d->Alloc<unsigned int>(starts.size(), starts.data());
	}
	s.signal = d->Alloc<float>(m_Op_Exc->m_Exc->GetLength(), signal);
}

void HIP_Ext_Excitation::Apply(Sources& s, float* field)
{
	if (s.count==0)
		return;

	ExcitationParam param;
	param.numTS = m_Eng->GetNumberOfTimesteps();
	param.period = param.numTS+1;
	const bool periodic = m_Op_Exc->m_Exc->GetSignalPeriod()>0;
	if (periodic)
		param.period = int(m_Op_Exc->m_Exc->GetSignalPeriod()/m_Op_Exc->m_Exc->GetTimestep());
	param.length = m_Op_Exc->m_Exc->GetLength();
	param.count = s.count;
	param.zero_start = s.zero_start;
	// every entry outside its window (before the first delay, or after the last delay's whole signal): nothing to add
	if (s.zero_start && !periodic &&
	    ((param.numTS<=(int)s.min_delay) || (param.numTS>=(int)s.max_delay+param.length)))
		return;

	if (s.serial)
		HIP_Launch(d, "excitation_apply_segments", excitation_apply_segments, s.num_segments, 1, 1,
		            field, (const size_t*)s.index, (const float*)s.amp, (const unsigned int*)s.delay, (const float*)s.signal, param,
		            (const unsigned int*)s.seg_start, s.num_segments);
	else
		HIP_Launch(d, "excitation_apply", excitation_apply, s.count, 1, 1,
		            field, (const size_t*)s.index, (const float*)s.amp, (const unsigned int*)s.delay, (const float*)s.signal, param);
}

GPU_Extension* HIP_CreateExt_Excitation(GPU_Backend_HIP::Impl* d, Engine_Extension* eng_ext, Engine* eng)
{
	if (!dynamic_cast<Engine_Ext_Excitation*>(eng_ext))
		return NULL;
	Operator_Ext_Excitation* op_ext = dynamic_cast<Operator_Ext_Excitation*>(eng_ext->GetOperatorExtension());
	if (!op_ext)
		return NULL;
	return new HIP_Ext_Excitation(d, op_ext, eng);
}
