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

#ifndef OPERATOR_GPU_H
#define OPERATOR_GPU_H

#include "operator.h"

#include <string>
#include <vector>

class GPU_Backend;

//! Operator for the GPU engine
/*!
  The operator is built on the host like the basic operator. Engine_GPU uploads
  the final coefficients to the device.
  */
class Operator_GPU : public Operator
{
public:
	//! Create a new operator for the GPU backend \a backend (see GPU_Backend::New), which it takes over and hands on to its engine
	static Operator_GPU* New(GPU_Backend* backend);
	virtual ~Operator_GPU();

	virtual Engine* CreateEngine();

	//! the GPU backend kind a later engine of this operator is created on (GPU_Backend::New): "auto" or "reference"
	void SetBackendKind(const std::string& kind) {m_BackendKind = kind;}
	//! keep the host coefficients after CreateEngine(), for a later engine of the same operator (openEMS::RestartFDTD)
	void SetKeepCoefficients(bool keep) {m_KeepCoefficients = keep;}

protected:
	//! use New() for creating a new Operator
	Operator_GPU(GPU_Backend* backend);

	//! Calc_EC_Range(), CalcPEC_Range() and CalcUpdateCoefficientsRange() over x line ranges in parallel
	/*!
	  Operator_GPU derives from the basic Operator, whose storage the host mirror
	  of Engine_GPU uses; Operator_Multithread is an Operator_SSE_Compressed and
	  brings the wrong one. Only these three build steps are worth repeating here.
	  */
	virtual bool Calc_EC();
	virtual bool CalcPEC();
	virtual void CalcUpdateCoefficients();
	virtual double DeviceLanczosMaxEig(const float* SC, const float* IL, size_t i0, size_t sx, size_t sy, size_t sz, int iterations) const;

	//! x line ranges of the threads, at most one per available CPU, bounds inclusive
	void ThreadRanges(std::vector<unsigned int>& start, std::vector<unsigned int>& stop) const;

	GPU_Backend* m_Backend; //!< until CreateEngine()
	std::string m_BackendKind = "auto";
	bool m_KeepCoefficients = false;
};

#endif // OPERATOR_GPU_H
