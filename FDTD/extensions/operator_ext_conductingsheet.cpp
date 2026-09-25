/*
*	Copyright (C) 2012 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#include "operator_ext_conductingsheet.h"
#include "tools/arraylib/array_nijk.h"
#include "tools/constants.h"
#include "cond_sheet_parameter.h"
#include "cond_sheet_parameter_complex4.h"

#include "CSPropConductingSheet.h"
#include "tools/useful.h"

#include <map>
#include <set>

using std::cerr;
using std::endl;

Operator_Ext_ConductingSheet::Operator_Ext_ConductingSheet(Operator* op, double f_max) : Operator_Ext_LorentzMaterial(op)
{
	m_f_max = f_max;
}

Operator_Ext_ConductingSheet::Operator_Ext_ConductingSheet(Operator* op, Operator_Ext_ConductingSheet* op_ext) : Operator_Ext_LorentzMaterial(op, op_ext)
{
	m_f_max = op_ext->m_f_max;
}

Operator_Extension* Operator_Ext_ConductingSheet::Clone(Operator* op)
{
	return new Operator_Ext_ConductingSheet(op, this);
}

bool Operator_Ext_ConductingSheet::BuildExtension()
{
	double dT = m_Op->GetTimestep();
	unsigned int pos[] = {0,0,0};
	double coord[3];
	unsigned int numLines[3] = {m_Op->GetNumberOfLines(0,true),m_Op->GetNumberOfLines(1,true),m_Op->GetNumberOfLines(2,true)};

	m_Order = 0;
	std::vector<unsigned int> v_pos[3];
	ArrayLib::ArrayNIJK<int8_t> tanDir("tanDir", numLines);
	ArrayLib::ArrayNIJK<float> Conductivity("Conductivity", numLines);
	ArrayLib::ArrayNIJK<float> Thickness("Thickness", numLines);

	// Each x line fills its own cells and position list; the lists are joined
	// in x order, so the result is identical to the serial scan.
	struct ScanResult
	{
		std::vector<unsigned int> pos[3];
		unsigned int nr_pec[3] = {0,0,0};
		std::set<CSPrimitives*> used;
		bool failed = false;
	};
	// CSXCAD primitives recompute their dimension inside GetBoundBox (reset to 0,
	// then counted up), so calling it from several scan threads lets another
	// thread read a transient dimension and wrongly fall back to PEC. Read each
	// sheet primitive's dimension and bounding box once, before the scan.
	struct SheetShape
	{
		int dimension;
		double box[6];
	};
	std::map<CSPrimitives*, SheetShape> sheetShapes;
	for (CSPrimitives* prim : m_Op->GetGeometryCSX()->GetAllPrimitives(false, CSProperties::ANY))
	{
		if (dynamic_cast<CSPropConductingSheet*>(prim->GetProperty())==NULL)
			continue;
		SheetShape& shape = sheetShapes[prim];
		prim->GetBoundBox(shape.box);
		shape.dimension = prim->GetDimension();
	}

	auto scan = [&](unsigned int xStart, unsigned int xStop, ScanResult* result) -> void
	{
		unsigned int pos[3] = {0,0,0};
		double coord[3];
		CSPrimitives* cs_sheet = NULL;
		int nP, nPP;
		bool b_pos_on;
		bool disable_pos;
		std::vector<unsigned int>* my_pos = result->pos;
		unsigned int* nr_pec = result->nr_pec;
		std::set<CSPrimitives*>& used = result->used;
		bool& failed = result->failed;
		for (pos[0]=xStart; pos[0]<=xStop; ++pos[0])
		{
			for (pos[1]=0; pos[1]<numLines[1]; ++pos[1])
			{
				std::vector<CSPrimitives*> vPrims = m_Op->GetPrimitivesBoundBox(
					pos[0], pos[1], -1,
					(CSProperties::PropertyType)(CSProperties::MATERIAL | CSProperties::METAL)
				);

				std::vector<CSPrimitives*> vPrimsZ;
				for (pos[2]=0; pos[2]<numLines[2]; ++pos[2])
				{
					m_Op->NarrowPrimitives(vPrims, vPrimsZ, pos[0], pos[1], pos[2]);
					b_pos_on = false;
					disable_pos = false;
					// disable conducting sheet model inside the boundary conditions, especially inside a pml
					for (int m=0;m<3;++m)
						if ((pos[m]<=(unsigned int)m_Op->GetBCSize(2*m)) || (pos[m]>=(numLines[m]-m_Op->GetBCSize(2*m+1)-1)))
							disable_pos = true;

					for (int n=0; n<3; ++n)
					{
						nP = (n+1)%3;
						nPP = (n+2)%3;

						tanDir(n, pos[0], pos[1], pos[2]) = -1; //deactivate by default
						Conductivity(n, pos[0], pos[1], pos[2]) = 0; //deactivate by default
						Thickness(n, pos[0], pos[1], pos[2]) = 0; //deactivate by default

						if (m_Op->GetYeeCoords(n,pos,coord,false)==false)
							continue;

						// Ez at r==0 not supported --> set to PEC
						if (m_CC_R0_included && (n==2) && (pos[0]==0))
							disable_pos = true;

	//					CSProperties* prop = m_Op->GetGeometryCSX()->GetPropertyByCoordPriority(coord,(CSProperties::PropertyType)(CSProperties::METAL | CSProperties::MATERIAL), false, &cs_sheet);
						CSProperties* prop = m_Op->GetGeometryCSX()->GetPropertyByCoordPriority(coord, vPrimsZ, false, &cs_sheet);
						CSPropConductingSheet* cs_prop = dynamic_cast<CSPropConductingSheet*>(prop);
						if (cs_prop)
						{
							if (cs_sheet==NULL)
							{
								failed = true; //sanity check, this should never happen
								return;
							}
							auto shape = sheetShapes.find(cs_sheet);
							if (shape==sheetShapes.end())
							{
								failed = true; //sanity check, every sheet primitive was recorded above
								return;
							}
							if (shape->second.dimension!=2)
							{
								cerr << "Operator_Ext_ConductingSheet::BuildExtension: A conducting sheet primitive (ID: " << cs_sheet->GetID() << ") with dimension: " << shape->second.dimension << " found, fallback to PEC!" << endl;
								m_Op->SetVV(n,pos[0],pos[1],pos[2], 0 );
								m_Op->SetVI(n,pos[0],pos[1],pos[2], 0 );
								++nr_pec[n];
								continue;
							}
							used.insert(cs_sheet);

							if (disable_pos)
							{
								m_Op->SetVV(n,pos[0],pos[1],pos[2], 0 );
								m_Op->SetVI(n,pos[0],pos[1],pos[2], 0 );
								++nr_pec[n];
								continue;
							}

							Conductivity(n, pos[0], pos[1], pos[2]) = cs_prop->GetConductivity();
							Thickness(n, pos[0], pos[1], pos[2]) = cs_prop->GetThickness();

							if ((Conductivity(n, pos[0], pos[1], pos[2])<=0) || (Thickness(n, pos[0], pos[1], pos[2])<=0))
							{
								cerr << "Operator_Ext_ConductingSheet::BuildExtension: Warning: Zero conductivity or thickness detected... fallback to PEC!" << endl;
								m_Op->SetVV(n,pos[0],pos[1],pos[2], 0 );
								m_Op->SetVI(n,pos[0],pos[1],pos[2], 0 );
								++nr_pec[n];
								continue;
							}

							const double* box = shape->second.box;
							if (box[2*nP]!=box[2*nP+1])
								tanDir(n, pos[0], pos[1], pos[2]) = nP;
							if (box[2*nPP]!=box[2*nPP+1])
								tanDir(n, pos[0], pos[1], pos[2]) = nPP;
							b_pos_on = true;
						}
					}
					if (b_pos_on)
					{
						for (int n=0; n<3; ++n)
							my_pos[n].push_back(pos[n]);
					}
				}
			}
		}
	};

	std::vector<ScanResult> results(numLines[0]);
	ParallelLines(numLines[0], SetupThreads(numLines[0]),
		[&](unsigned int x, unsigned int) { scan(x, x, &results[x]); });
	for (size_t t=0; t<results.size(); ++t)
	{
		if (results[t].failed)
			return false;
		for (int n=0; n<3; ++n)
		{
			v_pos[n].insert(v_pos[n].end(), results[t].pos[n].begin(), results[t].pos[n].end());
			m_Op->m_Nr_PEC[n] += results[t].nr_pec[n];
		}
		for (CSPrimitives* prim : results[t].used)
			prim->SetPrimitiveUsed(true);
	}

	size_t numCS = v_pos[0].size();
	if (numCS==0)
		return false;

	m_Order	= 4;
	for (int order=0; order<m_Order; ++order)
		m_LM_Count.push_back(numCS);

	m_volt_ADE_On = new bool[m_Order];
	m_curr_ADE_On = new bool[m_Order];
	m_volt_Lor_ADE_On = new bool[m_Order];
	m_curr_Lor_ADE_On = new bool[m_Order];
	for (int order=0; order<m_Order; ++order)
	{
		m_volt_ADE_On[order]=true;
		m_curr_ADE_On[order]=false;
		m_volt_Lor_ADE_On[order]=false;
		m_curr_Lor_ADE_On[order]=false;
	}

	m_LM_pos = new unsigned int**[m_Order];
	v_int_ADE = new FDTD_FLOAT**[m_Order];
	v_ext_ADE = new FDTD_FLOAT**[m_Order];
	for (int order=0; order<m_Order; ++order)
	{
		m_LM_pos[order] = new unsigned int*[3];
		v_int_ADE[order] = new FDTD_FLOAT*[3];
		v_ext_ADE[order] = new FDTD_FLOAT*[3];
	}

	for (int n=0; n<3; ++n)
	{
		for (int order=0; order<m_Order; ++order)
		{
			m_LM_pos[order][n] = new unsigned int[numCS];
			for (unsigned int i=0; i<numCS; ++i)
				m_LM_pos[order][n][i] = v_pos[n].at(i);
			v_int_ADE[order][n] = new FDTD_FLOAT[numCS];
			v_ext_ADE[order][n] = new FDTD_FLOAT[numCS];
		}
	}

	unsigned int index;
	float w_stop = m_f_max*2*PI;
	float Omega_max=0;
	float G,L[4],R[4],Lmin;
	float G0, w0;
	float wtl; //width to length factor
	float factor=1;
	int t_dir=0; //tangential sheet direction
	unsigned int tpos[] = {0,0,0};
	unsigned int optParaPos;
	for (unsigned int i=0;i<numCS;++i)
	{
		pos[0]=m_LM_pos[0][0][i];pos[1]=m_LM_pos[0][1][i];pos[2]=m_LM_pos[0][2][i];
		tpos[0]=pos[0];tpos[1]=pos[1];tpos[2]=pos[2];
		index = m_Op->MainOp->SetPos(pos[0],pos[1],pos[2]);
		for (int n=0;n<3;++n)
		{
			tpos[0]=pos[0];tpos[1]=pos[1];tpos[2]=pos[2];
			t_dir = tanDir(n, pos[0], pos[1], pos[2]);
			G0 = Conductivity(n, pos[0], pos[1], pos[2])*Thickness(n, pos[0], pos[1], pos[2]);
			w0 = 8.0/ G0 / Thickness(n, pos[0], pos[1], pos[2])/MUE0;
			Omega_max = w_stop/w0;
			for (optParaPos=0;optParaPos<numOptPara;++optParaPos)
				if (omega_stop[optParaPos]>Omega_max)
					break;
			if (optParaPos>=numOptPara)
			{
				cerr << "Operator_Ext_ConductingSheet::BuildExtension(): Error, conductor thickness, conductivity or max. simulation frequency of interest is too high! Check parameter!" << endl;
				cerr << " --> max f: " << m_f_max << "Hz,  Conductivity: " << Conductivity(n, pos[0], pos[1], pos[2]) << "S/m, Thickness " << Thickness(n, pos[0], pos[1], pos[2])*1e6 << "um" << endl;
				optParaPos = numOptPara-1;
			}
			for (int order=0; order<m_Order; ++order)
			{
				v_int_ADE[order][n][i]=0;
				v_ext_ADE[order][n][i]=0;
			}
			if (t_dir>=0)
			{
				wtl = m_Op->GetEdgeLength(n,pos)/m_Op->GetNodeWidth(t_dir,pos);
				factor = 1;
				if (tanDir(t_dir, tpos[0], tpos[1], tpos[2])<0)
					factor = 2;
				--tpos[t_dir];
				if (tanDir(t_dir, tpos[0], tpos[1], tpos[2])<0)
					factor = 2;

				// The legacy two-branch model is already accurate for Omega_max<1 and
				// has less stabilization capacitance.  Above that range, four passive
				// branches approximate the full complex sheet impedance.  The unused
				// branches below Omega=1 are open circuits.
				double normalized_L[4];
				double normalized_R[4];
				double normalized_G;
				if (Omega_max<1.0)
				{
					normalized_L[0]=l1[optParaPos]; normalized_L[1]=l2[optParaPos];
					normalized_R[0]=r1[optParaPos]; normalized_R[1]=r2[optParaPos];
					normalized_L[2]=normalized_L[3]=1e20;
					normalized_R[2]=normalized_R[3]=1e20;
					normalized_G=g[optParaPos];
				}
				else
				{
					normalized_L[0]=l1_complex4[optParaPos]; normalized_L[1]=l2_complex4[optParaPos];
					normalized_L[2]=l3_complex4[optParaPos]; normalized_L[3]=l4_complex4[optParaPos];
					normalized_R[0]=r1_complex4[optParaPos]; normalized_R[1]=r2_complex4[optParaPos];
					normalized_R[2]=r3_complex4[optParaPos]; normalized_R[3]=r4_complex4[optParaPos];
					normalized_G=g_complex4[optParaPos];
				}
				G = G0*normalized_G/factor;
				for (int order=0; order<m_Order; ++order)
				{
					L[order] = normalized_L[order]/G0/w0*factor*wtl;
					R[order] = normalized_R[order]/G0*factor*wtl;
				}
				G/=wtl;

				Lmin = L[0];
				for (int order=1; order<m_Order; ++order)
					if (L[order]<Lmin)
						Lmin = L[order];
				m_Op->EC_G[n][index]= G;
				m_Op->EC_C[n][index]= dT*dT/4.0*16.0/Lmin;
				for (int order=0; order<m_Order; ++order)
					m_Op->EC_C[n][index] += dT*dT/4.0/L[order];
				m_Op->Calc_ECOperatorPos(n,pos);

				for (int order=0; order<m_Order; ++order)
				{
					v_int_ADE[order][n][i]=(2.0*L[order]-dT*R[order])/(2.0*L[order]+dT*R[order]);
					v_ext_ADE[order][n][i]=dT/(L[order]+dT*R[order]/2.0)*m_Op->GetVI(n,pos[0],pos[1],pos[2]);
				}
			}
		}
	}
	return true;
}
