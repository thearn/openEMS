/*
*	Copyright (C) 2010 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#include "processfields_fd.h"
#include "Common/operator_base.h"
#include "tools/vtk_file_writer.h"
#include "tools/hdf5_file_writer.h"
#include "async_dumps.h"
#include <iomanip>
#include <sstream>
#include <string>

using namespace std;

ProcessFieldsFD::ProcessFieldsFD(Engine_Interface_Base* eng_if) : ProcessFields(eng_if)
{
	m_FieldDFT = -1;
	m_Snapshots = false;
	m_AsyncUsed = false;
	m_AsyncFailed = false;
}

ProcessFieldsFD::~ProcessFieldsFD()
{
	FinishAsync();
	delete m_CP_File;
	m_CP_File = NULL;
	for (size_t n = 0; n<m_FD_Fields.size(); ++n)
	{
		delete m_FD_Fields.at(n);
	}
	m_FD_Fields.clear();
}

void ProcessFieldsFD::InitProcess()
{
	m_FieldDFT = -1;
	m_Snapshots = false;
	if (Enabled==false) return;

	if (m_FD_Samples.size()==0)
	{
		cerr << "ProcessFieldsFD::InitProcess: No frequencies found... skipping this dump!" << endl;
		Enabled=false;
		return;
	}

	//setup the hdf5 file
	ProcessFields::InitProcess();

	if (m_Vtk_Dump_File)
		m_Vtk_Dump_File->SetHeader(string("openEMS FD Field Dump -- Interpolation: ")+m_Eng_Interface->GetInterpolationTypeString());

	if (m_HDF5_Dump_File)
	{
		m_HDF5_Dump_File->SetCurrentGroup("/FieldData/FD");
		m_HDF5_Dump_File->WriteAttribute("/FieldData/FD","frequency",m_FD_Samples);
		const char* every = getenv("OPENEMS_FD_CHECKPOINT_EVERY");
		const char* start = getenv("OPENEMS_FD_CHECKPOINT_START_S");
		if (every && atoi(every)>0)
		{
			m_CP_Every = atoi(every);
			m_CP_Start = start ? atof(start) : 0;
			delete m_CP_File;
			// OPENEMS_FD_CHECKPOINT_DIR places the checkpoints elsewhere (e.g. a RAM disk)
			const char* dir = getenv("OPENEMS_FD_CHECKPOINT_DIR");
			std::string name = m_filename;
			if (dir && *dir)
			{
				size_t slash = m_filename.find_last_of('/');
				name = std::string(dir) + "/" + (slash==std::string::npos ? m_filename : m_filename.substr(slash+1));
			}
			m_CP_File = new HDF5_File_Writer(name+"_cp.h5");
			m_CP_File->SetKeepOpen(true);
			m_CP_File->SetCurrentGroup("/FD_Checkpoints");
			m_CP_File->WriteAttribute("/FD_Checkpoints","frequency",m_FD_Samples);
			m_CP_File->WriteAttribute("/FD_Checkpoints","sample_interval_timesteps",(float)m_FD_Interval);
			m_CP_File->WriteAttribute("/FD_Checkpoints","timestep_s",(float)Op->GetTimestep());
		}
	}

	//create data structures...
	for (size_t n = 0; n<m_FD_Samples.size(); ++n)
	{
		ArrayLib::ArrayNIJK<std::complex<float>>* field_fd = new ArrayLib::ArrayNIJK<std::complex<float>>("FD field", numLines);
		m_FD_Fields.push_back(field_fd);
	}

	// let the engine keep the sums (e.g. on the device), else sum them from field snapshots
	// in the background; both need the node gather and have to be set up before the first
	// snapshot, so not for the SAR dumps, whose dump type has no gather
	if (GetGather())
	{
		m_FieldDFT = m_Eng_Interface->CreateFieldDFT(m_Gather, m_FD_Samples.size());
		if (m_FieldDFT<0)
			m_Snapshots = m_Eng_Interface->PrepareSnapshotGather(m_Gather);
	}
}

int ProcessFieldsFD::Process()
{
	if (Enabled==false) return -1;
	if (CheckTimestep()==false) return GetNextInterval();

	if ((m_FD_Interval==0) || (m_Eng_Interface->GetNumberOfTimesteps()%m_FD_Interval!=0))
		return GetNextInterval();

	std::vector<std::complex<float>> weights(m_FD_Samples.size());
	double T = m_Eng_Interface->GetTime(m_dualTime);
	for (size_t n = 0; n<m_FD_Samples.size(); ++n)
	{
		std::complex<float> exp_jwt_2_dt = std::exp( (std::complex<float>)(-2.0 * I_UNIT * PI * m_FD_Samples.at(n) * T) );
		exp_jwt_2_dt *= 2; // *2 for single-sided spectrum
		exp_jwt_2_dt *= Op->GetTimestep() * m_FD_Interval; // multiply with timestep-interval
		weights[n] = exp_jwt_2_dt;
	}

	++m_FD_SampleCount;

	// summed by the engine with the same weights: the fields never leave the device
	if (m_FieldDFT>=0)
	{
		m_Eng_Interface->AccumulateFieldDFT(m_FieldDFT, weights);
		if (m_CP_File) Checkpoint(T);
		return GetNextInterval();
	}

	// else from a field snapshot, summed in the background while the engine continues;
	// AsyncDumps keeps the task order, so the samples are added in the order taken
	int slot = -1;
	const float *volt = NULL, *curr = NULL;
	if (m_Snapshots && AsyncDumps::Get().Snapshot(m_Eng_Interface, m_Eng_Interface->GetNumberOfTimesteps(), slot, volt, curr))
	{
		const float* src = (m_DumpType==H_FIELD_DUMP) ? curr : volt;
		m_AsyncUsed = true;
		AsyncDumps::Get().Push(this, slot, [this, src, slot, weights]()
		{
			m_Eng_Interface->WaitFieldSnapshot(slot);   // the copy may still run on the device
			ArrayLib::ArrayNIJK<FDTD_FLOAT> field;
			if (CalcField(field, src))
				AddSample(field, weights);
			else
				m_AsyncFailed = true;
		});
		return GetNextInterval();
	}

	ArrayLib::ArrayNIJK<FDTD_FLOAT> tmp_field_td;
	if (!CalcField(tmp_field_td))
		return -1;
	AddSample(tmp_field_td, weights);
	if (m_CP_File) Checkpoint(T);
	return GetNextInterval();
}

void ProcessFieldsFD::Checkpoint(double T)
{
	if (T < m_CP_Start)
		return;
	++m_CP_Samples;
	if (m_CP_Samples % m_CP_Every != 0)
		return;
	FinishAsync();
	if ((m_FieldDFT>=0) && !m_Eng_Interface->ReadFieldDFT(m_FieldDFT, m_FD_Fields))
	{
		cerr << "ProcessFieldsFD::Checkpoint: can't read the frequency domain fields of the engine!" << endl;
		return;
	}
	for (size_t n = 0; n<m_FD_Samples.size(); ++n)
	{
		stringstream ss;
		ss << "cp" << m_CP_Count << "_f" << n;
		if (!m_CP_File->WriteVectorField<std::complex<float>>(ss.str(), *m_FD_Fields.at(n)))
			cerr << "ProcessFieldsFD::Checkpoint: can't write " << ss.str() << endl;
		m_CP_File->WriteAttribute("/FD_Checkpoints/"+ss.str(), "time", (double)T);
		m_CP_File->WriteAttribute("/FD_Checkpoints/"+ss.str(), "timestep", (float)m_Eng_Interface->GetNumberOfTimesteps());
		m_CP_File->WriteAttribute("/FD_Checkpoints/"+ss.str(), "samples", (float)m_FD_SampleCount);
	}
	++m_CP_Count;
}

void ProcessFieldsFD::AddSample(const ArrayLib::ArrayNIJK<FDTD_FLOAT>& field, const std::vector<std::complex<float>>& weights)
{
	const FDTD_FLOAT* field_td = field.data();
	for (size_t n = 0; n<m_FD_Samples.size(); ++n)
	{
		unsigned int N = m_FD_Fields.at(n)->size();
		std::complex<float>* field_fd = m_FD_Fields.at(n)->data();
		for (unsigned int ijk=0;ijk<N;++ijk)
			field_fd[ijk] += field_td[ijk] * weights[n];
	}
}

void ProcessFieldsFD::FinishAsync()
{
	if (!m_AsyncUsed)
		return;
	AsyncDumps::Get().Wait(this);
	m_AsyncUsed = false;
	if (m_AsyncFailed)
	{
		cerr << "ProcessFieldsFD: can't calculate the field of a sample... the frequency domain dump is incomplete!" << endl;
		m_AsyncFailed = false;
	}
}

void ProcessFieldsFD::PostProcess()
{
	FinishAsync();
	if (m_CP_File)
		m_CP_File->Close();
	if ((m_FieldDFT>=0) && !m_Eng_Interface->ReadFieldDFT(m_FieldDFT, m_FD_Fields))
		cerr << "ProcessFieldsFD::PostProcess: can't read the frequency domain fields of the engine!" << endl;
	DumpFDData();
}

void ProcessFieldsFD::DumpFDData()
{
	if (m_fileType==VTK_FILETYPE)
	{
		ArrayLib::ArrayNIJK<float> tmp_field("field", numLines);
		std::complex<float>* field_fd = NULL;
		float* field = tmp_field.data();
		double angle=0;
		int Nr_Ph = 21;

		for (size_t n = 0; n<m_FD_Samples.size(); ++n)
		{
			std::string str_freq;
			double freq = m_FD_Samples.at(n);
			if ((freq-long(freq))==0)
				// if the frequency is a natural number, export as long
				str_freq = std::to_string(long(freq));
			else
				str_freq = std::to_string(freq);

			unsigned int N = m_FD_Fields.at(n)->size();
			field_fd = m_FD_Fields.at(n)->data();
			//dump multiple phase to vtk-files
			for (int p=0; p<Nr_Ph; ++p)
			{
				angle = 2.0 * PI * p / Nr_Ph;
				std::complex<float> exp_jwt = std::exp( (std::complex<float>)(I_UNIT * angle) );
				for (unsigned int ijk=0;ijk<N;++ijk)
					field[ijk] = real(field_fd[ijk] * exp_jwt);

				stringstream ss;
				ss << m_filename << fixed << "_f=" << str_freq << "_p=" << std::setw( 3 ) << std::setfill( '0' ) <<(int)(angle * 180 / PI);

				m_Vtk_Dump_File->SetFilename(ss.str());
				m_Vtk_Dump_File->ClearAllFields();
				m_Vtk_Dump_File->AddVectorField(GetFieldNameByType(m_DumpType),tmp_field);
				if (m_Vtk_Dump_File->Write()==false)
					cerr << "ProcessFieldsFD::Process: can't dump to file... abort! " << endl;
			}

			{
				//dump magnitude to vtk-files
				for (unsigned int ijk=0;ijk<N;++ijk)
					field[ijk] = abs(field_fd[ijk]);

				stringstream ss;
				ss << m_filename << fixed << "_f=" << str_freq << "_abs";
				m_Vtk_Dump_File->SetFilename(ss.str());
				m_Vtk_Dump_File->ClearAllFields();
				m_Vtk_Dump_File->AddVectorField(GetFieldNameByType(m_DumpType),tmp_field);
				if (m_Vtk_Dump_File->Write()==false)
					cerr << "ProcessFieldsFD::Process: can't dump to file... abort! " << endl;
			}

			{
				//dump phase to vtk-files
				for (unsigned int ijk=0;ijk<N;++ijk)
					field[ijk] = arg(field_fd[ijk]);

				stringstream ss;
				ss << m_filename << fixed << "_f=" << str_freq << "_arg";
				m_Vtk_Dump_File->SetFilename(ss.str());
				m_Vtk_Dump_File->ClearAllFields();
				m_Vtk_Dump_File->AddVectorField(GetFieldNameByType(m_DumpType),tmp_field);
				if (m_Vtk_Dump_File->Write()==false)
					cerr << "ProcessFieldsFD::Process: can't dump to file... abort! " << endl;
			}
		}
		return;
	}

	if (m_fileType==HDF5_FILETYPE)
	{
		for (size_t n = 0; n<m_FD_Samples.size(); ++n)
		{
			stringstream ss;
			ss << "f" << n;
			// size_t datasize[]={numLines[0],numLines[1],numLines[2]};
			if (m_HDF5_Dump_File->WriteVectorField<std::complex<float>>(ss.str(), *m_FD_Fields.at(n), g_settings.GetLegacyHDF5Dumps())==false)
				cerr << "ProcessFieldsFD::Process: can't dump to file...! " << endl;

			if (g_settings.GetLegacyHDF5Dumps())
			{
				if (m_HDF5_Dump_File->WriteAttribute("/FieldData/FD/"+ss.str()+"_real", "frequency", m_FD_Samples.at(n))==false)
					cerr << "ProcessFieldsFD::Process: can't dump to file...! " << endl;
				if (m_HDF5_Dump_File->WriteAttribute("/FieldData/FD/"+ss.str()+"_imag", "frequency", m_FD_Samples.at(n))==false)
					cerr << "ProcessFieldsFD::Process: can't dump to file...! " << endl;
			}
			else if (m_HDF5_Dump_File->WriteAttribute("/FieldData/FD/"+ss.str(), "frequency", m_FD_Samples.at(n))==false)
				cerr << "ProcessFieldsFD::Process: can't dump to file...! " << endl;
		}
		return;
	}

	cerr << "ProcessFieldsFD::Process: unknown File-Type" << endl;
}
