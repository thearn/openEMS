/*
* Copyright (C) 2026 Sean Mollet (sean@malmoset.com)
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

#include "cuda_palette_benchmark.h"
#include "cuda_local_palette.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace
{
const unsigned int TY=7, TZ=31;

void check(cudaError_t e, const char* what)
{
	if (e!=cudaSuccess) throw std::runtime_error(std::string("CUDA palette benchmark: ")+what+": "+cudaGetErrorString(e));
}

struct DeviceBuffers
{
	std::vector<void*> p;
	~DeviceBuffers() {for (void* x:p) cudaFree(x);}
	template<class T> T* copy(const std::vector<T>& v, cudaStream_t stream)
	{
		T* d=NULL; check(cudaMalloc(&d,std::max((size_t)1,v.size())*sizeof(T)),"cudaMalloc"); p.push_back(d);
		if (!v.empty()) check(cudaMemcpyAsync(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice,stream),"cudaMemcpy");
		return d;
	}
	template<class T> T* alloc(size_t n)
	{
		T* d=NULL; check(cudaMalloc(&d,std::max((size_t)1,n)*sizeof(T)),"cudaMalloc"); p.push_back(d); return d;
	}
};

__device__ __forceinline__ uint32_t mix_coefficients(const float* table, unsigned int set, uint32_t h)
{
	#pragma unroll
	for (unsigned int k=0;k<12;++k)
	{
		h=(h<<5)|(h>>27);
		h^=__float_as_uint(table[12*set+k])+0x9e3779b9u+k;
	}
	return h;
}

template<bool LOCAL>
__global__ void palette_decode(const void* global_index, const float* global_table, unsigned int global_mode,
	const unsigned char* local_index, const float* local_table, const CUDA_PaletteTile* meta,
	unsigned int nx, unsigned int ny, unsigned int nz, unsigned int xc,
	unsigned int gy, unsigned int gz, uint32_t* output)
{
	const unsigned int tile=blockIdx.x;
	const unsigned int bz=tile%gz, by=(tile/gz)%gy, bx=tile/(gz*gy);
	const unsigned int x0=bx*xc, y0=by*TY, z0=bz*TZ;
	const unsigned int owned_x=min(xc,nx-x0), owned_y=min(TY,ny-y0), owned_z=min(TZ,nz-z0);
	const unsigned int ly=threadIdx.y, lz=threadIdx.x;
	const CUDA_PaletteTile m=LOCAL?meta[tile]:CUDA_PaletteTile{0,0,(uint16_t)min(owned_x+1,nx-x0),
		(uint16_t)min(owned_y+1,ny-y0),(uint16_t)min(owned_z+1,nz-z0),0,0};
	uint32_t h=0x811c9dc5u^(tile*257u+ly*32u+lz);
	auto consume = [&](unsigned int lx)
	{
		unsigned int set;
		if (LOCAL)
		{
			const size_t q=(size_t)m.index_offset+(((size_t)lx*m.ny+ly)*m.nz+lz)*m.width;
			const unsigned int local=m.width==1 ? local_index[q] : ((const uint16_t*)local_index)[q/2];
			set=m.dictionary_offset+local;
			h=mix_coefficients(local_table,set,h);
		}
		else
		{
			const size_t q=((size_t)(x0+lx)*ny+(y0+ly))*nz+(z0+lz);
			set=global_mode==1 ? ((const uint16_t*)global_index)[q] : ((const uint32_t*)global_index)[q];
			h=mix_coefficients(global_table,set,h);
		}
	};
	if (ly<m.ny && lz<m.nz)
		for (unsigned int lx=0;lx<m.nx;++lx) consume(lx);
	if (ly<owned_y && lz<owned_z && y0+ly+1<ny && z0+lz+1<nz)
		for (unsigned int lx=0;lx<owned_x && x0+lx+1<nx;++lx) consume(lx);
	output[(size_t)tile*256+ly*32+lz]=h;
}

uint64_t hash64(const std::vector<uint32_t>& v)
{
	uint64_t h=1469598103934665603ull;
	for (uint32_t x:v) for (unsigned int k=0;k<4;++k) {h^=(x>>(8*k))&255;h*=1099511628211ull;}
	return h;
}

}

void CUDA_RunPaletteBenchmark(const CUDA_PaletteBenchmarkInput& in)
{
	if (in.path.empty() || !in.nx || !in.ny || !in.nz || !in.xc || !in.global_index || !in.global_table ||
	    (in.coefficients.mode!=1 && in.coefficients.mode!=2) ||
	    in.coefficients.index.size()!=(size_t)in.nx*in.ny*in.nz || in.coefficients.table.size()!=12*in.coefficients.count)
		throw std::runtime_error("invalid CUDA palette benchmark input");
	CUDA_LocalPalette palette=CUDA_BuildLocalPalette(in.coefficients,in.nx,in.ny,in.nz,in.xc);
	const auto verification_start=std::chrono::steady_clock::now();
	CUDA_ValidateLocalPalette(palette,in.coefficients);
	const double verification_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-verification_start).count();
	const double encoding_s=palette.encoding_s;
	const double construction_and_verification_s=encoding_s+verification_s;
	const unsigned int gy=palette.gy,gz=palette.gz;
	const size_t tiles=palette.meta.size();
	DeviceBuffers dev;const std::chrono::steady_clock::time_point upload_start=std::chrono::steady_clock::now();
	CUDA_PaletteTile* dmeta=dev.copy(palette.meta,in.stream);unsigned char* dindex=dev.copy(palette.index,in.stream);float* dtable=dev.copy(palette.table,in.stream);
	const size_t outputs=tiles*256;uint32_t* dout=dev.alloc<uint32_t>(outputs);
	check(cudaStreamSynchronize(in.stream),"encoding upload");
	const double upload_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-upload_start).count();
	const dim3 block(32,8,1),grid(tiles,1,1);
	auto launch_global=[&](){palette_decode<false><<<grid,block,0,in.stream>>>(in.global_index,in.global_table,in.coefficients.mode,dindex,dtable,dmeta,in.nx,in.ny,in.nz,in.xc,gy,gz,dout);};
	auto launch_local=[&](){palette_decode<true><<<grid,block,0,in.stream>>>(in.global_index,in.global_table,in.coefficients.mode,dindex,dtable,dmeta,in.nx,in.ny,in.nz,in.xc,gy,gz,dout);};
	std::vector<uint32_t> control(outputs),candidate(outputs);
	launch_global();check(cudaMemcpyAsync(control.data(),dout,outputs*sizeof(uint32_t),cudaMemcpyDeviceToHost,in.stream),"control digest copy");check(cudaStreamSynchronize(in.stream),"control digest");
	launch_local();check(cudaMemcpyAsync(candidate.data(),dout,outputs*sizeof(uint32_t),cudaMemcpyDeviceToHost,in.stream),"local digest copy");check(cudaStreamSynchronize(in.stream),"local digest");
	if(control!=candidate)throw std::runtime_error("CUDA palette benchmark device digest mismatch");
	for(int i=0;i<2;++i){launch_global();launch_local();}check(cudaStreamSynchronize(in.stream),"warmup");
	cudaEvent_t start,stop;check(cudaEventCreate(&start),"event");check(cudaEventCreate(&stop),"event");
	auto measure=[&](bool local,unsigned int repeats)
	{
		check(cudaEventRecord(start,in.stream),"event start");
		for(unsigned int i=0;i<repeats;++i) {if(local)launch_local();else launch_global();}
		check(cudaEventRecord(stop,in.stream),"event stop");check(cudaEventSynchronize(stop),"event sync");float ms=0;check(cudaEventElapsedTime(&ms,start,stop),"event elapsed");return ms;
	};
	const float control_probe=measure(false,1),candidate_probe=measure(true,1);
	unsigned int repeats=std::max(1u,(unsigned int)std::ceil(300.0/std::max(0.01f,std::min(control_probe,candidate_probe))));
	float control_calibration=0,candidate_calibration=0;
	for(unsigned int attempt=0;attempt<3;++attempt)
	{
		control_calibration=measure(false,repeats);candidate_calibration=measure(true,repeats);
		const float shortest=std::min(control_calibration,candidate_calibration);
		if(shortest>=275)break;
		repeats=std::max(repeats+1,(unsigned int)std::ceil(repeats*300.0/std::max(1.0f,shortest)));
	}
	std::vector<float> global_ms,local_ms;global_ms.reserve(21);local_ms.reserve(21);
	for(unsigned int pair=0;pair<21;++pair)
	{
		float g,l;if(pair&1){l=measure(true,repeats);g=measure(false,repeats);}else{g=measure(false,repeats);l=measure(true,repeats);}
		global_ms.push_back(g);local_ms.push_back(l);
	}
	cudaEventDestroy(start);cudaEventDestroy(stop);
	auto median=[](std::vector<float> v){std::sort(v.begin(),v.end());return v[v.size()/2];};
	const float gm=median(global_ms),lm=median(local_ms);
	std::vector<float> global_deviation,local_deviation,pair_change;
	for(size_t i=0;i<global_ms.size();++i)
	{
		global_deviation.push_back(std::fabs(global_ms[i]-gm));
		local_deviation.push_back(std::fabs(local_ms[i]-lm));
		pair_change.push_back(100*(local_ms[i]/global_ms[i]-1));
	}
	const float gmad=median(global_deviation),lmad=median(local_deviation),pm=median(pair_change);
	std::vector<float> pair_deviation;for(float x:pair_change)pair_deviation.push_back(std::fabs(x-pm));
	const float pmad=median(pair_deviation);
	const uint64_t cells=(uint64_t)in.nx*in.ny*in.nz;
	const uint64_t global_index_bytes=cells*(in.coefficients.mode==1?2:4),global_table_bytes=in.coefficients.table.size()*sizeof(float);
	const uint64_t local_bytes=palette.bytes();
	const uint64_t host_payload_bytes=in.coefficients.index.capacity()*sizeof(uint32_t)+in.coefficients.table.capacity()*sizeof(float)+
		palette.index.capacity()+palette.table.capacity()*sizeof(float)+palette.meta.capacity()*sizeof(CUDA_PaletteTile)+palette.dictionary_id_entries*sizeof(uint32_t);
	std::ofstream out(in.path.c_str(),std::ios::out|std::ios::trunc);if(!out)throw std::runtime_error("cannot write CUDA palette report");
	out<<std::setprecision(10)<<"{\n  \"schema\":1,\n  \"dimensions\":{\"x\":"<<in.nx<<",\"y\":"<<in.ny<<",\"z\":"<<in.nz<<"},\n"
	   <<"  \"tiles\":"<<tiles<<",\"one_byte_tiles\":"<<palette.one_byte_tiles<<",\"two_byte_tiles\":"<<palette.two_byte_tiles<<",\"shared_dictionaries\":"<<palette.shared_dictionaries<<",\n"
	   <<"  \"logical_lookups_per_launch\":"<<palette.logical_lookups<<",\"encoding_construction_s\":"<<encoding_s<<",\"host_verification_s\":"<<verification_s
	   <<",\"construction_and_verification_s\":"<<construction_and_verification_s<<",\"upload_s\":"<<upload_s<<",\n"
	   <<"  \"global\":{\"index_bytes\":"<<global_index_bytes<<",\"table_bytes\":"<<global_table_bytes<<",\"total_bytes\":"<<global_index_bytes+global_table_bytes<<"},\n"
	   <<"  \"local\":{\"index_bytes\":"<<palette.index.size()<<",\"index_padding_bytes\":"<<palette.index_padding<<",\"table_bytes\":"<<palette.table.size()*sizeof(float)<<",\"metadata_bytes\":"<<palette.meta.size()*sizeof(CUDA_PaletteTile)<<",\"total_bytes\":"<<local_bytes<<"},\n"
	   <<"  \"calculated_bytes_per_launch\":{\"coefficient_values\":"<<palette.logical_lookups*48<<",\"global_indices\":"<<palette.logical_lookups*(in.coefficients.mode==1?2:4)<<",\"local_indices\":"<<palette.local_index_reads<<",\"local_metadata_requests\":"<<tiles*256*sizeof(CUDA_PaletteTile)<<"},\n"
	   <<"  \"temporary_device_bytes\":"<<local_bytes+outputs*sizeof(uint32_t)<<",\"host_payload_bytes\":"<<host_payload_bytes<<",\"output_digest_fnv64\":\""<<std::hex<<hash64(control)<<std::dec<<"\",\n"
	   <<"  \"benchmark\":{\"interval_target_ms\":250,\"calibration_target_ms\":300,\"repeats\":"<<repeats<<",\"control_probe_ms\":"<<control_probe<<",\"candidate_probe_ms\":"<<candidate_probe
	   <<",\"control_calibration_ms\":"<<control_calibration<<",\"candidate_calibration_ms\":"<<candidate_calibration<<",\"control_ms\":[";
	for(size_t i=0;i<global_ms.size();++i){if(i)out<<',';out<<global_ms[i];}out<<"],\"candidate_ms\":[";
	for(size_t i=0;i<local_ms.size();++i){if(i)out<<',';out<<local_ms[i];}out<<"],\"paired_change_percent\":[";
	for(size_t i=0;i<pair_change.size();++i){if(i)out<<',';out<<pair_change[i];}out<<"],\"control_median_ms\":"<<gm<<",\"candidate_median_ms\":"<<lm
	   <<",\"change_percent\":"<<100*(lm/gm-1)<<",\"paired_change_range_percent\":["<<*std::min_element(pair_change.begin(),pair_change.end())<<','<<*std::max_element(pair_change.begin(),pair_change.end())
	   <<"],\"control_mad_ms\":"<<gmad<<",\"candidate_mad_ms\":"<<lmad<<",\"paired_change_mad_percent\":"<<pmad
	   <<",\"control_effective_lookups_per_s\":"<<(double)palette.logical_lookups*repeats*1000/gm<<",\"candidate_effective_lookups_per_s\":"<<(double)palette.logical_lookups*repeats*1000/lm<<"},\n"
	   <<"  \"checks\":{\"host_decode_exact\":true,\"device_digest_exact\":true}\n}\n";
	if(!out)throw std::runtime_error("failed writing CUDA palette report");
}
