/*
*	Largest eigenvalue of the main-grid leapfrog operator on the GPU (solvers-2026-09 S1.6).
*
*	A = sqrt(C^-1) Ci L^-1 Ci^T sqrt(C^-1) with the engine's own stencils; SC = sqrt(1/C) and
*	IL = 1/L per edge (3 components, openEMS AdrOp layout). Lanczos with double reductions. The
*	current-like intermediate is recomputed inside the fused kernel, so only five vectors of 3N
*	floats are resident. Returns -1 when CUDA or the memory is unavailable (caller falls back).
*/
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cstdlib>

namespace {

struct Grid { unsigned N0, N1, N2; size_t i0, sx, sy, sz, total; int ax[3]; };

__device__ __forceinline__ size_t gidx(const Grid& g, unsigned x, unsigned y, unsigned z)
{ return g.i0 + x*g.sx + y*g.sy + z*g.sz; }

__device__ __forceinline__ float uval(const Grid& g, const float* SC, const float* in, int n, unsigned x, unsigned y, unsigned z)
{ size_t i = n*g.total + gidx(g,x,y,z); return SC[i]*in[i]; }

// t_n at (x,y,z): negative engine current-update curl of u, zero outside [0,N-1)^3
__device__ float tval(const Grid& g, const float* SC, const float* IL, const float* in, int n, unsigned x, unsigned y, unsigned z)
{
	if (x+1>=g.N0 || y+1>=g.N1 || z+1>=g.N2) return 0.f;
	size_t i = gidx(g,x,y,z);
	float c;
	if (n==0) c = uval(g,SC,in,2,x,y,z)-uval(g,SC,in,2,x,y+1,z)-uval(g,SC,in,1,x,y,z)+uval(g,SC,in,1,x,y,z+1);
	else if (n==1) c = uval(g,SC,in,0,x,y,z)-uval(g,SC,in,0,x,y,z+1)-uval(g,SC,in,2,x,y,z)+uval(g,SC,in,2,x+1,y,z);
	else c = uval(g,SC,in,1,x,y,z)-uval(g,SC,in,1,x+1,y,z)-uval(g,SC,in,0,x,y,z)+uval(g,SC,in,0,x,y+1,z);
	return -IL[n*g.total+i]*c;
}

__global__ void apply_kernel(Grid g, const float* SC, const float* IL, const float* in, float* out)
{
	size_t cell = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
	size_t cells = (size_t)g.N0*g.N1*g.N2;
	if (cell>=cells) return;
	// decode with the smallest-stride axis fastest (coalesced loads for any AdrOp layout)
	unsigned Nn[3] = {g.N0, g.N1, g.N2}, c[3];
	unsigned rem = (unsigned)cell;
	c[g.ax[0]] = rem % Nn[g.ax[0]]; rem /= Nn[g.ax[0]];
	c[g.ax[1]] = rem % Nn[g.ax[1]]; rem /= Nn[g.ax[1]];
	c[g.ax[2]] = rem;
	unsigned x = c[0], y = c[1], z = c[2];
	size_t i = gidx(g,x,y,z);
	float t0 = tval(g,SC,IL,in,0,x,y,z), t1 = tval(g,SC,IL,in,1,x,y,z), t2 = tval(g,SC,IL,in,2,x,y,z);
	float d2y = y>0 ? t2 - tval(g,SC,IL,in,2,x,y-1,z) : 0.f;
	float d1z = z>0 ? t1 - tval(g,SC,IL,in,1,x,y,z-1) : 0.f;
	float d0z = z>0 ? t0 - tval(g,SC,IL,in,0,x,y,z-1) : 0.f;
	float d2x = x>0 ? t2 - tval(g,SC,IL,in,2,x-1,y,z) : 0.f;
	float d1x = x>0 ? t1 - tval(g,SC,IL,in,1,x-1,y,z) : 0.f;
	float d0y = y>0 ? t0 - tval(g,SC,IL,in,0,x,y-1,z) : 0.f;
	out[0*g.total+i] = SC[0*g.total+i]*(d2y-d1z);
	out[1*g.total+i] = SC[1*g.total+i]*(d0z-d2x);
	out[2*g.total+i] = SC[2*g.total+i]*(d1x-d0y);
}

static const int DOT_BLOCKS = 1024;

__global__ void dot_kernel(const float* a, const float* b, size_t n, double* result)
{
	__shared__ double buf[256];
	double s = 0;
	for (size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i<n; i += (size_t)gridDim.x*blockDim.x)
		s += (double)a[i]*b[i];
	buf[threadIdx.x] = s; __syncthreads();
	for (int k=blockDim.x/2; k>0; k>>=1) { if (threadIdx.x<k) buf[threadIdx.x]+=buf[threadIdx.x+k]; __syncthreads(); }
	if (threadIdx.x==0) result[blockIdx.x] = buf[0];   // summed on the host in block order: run-to-run identical
}

__global__ void lanczos_update(float* w, const float* q, const float* qp, float a, float b, size_t n)
{
	for (size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i<n; i += (size_t)gridDim.x*blockDim.x)
		w[i] = w[i] - a*q[i] - b*qp[i];
}

__global__ void scale_copy(float* dst, const float* src, float s, size_t n)
{
	for (size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i<n; i += (size_t)gridDim.x*blockDim.x)
		dst[i] = src[i]*s;
}

__global__ void random_fill(float* v, size_t n)
{
	for (size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i<n; i += (size_t)gridDim.x*blockDim.x)
	{
		unsigned long long s = i*0x9E3779B97F4A7C15ULL + 88172645463325252ULL;
		s ^= s>>33; s *= 0xff51afd7ed558ccdULL; s ^= s>>33; s *= 0xc4ceb9fe1a85ec53ULL; s ^= s>>33;
		v[i] = (float)((s>>11)*(1.0/9007199254740992.0) - 0.5);
	}
}

double tridiag_max(const std::vector<double>& a, const std::vector<double>& b)
{
	const size_t k = a.size();
	double lo=1e300, hi=-1e300;
	for (size_t i=0;i<k;++i) { double r=(i>0?fabs(b[i-1]):0)+(i+1<k?fabs(b[i]):0); lo=std::min(lo,a[i]-r); hi=std::max(hi,a[i]+r); }
	for (int it=0; it<200; ++it)
	{
		double mid=0.5*(lo+hi); int count=0; double d=1;
		for (size_t i=0;i<k;++i) { d=a[i]-mid-(i>0?b[i-1]*b[i-1]/d:0); if (d==0) d=1e-300; if (d<0) ++count; }
		if (count<(int)k) lo=mid; else hi=mid;
		if (hi-lo<=1e-12*fabs(hi)) break;
	}
	return hi;
}

} // namespace

double CUDA_LanczosMaxEig(const float* SC_h, const float* IL_h, unsigned N0, unsigned N1, unsigned N2,
                          size_t i0, size_t sx, size_t sy, size_t sz, int iterations)
{
	Grid g{N0,N1,N2,i0,sx,sy,sz,(size_t)N0*N1*N2,{0,1,2}};
	{
		size_t st[3] = {sx,sy,sz};
		std::sort(g.ax, g.ax+3, [&](int a, int b){return st[a]<st[b];});
	}
	const size_t n = 3*g.total;
	auto T0 = std::chrono::steady_clock::now();
	auto since = [&]() {return std::chrono::duration<double>(std::chrono::steady_clock::now()-T0).count();};
	const bool timing = getenv("OPENEMS_SETUP_TIMES") && atoi(getenv("OPENEMS_SETUP_TIMES"));
	int dev=0;
	if (cudaGetDeviceCount(&dev)!=cudaSuccess || dev==0) { cudaGetLastError(); return -1; }
	cudaFree(0);
	if (timing) std::cout << "OPENEMS_SETUP_TIME lanczos_cuda_context " << since() << " s (strides " << sx << "," << sy << "," << sz << ")" << std::endl;
	size_t free_b=0, total_b=0;
	if (cudaMemGetInfo(&free_b,&total_b)!=cudaSuccess) { cudaGetLastError(); return -1; }
	if (free_b < 5*n*sizeof(float) + (256u<<20)) return -1;
	float *SC=0,*IL=0,*q=0,*qp=0,*w=0; double* red=0;
	bool ok = cudaMalloc(&SC,n*4)==cudaSuccess && cudaMalloc(&IL,n*4)==cudaSuccess && cudaMalloc(&q,n*4)==cudaSuccess
	          && cudaMalloc(&qp,n*4)==cudaSuccess && cudaMalloc(&w,n*4)==cudaSuccess && cudaMalloc(&red,DOT_BLOCKS*sizeof(double))==cudaSuccess;
	double lambda = -1;
	if (ok)
	{
		cudaMemcpy(SC,SC_h,n*4,cudaMemcpyHostToDevice); cudaMemcpy(IL,IL_h,n*4,cudaMemcpyHostToDevice);
		cudaMemset(qp,0,n*4);
		cudaDeviceSynchronize();
		if (timing) std::cout << "OPENEMS_SETUP_TIME lanczos_upload " << since() << " s" << std::endl;
		const int T=256; const int B=std::min<size_t>((n+T-1)/T, 65535*8);
		const size_t cells=g.total; const unsigned AB=(unsigned)((cells+T-1)/T);
		std::vector<double> part(DOT_BLOCKS);
		auto dot=[&](const float* a,const float* b){ dot_kernel<<<DOT_BLOCKS,T>>>(a,b,n,red); cudaMemcpy(part.data(),red,DOT_BLOCKS*sizeof(double),cudaMemcpyDeviceToHost); double r=0; for (double v : part) r+=v; return r; };
		random_fill<<<B,T>>>(q,n);
		// the start vector must vanish where SC = 0 (no degree of freedom); scaling by SC handles it in A
		double nrm = sqrt(dot(q,q));
		scale_copy<<<B,T>>>(q,q,(float)(1.0/nrm),n);
		std::vector<double> al, be; double bprev=0;
		cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float t_apply=0, t_rest=0;
		for (int k=0;k<iterations;++k)
		{
			cudaEventRecord(e0);
			apply_kernel<<<AB,T>>>(g,SC,IL,q,w);
			cudaEventRecord(e1); cudaEventSynchronize(e1); { float ms; cudaEventElapsedTime(&ms,e0,e1); t_apply+=ms; }
			cudaEventRecord(e0);
			double a = dot(q,w);
			lanczos_update<<<B,T>>>(w,q,qp,(float)a,(float)bprev,n);
			double b = sqrt(dot(w,w));
			al.push_back(a);
			lambda = tridiag_max(al,be);
			if (!(b>0)) break;
			be.push_back(b);
			std::swap(q,qp);
			scale_copy<<<B,T>>>(q,w,(float)(1.0/b),n);
			bprev=b;
			cudaEventRecord(e1); cudaEventSynchronize(e1); { float ms; cudaEventElapsedTime(&ms,e0,e1); t_rest+=ms; }
		}
		if (timing) std::cout << "OPENEMS_SETUP_TIME lanczos_apply_ms " << t_apply << " rest_ms " << t_rest << std::endl;
		if (cudaDeviceSynchronize()!=cudaSuccess) lambda=-1;
		if (timing) std::cout << "OPENEMS_SETUP_TIME lanczos_iterations " << since() << " s" << std::endl;
	}
	cudaFree(SC); cudaFree(IL); cudaFree(q); cudaFree(qp); cudaFree(w); cudaFree(red);
	cudaGetLastError();
	return lambda;
}
