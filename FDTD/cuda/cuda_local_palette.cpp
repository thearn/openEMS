/* Copyright (C) 2026 Sean Mollet (sean@malmoset.com) */
#include "cuda_local_palette.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace
{
const unsigned int TY=7, TZ=31;

struct VectorHash
{
	size_t operator()(const std::vector<uint32_t>& values) const
	{
		size_t h=(size_t)1469598103934665603ull;
		for(uint32_t value:values)h=(h^value)*(size_t)1099511628211ull;
		return h;
	}
};
}

CUDA_LocalPalette CUDA_BuildLocalPalette(const GPU_CoeffSets& c,
	unsigned int nx, unsigned int ny, unsigned int nz, unsigned int xc)
{
	if(!nx || !ny || !nz || !xc || (c.mode!=1 && c.mode!=2) ||
	   c.index.size()!=(size_t)nx*ny*nz || c.table.size()!=12*c.count)
		throw std::runtime_error("invalid CUDA local-palette input");
	const std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
	CUDA_LocalPalette p;
	p.nx=nx;p.ny=ny;p.nz=nz;p.xc=xc;
	p.gx=(nx+xc-1)/xc;p.gy=(ny+TY-1)/TY;p.gz=(nz+TZ-1)/TZ;
	p.meta.reserve((size_t)p.gx*p.gy*p.gz);
	std::unordered_map<std::vector<uint32_t>,uint32_t,VectorHash> dictionaries;
	std::vector<uint32_t> marks(c.count,0),local_ids(c.count),dict;
	uint32_t generation=0;
	for(unsigned int bx=0;bx<p.gx;++bx)for(unsigned int by=0;by<p.gy;++by)for(unsigned int bz=0;bz<p.gz;++bz)
	{
		const unsigned int x0=bx*xc,y0=by*TY,z0=bz*TZ;
		const unsigned int fx=std::min(xc+1,nx-x0),fy=std::min(TY+1,ny-y0),fz=std::min(TZ+1,nz-z0);
		if(++generation==0){std::fill(marks.begin(),marks.end(),0);generation=1;}
		dict.clear();
		for(unsigned int x=0;x<fx;++x)for(unsigned int y=0;y<fy;++y)for(unsigned int z=0;z<fz;++z)
		{
			const uint32_t set=c.index[((size_t)(x0+x)*ny+y0+y)*nz+z0+z];
			if(marks[set]!=generation){marks[set]=generation;dict.push_back(set);}
		}
		std::sort(dict.begin(),dict.end());
		p.dictionary_id_entries+=dict.size();
		const unsigned int width=dict.size()<=256?1:2;
		if(width==1)++p.one_byte_tiles;else ++p.two_byte_tiles;
		if(width==2 && (p.index.size()&1)){p.index.push_back(0);++p.index_padding;}
		auto found=dictionaries.find(dict);
		uint32_t dictionary_offset;
		if(found==dictionaries.end())
		{
			dictionary_offset=p.table.size()/12;
			dictionaries.insert(std::make_pair(dict,dictionary_offset));
			for(uint32_t set:dict)p.table.insert(p.table.end(),c.table.begin()+12*set,c.table.begin()+12*(set+1));
		}
		else dictionary_offset=found->second;
		p.meta.push_back({(uint32_t)p.index.size(),dictionary_offset,(uint16_t)fx,(uint16_t)fy,(uint16_t)fz,(uint8_t)width,0});
		for(uint32_t local=0;local<dict.size();++local)local_ids[dict[local]]=local;
		for(unsigned int x=0;x<fx;++x)for(unsigned int y=0;y<fy;++y)for(unsigned int z=0;z<fz;++z)
		{
			const uint32_t set=c.index[((size_t)(x0+x)*ny+y0+y)*nz+z0+z];
			const uint32_t local=local_ids[set];
			if(width==1)p.index.push_back((unsigned char)local);
			else {p.index.push_back(local&255);p.index.push_back(local>>8);}
		}
		const unsigned int ox=std::min(xc,nx-x0),oy=std::min(TY,ny-y0),oz=std::min(TZ,nz-z0);
		p.logical_lookups+=(size_t)fx*fy*fz;p.local_index_reads+=(size_t)fx*fy*fz*width;
		for(unsigned int x=0;x<ox && x0+x+1<nx;++x)
		{
			const size_t current=(size_t)std::min(oy,ny-y0-1)*std::min(oz,nz-z0-1);
			p.logical_lookups+=current;p.local_index_reads+=current*width;
		}
	}
	p.shared_dictionaries=dictionaries.size();
	p.encoding_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
	return p;
}

void CUDA_ValidateLocalPalette(const CUDA_LocalPalette& p,const GPU_CoeffSets& c)
{
	if(p.meta.size()!=(size_t)p.gx*p.gy*p.gz || p.table.size()%12)
		throw std::runtime_error("invalid CUDA local-palette metadata");
	for(size_t tile=0;tile<p.meta.size();++tile)
	{
		const CUDA_PaletteTile& m=p.meta[tile];
		const unsigned int bz=tile%p.gz,by=(tile/p.gz)%p.gy,bx=tile/(p.gz*p.gy);
		if((m.width!=1 && m.width!=2) || m.index_offset+(size_t)m.nx*m.ny*m.nz*m.width>p.index.size())
			throw std::runtime_error("CUDA local-palette index bound mismatch");
		for(unsigned int x=0;x<m.nx;++x)for(unsigned int y=0;y<m.ny;++y)for(unsigned int z=0;z<m.nz;++z)
		{
			const size_t q=m.index_offset+(((size_t)x*m.ny+y)*m.nz+z)*m.width;
			const unsigned int local=m.width==1?p.index[q]:(unsigned int)p.index[q]|((unsigned int)p.index[q+1]<<8);
			const uint32_t global=c.index[((size_t)(bx*p.xc+x)*p.ny+by*TY+y)*p.nz+bz*TZ+z];
			if(m.dictionary_offset+local>=p.table.size()/12 ||
			   std::memcmp(&p.table[12*(m.dictionary_offset+local)],&c.table[12*global],12*sizeof(float)))
				throw std::runtime_error("CUDA local-palette host decode mismatch");
		}
	}
}

void CUDA_SelfTestLocalPalette()
{
	auto make=[](unsigned int nx,unsigned int ny,unsigned int nz,unsigned int count,bool many)
	{
		GPU_CoeffSets c;c.mode=1;c.count=count;c.index.resize((size_t)nx*ny*nz);c.table.resize(12*count);
		for(size_t s=0;s<count;++s)for(unsigned int k=0;k<12;++k)
		{
			uint32_t bits=0x3f000000u+(uint32_t)(s*16+k);
			std::memcpy(&c.table[12*s+k],&bits,sizeof(bits));
		}
		for(unsigned int x=0;x<nx;++x)for(unsigned int y=0;y<ny;++y)for(unsigned int z=0;z<nz;++z)
			c.index[((size_t)x*ny+y)*nz+z]=many?(uint32_t)(((size_t)x*ny*nz+y*nz+z)%count):(uint32_t)((y+z)%count);
		return c;
	};
	{
		GPU_CoeffSets c=make(9,8,33,4,false);
		CUDA_LocalPalette a=CUDA_BuildLocalPalette(c,9,8,33,4),b=CUDA_BuildLocalPalette(c,9,8,33,4);
		CUDA_ValidateLocalPalette(a,c);
		if(a.two_byte_tiles || !a.one_byte_tiles || a.shared_dictionaries>=a.meta.size() ||
		   a.index!=b.index || a.table!=b.table || a.meta.size()!=b.meta.size() ||
		   std::memcmp(a.meta.data(),b.meta.data(),a.meta.size()*sizeof(CUDA_PaletteTile)))
			throw std::runtime_error("CUDA local-palette deterministic one-byte self-test failed");
		if(a.meta.back().nx>=5 || a.meta.back().ny>=8 || a.meta.back().nz>=32)
			throw std::runtime_error("CUDA local-palette clipped-edge self-test failed");
		a.index[0]=255;
		bool rejected=false;try{CUDA_ValidateLocalPalette(a,c);}catch(const std::runtime_error&){rejected=true;}
		if(!rejected)throw std::runtime_error("CUDA local-palette corruption self-test failed");
	}
	{
		GPU_CoeffSets c=make(5,8,32,300,true);
		CUDA_LocalPalette p=CUDA_BuildLocalPalette(c,5,8,32,4);
		CUDA_ValidateLocalPalette(p,c);
		if(!p.two_byte_tiles)throw std::runtime_error("CUDA local-palette two-byte self-test failed");
	}
}
