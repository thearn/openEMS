/*
* Copyright (C) 2026 Sean Mollet (sean@malmoset.com)
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*/

#include "cuda_opportunity_report.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace
{
uint64_t volume(const GPU_OpportunityDim& d)
{
	return (uint64_t)d.x*d.y*d.z;
}

unsigned int ceil_div(unsigned int n, unsigned int d)
{
	return n/d + (n%d != 0);
}

std::string json_string(const std::string& value)
{
	std::ostringstream out;
	out << '"';
	for (size_t i=0; i<value.size(); ++i)
	{
		const unsigned char c = (unsigned char)value[i];
		switch (c)
		{
		case '"': out << "\\\""; break;
		case '\\': out << "\\\\"; break;
		case '\b': out << "\\b"; break;
		case '\f': out << "\\f"; break;
		case '\n': out << "\\n"; break;
		case '\r': out << "\\r"; break;
		case '\t': out << "\\t"; break;
		default:
			if (c<0x20)
				out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (unsigned int)c << std::dec;
			else
				out << value[i];
		}
	}
	out << '"';
	return out.str();
}

// Small self-contained SHA-256 used only for exact census identities. Feed integers in an
// explicit byte order so reports made on different hosts have the same identity.
class SHA256
{
public:
	SHA256() : m_Total(0), m_Used(0)
	{
		const uint32_t initial[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
		                             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
		std::copy(initial, initial+8, m_State);
	}

	void update(const void* data, size_t size)
	{
		const unsigned char* p = (const unsigned char*)data;
		m_Total += size;
		while (size)
		{
			const size_t n = std::min(size, sizeof(m_Buffer)-m_Used);
			std::memcpy(m_Buffer+m_Used, p, n);
			m_Used += n; p += n; size -= n;
			if (m_Used==sizeof(m_Buffer))
			{
				block(m_Buffer);
				m_Used = 0;
			}
		}
	}

	void u32(uint32_t value)
	{
		const unsigned char bytes[4] = {(unsigned char)value, (unsigned char)(value>>8),
		                                (unsigned char)(value>>16), (unsigned char)(value>>24)};
		update(bytes, sizeof(bytes));
	}

	std::string finish()
	{
		const uint64_t bits = m_Total*8;
		const unsigned char one = 0x80, zero = 0;
		update(&one, 1);
		while (m_Used!=56)
			update(&zero, 1);
		unsigned char length[8];
		for (int i=0; i<8; ++i)
			length[i] = (unsigned char)(bits>>(56-8*i));
		update(length, sizeof(length));
		std::ostringstream out;
		out << std::hex << std::setfill('0');
		for (int i=0; i<8; ++i)
			out << std::setw(8) << m_State[i];
		return out.str();
	}

private:
	static uint32_t rotate(uint32_t x, unsigned int n) {return (x>>n) | (x<<(32-n));}
	void block(const unsigned char* p)
	{
		static const uint32_t K[64] = {
			0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
			0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
			0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
			0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
			0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
			0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
			0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
			0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
		uint32_t w[64];
		for (int i=0; i<16; ++i)
			w[i] = ((uint32_t)p[4*i]<<24) | ((uint32_t)p[4*i+1]<<16) | ((uint32_t)p[4*i+2]<<8) | p[4*i+3];
		for (int i=16; i<64; ++i)
		{
			const uint32_t s0 = rotate(w[i-15],7)^rotate(w[i-15],18)^(w[i-15]>>3);
			const uint32_t s1 = rotate(w[i-2],17)^rotate(w[i-2],19)^(w[i-2]>>10);
			w[i] = w[i-16]+s0+w[i-7]+s1;
		}
		uint32_t a=m_State[0], b=m_State[1], c=m_State[2], d=m_State[3];
		uint32_t e=m_State[4], f=m_State[5], g=m_State[6], h=m_State[7];
		for (int i=0; i<64; ++i)
		{
			const uint32_t S1=rotate(e,6)^rotate(e,11)^rotate(e,25);
			const uint32_t ch=(e&f)^((~e)&g);
			const uint32_t t1=h+S1+ch+K[i]+w[i];
			const uint32_t S0=rotate(a,2)^rotate(a,13)^rotate(a,22);
			const uint32_t maj=(a&b)^(a&c)^(b&c);
			const uint32_t t2=S0+maj;
			h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
		}
		m_State[0]+=a; m_State[1]+=b; m_State[2]+=c; m_State[3]+=d;
		m_State[4]+=e; m_State[5]+=f; m_State[6]+=g; m_State[7]+=h;
	}
	uint32_t m_State[8];
	uint64_t m_Total;
	unsigned char m_Buffer[64];
	size_t m_Used;
};

struct VectorHash
{
	size_t operator()(const std::vector<uint32_t>& values) const
	{
		size_t h = (size_t)1469598103934665603ULL;
		for (size_t i=0; i<values.size(); ++i)
			h = (h ^ values[i]) * (size_t)1099511628211ULL;
		return h;
	}
};

bool intersects(unsigned int x0, unsigned int x1, unsigned int y0, unsigned int y1,
	            unsigned int z0, unsigned int z1, const GPU_OpportunityBox& b)
{
	return x0 < b.start.x+b.length.x && b.start.x < x1 &&
	       y0 < b.start.y+b.length.y && b.start.y < y1 &&
	       z0 < b.start.z+b.length.z && b.start.z < z1;
}

void write_dim(std::ostream& out, const GPU_OpportunityDim& d)
{
	out << "{\"x\":" << d.x << ",\"y\":" << d.y << ",\"z\":" << d.z << '}';
}

void write_hist(std::ostream& out, const std::map<unsigned int,uint64_t>& hist)
{
	out << '{';
	bool first = true;
	for (std::map<unsigned int,uint64_t>::const_iterator it=hist.begin(); it!=hist.end(); ++it)
	{
		if (!first) out << ',';
		first = false;
		out << '"' << it->first << "\":" << it->second;
	}
	out << '}';
}

unsigned int local_width(size_t sets)
{
	return sets<=256 ? 1 : (sets<=65536 ? 2 : 4);
}
}

void GPU_WriteOpportunityReport(const GPU_OpportunityInput& in)
{
	if (in.path.empty() || !in.xc || !in.useful_y || !in.useful_z)
		throw std::runtime_error("invalid CUDA opportunity report configuration");
	if (in.begin.x>in.end.x || in.begin.y>in.end.y || in.begin.z>in.end.z ||
	    in.end.x>in.dim.x || in.end.y>in.dim.y || in.end.z>in.dim.z)
		throw std::runtime_error("invalid CUDA opportunity launch range");
	const uint64_t cells = volume(in.dim);
	if (in.coefficients.index.size()!=cells || in.coefficients.table.size()!=12*in.coefficients.count ||
	    (in.coefficients.mode!=1 && in.coefficients.mode!=2))
		throw std::runtime_error("CUDA opportunity report requires exact compressed coefficients");

	SHA256 index_sha, table_sha;
	for (size_t i=0; i<in.coefficients.index.size(); ++i)
	{
		if (in.coefficients.index[i]>=in.coefficients.count)
			throw std::runtime_error("coefficient set index out of range");
		index_sha.u32(in.coefficients.index[i]);
	}
	for (size_t i=0; i<in.coefficients.table.size(); ++i)
	{
		uint32_t bits;
		std::memcpy(&bits, &in.coefficients.table[i], sizeof(bits));
		table_sha.u32(bits);
	}
	const std::string index_digest = index_sha.finish(), table_digest = table_sha.finish();

	std::unordered_set<unsigned int> folded;
	if (in.folded_ade>=0)
		folded.insert(in.ade.at((size_t)in.folded_ade).flat.begin(), in.ade.at((size_t)in.folded_ade).flat.end());
	std::unordered_set<unsigned int> fixup;
	for (size_t i=0; i<in.fixup.size(); ++i)
		fixup.insert(in.fixup[i].flat);
	if (in.modified.size()>63)
		throw std::runtime_error("too many modified-voltage sources for census mask");
	std::unordered_map<unsigned int,uint64_t> modified;
	for (size_t s=0; s<in.modified.size(); ++s)
		for (size_t i=0; i<in.modified[s].flat.size(); ++i)
			modified[in.modified[s].flat[i]] |= (uint64_t)1<<s;

	const unsigned int dx=in.end.x-in.begin.x, dy=in.end.y-in.begin.y, dz=in.end.z-in.begin.z;
	const unsigned int gx=ceil_div(dx,in.xc), gy=ceil_div(dy,in.useful_y), gz=ceil_div(dz,in.useful_z);
	const uint64_t all_tiles=(uint64_t)gx*gy*gz, all_nodes=(uint64_t)dx*dy*dz;
	uint64_t enumerated_tiles=0, enumerated_nodes=0, partial_tiles=0;
	uint64_t branch_tiles=0, branch_nodes=0, isolated_tiles=0, isolated_nodes=0;
	std::array<uint64_t,16> reason_tiles = {{0}}, reason_nodes = {{0}};
	std::vector<uint64_t> source_tiles(in.modified.size(),0), source_nodes(in.modified.size(),0);
	std::map<unsigned int,uint64_t> palette_tiles, palette_lookups;
	uint64_t lookups8=0, lookups16=0, total_lookups=0;
	uint64_t owned_index_bytes=0, footprint_index_bytes=0, duplicated_dictionary_entries=0;
	uint64_t shared_dictionary_entries=0;
	std::unordered_map<std::vector<uint32_t>, unsigned int, VectorHash> shared_palettes;
	std::vector<uint32_t> marks(in.coefficients.count,0), palette;
	uint32_t generation=0;

	for (unsigned int bx=0; bx<gx; ++bx)
		for (unsigned int by=0; by<gy; ++by)
			for (unsigned int bz=0; bz<gz; ++bz)
			{
				const unsigned int x0=in.begin.x+bx*in.xc, x1=std::min(x0+in.xc,in.end.x);
				const unsigned int y0=in.begin.y+by*in.useful_y, y1=std::min(y0+in.useful_y,in.end.y);
				const unsigned int z0=in.begin.z+bz*in.useful_z, z1=std::min(z0+in.useful_z,in.end.z);
				const unsigned int cx1=std::min(x1+1,in.end.x), cy1=std::min(y1+1,in.end.y), cz1=std::min(z1+1,in.end.z);
				const unsigned int hx1=std::min(x1+1,in.dim.x), hy1=std::min(y1+1,in.dim.y), hz1=std::min(z1+1,in.dim.z);
				const uint64_t owned=(uint64_t)(x1-x0)*(y1-y0)*(z1-z0);
				const uint64_t footprint=(uint64_t)(cx1-x0)*(cy1-y0)*(cz1-z0);
				++enumerated_tiles; enumerated_nodes += owned;
				if (owned!=(uint64_t)in.xc*in.useful_y*in.useful_z) ++partial_tiles;

				bool pml=false, has_folded=false, has_fixup=false;
				uint64_t source_mask=0;
				for (size_t r=0; r<in.upml.size(); ++r)
					pml |= intersects(x0,cx1,y0,cy1,z0,cz1,in.upml[r]);
				for (unsigned int x=x0; x<hx1; ++x)
					for (unsigned int y=y0; y<hy1; ++y)
						for (unsigned int z=z0; z<hz1; ++z)
						{
							const unsigned int flat=(x*in.dim.y+y)*in.dim.z+z;
							if (x<cx1 && y<cy1 && z<cz1 && folded.count(flat)) has_folded=true;
							if (fixup.count(flat)) has_fixup=true;
							std::unordered_map<unsigned int,uint64_t>::const_iterator m=modified.find(flat);
							if (m!=modified.end()) source_mask |= m->second;
						}
				const bool has_modified=source_mask!=0;
				const unsigned int reason=(pml?1:0)|(has_folded?2:0)|(has_modified?4:0)|(has_fixup?8:0);
				++reason_tiles[reason]; reason_nodes[reason]+=owned;
				const bool branch_free=!pml && !has_folded;
				if (branch_free) {++branch_tiles; branch_nodes+=owned;}
				if (branch_free && !has_modified && !has_fixup) {++isolated_tiles; isolated_nodes+=owned;}
				for (size_t s=0; s<in.modified.size(); ++s)
					if (source_mask&((uint64_t)1<<s)) {++source_tiles[s]; source_nodes[s]+=owned;}

				if (++generation==0)
				{
					std::fill(marks.begin(),marks.end(),0);
					generation=1;
				}
				palette.clear();
				for (unsigned int x=x0; x<cx1; ++x)
					for (unsigned int y=y0; y<cy1; ++y)
						for (unsigned int z=z0; z<cz1; ++z)
						{
							const uint32_t set=in.coefficients.index[(x*in.dim.y+y)*in.dim.z+z];
							if (marks[set]!=generation) {marks[set]=generation; palette.push_back(set);}
						}
				std::sort(palette.begin(),palette.end());
				const unsigned int width=local_width(palette.size());
				++palette_tiles[(unsigned int)palette.size()];
				palette_lookups[(unsigned int)palette.size()]+=footprint;
				total_lookups+=footprint;
				if (width<=1) lookups8+=footprint;
				if (width<=2) lookups16+=footprint;
				owned_index_bytes+=owned*width;
				footprint_index_bytes+=footprint*width;
				duplicated_dictionary_entries+=palette.size();
				if (shared_palettes.insert(std::make_pair(palette,(unsigned int)shared_palettes.size())).second)
					shared_dictionary_entries+=palette.size();
			}

	if (enumerated_tiles!=all_tiles || enumerated_nodes!=all_nodes)
		throw std::runtime_error("tile census conservation failure");
	uint64_t reason_tile_sum=0, reason_node_sum=0;
	for (size_t i=0;i<16;++i) {reason_tile_sum+=reason_tiles[i]; reason_node_sum+=reason_nodes[i];}
	if (reason_tile_sum!=all_tiles || reason_node_sum!=all_nodes || branch_tiles>all_tiles || isolated_tiles>branch_tiles)
		throw std::runtime_error("reason census conservation failure");

	uint64_t ade_state=0, ade_coeff=0, ade_compact_state=0, ade_compact_coeff=0;
	uint64_t ade_active_occurrences=0, ade_nodes=0, ade_position_mask=0;
	for (size_t g=0; g<in.ade.size(); ++g)
	{
		if (in.ade[g].flat.size()!=in.ade[g].mask.size()) throw std::runtime_error("ADE census size mismatch");
		uint64_t active=0;
		for (size_t i=0;i<in.ade[g].mask.size();++i)
		{
			active += (in.ade[g].mask[i]&1)!=0;
			active += (in.ade[g].mask[i]&2)!=0;
			active += (in.ade[g].mask[i]&4)!=0;
		}
		const uint64_t state_factors=in.ade[g].lorentz?2:1;
		const uint64_t coefficient_factors=in.ade[g].lorentz?3:2;
		ade_state += (uint64_t)in.ade[g].orders*3*in.ade[g].flat.size()*state_factors*sizeof(float);
		ade_coeff += (uint64_t)in.ade[g].orders*3*in.ade[g].flat.size()*coefficient_factors*sizeof(float);
		ade_compact_state += (uint64_t)in.ade[g].orders*active*state_factors*sizeof(float);
		ade_compact_coeff += (uint64_t)in.ade[g].orders*active*coefficient_factors*sizeof(float);
		ade_active_occurrences += active;
		ade_nodes += in.ade[g].flat.size();
		ade_position_mask += in.ade[g].flat.size()*(sizeof(uint32_t)+sizeof(unsigned char));
	}
	const uint64_t folded_index_bytes = in.folded_ade<0 ? 0 :
		((cells+31)/32)*2*sizeof(uint32_t) + in.dim.z*sizeof(unsigned char) + in.ade[(size_t)in.folded_ade].mask.size();

	const uint64_t current_index_bytes=cells*(in.coefficients.mode==1?2:4);
	const uint64_t global_table_bytes=in.coefficients.table.size()*sizeof(float);
	const uint64_t duplicate_dict_bytes=duplicated_dictionary_entries*12*sizeof(float);
	const uint64_t shared_dict_bytes=shared_dictionary_entries*12*sizeof(float);
	const uint64_t shared_selector_bytes=all_tiles*4;

	std::ofstream out(in.path.c_str(), std::ios::out|std::ios::trunc);
	if (!out) throw std::runtime_error("cannot open CUDA opportunity report: "+in.path);
	out << "{\n  \"schema\": \"openems.cuda-opportunity.v1\",\n  \"dimensions\": "; write_dim(out,in.dim);
	out << ",\n  \"launch\": {\"begin\":"; write_dim(out,in.begin); out << ",\"end\":"; write_dim(out,in.end);
	out << ",\"tile\":{\"x\":"<<in.xc<<",\"y\":"<<in.useful_y<<",\"z\":"<<in.useful_z
	    << "},\"grid\":{\"x\":"<<gx<<",\"y\":"<<gy<<",\"z\":"<<gz<<"}},\n";
	out << "  \"tiles\": {\"total\":"<<all_tiles<<",\"partial\":"<<partial_tiles<<",\"owned_nodes\":"<<all_nodes<<",\"upml_regions\":"<<in.upml.size()
	    << ",\"branch_free\":{\"tiles\":"<<branch_tiles<<",\"owned_nodes\":"<<branch_nodes
	    << "},\"schedule_isolated\":{\"tiles\":"<<isolated_tiles<<",\"owned_nodes\":"<<isolated_nodes<<"},\"reason_masks\":[";
	for (size_t i=0;i<16;++i) {if(i)out<<',';out<<"{\"mask\":"<<i<<",\"tiles\":"<<reason_tiles[i]<<",\"owned_nodes\":"<<reason_nodes[i]<<'}';}
	out << "]},\n  \"modified_sources\": [";
	for(size_t s=0;s<in.modified.size();++s){if(s)out<<',';out<<"{\"name\":"<<json_string(in.modified[s].source)<<",\"registered_nodes\":"<<in.modified[s].flat.size()<<",\"tiles\":"<<source_tiles[s]<<",\"owned_nodes_in_tiles\":"<<source_nodes[s]<<'}';}
	out << "],\n  \"fixups\": {\"entries\":"<<in.fixup.size()<<",\"unique_nodes\":"<<fixup.size()<<"},\n";
	out << "  \"ade\": {\"folded_group\":"<<in.folded_ade<<",\"nodes\":"<<ade_nodes<<",\"active_component_occurrences\":"<<ade_active_occurrences
	    << ",\"allocated_state_bytes\":"<<ade_state<<",\"allocated_coefficient_bytes\":"<<ade_coeff
	    << ",\"active_packed_state_bytes\":"<<ade_compact_state<<",\"active_packed_coefficient_bytes\":"<<ade_compact_coeff
	    << ",\"position_mask_bytes\":"<<ade_position_mask<<",\"folded_flag_prefix_mask_bytes\":"<<folded_index_bytes<<",\"groups\":[";
	for(size_t g=0;g<in.ade.size();++g){
		std::array<uint64_t,8> masks={{0}}; std::unordered_set<unsigned int> xs,ys,zs; uint64_t active=0;
		GPU_OpportunityDim lo={in.dim.x,in.dim.y,in.dim.z}, hi={0,0,0};
		for(size_t i=0;i<in.ade[g].flat.size();++i){const unsigned int f=in.ade[g].flat[i],x=f/(in.dim.y*in.dim.z),y=(f/in.dim.z)%in.dim.y,z=f%in.dim.z; const unsigned int m=in.ade[g].mask[i]&7; ++masks[m]; active+=(m&1)!=0;active+=(m&2)!=0;active+=(m&4)!=0;xs.insert(x);ys.insert(y);zs.insert(z);lo.x=std::min(lo.x,x);lo.y=std::min(lo.y,y);lo.z=std::min(lo.z,z);hi.x=std::max(hi.x,x+1);hi.y=std::max(hi.y,y+1);hi.z=std::max(hi.z,z+1);}
		if(g)out<<',';out<<"{\"index\":"<<g<<",\"folded\":"<<(in.folded_ade==(int)g?"true":"false")<<",\"nodes\":"<<in.ade[g].flat.size()<<",\"orders\":"<<in.ade[g].orders<<",\"lorentz\":"<<(in.ade[g].lorentz?"true":"false")<<",\"active_component_occurrences\":"<<active<<",\"mask_histogram\":[";
		for(size_t m=0;m<8;++m){if(m)out<<',';out<<masks[m];}out<<"],\"unique_planes\":{\"x\":"<<xs.size()<<",\"y\":"<<ys.size()<<",\"z\":"<<zs.size()<<"},\"bounds\":{\"begin\":";write_dim(out,lo);out<<",\"end\":";write_dim(out,hi);out<<"}}";
	}
	out << "]},\n  \"coefficients\": {\"sets\":"<<in.coefficients.count<<",\"global_index_bits\":"<<(in.coefficients.mode==1?16:32)
	    << ",\"global_index_bytes\":"<<current_index_bytes<<",\"global_table_bytes\":"<<global_table_bytes
	    << ",\"global_total_bytes\":"<<current_index_bytes+global_table_bytes<<",\"index_u32_le_sha256\":"<<json_string(index_digest)
	    << ",\"table_f32_bits_le_sha256\":"<<json_string(table_digest)<<",\"tile_distinct_set_histogram\":";write_hist(out,palette_tiles);
	out << ",\"tile_lookup_histogram\":";write_hist(out,palette_lookups);
	out << ",\"footprint_lookups\":"<<total_lookups<<",\"lookups_fitting_u8\":"<<lookups8<<",\"lookups_fitting_u16\":"<<lookups16
	    << ",\"encodings\":{\"owned_private\":{\"index_bytes\":"<<owned_index_bytes<<",\"dictionary_bytes\":"<<duplicate_dict_bytes<<",\"total_bytes\":"<<owned_index_bytes+duplicate_dict_bytes
	    << "},\"footprint_private\":{\"index_bytes\":"<<footprint_index_bytes<<",\"dictionary_bytes\":"<<duplicate_dict_bytes<<",\"total_bytes\":"<<footprint_index_bytes+duplicate_dict_bytes
	    << "},\"owned_shared_dictionary\":{\"index_bytes\":"<<owned_index_bytes<<",\"dictionary_bytes\":"<<shared_dict_bytes<<",\"selector_bytes\":"<<shared_selector_bytes<<",\"total_bytes\":"<<owned_index_bytes+shared_dict_bytes+shared_selector_bytes<<",\"unique_dictionaries\":"<<shared_palettes.size()
	    << "},\"footprint_shared_dictionary\":{\"index_bytes\":"<<footprint_index_bytes<<",\"dictionary_bytes\":"<<shared_dict_bytes<<",\"selector_bytes\":"<<shared_selector_bytes<<",\"total_bytes\":"<<footprint_index_bytes+shared_dict_bytes+shared_selector_bytes<<",\"unique_dictionaries\":"<<shared_palettes.size()<<"}}},\n";
	out << "  \"layout\": {\"current_useful_z\":"<<in.useful_z<<",\"current_z_tiles\":"<<gz<<",\"current_z_padding_per_xy_line\":"<<(uint64_t)gz*in.useful_z-dz
	    << ",\"hypothetical_useful_z\":32,\"hypothetical_z_tiles\":"<<ceil_div(dz,32)<<",\"hypothetical_z_padding_per_xy_line\":"<<(uint64_t)ceil_div(dz,32)*32-dz<<"},\n";
	out << "  \"device\": {\"name\":"<<json_string(in.device.name)<<",\"compute_capability\":"<<json_string(std::to_string(in.device.compute_major)+"."+std::to_string(in.device.compute_minor))
	    << ",\"l2_bytes\":"<<in.device.l2_bytes<<",\"persisting_l2_max_bytes\":"<<in.device.persisting_l2_max_bytes<<",\"free_bytes_at_decision\":"<<in.device.free_bytes
	    << ",\"total_bytes\":"<<in.device.total_bytes<<",\"fused_field_bytes\":"<<in.device.fused_field_bytes<<",\"fused_extension_bytes\":"<<in.device.fused_extension_bytes<<",\"reserve_bytes\":"<<in.device.reserve_bytes<<"},\n";
	out << "  \"checks\": {\"tile_count_conserved\":true,\"owned_nodes_conserved\":true,\"reason_masks_conserved\":true,\"coefficient_indices_in_range\":true}\n}\n";
	if (!out) throw std::runtime_error("failed to write CUDA opportunity report: "+in.path);
}
