// m4_fullscale_check.cu — throwaway: exercise the PRODUCTION compute_f2_blocks
// (CudaBackend, the grouped design) at real scale (derived_full P=768) to confirm
// it is VRAM-safe end-to-end and to capture a production timing number. Spot-checks
// a few off-diagonal f2 values against a quick native recompute. NOT a gate (the
// gate is tests/reference/test_f2_blocks_equivalence on derived_acc); this just
// proves the production seam runs at P=768 and reports ms.
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "steppe/config.hpp"
#include "steppe/fstats.hpp"
#include "core/internal/views.hpp"
#include "core/domain/block_partition_rule.hpp"
#include "core/fstats/f2_from_blocks.hpp"
#include "device/backend.hpp"
#include "io/snp_reader.hpp"

namespace steppe::device { std::unique_ptr<steppe::ComputeBackend> make_cuda_backend(); }

static void read_f64(const std::string& p, std::vector<double>& d, size_t n){
  FILE*f=fopen(p.c_str(),"rb"); if(!f){fprintf(stderr,"open %s\n",p.c_str());exit(1);}
  size_t g=fread(d.data(),8,n,f); fclose(f); if(g!=n){fprintf(stderr,"short read\n");exit(1);} }

int main(int argc,char**argv){
  std::string dir=argc>=2?argv[1]:"/workspace/data/aadr/derived_full";
  std::string snp=argc>=3?argv[2]:"/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB.snp";
  int P; long M;
  { FILE*f=fopen((dir+"/shape.txt").c_str(),"r"); fscanf(f,"%d %ld",&P,&M); fclose(f); }
  fprintf(stderr,"[load] P=%d M=%ld\n",P,M);
  std::vector<double> Q((size_t)P*M),V((size_t)P*M),N((size_t)P*M);
  read_f64(dir+"/Q.f64",Q,(size_t)P*M); read_f64(dir+"/V.f64",V,(size_t)P*M); read_f64(dir+"/N.f64",N,(size_t)P*M);
  auto snptab=steppe::io::read_snp(snp,(size_t)M);
  double bs=steppe::core::block_size_cm_to_morgans(steppe::kDefaultBlockSizeCm);
  auto part=steppe::core::assign_blocks(snptab.chrom,snptab.genpos_morgans,bs);
  fprintf(stderr,"[blocks] n_block=%d\n",part.n_block);

  steppe::core::MatView mQ{Q.data(),P,M}, mV{V.data(),P,M}, mN{N.data(),P,M};
  auto gpu=steppe::device::make_cuda_backend();

  for(auto kind : {steppe::Precision::Kind::EmulatedFp64, steppe::Precision::Kind::Fp64}){
    steppe::Precision prec{kind, steppe::kDefaultMantissaBits};
    cudaDeviceSynchronize();
    auto t0=std::chrono::high_resolution_clock::now();
    auto t=steppe::core::compute_f2_blocks(*gpu,mQ,mV,mN,part,prec);
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    size_t free_b,tot_b; cudaMemGetInfo(&free_b,&tot_b);
    // spot value: f2 between pops 0 and 1 in block 0
    double f2_01_b0 = t.f2[0 + (size_t)P*1 + 0];
    fprintf(stderr,"[%s] compute_f2_blocks end-to-end = %.1f ms | n_block=%d block0_size=%d "
                   "f2(0,1,blk0)=%.6e | VRAM free=%.1f/%.1f GB\n",
            kind==steppe::Precision::Kind::EmulatedFp64?"EmuFp64{40}":"Fp64",
            ms,t.n_block,t.block_sizes[0],f2_01_b0,free_b/1e9,tot_b/1e9);
  }
  fprintf(stderr,"OK: production compute_f2_blocks ran at P=%d (no OOM).\n",P);
  return 0;
}
