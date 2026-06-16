// f2_block_batched_spike.cu — M4 GATING SPIKE (throwaway; experiments/, NOT production).
//
// THE QUESTION (ROADMAP §3 M4; architecture.md §5 S2, §11.1/§11.2, §12):
//   M0 measured ONE big GEMM over ALL SNPs (high arithmetic intensity). M4 must
//   produce f2 PER BLOCK (~719 autosome / 757 all-chr blocks on real AADR, avg
//   ~800 SNPs/block, VARIABLE sizes) → a batch of ~700 SMALL GEMMs. That is a
//   different regime (lower per-GEMM arithmetic intensity, possible launch-bound,
//   padding waste). The fixed-slice Ozaki 40-bit speedup + accuracy were validated
//   for the BIG GEMM only — NOT yet per-block. This spike MEASURES whether they
//   carry over, and which batched design is fastest, on REAL AADR.
//
// WHAT IT DOES, all on REAL data (no synthetic — ROADMAP §0 cautionary tale):
//   1. Load Q/V/N (column-major [P×M], shape.txt) from <dir>.
//   2. Read the first M rows of the AADR .snp (chrom, genpos in Morgans) and run
//      the SHARED block rule assign_blocks() → block_id[M], n_block. (derived_acc =
//      first 100k SNP prefix; derived_full = all 584131 — both are file-order
//      prefixes of the .snp, see build_tgeno_matrix.py snp_cap.)
//   3. Build the fused feeder outputs Q(masked), V, S=[Qsq;Hc] for all SNPs.
//   4. Reorder columns so each block is contiguous; record per-block sizes/offsets.
//   5. Compute per-block f2 [P×P×n_block] + Vpair THREE ways:
//        (A) cublasDgemmStridedBatched over blocks, padded to s_max (V=0 pad cols
//            ⇒ contribute nothing). ONE strided-batched call per GEMM (G, Vpair, R).
//        (B) loop of per-block cublasGemmEx (no padding waste, ~n_block launches).
//        (C) size-grouped strided-batched: bucket blocks by size into power-of-2
//            groups, one strided-batched call per group → bounds padding waste.
//      Each at native FP64, Ozaki 40-bit, Ozaki 32-bit. The numerator/divide
//      (assemble) is ALWAYS native FP64 in a custom kernel (architecture.md §12).
//   6. ACCURACY: per-block f2 at 40b/32b vs native-FP64 (same design) AND vs a
//      long-double CPU per-block oracle. Worst-case relative f2 error over ALL
//      blocks × pairs.
//   7. Report block-size distribution (min/median/max) + padding-waste factor.
//
// Build (REMOTE sm_120 / CUDA-13 box; NOT locally):
//   nvcc -O3 -std=c++17 -arch=sm_120 -DSTEPPE_HAVE_EMU_TUNING=1 \
//        f2_block_batched_spike.cu -lcublas -o f2_block_batched_spike
// Run:
//   ./f2_block_batched_spike <derived_dir> <snp_path> [iters]
//   e.g. ./f2_block_batched_spike /workspace/data/aadr/derived_full \
//             /workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB.snp 5
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <library_types.h>

#define CUDA_CHECK(e) do{ cudaError_t e_=(e); if(e_!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} }while(0)
#define CUBLAS_CHECK(e) do{ cublasStatus_t s_=(e); if(s_!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS %s:%d status %d\n",__FILE__,__LINE__,(int)s_); exit(1);} }while(0)

#ifndef STEPPE_HAVE_EMU_TUNING
#define STEPPE_HAVE_EMU_TUNING 0
#endif

// ---- AT2 block rule (mirrors core/domain/block_partition_rule: blgsize 0.05 Morgans = 5 cM) ----
static const double kBlockSizeMorgans = 0.05;  // architecture.md §9 / config kDefaultBlockSizeCm=5.0

static int block_of(double genpos_morgans, double bs){ return (int)std::floor(genpos_morgans/bs); }

// assign_blocks: one file-order pass, per-chrom reset + dense renumber of occupied bins.
static int assign_blocks(const std::vector<int>& chrom, const std::vector<double>& gpos,
                         double bs, std::vector<int>& block_id){
  const long m=(long)chrom.size(); block_id.assign((size_t)m,0);
  if(m<=0) return 0;
  int prev_c=0, prev_b=0; long g=-1;
  for(long s=0;s<m;s++){
    int c=chrom[(size_t)s]; int b=block_of(gpos[(size_t)s],bs);
    bool open=(s==0)||(c!=prev_c)||(b!=prev_b);
    if(open) ++g;
    block_id[(size_t)s]=(int)g; prev_c=c; prev_b=b;
  }
  return (int)g+1;
}

static void read_f64(const std::string& p, std::vector<double>& d, size_t n){
  FILE*f=fopen(p.c_str(),"rb"); if(!f){fprintf(stderr,"open %s\n",p.c_str());exit(1);}
  size_t got=fread(d.data(),8,n,f); fclose(f);
  if(got!=n){fprintf(stderr,"%s short read %zu/%zu\n",p.c_str(),got,n);exit(1);}
}

// shared het correction + assemble (verbatim formula from core/internal/f2_estimator.hpp)
static inline double het_corr(double q,double n,bool v){ if(!v)return 0.0; double d=(n-1.0>1.0)?(n-1.0):1.0; return q*(1.0-q)/d; }

// ---- per-block timing of a design; returns ms for the GEMM phase only ----
struct BlockLayout {
  int P, M, n_block, s_max;
  std::vector<int>    sizes;      // SNPs per block
  std::vector<long>   offsets;    // column offset of each block in the contiguous layout
  // contiguous (loop) layout: blocks back-to-back, total M columns
  std::vector<double> Qc, Vc, Sc; // [P×M], [P×M], [2P×M]
  // padded (strided) layout: each block padded to s_max, n_block slabs
  std::vector<double> Qp, Vp_, Sp; // [P×s_max × n_block], [..], [2P×s_max × n_block]
};

static void engage(cublasHandle_t h, int mode, int bits){
  // mode: 0 native, 1 emu
  if(mode==0){ CUBLAS_CHECK(cublasSetMathMode(h,CUBLAS_PEDANTIC_MATH)); return; }
  CUBLAS_CHECK(cublasSetMathMode(h,CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
#if STEPPE_HAVE_EMU_TUNING
  CUBLAS_CHECK(cublasSetEmulationStrategy(h,CUBLAS_EMULATION_STRATEGY_EAGER));
  CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(h,CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
  CUBLAS_CHECK(cublasSetFixedPointEmulationMaxMantissaBitCount(h,bits));
#else
  (void)bits;
#endif
}
static cublasComputeType_t ctype(int mode){ return mode==0?CUBLAS_COMPUTE_64F:CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT; }

// assemble f2 [P×P] per block from G,Vpair,R (host side, native double).
static void assemble_block(int P,const double*G,const double*Vp,const double*R,double*f2){
  for(int j=0;j<P;j++)for(int i=0;i<P;i++){
    double Rd_ij=R[(size_t)i+(size_t)2*P*j], Rd_ji=R[(size_t)j+(size_t)2*P*i];
    double H_ij =R[(size_t)(P+i)+(size_t)2*P*j], H_ji=R[(size_t)(P+j)+(size_t)2*P*i];
    double Gij=G[(size_t)i+(size_t)P*j], Vij=Vp[(size_t)i+(size_t)P*j];
    double num=Rd_ij+Rd_ji-2.0*Gij-H_ij-H_ji;
    f2[(size_t)i+(size_t)P*j]= Vij>0.0? num/Vij : 0.0;
  }
}

// ============================ DESIGN A: strided-batched, padded to s_max ============================
// One cublasDgemmStridedBatched per GEMM over all n_block slabs.
// timing returns GEMM ms; fills f2_all[P*P*n_block] + vpair_all (host).
static float design_strided(cublasHandle_t h,int mode,int bits,const BlockLayout&L,int iters,
                            std::vector<double>&f2_all,std::vector<double>&vpair_all){
  const int P=L.P, B=L.n_block, sm=L.s_max, twoP=2*P;
  const long Psm=(long)P*sm, twoPsm=(long)twoP*sm;
  // VRAM budget: padded inputs Q+V (Psm*B each) + S (twoPsm*B) + outputs G+Vpair
  // (P*P*B each) + R (twoP*P*B). At global s_max this can EXCEED 32 GB on real
  // scale (the global-pad waste finding). Probe free mem; skip (return -1) if the
  // input footprint alone won't fit — that is itself a verdict for design A.
  double need_gb = ((double)Psm*B*2 + (double)twoPsm*B + (double)P*P*B*2 + (double)twoP*P*B)*8.0/1e9;
  size_t freeb=0,totb=0; CUDA_CHECK(cudaMemGetInfo(&freeb,&totb));
  fprintf(stderr,"   [strided] needs ~%.1f GB, free=%.1f GB\n",need_gb,freeb/1e9);
  if(need_gb*1e9 > 0.92*(double)freeb){ fprintf(stderr,"   [strided] SKIP — global-s_max footprint exceeds VRAM\n"); return -1.0f; }
  // device padded inputs
  double *dQ,*dV,*dS; CUDA_CHECK(cudaMalloc(&dQ,(size_t)Psm*B*8));
  CUDA_CHECK(cudaMalloc(&dV,(size_t)Psm*B*8)); CUDA_CHECK(cudaMalloc(&dS,(size_t)twoPsm*B*8));
  CUDA_CHECK(cudaMemcpy(dQ,L.Qp.data(),(size_t)Psm*B*8,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV,L.Vp_.data(),(size_t)Psm*B*8,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dS,L.Sp.data(),(size_t)twoPsm*B*8,cudaMemcpyHostToDevice));
  // device outputs: G,Vpair [P×P×B], R [2P×P×B]
  double *dG,*dVp,*dR; CUDA_CHECK(cudaMalloc(&dG,(size_t)P*P*B*8));
  CUDA_CHECK(cudaMalloc(&dVp,(size_t)P*P*B*8)); CUDA_CHECK(cudaMalloc(&dR,(size_t)twoP*P*B*8));
  const double a=1,b=0;
  cublasComputeType_t ct=ctype(mode); engage(h,mode,bits);
  auto run=[&](){
    // G[P×P×B] = Q·Qᵀ per slab.  A=Q (m=P,k=sm,lda=P,strideA=P*sm), B=Q (n=P,ldb=P,strideB=P*sm), C=G ldc=P stride=P*P
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(h,CUBLAS_OP_N,CUBLAS_OP_T,P,P,sm,
        &a,dQ,CUDA_R_64F,P,Psm, dQ,CUDA_R_64F,P,Psm, &b,dG,CUDA_R_64F,P,(long)P*P, B,ct,CUBLAS_GEMM_DEFAULT));
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(h,CUBLAS_OP_N,CUBLAS_OP_T,P,P,sm,
        &a,dV,CUDA_R_64F,P,Psm, dV,CUDA_R_64F,P,Psm, &b,dVp,CUDA_R_64F,P,(long)P*P, B,ct,CUBLAS_GEMM_DEFAULT));
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(h,CUBLAS_OP_N,CUBLAS_OP_T,twoP,P,sm,
        &a,dS,CUDA_R_64F,twoP,twoPsm, dV,CUDA_R_64F,P,Psm, &b,dR,CUDA_R_64F,twoP,(long)twoP*P, B,ct,CUBLAS_GEMM_DEFAULT));
  };
  cudaEvent_t e0,e1;CUDA_CHECK(cudaEventCreate(&e0));CUDA_CHECK(cudaEventCreate(&e1));
  for(int w=0;w<2;w++) run(); CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(e0)); for(int it=0;it<iters;it++) run();
  CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
  float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1)); ms/=iters;
  // copy back, assemble per block
  std::vector<double> G((size_t)P*P*B),Vp((size_t)P*P*B),R((size_t)twoP*P*B);
  CUDA_CHECK(cudaMemcpy(G.data(),dG,(size_t)P*P*B*8,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(Vp.data(),dVp,(size_t)P*P*B*8,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(R.data(),dR,(size_t)twoP*P*B*8,cudaMemcpyDeviceToHost));
  f2_all.assign((size_t)P*P*B,0.0); vpair_all.assign((size_t)P*P*B,0.0);
  for(int bk=0;bk<B;bk++){
    assemble_block(P,&G[(size_t)P*P*bk],&Vp[(size_t)P*P*bk],&R[(size_t)twoP*P*bk],&f2_all[(size_t)P*P*bk]);
    memcpy(&vpair_all[(size_t)P*P*bk],&Vp[(size_t)P*P*bk],(size_t)P*P*8);
  }
  cudaFree(dQ);cudaFree(dV);cudaFree(dS);cudaFree(dG);cudaFree(dVp);cudaFree(dR);
  cudaEventDestroy(e0);cudaEventDestroy(e1);
  return ms;
}

// ============================ DESIGN B: loop of per-block GEMMs (no padding) ============================
static float design_loop(cublasHandle_t h,int mode,int bits,const BlockLayout&L,int iters,
                         std::vector<double>&f2_all,std::vector<double>&vpair_all){
  const int P=L.P, B=L.n_block, twoP=2*P;
  // contiguous device inputs (all M columns, blocks back-to-back)
  const long M=L.M;
  double *dQ,*dV,*dS; CUDA_CHECK(cudaMalloc(&dQ,(size_t)P*M*8));
  CUDA_CHECK(cudaMalloc(&dV,(size_t)P*M*8)); CUDA_CHECK(cudaMalloc(&dS,(size_t)twoP*M*8));
  CUDA_CHECK(cudaMemcpy(dQ,L.Qc.data(),(size_t)P*M*8,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV,L.Vc.data(),(size_t)P*M*8,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dS,L.Sc.data(),(size_t)twoP*M*8,cudaMemcpyHostToDevice));
  double *dG,*dVp,*dR; CUDA_CHECK(cudaMalloc(&dG,(size_t)P*P*B*8));
  CUDA_CHECK(cudaMalloc(&dVp,(size_t)P*P*B*8)); CUDA_CHECK(cudaMalloc(&dR,(size_t)twoP*P*B*8));
  const double a=1,b=0; cublasComputeType_t ct=ctype(mode); engage(h,mode,bits);
  auto run=[&](){
    for(int bk=0;bk<B;bk++){
      int s=L.sizes[bk]; long off=L.offsets[bk];
      const double* Qb=dQ+(size_t)P*off; const double* Vb=dV+(size_t)P*off; const double* Sb=dS+(size_t)twoP*off;
      double* Gb=dG+(size_t)P*P*bk; double* Vpb=dVp+(size_t)P*P*bk; double* Rb=dR+(size_t)twoP*P*bk;
      CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,P,P,s,&a,Qb,CUDA_R_64F,P,Qb,CUDA_R_64F,P,&b,Gb,CUDA_R_64F,P,ct,CUBLAS_GEMM_DEFAULT));
      CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,P,P,s,&a,Vb,CUDA_R_64F,P,Vb,CUDA_R_64F,P,&b,Vpb,CUDA_R_64F,P,ct,CUBLAS_GEMM_DEFAULT));
      CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,twoP,P,s,&a,Sb,CUDA_R_64F,twoP,Vb,CUDA_R_64F,P,&b,Rb,CUDA_R_64F,twoP,ct,CUBLAS_GEMM_DEFAULT));
    }
  };
  cudaEvent_t e0,e1;CUDA_CHECK(cudaEventCreate(&e0));CUDA_CHECK(cudaEventCreate(&e1));
  for(int w=0;w<2;w++) run(); CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(e0)); for(int it=0;it<iters;it++) run();
  CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
  float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1)); ms/=iters;
  std::vector<double> G((size_t)P*P*B),Vp((size_t)P*P*B),R((size_t)twoP*P*B);
  CUDA_CHECK(cudaMemcpy(G.data(),dG,(size_t)P*P*B*8,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(Vp.data(),dVp,(size_t)P*P*B*8,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(R.data(),dR,(size_t)twoP*P*B*8,cudaMemcpyDeviceToHost));
  f2_all.assign((size_t)P*P*B,0.0); vpair_all.assign((size_t)P*P*B,0.0);
  for(int bk=0;bk<B;bk++){
    assemble_block(P,&G[(size_t)P*P*bk],&Vp[(size_t)P*P*bk],&R[(size_t)twoP*P*bk],&f2_all[(size_t)P*P*bk]);
    memcpy(&vpair_all[(size_t)P*P*bk],&Vp[(size_t)P*P*bk],(size_t)P*P*8);
  }
  cudaFree(dQ);cudaFree(dV);cudaFree(dS);cudaFree(dG);cudaFree(dVp);cudaFree(dR);
  cudaEventDestroy(e0);cudaEventDestroy(e1);
  return ms;
}

// ============================ DESIGN C: size-grouped strided-batched ============================
// Bucket blocks into power-of-2 size groups; pad each group only to ITS group s_max;
// one strided-batched call per group. Bounds padding waste vs design A's global s_max.
struct Group { int s_pad; std::vector<int> blocks; };
static float design_grouped(cublasHandle_t h,int mode,int bits,const BlockLayout&L,int iters,
                            std::vector<double>&f2_all,std::vector<double>&vpair_all){
  const int P=L.P, B=L.n_block, twoP=2*P;
  // bucket by ceil_pow2(size) -> group pad = that pow2 (bounded waste < 2x within group)
  std::vector<Group> groups;
  auto cp2=[](int x){ int p=1; while(p<x)p<<=1; return p; };
  std::vector<std::pair<int,int>> seen; // (s_pad, group index)
  for(int bk=0;bk<B;bk++){ int sp=cp2(L.sizes[bk]); int gi=-1;
    for(size_t k=0;k<seen.size();k++) if(seen[k].first==sp){gi=seen[k].second;break;}
    if(gi<0){ gi=(int)groups.size(); groups.push_back({sp,{}}); seen.push_back({sp,gi}); }
    groups[gi].blocks.push_back(bk);
  }
  std::sort(groups.begin(),groups.end(),[](const Group&x,const Group&y){return x.s_pad<y.s_pad;});
  // VRAM-FRUGAL: process ONE group at a time (the realistic M4 design). Only one
  // group's padded inputs + its outputs are resident; f2_all (the [P²×B] tensor)
  // lives on the HOST here. Padding waste is bounded < 2x WITHIN a group (vs the
  // global-s_max design's 2.76x), so this fits VRAM where design A does not.
  const double a=1,b=0; cublasComputeType_t ct=ctype(mode); engage(h,mode,bits);
  f2_all.assign((size_t)P*P*B,0.0); vpair_all.assign((size_t)P*P*B,0.0);
  double padded=0, real=0;
  cudaEvent_t e0,e1;CUDA_CHECK(cudaEventCreate(&e0));CUDA_CHECK(cudaEventCreate(&e1));
  float ms_total=0;
  for(auto&g:groups){
    int sp=g.s_pad, nb=(int)g.blocks.size(); long Psp=(long)P*sp, twoPsp=(long)twoP*sp;
    padded+=(double)sp*nb; for(int bk:g.blocks) real+=L.sizes[bk];
    std::vector<double> Qp((size_t)Psp*nb,0.0),Vp((size_t)Psp*nb,0.0),Sp((size_t)twoPsp*nb,0.0);
    for(int k=0;k<nb;k++){ int bk=g.blocks[k]; int s=L.sizes[bk]; long off=L.offsets[bk];
      for(int c=0;c<s;c++){
        memcpy(&Qp[(size_t)Psp*k+(size_t)P*c], &L.Qc[(size_t)P*(off+c)], (size_t)P*8);
        memcpy(&Vp[(size_t)Psp*k+(size_t)P*c], &L.Vc[(size_t)P*(off+c)], (size_t)P*8);
        memcpy(&Sp[(size_t)twoPsp*k+(size_t)twoP*c], &L.Sc[(size_t)twoP*(off+c)], (size_t)twoP*8);
      }
    }
    double *dQ,*dV,*dS,*dG,*dVp,*dR;
    CUDA_CHECK(cudaMalloc(&dQ,(size_t)Psp*nb*8)); CUDA_CHECK(cudaMalloc(&dV,(size_t)Psp*nb*8));
    CUDA_CHECK(cudaMalloc(&dS,(size_t)twoPsp*nb*8));
    CUDA_CHECK(cudaMemcpy(dQ,Qp.data(),(size_t)Psp*nb*8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV,Vp.data(),(size_t)Psp*nb*8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dS,Sp.data(),(size_t)twoPsp*nb*8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dG,(size_t)P*P*nb*8)); CUDA_CHECK(cudaMalloc(&dVp,(size_t)P*P*nb*8));
    CUDA_CHECK(cudaMalloc(&dR,(size_t)twoP*P*nb*8));
    auto run=[&](){
      CUBLAS_CHECK(cublasGemmStridedBatchedEx(h,CUBLAS_OP_N,CUBLAS_OP_T,P,P,sp,
          &a,dQ,CUDA_R_64F,P,Psp, dQ,CUDA_R_64F,P,Psp, &b,dG,CUDA_R_64F,P,(long)P*P, nb,ct,CUBLAS_GEMM_DEFAULT));
      CUBLAS_CHECK(cublasGemmStridedBatchedEx(h,CUBLAS_OP_N,CUBLAS_OP_T,P,P,sp,
          &a,dV,CUDA_R_64F,P,Psp, dV,CUDA_R_64F,P,Psp, &b,dVp,CUDA_R_64F,P,(long)P*P, nb,ct,CUBLAS_GEMM_DEFAULT));
      CUBLAS_CHECK(cublasGemmStridedBatchedEx(h,CUBLAS_OP_N,CUBLAS_OP_T,twoP,P,sp,
          &a,dS,CUDA_R_64F,twoP,twoPsp, dV,CUDA_R_64F,P,Psp, &b,dR,CUDA_R_64F,twoP,(long)twoP*P, nb,ct,CUBLAS_GEMM_DEFAULT));
    };
    for(int w=0;w<2;w++) run(); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0)); for(int it=0;it<iters;it++) run();
    CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
    float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1)); ms_total+=ms/iters;
    std::vector<double> G((size_t)P*P*nb),Vc((size_t)P*P*nb),R((size_t)twoP*P*nb);
    CUDA_CHECK(cudaMemcpy(G.data(),dG,(size_t)P*P*nb*8,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Vc.data(),dVp,(size_t)P*P*nb*8,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(R.data(),dR,(size_t)twoP*P*nb*8,cudaMemcpyDeviceToHost));
    for(int k=0;k<nb;k++){ int bk=g.blocks[k];
      assemble_block(P,&G[(size_t)P*P*k],&Vc[(size_t)P*P*k],&R[(size_t)twoP*P*k],&f2_all[(size_t)P*P*bk]);
      memcpy(&vpair_all[(size_t)P*P*bk],&Vc[(size_t)P*P*k],(size_t)P*P*8);
    }
    cudaFree(dQ);cudaFree(dV);cudaFree(dS);cudaFree(dG);cudaFree(dVp);cudaFree(dR);
  }
  cudaEventDestroy(e0);cudaEventDestroy(e1);
  fprintf(stderr,"   [grouped] %zu pow2 size-groups, per-group resident; pad-waste=%.2fx\n",
          groups.size(),padded/real);
  return ms_total;
}

// ---- long-double per-block CPU oracle (cancellation-free): f2_all_ref[P*P*B] ----
static void cpu_oracle(const BlockLayout&L,const std::vector<double>&Qraw,const std::vector<double>&Vraw,
                       const std::vector<double>&Nraw,const std::vector<int>&block_id,
                       std::vector<double>&f2_ref){
  const int P=L.P; const long M=L.M; const int B=L.n_block;
  f2_ref.assign((size_t)P*P*B,0.0);
  // group SNP indices per block (file order)
  std::vector<std::vector<long>> bsnps(B);
  for(long s=0;s<M;s++) bsnps[block_id[(size_t)s]].push_back(s);
  #pragma omp parallel for schedule(dynamic)
  for(int bk=0;bk<B;bk++){
    const auto& idx=bsnps[bk];
    for(int i=0;i<P;i++) for(int j=i+1;j<P;j++){
      long double sum=0; long cnt=0;
      for(long s : idx){ size_t ki=(size_t)i+(size_t)P*s, kj=(size_t)j+(size_t)P*s;
        if(Vraw[ki]>0.5 && Vraw[kj]>0.5){
          long double pi=Qraw[ki],pj=Qraw[kj]; long double d=pi-pj;
          long double ni=Nraw[ki],nj=Nraw[kj];
          long double hci=pi*(1.0L-pi)/((ni>1.0L)?(ni-1.0L):1.0L);
          long double hcj=pj*(1.0L-pj)/((nj>1.0L)?(nj-1.0L):1.0L);
          sum+= d*d-hci-hcj; cnt++;
        }
      }
      double f=cnt? (double)(sum/(long double)cnt):0.0;
      f2_ref[(size_t)i+(size_t)P*j+(size_t)P*P*bk]=f;
      f2_ref[(size_t)j+(size_t)P*i+(size_t)P*P*bk]=f;
    }
  }
}

// Accuracy diagnostics over ALL blocks × off-diagonal pairs.
struct Acc {
  double maxRel=0;       // worst pure-relative error (|ref|>1e-12)
  double atWorstAbs=0;   // absolute error at the worst-relative point
  double atWorstRef=0;   // |ref| at the worst-relative point
  double maxAbs=0;       // worst ABSOLUTE error anywhere
  double maxRelCombined=0; // worst (|c-r| / (atol + |r|)) — the combined tolerance form
  long scored=0; int signflips=0;
};
// combined-form atol: f2 values are O(1e-2..1e-1); a per-block atol of 1e-9 is far
// below any real f2 yet absorbs the near-zero entries that inflate pure relative.
static const double kAtol = 1e-9;
static Acc accuracy(int P,int B,const std::vector<double>&cand,const std::vector<double>&ref){
  Acc a;
  for(int bk=0;bk<B;bk++) for(int j=0;j<P;j++) for(int i=0;i<P;i++){
    if(i==j) continue; size_t o=(size_t)i+(size_t)P*j+(size_t)P*P*bk;
    double r=ref[o], c=cand[o]; double ae=fabs(c-r);
    if(ae>a.maxAbs) a.maxAbs=ae;
    double rc=ae/(kAtol+fabs(r)); if(rc>a.maxRelCombined) a.maxRelCombined=rc;
    if(fabs(r)<1e-12) continue;
    double e=ae/std::max(fabs(r),1e-300);
    if(e>a.maxRel){ a.maxRel=e; a.atWorstAbs=ae; a.atWorstRef=fabs(r); }
    a.scored++;
    if(((r>0)!=(c>0)) && c!=0) a.signflips++;
  }
  return a;
}

int main(int argc,char**argv){
  if(argc<3){ fprintf(stderr,"usage: %s <derived_dir> <snp_path> [iters]\n",argv[0]); return 2; }
  std::string dir=argv[1], snp=argv[2]; int iters=(argc>=4)?atoi(argv[3]):5;

  // ---- load shape + Q/V/N ----
  int P; long long M;
  { std::string sp=dir+"/shape.txt"; FILE*f=fopen(sp.c_str(),"r"); if(!f||fscanf(f,"%d %lld",&P,&M)!=2){fprintf(stderr,"bad shape\n");return 1;} fclose(f); }
  fprintf(stderr,"[load] dir=%s P=%d M=%lld iters=%d — REAL AADR\n",dir.c_str(),P,(long long)M,iters);
  size_t qN=(size_t)P*M; std::vector<double> Qraw(qN),Vraw(qN),Nraw(qN);
  read_f64(dir+"/Q.f64",Qraw,qN); read_f64(dir+"/V.f64",Vraw,qN); read_f64(dir+"/N.f64",Nraw,qN);

  // ---- read first M rows of .snp: col2 chrom, col3 genpos (Morgans) ----
  std::vector<int> chrom; std::vector<double> gpos; chrom.reserve((size_t)M); gpos.reserve((size_t)M);
  { FILE*f=fopen(snp.c_str(),"r"); if(!f){fprintf(stderr,"open snp %s\n",snp.c_str());return 1;}
    char id[256]; int c; double gp; long p4; char a1[8],a2[8]; long got=0;
    while(got<M && fscanf(f,"%255s %d %lf %ld %7s %7s",id,&c,&gp,&p4,a1,a2)==6){ chrom.push_back(c); gpos.push_back(gp); got++; }
    fclose(f);
    if(got!=M){ fprintf(stderr,"snp rows %ld != M %lld\n",got,(long long)M); return 1; }
  }
  std::vector<int> block_id; int n_block=assign_blocks(chrom,gpos,kBlockSizeMorgans,block_id);
  fprintf(stderr,"[blocks] n_block=%d (blgsize=%.3f Morgans)\n",n_block,kBlockSizeMorgans);

  // ---- block-size distribution + offsets (file order, blocks already contiguous in file order) ----
  BlockLayout L; L.P=P; L.M=(int)M; L.n_block=n_block;
  L.sizes.assign(n_block,0);
  for(long s=0;s<(long)M;s++) L.sizes[block_id[(size_t)s]]++;
  L.offsets.assign(n_block,0); { long off=0; for(int bk=0;bk<n_block;bk++){ L.offsets[bk]=off; off+=L.sizes[bk]; } }
  std::vector<int> srt=L.sizes; std::sort(srt.begin(),srt.end());
  int smin=srt.front(), smax=srt.back(), smed=srt[srt.size()/2];
  L.s_max=smax;
  double tot_real=(double)M, tot_pad=(double)smax*n_block;
  fprintf(stderr,"[dist] SNPs/block  min=%d  median=%d  max=%d   total=%lld\n",smin,smed,smax,(long long)M);
  fprintf(stderr,"[pad ] strided global s_max=%d  -> padded cols=%.0f  REAL cols=%lld  waste=%.2fx\n",
          smax,tot_pad,(long long)M,tot_pad/tot_real);

  // ---- fused feeder: contiguous layout (file order == block-contiguous already) ----
  int twoP=2*P;
  L.Qc.assign((size_t)P*M,0.0); L.Vc.assign((size_t)P*M,0.0); L.Sc.assign((size_t)twoP*M,0.0);
  #pragma omp parallel for schedule(static)
  for(long s=0;s<(long)M;s++) for(int i=0;i<P;i++){
    size_t k=(size_t)i+(size_t)P*s; bool v=Vraw[k]>0.5; double q=v?Qraw[k]:0.0; double hc=het_corr(Qraw[k],Nraw[k],v);
    L.Qc[k]=q; L.Vc[k]=v?1.0:0.0;
    L.Sc[(size_t)i+(size_t)twoP*s]=q*q; L.Sc[(size_t)(P+i)+(size_t)twoP*s]=hc;
  }
  // ---- padded layout for design A (each block -> s_max slab; pad cols zero) ----
  // Build the global-padded layout for design A only when it could plausibly fit
  // VRAM (input footprint < device total); otherwise skip the ~40 GB host build —
  // design_strided's own cudaMemGetInfo guard will SKIP it. We probe device total.
  size_t fb0=0,tb0=0; CUDA_CHECK(cudaMemGetInfo(&fb0,&tb0));
  double aInGB=((double)P*smax*n_block*2 + (double)twoP*smax*n_block)*8.0/1e9;
  if(aInGB < (double)tb0/1e9){
    L.Qp.assign((size_t)P*smax*n_block,0.0); L.Vp_.assign((size_t)P*smax*n_block,0.0);
    L.Sp.assign((size_t)twoP*smax*n_block,0.0);
    #pragma omp parallel for schedule(dynamic)
    for(int bk=0;bk<n_block;bk++){ int s=L.sizes[bk]; long off=L.offsets[bk];
      for(int c=0;c<s;c++){
        memcpy(&L.Qp[(size_t)P*smax*bk+(size_t)P*c], &L.Qc[(size_t)P*(off+c)], (size_t)P*8);
        memcpy(&L.Vp_[(size_t)P*smax*bk+(size_t)P*c], &L.Vc[(size_t)P*(off+c)], (size_t)P*8);
        memcpy(&L.Sp[(size_t)twoP*smax*bk+(size_t)twoP*c], &L.Sc[(size_t)twoP*(off+c)], (size_t)twoP*8);
      }
    }
  } else {
    fprintf(stderr,"[pad ] design-A global-pad host build SKIPPED (~%.0f GB inputs > %.0f GB device total)\n",
            aInGB,(double)tb0/1e9);
  }

  // ---- CPU oracle (per-block long double) ----
  fprintf(stderr,"[oracle] computing per-block long-double reference (P=%d, may take a bit)...\n",P);
  std::vector<double> f2_ref; cpu_oracle(L,Qraw,Vraw,Nraw,block_id,f2_ref);

  cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h));
  void* ws; CUDA_CHECK(cudaMalloc(&ws,(size_t)64*1024*1024)); CUBLAS_CHECK(cublasSetWorkspace(h,ws,(size_t)64*1024*1024));

  // ---- run each design at native / 40b / 32b ----
  struct Res { const char* name; bool ok; float ms_nat,ms_40,ms_32; Acc anat,a40,a32; };
  std::vector<Res> R;
  printf("\n==== THROUGHPUT (GEMM phase, ms; speedup vs native FP64 same design) ====\n");
  printf("%-10s %10s %10s %8s %10s %8s\n","design","nat_ms","40b_ms","40/nat","32b_ms","32/nat");

  auto bench=[&](const char* nm, auto fn){
    std::vector<double> f2n,vpn, f240,vp40, f232,vp32;
    float mn=fn(h,0,53,L,iters,f2n,vpn);
    if(mn<0){ printf("%-10s %10s %10s %8s %10s %8s\n",nm,"SKIP(OOM)","-","-","-","-");
              Res r; r.name=nm; r.ok=false; R.push_back(r); return; }
    float m40=fn(h,1,40,L,iters,f240,vp40);
    float m32=fn(h,1,32,L,iters,f232,vp32);
    Res r; r.name=nm; r.ok=true; r.ms_nat=mn; r.ms_40=m40; r.ms_32=m32;
    r.anat=accuracy(P,n_block,f2n,f2_ref);
    r.a40 =accuracy(P,n_block,f240,f2_ref);
    r.a32 =accuracy(P,n_block,f232,f2_ref);
    printf("%-10s %10.3f %10.3f %8.2f %10.3f %8.2f\n",nm,mn,m40,mn/m40,m32,mn/m32);
    R.push_back(r);
  };
  bench("strided", design_strided);
  bench("loop",    design_loop);
  bench("grouped", design_grouped);

  printf("\n==== ACCURACY vs long-double CPU oracle, over ALL %d blocks × off-diag pairs ====\n",n_block);
  printf("PURE-RELATIVE (|c-r|/|r|, |r|>1e-12) — inflates on tiny-|f2| block entries:\n");
  printf("%-10s %13s %13s %13s\n","design","native","40b","32b");
  for(auto&r:R){ if(!r.ok){printf("%-10s %13s\n",r.name,"SKIP(OOM)");continue;}
    printf("%-10s %13.3e %13.3e %13.3e\n",r.name,r.anat.maxRel,r.a40.maxRel,r.a32.maxRel); }
  printf("\nMAX-ABSOLUTE (|c-r|) — the true precision floor (f2 magnitudes are O(1e-2..1e-1)):\n");
  printf("%-10s %13s %13s %13s\n","design","native","40b","32b");
  for(auto&r:R){ if(!r.ok){printf("%-10s %13s\n",r.name,"SKIP(OOM)");continue;}
    printf("%-10s %13.3e %13.3e %13.3e\n",r.name,r.anat.maxAbs,r.a40.maxAbs,r.a32.maxAbs); }
  printf("\nCOMBINED |c-r|/(%g+|r|) — the production tolerance form (atol absorbs near-zero f2):\n",kAtol);
  printf("%-10s %13s %13s %13s\n","design","native","40b","32b");
  for(auto&r:R){ if(!r.ok){printf("%-10s %13s\n",r.name,"SKIP(OOM)");continue;}
    printf("%-10s %13.3e %13.3e %13.3e\n",r.name,r.anat.maxRelCombined,r.a40.maxRelCombined,r.a32.maxRelCombined); }
  printf("\nWORST-RELATIVE POINT (where pure-relative peaks) — is it a tiny-|f2| entry?\n");
  printf("%-10s %14s %14s %14s\n","design","|ref|@worst40","abserr@worst40","ratio40/nat");
  for(auto&r:R){ if(!r.ok){printf("%-10s %14s\n",r.name,"SKIP(OOM)");continue;}
    printf("%-10s %14.3e %14.3e %14.3e\n",r.name,r.a40.atWorstRef,r.a40.atWorstAbs,r.a40.maxRel/std::max(r.anat.maxRel,1e-300)); }

  printf("\n# Reference targets (M0 big-GEMM over ALL SNPs, ROADMAP §0): native ~1e-11, 40b ~2.2e-11, 32b ~8.6e-9.\n");
  printf("# Per-block pure-relative is EXPECTED higher (small block ⇒ some tiny-|f2| entries blow up the ratio);\n");
  printf("# the load-bearing checks are 40b≈native AND the max-ABSOLUTE / combined-tolerance floors.\n");

  cudaFree(ws); cublasDestroy(h);
  return 0;
}
