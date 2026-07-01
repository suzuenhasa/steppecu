// f2_timing.cu  — throughput crossover for the f2 3-GEMM reformulation:
//   native FP64  vs  Ozaki emulated FP64 (CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, EAGER),
// TIMING ONLY (accuracy already established in f2_emu_spike). Two modes:
//   SWEEP  : ./f2_timing M iters P1 [P2 ...]            synthetic random freqs, sweep P
//   LOAD   : ./f2_timing --load <dir> [iters]           real Q/V/N from build_matrix.py
//
// The three f2 GEMMs (column-major), per the reference reformulation:
//   G[P x P]     = Q[P x M] * Q^T        (OP_N, OP_T, m=P,  n=P, k=M)
//   Vpair[P x P] = V[P x M] * V^T        (OP_N, OP_T, m=P,  n=P, k=M)
//   R[2P x P]    = S[2P x M] * V^T       (OP_N, OP_T, m=2P, n=P, k=M)
// Arithmetic intensity of G ~ P/8 FLOP/byte -> compute-bound only once P is large,
// which is the all-pairs precompute regime (P = hundreds..thousands of populations).
//
// Build: nvcc -O3 -std=c++17 -arch=sm_120 -DSTEPPE_HAVE_EMU_TUNING=1 f2_timing.cu -lcublas -o f2_timing
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <library_types.h>

#define CUDA_CHECK(e) do{ cudaError_t e_=(e); if(e_!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} }while(0)
#define CUBLAS_CHECK(e) do{ cublasStatus_t s_=(e); if(s_!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS %s:%d status %d\n",__FILE__,__LINE__,(int)s_); exit(1);} }while(0)

static void three_gemms(cublasHandle_t h, cublasComputeType_t ct, int P, int M,
                        const double* dQ,const double* dV,const double* dS,
                        double* dG,double* dVp,double* dR){
  const double a=1.0,b=0.0;
  CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,P,P,M,&a,dQ,CUDA_R_64F,P,dQ,CUDA_R_64F,P,&b,dG,CUDA_R_64F,P,ct,CUBLAS_GEMM_DEFAULT));
  CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,P,P,M,&a,dV,CUDA_R_64F,P,dV,CUDA_R_64F,P,&b,dVp,CUDA_R_64F,P,ct,CUBLAS_GEMM_DEFAULT));
  CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,2*P,P,M,&a,dS,CUDA_R_64F,2*P,dV,CUDA_R_64F,P,&b,dR,CUDA_R_64F,2*P,ct,CUBLAS_GEMM_DEFAULT));
}

static float time_ms(cublasHandle_t h,cublasComputeType_t ct,int P,int M,int iters,
                     const double*dQ,const double*dV,const double*dS,double*dG,double*dVp,double*dR){
  cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
  for(int i=0;i<2;i++) three_gemms(h,ct,P,M,dQ,dV,dS,dG,dVp,dR);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(s));
  for(int i=0;i<iters;i++) three_gemms(h,ct,P,M,dQ,dV,dS,dG,dVp,dR);
  CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
  float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,s,e));
  cudaEventDestroy(s); cudaEventDestroy(e); return ms/iters;
}

// FP32-data TF32-tensor-core versions (cost is data-INDEPENDENT, unlike Ozaki).
static void three_gemms_f32(cublasHandle_t h, cublasComputeType_t ct, int P, int M,
                        const float* dQ,const float* dV,const float* dS,
                        float* dG,float* dVp,float* dR){
  const float a=1.0f,b=0.0f;
  CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,P,P,M,&a,dQ,CUDA_R_32F,P,dQ,CUDA_R_32F,P,&b,dG,CUDA_R_32F,P,ct,CUBLAS_GEMM_DEFAULT));
  CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,P,P,M,&a,dV,CUDA_R_32F,P,dV,CUDA_R_32F,P,&b,dVp,CUDA_R_32F,P,ct,CUBLAS_GEMM_DEFAULT));
  CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,2*P,P,M,&a,dS,CUDA_R_32F,2*P,dV,CUDA_R_32F,P,&b,dR,CUDA_R_32F,2*P,ct,CUBLAS_GEMM_DEFAULT));
}
static float time_ms_f32(cublasHandle_t h,cublasComputeType_t ct,int P,int M,int iters,
                     const float*dQ,const float*dV,const float*dS,float*dG,float*dVp,float*dR){
  cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
  for(int i=0;i<2;i++) three_gemms_f32(h,ct,P,M,dQ,dV,dS,dG,dVp,dR);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(s));
  for(int i=0;i<iters;i++) three_gemms_f32(h,ct,P,M,dQ,dV,dS,dG,dVp,dR);
  CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
  float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,s,e));
  cudaEventDestroy(s); cudaEventDestroy(e); return ms/iters;
}

// alloc device, copy host inputs, time both precisions, print one row.
static void run_one(cublasHandle_t hNat,cublasHandle_t hEmu,int P,int M,int iters,
                    const std::vector<double>&hQ,const std::vector<double>&hV,const std::vector<double>&hS,
                    const char* tag){
  size_t qN=(size_t)P*M, sN=(size_t)2*P*M;
  double *dQ,*dV,*dS,*dG,*dVp,*dR;
  CUDA_CHECK(cudaMalloc(&dQ,qN*8)); CUDA_CHECK(cudaMalloc(&dV,qN*8)); CUDA_CHECK(cudaMalloc(&dS,sN*8));
  CUDA_CHECK(cudaMalloc(&dG,(size_t)P*P*8)); CUDA_CHECK(cudaMalloc(&dVp,(size_t)P*P*8)); CUDA_CHECK(cudaMalloc(&dR,(size_t)2*P*P*8));
  CUDA_CHECK(cudaMemcpy(dQ,hQ.data(),qN*8,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV,hV.data(),qN*8,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dS,hS.data(),sN*8,cudaMemcpyHostToDevice));
  float tn=time_ms(hNat,CUBLAS_COMPUTE_64F,P,M,iters,dQ,dV,dS,dG,dVp,dR);
  std::vector<double> gNat((size_t)P*P),gEmu((size_t)P*P);
  CUDA_CHECK(cudaMemcpy(gNat.data(),dG,(size_t)P*P*8,cudaMemcpyDeviceToHost));
  float te=time_ms(hEmu,CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT,P,M,iters,dQ,dV,dS,dG,dVp,dR);
  CUDA_CHECK(cudaMemcpy(gEmu.data(),dG,(size_t)P*P*8,cudaMemcpyDeviceToHost));
  bool eq=true; for(size_t i=0;i<gNat.size();i++) if(gNat[i]!=gEmu[i]){eq=false;break;}
  // max relative emu-vs-native diff on G (cheap P^2 sanity, not ground truth)
  double mrel=0; for(size_t i=0;i<gNat.size();i++){ double d=fabs(gEmu[i]-gNat[i]); double r=fabs(gNat[i])>1e-300?d/fabs(gNat[i]):d; if(r>mrel)mrel=r; }
#if STEPPE_HAVE_EMU_TUNING
  // ---- FIXED-slice Ozaki at capped mantissa bits (fewer slices than dynamic -> faster) ----
  CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(hEmu,CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
  int fb[]={32,40,48};
  for(int k=0;k<3;k++){ CUBLAS_CHECK(cublasSetFixedPointEmulationMaxMantissaBitCount(hEmu,fb[k]));
    float tfx=time_ms(hEmu,CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT,P,M,iters,dQ,dV,dS,dG,dVp,dR);
    printf("   ozaki-fixed %2db: %10.3f ms   vs_nat=%6.2fx\n",fb[k],tfx,tn/tfx); }
  CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(hEmu,CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC));
#endif
  // ---- TF32 arm: FP32 data, TF32 tensor-core compute (cost data-INDEPENDENT) ----
  std::vector<float> fQ(hQ.begin(),hQ.end()), fV(hV.begin(),hV.end()), fS(hS.begin(),hS.end());
  float *fdQ,*fdV,*fdS,*fdG,*fdVp,*fdR;
  CUDA_CHECK(cudaMalloc(&fdQ,qN*4)); CUDA_CHECK(cudaMalloc(&fdV,qN*4)); CUDA_CHECK(cudaMalloc(&fdS,sN*4));
  CUDA_CHECK(cudaMalloc(&fdG,(size_t)P*P*4)); CUDA_CHECK(cudaMalloc(&fdVp,(size_t)P*P*4)); CUDA_CHECK(cudaMalloc(&fdR,(size_t)2*P*P*4));
  CUDA_CHECK(cudaMemcpy(fdQ,fQ.data(),qN*4,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(fdV,fV.data(),qN*4,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(fdS,fS.data(),sN*4,cudaMemcpyHostToDevice));
  cublasHandle_t hTf; CUBLAS_CHECK(cublasCreate(&hTf)); CUBLAS_CHECK(cublasSetMathMode(hTf,CUBLAS_TF32_TENSOR_OP_MATH));
  float tt=time_ms_f32(hTf,CUBLAS_COMPUTE_32F_FAST_TF32,P,M,iters,fdQ,fdV,fdS,fdG,fdVp,fdR);
  std::vector<float> gTf((size_t)P*P); CUDA_CHECK(cudaMemcpy(gTf.data(),fdG,(size_t)P*P*4,cudaMemcpyDeviceToHost));
  double tfrel=0; for(size_t i=0;i<gNat.size();i++){ double d=fabs((double)gTf[i]-gNat[i]); double r=fabs(gNat[i])>1e-300?d/fabs(gNat[i]):d; if(r>tfrel)tfrel=r; }
  cublasDestroy(hTf);
  cudaFree(fdQ);cudaFree(fdV);cudaFree(fdS);cudaFree(fdG);cudaFree(fdVp);cudaFree(fdR);
  double gflop=3.0*2.0*(double)P*P*M/1e9;
  printf("%10s %6d %9d %8.1f %10.3f %10.3f %9.2f %10.3f %9.2f %8s %10.2e %10.2e %9.1f\n",
         tag,P,M,P/8.0,tn,te,tn/te,tt,tn/tt, eq?"EQ!":"no", mrel, tfrel, gflop);
  fflush(stdout);
  cudaFree(dQ);cudaFree(dV);cudaFree(dS);cudaFree(dG);cudaFree(dVp);cudaFree(dR);
}

static void hdr(){ printf("%10s %6s %9s %8s %10s %10s %9s %10s %9s %8s %10s %10s %9s\n",
  "tag","P","M","AI~P/8","t_nat_ms","t_emu_ms","emu/nat","t_tf32_ms","tf32/nat","emuEqNat","emuVsNat","tf32VsNat","GFLOP"); }

// read N doubles from path; exit on short read.
static void read_f64(const char* path,double* dst,size_t n){
  FILE* f=fopen(path,"rb"); if(!f){fprintf(stderr,"open %s failed\n",path);exit(1);}
  size_t got=fread(dst,sizeof(double),n,f);
  // ensure no extra trailing data (size mismatch => layout/shape error)
  size_t extra=fread(dst+ (n? n-1:0),1,1,f); (void)extra;
  fclose(f);
  if(got!=n){fprintf(stderr,"%s: read %zu of %zu doubles\n",path,got,n);exit(1);}
}

int main(int argc,char**argv){
  cublasHandle_t hNat,hEmu;
  CUBLAS_CHECK(cublasCreate(&hNat)); CUBLAS_CHECK(cublasCreate(&hEmu));
  CUBLAS_CHECK(cublasSetMathMode(hNat,CUBLAS_PEDANTIC_MATH));
#if STEPPE_HAVE_EMU_TUNING
  CUBLAS_CHECK(cublasSetMathMode(hEmu,CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
  CUBLAS_CHECK(cublasSetEmulationStrategy(hEmu,CUBLAS_EMULATION_STRATEGY_EAGER));
  // The headline t_emu_ms / emu/nat column below measures DYNAMIC control -- this
  // is deliberately the rejected TRAP arm (it overshoots to ~60 bits on real data
  // -> parity with native, no win). The production-relevant numbers are the
  // "ozaki-fixed 32/40/48b" rows printed in run_one(), which use FIXED control and
  // show the 8-17x win. Production never selects dynamic (architecture.md 12, ROADMAP 0).
  CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(hEmu,CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC));
#endif

  if(argc>=3 && strcmp(argv[1],"--load")==0){
    const char* dir=argv[2];
    int iters=(argc>=4)?atoi(argv[3]):10;
    char path[1024]; int P=0; long long M=0;
    snprintf(path,sizeof path,"%s/shape.txt",dir);
    { FILE* f=fopen(path,"r"); if(!f){fprintf(stderr,"open %s failed\n",path);return 1;}
      if(fscanf(f,"%d %lld",&P,&M)!=2){fprintf(stderr,"bad shape.txt\n");return 1;} fclose(f); }
    fprintf(stderr,"[load] dir=%s P=%d M=%lld iters=%d\n",dir,P,(long long)M,iters);
    size_t qN=(size_t)P*M;
    std::vector<double> Qraw(qN),V(qN),N(qN),Q(qN),S((size_t)2*P*M);
    snprintf(path,sizeof path,"%s/Q.f64",dir); read_f64(path,Qraw.data(),qN);
    snprintf(path,sizeof path,"%s/V.f64",dir); read_f64(path,V.data(),qN);
    snprintf(path,sizeof path,"%s/N.f64",dir); read_f64(path,N.data(),qN);
    // build masked Q and stacked S=[Qsq;Hc] (column-major; element (i,s) at i+P*s / i+2P*s)
    for(long long s=0;s<M;s++) for(int i=0;i<P;i++){
      size_t k=(size_t)i+(size_t)P*s; double v=V[k], q=Qraw[k]*v; double n=N[k];
      Q[k]=q;
      S[(size_t)i+(size_t)2*P*s]     = q*q;
      S[(size_t)(P+i)+(size_t)2*P*s] = q*(1.0-q)/((n>1.0)?(n-1.0):1.0)*v;
    }
    hdr();
    run_one(hNat,hEmu,P,(int)M,iters,Q,V,S,"AADR");
  } else if(argc>=4){
    int M=atoi(argv[1]); int iters=atoi(argv[2]);
    std::mt19937_64 rng(1234); std::uniform_real_distribution<double> U(0.0,1.0);
    hdr();
    for(int ai=3;ai<argc;ai++){
      int P=atoi(argv[ai]); size_t qN=(size_t)P*M;
      std::vector<double> Q(qN),V(qN),S((size_t)2*P*M);
      for(size_t i=0;i<qN;i++){ Q[i]=U(rng); V[i]=1.0; }
      for(size_t s=0;s<(size_t)M;s++) for(int i=0;i<P;i++){ double q=Q[(size_t)i+(size_t)P*s];
        S[(size_t)i+(size_t)2*P*s]=q*q; S[(size_t)(P+i)+(size_t)2*P*s]=q*(1.0-q)/198.0; }
      run_one(hNat,hEmu,P,M,iters,Q,V,S,"synth");
    }
  } else {
    fprintf(stderr,"usage:\n  %s M iters P1 [P2 ...]\n  %s --load <dir> [iters]\n",argv[0],argv[0]);
    return 2;
  }
  printf("# emu/nat>1 => emulation FASTER. emuEqNat must be 'no' (else emulation didn't engage).\n");
  cublasDestroy(hNat); cublasDestroy(hEmu); return 0;
}
