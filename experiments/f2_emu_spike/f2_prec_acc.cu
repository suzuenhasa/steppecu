// f2_prec_acc.cu  — how close to exact FP64 do the fast tensor-core precisions get
// on REAL f2, after the catastrophic cancellation? Computes the f2 matrix several ways
// and compares each to a cancellation-free long-double CPU reference.
//
// Arms (the 3 GEMMs G=Q*Q^T, Vpair=V*V^T, R=[Qsq;Hc]*V^T run in each precision; the
// numerator+divide is ALWAYS done in double, isolating the GEMM precision's effect):
//   nat64   : CUBLAS_COMPUTE_64F                     (exact baseline, FP64 hardware)
//   fp32    : CUBLAS_COMPUTE_32F                     (plain FP32, ~24-bit, no tensor core)
//   tf32    : CUBLAS_COMPUTE_32F_FAST_TF32           (1 tensor-core pass, ~10-bit mantissa)
//   bf16x9  : CUBLAS_COMPUTE_32F_EMULATED_16BFX9     (multi-pass tensor-core "Ozaki" -> ~FP32)
//
// Build: nvcc -O3 -std=c++17 -arch=sm_120 -DSTEPPE_HAVE_EMU_TUNING=1 f2_prec_acc.cu -lcublas -o f2_prec_acc
// Run:   ./f2_prec_acc --load <dir>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <library_types.h>

#define CUDA_CHECK(e) do{cudaError_t e_=(e); if(e_!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);} }while(0)
#define CUBLAS_CHECK(e) do{cublasStatus_t s_=(e); if(s_!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS %s:%d status %d\n",__FILE__,__LINE__,(int)s_);exit(1);} }while(0)

static void read_f64(const char*p,double*d,size_t n){FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p);exit(1);} size_t g=fread(d,8,n,f); fclose(f); if(g!=n){fprintf(stderr,"%s short read %zu/%zu\n",p,g,n);exit(1);} }

// assemble f2 (double) from host-side G[P*P], Vp[P*P], R[2P*P] (all column-major).
static void assemble(int P,const std::vector<double>&G,const std::vector<double>&Vp,const std::vector<double>&R,std::vector<double>&f2){
  f2.assign((size_t)P*P,0.0);
  for(int j=0;j<P;j++)for(int i=0;i<P;i++){
    double Rd_ij=R[(size_t)i+(size_t)2*P*j], Rd_ji=R[(size_t)j+(size_t)2*P*i];
    double H_ij =R[(size_t)(P+i)+(size_t)2*P*j], H_ji=R[(size_t)(P+j)+(size_t)2*P*i];
    double Gij=G[(size_t)i+(size_t)P*j], Vij=Vp[(size_t)i+(size_t)P*j];
    double num=Rd_ij+Rd_ji-2.0*Gij-H_ij-H_ji;
    f2[(size_t)i+(size_t)P*j]= Vij>0.0? num/Vij : 0.0;
  }
}

// time the 3 GEMMs (double path) and return host G,Vp,R
static float gemms64(cublasHandle_t h,cublasComputeType_t ct,int P,int M,const double*dQ,const double*dV,const double*dS,
                     std::vector<double>&G,std::vector<double>&Vp,std::vector<double>&R){
  double*dG,*dVp,*dR; CUDA_CHECK(cudaMalloc(&dG,(size_t)P*P*8));CUDA_CHECK(cudaMalloc(&dVp,(size_t)P*P*8));CUDA_CHECK(cudaMalloc(&dR,(size_t)2*P*P*8));
  const double a=1,b=0; cudaEvent_t e0,e1;CUDA_CHECK(cudaEventCreate(&e0));CUDA_CHECK(cudaEventCreate(&e1));
  for(int w=0;w<2;w++){
    CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,P,P,M,&a,dQ,CUDA_R_64F,P,dQ,CUDA_R_64F,P,&b,dG,CUDA_R_64F,P,ct,CUBLAS_GEMM_DEFAULT));
    CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,P,P,M,&a,dV,CUDA_R_64F,P,dV,CUDA_R_64F,P,&b,dVp,CUDA_R_64F,P,ct,CUBLAS_GEMM_DEFAULT));
    CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,2*P,P,M,&a,dS,CUDA_R_64F,2*P,dV,CUDA_R_64F,P,&b,dR,CUDA_R_64F,2*P,ct,CUBLAS_GEMM_DEFAULT));
  }
  CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaEventRecord(e0));
  CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,P,P,M,&a,dQ,CUDA_R_64F,P,dQ,CUDA_R_64F,P,&b,dG,CUDA_R_64F,P,ct,CUBLAS_GEMM_DEFAULT));
  CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,P,P,M,&a,dV,CUDA_R_64F,P,dV,CUDA_R_64F,P,&b,dVp,CUDA_R_64F,P,ct,CUBLAS_GEMM_DEFAULT));
  CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,2*P,P,M,&a,dS,CUDA_R_64F,2*P,dV,CUDA_R_64F,P,&b,dR,CUDA_R_64F,2*P,ct,CUBLAS_GEMM_DEFAULT));
  CUDA_CHECK(cudaEventRecord(e1));CUDA_CHECK(cudaEventSynchronize(e1)); float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));
  G.resize((size_t)P*P);Vp.resize((size_t)P*P);R.resize((size_t)2*P*P);
  CUDA_CHECK(cudaMemcpy(G.data(),dG,(size_t)P*P*8,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(Vp.data(),dVp,(size_t)P*P*8,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(R.data(),dR,(size_t)2*P*P*8,cudaMemcpyDeviceToHost));
  cudaFree(dG);cudaFree(dVp);cudaFree(dR);cudaEventDestroy(e0);cudaEventDestroy(e1); return ms;
}
// float path; outputs cast to double host arrays
static float gemms32(cublasHandle_t h,cublasComputeType_t ct,int P,int M,const float*dQ,const float*dV,const float*dS,
                     std::vector<double>&G,std::vector<double>&Vp,std::vector<double>&R){
  float*dG,*dVp,*dR; CUDA_CHECK(cudaMalloc(&dG,(size_t)P*P*4));CUDA_CHECK(cudaMalloc(&dVp,(size_t)P*P*4));CUDA_CHECK(cudaMalloc(&dR,(size_t)2*P*P*4));
  const float a=1,b=0; cudaEvent_t e0,e1;CUDA_CHECK(cudaEventCreate(&e0));CUDA_CHECK(cudaEventCreate(&e1));
  for(int w=0;w<2;w++){
    CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,P,P,M,&a,dQ,CUDA_R_32F,P,dQ,CUDA_R_32F,P,&b,dG,CUDA_R_32F,P,ct,CUBLAS_GEMM_DEFAULT));
    CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,P,P,M,&a,dV,CUDA_R_32F,P,dV,CUDA_R_32F,P,&b,dVp,CUDA_R_32F,P,ct,CUBLAS_GEMM_DEFAULT));
    CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,2*P,P,M,&a,dS,CUDA_R_32F,2*P,dV,CUDA_R_32F,P,&b,dR,CUDA_R_32F,2*P,ct,CUBLAS_GEMM_DEFAULT));
  }
  CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaEventRecord(e0));
  CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,P,P,M,&a,dQ,CUDA_R_32F,P,dQ,CUDA_R_32F,P,&b,dG,CUDA_R_32F,P,ct,CUBLAS_GEMM_DEFAULT));
  CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,P,P,M,&a,dV,CUDA_R_32F,P,dV,CUDA_R_32F,P,&b,dVp,CUDA_R_32F,P,ct,CUBLAS_GEMM_DEFAULT));
  CUBLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,2*P,P,M,&a,dS,CUDA_R_32F,2*P,dV,CUDA_R_32F,P,&b,dR,CUDA_R_32F,2*P,ct,CUBLAS_GEMM_DEFAULT));
  CUDA_CHECK(cudaEventRecord(e1));CUDA_CHECK(cudaEventSynchronize(e1)); float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));
  std::vector<float> g((size_t)P*P),v((size_t)P*P),r((size_t)2*P*P);
  CUDA_CHECK(cudaMemcpy(g.data(),dG,(size_t)P*P*4,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(v.data(),dVp,(size_t)P*P*4,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(r.data(),dR,(size_t)2*P*P*4,cudaMemcpyDeviceToHost));
  G.assign(g.begin(),g.end());Vp.assign(v.begin(),v.end());R.assign(r.begin(),r.end());
  cudaFree(dG);cudaFree(dVp);cudaFree(dR);cudaEventDestroy(e0);cudaEventDestroy(e1); return ms;
}

static void relerr(int P,const std::vector<double>&f2,const std::vector<double>&ref,double&mx,double&med){
  std::vector<double> rs; mx=0;
  for(int j=0;j<P;j++)for(int i=0;i<j;i++){ double r=ref[(size_t)i+(size_t)P*j]; if(fabs(r)<1e-12)continue;
    double e=fabs(f2[(size_t)i+(size_t)P*j]-r)/fabs(r); rs.push_back(e); if(e>mx)mx=e; }
  std::sort(rs.begin(),rs.end()); med= rs.empty()?0:rs[rs.size()/2];
}

int main(int argc,char**argv){
  if(argc<3||strcmp(argv[1],"--load")){fprintf(stderr,"usage: %s --load <dir>\n",argv[0]);return 2;}
  const char*dir=argv[2]; char p[1024]; int P;long long M;
  snprintf(p,sizeof p,"%s/shape.txt",dir);{FILE*f=fopen(p,"r");if(!f||fscanf(f,"%d %lld",&P,&M)!=2){fprintf(stderr,"bad shape\n");return 1;}fclose(f);}
  fprintf(stderr,"[load] %s  P=%d M=%lld\n",dir,P,(long long)M);
  size_t qN=(size_t)P*M; std::vector<double> Qr(qN),V(qN),N(qN);
  snprintf(p,sizeof p,"%s/Q.f64",dir);read_f64(p,Qr.data(),qN);
  snprintf(p,sizeof p,"%s/V.f64",dir);read_f64(p,V.data(),qN);
  snprintf(p,sizeof p,"%s/N.f64",dir);read_f64(p,N.data(),qN);
  // build masked Q, S=[Qsq;Hc]
  std::vector<double> Qm(qN),S((size_t)2*P*M);
  for(long long s=0;s<M;s++)for(int i=0;i<P;i++){size_t k=(size_t)i+(size_t)P*s;double v=V[k],q=Qr[k]*v,n=N[k];Qm[k]=q;
    S[(size_t)i+(size_t)2*P*s]=q*q; S[(size_t)(P+i)+(size_t)2*P*s]=q*(1.0-q)/((n>1.0)?(n-1.0):1.0)*v;}
  // long-double cancellation-free reference
  std::vector<double> ref((size_t)P*P,0.0);
  for(int i=0;i<P;i++)for(int j=i+1;j<P;j++){ long double sum=0; long long cnt=0;
    for(long long s=0;s<M;s++){size_t ki=(size_t)i+(size_t)P*s,kj=(size_t)j+(size_t)P*s; if(V[ki]>0.5&&V[kj]>0.5){
      long double pi=Qm[ki],pj=Qm[kj],ni=N[ki],nj=N[kj]; long double d=pi-pj;
      long double hci=pi*(1.0L-pi)/((ni>1.0L)?(ni-1.0L):1.0L), hcj=pj*(1.0L-pj)/((nj>1.0L)?(nj-1.0L):1.0L);
      sum+= d*d-hci-hcj; cnt++; }}
    double f=cnt? (double)(sum/(long double)cnt):0.0; ref[(size_t)i+(size_t)P*j]=f; ref[(size_t)j+(size_t)P*i]=f; }

  // upload double + float inputs
  double*dQ,*dV,*dS; CUDA_CHECK(cudaMalloc(&dQ,qN*8));CUDA_CHECK(cudaMalloc(&dV,qN*8));CUDA_CHECK(cudaMalloc(&dS,(size_t)2*P*M*8));
  CUDA_CHECK(cudaMemcpy(dQ,Qm.data(),qN*8,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dV,V.data(),qN*8,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dS,S.data(),(size_t)2*P*M*8,cudaMemcpyHostToDevice));
  std::vector<float> fQm(Qm.begin(),Qm.end()),fV(V.begin(),V.end()),fS(S.begin(),S.end());
  float*fdQ,*fdV,*fdS; CUDA_CHECK(cudaMalloc(&fdQ,qN*4));CUDA_CHECK(cudaMalloc(&fdV,qN*4));CUDA_CHECK(cudaMalloc(&fdS,(size_t)2*P*M*4));
  CUDA_CHECK(cudaMemcpy(fdQ,fQm.data(),qN*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(fdV,fV.data(),qN*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(fdS,fS.data(),(size_t)2*P*M*4,cudaMemcpyHostToDevice));

  cublasHandle_t h64,hf32,htf32,hbf9; CUBLAS_CHECK(cublasCreate(&h64));CUBLAS_CHECK(cublasCreate(&hf32));CUBLAS_CHECK(cublasCreate(&htf32));CUBLAS_CHECK(cublasCreate(&hbf9));
  CUBLAS_CHECK(cublasSetMathMode(h64,CUBLAS_PEDANTIC_MATH));
  CUBLAS_CHECK(cublasSetMathMode(htf32,CUBLAS_TF32_TENSOR_OP_MATH));
#if STEPPE_HAVE_EMU_TUNING
  CUBLAS_CHECK(cublasSetMathMode(hbf9,CUBLAS_FP32_EMULATED_BF16X9_MATH));
  CUBLAS_CHECK(cublasSetEmulationStrategy(hbf9,CUBLAS_EMULATION_STRATEGY_EAGER));
#endif

  printf("%8s %10s %9s %12s %12s\n","arm","gemm_ms","vs_nat","maxRel_f2","medRel_f2");
  std::vector<double> G,Vp,R,f2; double mx,md; float ms,ms0=1;
  ms=gemms64(h64,CUBLAS_COMPUTE_64F,P,M,dQ,dV,dS,G,Vp,R); assemble(P,G,Vp,R,f2); relerr(P,f2,ref,mx,md); ms0=ms;
  printf("%8s %10.3f %9.2f %12.3e %12.3e\n","nat64",ms,ms/ms0,mx,md);
  ms=gemms32(hf32,CUBLAS_COMPUTE_32F,P,M,fdQ,fdV,fdS,G,Vp,R); assemble(P,G,Vp,R,f2); relerr(P,f2,ref,mx,md);
  printf("%8s %10.3f %9.2f %12.3e %12.3e\n","fp32",ms,ms0/ms,mx,md);
  ms=gemms32(htf32,CUBLAS_COMPUTE_32F_FAST_TF32,P,M,fdQ,fdV,fdS,G,Vp,R); assemble(P,G,Vp,R,f2); relerr(P,f2,ref,mx,md);
  printf("%8s %10.3f %9.2f %12.3e %12.3e\n","tf32",ms,ms0/ms,mx,md);
  ms=gemms32(hbf9,CUBLAS_COMPUTE_32F_EMULATED_16BFX9,P,M,fdQ,fdV,fdS,G,Vp,R); assemble(P,G,Vp,R,f2); relerr(P,f2,ref,mx,md);
  printf("%8s %10.3f %9.2f %12.3e %12.3e\n","bf16x9",ms,ms0/ms,mx,md);
#if STEPPE_HAVE_EMU_TUNING
  // ---- FIXED-slice FP64 Ozaki: cap mantissa bits (data-independent cost, tunable accuracy) ----
  cublasHandle_t hoz; CUBLAS_CHECK(cublasCreate(&hoz));
  CUBLAS_CHECK(cublasSetMathMode(hoz,CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
  CUBLAS_CHECK(cublasSetEmulationStrategy(hoz,CUBLAS_EMULATION_STRATEGY_EAGER));
  CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(hoz,CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
  int bits[]={24,32,40,48,53};
  for(int bi=0;bi<5;bi++){
    CUBLAS_CHECK(cublasSetFixedPointEmulationMaxMantissaBitCount(hoz,bits[bi]));
    ms=gemms64(hoz,CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT,P,M,dQ,dV,dS,G,Vp,R); assemble(P,G,Vp,R,f2); relerr(P,f2,ref,mx,md);
    char nm[16]; snprintf(nm,sizeof nm,"ozk%db",bits[bi]);
    printf("%8s %10.3f %9.2f %12.3e %12.3e\n",nm,ms,ms0/ms,mx,md);
  }
  // dynamic = data-dependent slice count. This arm DEMONSTRATES the rejected TRAP:
  // on real AADR's wide dynamic range it overshoots to ~60 bits -> collapses to
  // parity with native (no speed win). Production uses FIXED control only (the
  // ozk32/40/48 arms above); dynamic is never selected (architecture.md 12, ROADMAP 0).
  CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(hoz,CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC));
  ms=gemms64(hoz,CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT,P,M,dQ,dV,dS,G,Vp,R); assemble(P,G,Vp,R,f2); relerr(P,f2,ref,mx,md);
  printf("%8s %10.3f %9.2f %12.3e %12.3e\n","ozk-dyn",ms,ms0/ms,mx,md);
#endif
  printf("# vs_nat>1 => faster than native FP64. maxRel_f2 = worst relative error vs long-double reference (tight tier ~1e-6).\n");
  return 0;
}
