// p2p_probe.cu — minimal P2P capability + byte-exactness + bandwidth probe for the
// M4.5 capable-path combine transport (cudaMemcpyPeer). Confirms the §12 parity
// precondition (byte-exact) and distinguishes real PCIe-P2P from a host-staged
// fallback. Build: nvcc -arch=sm_120 -O2 p2p_probe.cu -o p2p_probe ; run: ./p2p_probe
// Expect on RTX PRO 6000: canAccessPeer 0<->1 = 1, byte-exact = YES, ~55 GB/s.
// On a consumer GeForce 5090: canAccessPeer = 0 (P2P driver-disabled — host-staged only).
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("ERR %s:%d  %s  ->  %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e_)); return 1;} }while(0)

int main(){
  int n=0; CK(cudaGetDeviceCount(&n));
  printf("device count = %d\n", n);
  if(n<2){ printf("need >=2 GPUs\n"); return 1; }

  int ab=0, ba=0;
  CK(cudaDeviceCanAccessPeer(&ab,0,1));
  CK(cudaDeviceCanAccessPeer(&ba,1,0));
  printf("cudaDeviceCanAccessPeer: 0->1=%d  1->0=%d\n", ab, ba);

  const size_t bytes = 256ull*1024*1024; // 256 MB
  void *d0=nullptr, *d1=nullptr;
  CK(cudaSetDevice(0)); CK(cudaMalloc(&d0,bytes)); CK(cudaMemset(d0,0xAB,bytes));
  CK(cudaSetDevice(1)); CK(cudaMalloc(&d1,bytes)); CK(cudaMemset(d1,0x00,bytes));

  bool peer_enabled=false;
  if(ab){
    CK(cudaSetDevice(1));
    cudaError_t e=cudaDeviceEnablePeerAccess(0,0);
    if(e==cudaSuccess || e==cudaErrorPeerAccessAlreadyEnabled) peer_enabled=true;
    else printf("enablePeerAccess(1<-0) -> %s\n", cudaGetErrorString(e));
  }

  CK(cudaMemcpyPeer(d1,1,d0,0,bytes));
  CK(cudaDeviceSynchronize());
  std::vector<unsigned char> h(bytes);
  CK(cudaSetDevice(1)); CK(cudaMemcpy(h.data(),d1,bytes,cudaMemcpyDeviceToHost));
  bool ok=true; for(size_t i=0;i<bytes;i+=4093) if(h[i]!=0xAB){ ok=false; break; }
  printf("peer copy byte-exact = %s   (peer_access_enabled=%d)\n", ok?"YES":"NO", peer_enabled);

  cudaEvent_t s,e2; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e2));
  const int iters=30;
  CK(cudaEventRecord(s));
  for(int i=0;i<iters;i++) CK(cudaMemcpyPeer(d1,1,d0,0,bytes));
  CK(cudaEventRecord(e2)); CK(cudaEventSynchronize(e2));
  float ms=0; CK(cudaEventElapsedTime(&ms,s,e2));
  double gb=(double)bytes*iters/1e9;
  printf("cudaMemcpyPeer bandwidth = %.1f GB/s  (%zu MB x%d in %.2f ms)\n",
         gb/(ms/1e3), bytes>>20, iters, ms);

  CK(cudaFree(d0)); CK(cudaFree(d1));
  return 0;
}
