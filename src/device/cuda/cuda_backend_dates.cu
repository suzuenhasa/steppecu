// src/device/cuda/cuda_backend_dates.cu
//
// CudaBackend — DATES cuFFT autocorrelation LD engine subsystem TU (cuda_backend.cu
// split T2; docs/kimiactions/05-cuda-backend-split.md §2.3 TU-E). Out-of-line homes
// of `CudaBackend::dates_curve` + `CudaBackend::dates_repack` + `CudaBackend::dates_fit`
// — the DATES decay-curve pipeline (per-admixed-sample weight/residual grid → batched
// cufftExecD2Z over all chroms → |F|²/cross-power → batched cufftExecZ2D inverse →
// extract+accumulate lags → re-bin into the corr sufficient stats), the host bit-shuffle
// repack, and the per-curve exponential-decay fit. Bodies MOVED VERBATIM from
// cuda_backend.cu; nothing about codegen / math / precision changed by the split.
//
// PRECISION (§2.3 TU-E): native-FP64 FFT + native weight / normal-eq carve-out (§12);
// no BLAS / cuSOLVER on this path. The two cuFFT plans (CufftPlan, handles.hpp) are
// RAII-owned — they tear down on EVERY exit path (normal return AND any throw between
// create and tail), so the RAII intact is load-bearing for the no-leak guarantee
// ([16.1]). This TU joins the SAME steppe_device target.
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4).
#include <cufft.h>  // cufftExecD2Z / cufftExecZ2D, cufftDoubleComplex, CUFFT_D2Z / CUFFT_Z2D

#include <cstdint>  // std::uint8_t — the packed-genotype record byte type
#include <vector>   // std::vector — the per-curve fit outputs + host staging

#include "core/internal/nvtx.hpp"            // STEPPE_NVTX_RANGE (coarse phase-boundary marker)
#include "device/cuda/cuda_backend.cuh"      // the CudaBackend class declaration (split T0)
#include "device/cuda/check.cuh"             // STEPPE_CUDA_CHECK
#include "device/cuda/dates_kernel.cuh"      // DATES cuFFT autocorrelation LD engine kernels
#include "device/cuda/handles.hpp"           // CufftPlan (RAII) + CUFFT_CHECK

namespace steppe::device {

DatesMoments CudaBackend::dates_curve(
    const double* src1_freq, const double* src2_freq, const double* src_valid,
    const std::uint8_t* packed, std::size_t bytes_per_record, int n_target,
    const int* target_ploidy, const int* grid_cell, long M,
    const int* chrom_first, const int* chrom_last, int n_chrom,
    int numqbins, int n_bin, int diffmax, double binsize, int qbin,
    const Precision& precision) {
    (void)target_ploidy;  // DATES dosage = code/2 (ploidy-independent); see dates.c getgtypes.
    (void)binsize;        // the output-bin width is folded into the lag->bin re-bin (floor(d/qbin)).
    (void)precision;      // native double FFT + native weight carve-out (§12).
    guard_device();
    STEPPE_NVTX_RANGE("dates_curve");  // coarse phase boundary: the DATES FFT decay-curve pipeline
    DatesMoments out;
    out.n_chrom = n_chrom;
    out.n_bin = n_bin;
    if (n_chrom <= 0 || n_bin <= 0 || M <= 0 || n_target <= 0 || numqbins <= 0 ||
        diffmax <= 0) {
        out.status = Status::Ok;
        return out;
    }

    // ---- FFT length: pad ALL chroms to a COMMON power-of-2 n_fft >= 2·max_len so ONE
    // cufftPlanMany batches every chrom. The linear autocorrelation Σ_g W[g]·W[g+lag]
    // (lag<len) is INDEPENDENT of n_fft once n_fft >= 2·len (zero-pad avoids circular
    // wrap), and the /n_fft scale matches the FFTW reference — DATES uses a per-chrom
    // 2^(ceil(log2(len))+1); a common larger n is bit-equivalent for the autocorr values.
    int max_len = 1;
    for (int kc = 0; kc < n_chrom; ++kc) {
        const int slo = chrom_first[kc], shi = chrom_last[kc];
        if (slo >= 0 && shi >= slo) max_len = std::max(max_len, shi - slo + 1);
    }
    // ensure n_fft covers the linear autocorr support AND the requested lags.
    long need = std::max<long>(2L * max_len, static_cast<long>(diffmax) + 1L);
    int n_fft = 1;
    while (static_cast<long>(n_fft) < need) n_fft <<= 1;
    const int n_cplx = n_fft / 2 + 1;

    // ---- cuFFT plans (created ONCE, reused across samples): batched D2Z + Z2D over n_chrom.
    // RAII-owned by CufftPlan (handles.hpp): the plans back cuFFT device WORKSPACES, and the
    // function THROWS between create and the function tail (cufftSetStream below, the
    // cufftExec* in the per-sample loop, and every STEPPE_CUDA_CHECK in between), so a bare
    // cufftHandle + tail cufftDestroy would LEAK both plans (VRAM, scaling with retries) on any
    // unwind ([16.1]). As scope members they tear down on EVERY exit path; the previously
    // unchecked tail cufftDestroy is now routed through CufftError ([13.3]).
    CufftPlan plan_fwd, plan_inv;
    int n_dim = n_fft;
    plan_fwd.make(1, &n_dim, nullptr, 1, n_fft, nullptr, 1, n_cplx, CUFFT_D2Z, n_chrom);
    plan_inv.make(1, &n_dim, nullptr, 1, n_cplx, nullptr, 1, n_fft, CUFFT_Z2D, n_chrom);
    plan_fwd.set_stream(stream_.get());
    plan_inv.set_stream(stream_.get());

    // ---- device-resident inputs + scratch ----
    const std::size_t Mu = static_cast<std::size_t>(M);
    const std::size_t nq = static_cast<std::size_t>(numqbins);
    const std::size_t pad = static_cast<std::size_t>(n_chrom) * static_cast<std::size_t>(n_fft);
    const std::size_t cpx = static_cast<std::size_t>(n_chrom) * static_cast<std::size_t>(n_cplx);
    const std::size_t dm = static_cast<std::size_t>(n_chrom) *
                           (static_cast<std::size_t>(diffmax) + 1);
    const std::size_t cb = static_cast<std::size_t>(n_chrom) * static_cast<std::size_t>(n_bin);
    const std::size_t pk = static_cast<std::size_t>(n_target) * bytes_per_record;

    DeviceBuffer<double> dS1(Mu), dS2(Mu), dValid(Mu);
    DeviceBuffer<std::uint8_t> dPacked(pk);
    DeviceBuffer<int> dCell(Mu), dCfirst(static_cast<std::size_t>(n_chrom)),
        dClast(static_cast<std::size_t>(n_chrom));
    DeviceBuffer<double> dZ0(nq), dZ1(nq), dZ2(nq);
    DeviceBuffer<double> dPad0(pad), dPad1(pad), dPad2(pad);
    DeviceBuffer<double> dInv(pad);
    // complex scratch (cufftDoubleComplex == double2): forward outs for z0,z1,z2 + power.
    DeviceBuffer<double2> dF0(cpx), dF1(cpx), dF2(cpx), dPow(cpx);
    // per-chrom dd lag moments (summed across samples): dd00,dd11,dd02,dd20.
    DeviceBuffer<double> dDd00(dm), dDd11(dm), dDd02(dm), dDd20(dm);
    // output corr sufficient stats.
    DeviceBuffer<double> dS0(cb), dS11(cb), dS12(cb), dS22(cb);
    DeviceBuffer<double> dDot12(1), dDot22(1);

    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dS1.data(), src1_freq, Mu * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dS2.data(), src2_freq, Mu * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dValid.data(), src_valid, Mu * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dPacked.data(), packed, pk,
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dCell.data(), grid_cell, Mu * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dCfirst.data(), chrom_first,
                                      static_cast<std::size_t>(n_chrom) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dClast.data(), chrom_last,
                                      static_cast<std::size_t>(n_chrom) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dDd00.data(), 0, dm * sizeof(double), stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dDd11.data(), 0, dm * sizeof(double), stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dDd02.data(), 0, dm * sizeof(double), stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dDd20.data(), 0, dm * sizeof(double), stream_.get()));

    auto fft_fwd = [&](double* in, double2* fr) {
        CUFFT_CHECK(cufftExecD2Z(plan_fwd.get(), in,
                                 reinterpret_cast<cufftDoubleComplex*>(fr)));
    };
    auto fft_inv = [&](double2* fr, double* outr) {
        CUFFT_CHECK(cufftExecZ2D(plan_inv.get(),
                                 reinterpret_cast<cufftDoubleComplex*>(fr), outr));
    };

    // ---- per-sample loop (host-driven; the FFT/scatter are GPU-bound) -----------------
    for (int i = 0; i < n_target; ++i) {
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dDot12.data(), 0, sizeof(double), stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dDot22.data(), 0, sizeof(double), stream_.get()));
        launch_dates_regress_dots(dS1.data(), dS2.data(), dValid.data(), dPacked.data(),
                                  bytes_per_record, i, M, dDot12.data(), dDot22.data(),
                                  stream_.get());
        double h_dot12 = 0.0, h_dot22 = 0.0;
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(&h_dot12, dDot12.data(), sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(&h_dot22, dDot22.data(), sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        const double yreg = (h_dot22 != 0.0) ? (h_dot12 / h_dot22) : 0.0;

        STEPPE_CUDA_CHECK(cudaMemsetAsync(dZ0.data(), 0, nq * sizeof(double), stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dZ1.data(), 0, nq * sizeof(double), stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dZ2.data(), 0, nq * sizeof(double), stream_.get()));
        launch_dates_scatter(dS1.data(), dS2.data(), dValid.data(), dPacked.data(),
                             bytes_per_record, i, M, dCell.data(), yreg, dZ0.data(),
                             dZ1.data(), dZ2.data(), stream_.get());

        // pack chrom segments -> FFT rows (z0,z1,z2).
        launch_dates_pack_segments(dZ0.data(), dCfirst.data(), dClast.data(), n_chrom,
                                   numqbins, n_fft, dPad0.data(), stream_.get());
        launch_dates_pack_segments(dZ1.data(), dCfirst.data(), dClast.data(), n_chrom,
                                   numqbins, n_fft, dPad1.data(), stream_.get());
        launch_dates_pack_segments(dZ2.data(), dCfirst.data(), dClast.data(), n_chrom,
                                   numqbins, n_fft, dPad2.data(), stream_.get());

        // forward transforms.
        fft_fwd(dPad0.data(), dF0.data());
        fft_fwd(dPad1.data(), dF1.data());
        fft_fwd(dPad2.data(), dF2.data());

        // dd00 = autocorr(z0) = IFFT(|F0|²)/n.
        launch_dates_power_spectrum(dF0.data(), n_cplx, n_chrom, dPow.data(), stream_.get());
        fft_inv(dPow.data(), dInv.data());
        launch_dates_extract_lags(dInv.data(), n_fft, n_chrom, diffmax, dDd00.data(),
                                  stream_.get());
        // dd11 = autocorr(z1).
        launch_dates_power_spectrum(dF1.data(), n_cplx, n_chrom, dPow.data(), stream_.get());
        fft_inv(dPow.data(), dInv.data());
        launch_dates_extract_lags(dInv.data(), n_fft, n_chrom, diffmax, dDd11.data(),
                                  stream_.get());
        // dd02 = crosscorr(z0,z2) = IFFT(conj(F0)·F2)/n.
        launch_dates_cross_power(dF0.data(), dF2.data(), n_cplx, n_chrom, dPow.data(),
                                 stream_.get());
        fft_inv(dPow.data(), dInv.data());
        launch_dates_extract_lags(dInv.data(), n_fft, n_chrom, diffmax, dDd02.data(),
                                  stream_.get());
        // dd20 = crosscorr(z2,z0) = IFFT(conj(F2)·F0)/n.
        launch_dates_cross_power(dF2.data(), dF0.data(), n_cplx, n_chrom, dPow.data(),
                                 stream_.get());
        fft_inv(dPow.data(), dInv.data());
        launch_dates_extract_lags(dInv.data(), n_fft, n_chrom, diffmax, dDd20.data(),
                                  stream_.get());
    }

    // ---- re-bin lag -> output bin into the corr sufficient stats ----------------------
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dS0.data(), 0, cb * sizeof(double), stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dS11.data(), 0, cb * sizeof(double), stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dS12.data(), 0, cb * sizeof(double), stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dS22.data(), 0, cb * sizeof(double), stream_.get()));
    launch_dates_accumulate_bins(dDd00.data(), dDd11.data(), dDd02.data(), dDd20.data(),
                                 n_chrom, diffmax, n_bin, qbin, dS0.data(), dS11.data(),
                                 dS12.data(), dS22.data(), stream_.get());

    out.s0.assign(cb, 0.0);  out.s1.assign(cb, 0.0);  out.s2.assign(cb, 0.0);
    out.s11.assign(cb, 0.0); out.s12.assign(cb, 0.0); out.s22.assign(cb, 0.0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.s0.data(), dS0.data(), cb * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.s11.data(), dS11.data(), cb * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.s12.data(), dS12.data(), cb * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.s22.data(), dS22.data(), cb * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    // plan_fwd / plan_inv (CufftPlan) tear down via RAII on scope exit — on the
    // normal return here AND on any throw along the path above ([16.1]).
    out.status = Status::Ok;
    return out;
}

void CudaBackend::dates_repack(const std::uint8_t* src, std::size_t src_bpr, const long* kept_src,
                  long M_kept, int n_target, std::size_t dst_bpr,
                  std::uint8_t* dst) {
    guard_device();
    if (n_target <= 0 || M_kept <= 0 || dst_bpr == 0) return;
    const std::size_t src_bytes = static_cast<std::size_t>(n_target) * src_bpr;
    const std::size_t dst_bytes = static_cast<std::size_t>(n_target) * dst_bpr;
    DeviceBuffer<std::uint8_t> dSrc(src_bytes);
    DeviceBuffer<long> dKept(static_cast<std::size_t>(M_kept));
    DeviceBuffer<std::uint8_t> dDst(dst_bytes);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSrc.data(), src, src_bytes,
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dKept.data(), kept_src,
                                      static_cast<std::size_t>(M_kept) * sizeof(long),
                                      cudaMemcpyHostToDevice, stream_.get()));
    launch_dates_repack_target(dSrc.data(), src_bpr, dKept.data(), M_kept, n_target,
                               dst_bpr, dDst.data(), stream_.get());
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dst, dDst.data(), dst_bytes,
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
}

std::vector<DatesExpFit> CudaBackend::dates_fit(const double* curves, int win_len,
                                                 int n_curves, double step,
                                                 bool affine) {
    guard_device();
    std::vector<DatesExpFit> out(static_cast<std::size_t>(n_curves > 0 ? n_curves : 0));
    if (n_curves <= 0 || win_len <= 0 || !(step > 0.0)) return out;
    const std::size_t nc = static_cast<std::size_t>(n_curves);
    const std::size_t total = nc * static_cast<std::size_t>(win_len);
    DeviceBuffer<double> dCurves(total);
    DeviceBuffer<double> dDate(nc), dSd(nc);
    DeviceBuffer<int> dOk(nc);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dCurves.data(), curves, total * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    launch_dates_fit_curves(dCurves.data(), win_len, n_curves, step, affine,
                            dDate.data(), dSd.data(), dOk.data(), stream_.get());
    std::vector<double> h_date(nc, 0.0), h_sd(nc, 0.0);
    std::vector<int> h_ok(nc, 0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_date.data(), dDate.data(), nc * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_sd.data(), dSd.data(), nc * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_ok.data(), dOk.data(), nc * sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    for (std::size_t c = 0; c < nc; ++c) {
        out[c].date_gen = h_date[c];
        out[c].error_sd = h_sd[c];
        out[c].ok = h_ok[c];
    }
    return out;
}

}  // namespace steppe::device
