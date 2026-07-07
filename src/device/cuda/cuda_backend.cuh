#pragma once
// src/device/cuda/cuda_backend.cuh
//
// Private-to-steppe_device declaration of CudaBackend, the GPU ComputeBackend.
// One backend instance is bound to one physical GPU; the method bodies live
// out-of-line in the per-subsystem .cu TUs that compile into steppe_device.
//
// Reference: docs/reference/src_device_cuda_cuda_backend.cuh.md

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "device/backend.hpp"
#include "device/device_partial.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/device_decode_result.hpp"
#include "device/stream_f2_blocks.hpp"
#include "device/f2_blocks_out.hpp"
#include "device/vram_budget.hpp"
#include "steppe/config.hpp"
#include "steppe/fstats.hpp"
#include "device/cuda/handles.hpp"
#include "device/cuda/device_buffer.cuh"
#include "device/cuda/pinned_buffer.cuh"
#include "device/cuda/stream.hpp"
#include "device/cuda/check.cuh"

namespace steppe::device {

class BlockSink;

// CudaBackend: the GPU ComputeBackend — reference §1
class CudaBackend final : public ComputeBackend {
public:
    // Bind to one GPU — reference §2
    explicit CudaBackend(int device_id = 0);

    // ResidentBlocks (shared f2 GEMM output) — reference §3
    struct ResidentBlocks {
        DeviceBuffer<double> f2;
        DeviceBuffer<double> vpair;
        std::vector<int> block_sizes;
        int P = 0;
        int n_block = 0;
    };

    // f2 block computation family — reference §4
    [[nodiscard]] F2Result compute_f2(const core::MatView& Q,
                                      const core::MatView& V,
                                      const core::MatView& N,
                                      const Precision& precision) override;

    [[nodiscard]] F2BlockTensor compute_f2_blocks(const core::MatView& Q,
                                                  const core::MatView& V,
                                                  const core::MatView& N,
                                                  const int* block_id,
                                                  int n_block,
                                                  const Precision& precision) override;

    [[nodiscard]] DeviceF2Blocks compute_f2_blocks_device(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, const Precision& precision) override;

    [[nodiscard]] DevicePartial compute_f2_blocks_resident(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, int b0, const Precision& precision) override;

    void compute_f2_blocks_into(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, int b0,
        double* dst_f2, double* dst_vpair, int* block_sizes_dst,
        const Precision& precision) override;

    void compute_f2_blocks_streamed(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, const Precision& precision,
        StreamTarget& target, const RedecodeSource* redecode = nullptr) override;

    [[nodiscard]] ResidentBlocks run_f2_blocks_resident(const core::MatView& Q,
                                                        const core::MatView& V,
                                                        const core::MatView& N,
                                                        const int* block_id,
                                                        int n_block,
                                                        const Precision& precision);

    void stream_f2_blocks_impl(const core::MatView& Q, const core::MatView& V,
                               const core::MatView& N, const int* block_id, int n_block,
                               const Precision& precision, BlockSink& sink,
                               const RedecodeSource* redecode = nullptr);

    // Genotype decode and the format-reader front-end — reference §5
    void decode_af_resident(const DecodeTileView& tile, int P, long M,
                            DeviceBuffer<double>& dQ, DeviceBuffer<double>& dV,
                            DeviceBuffer<double>& dN, long s_lo = 0);

    [[nodiscard]] DecodeResult decode_af(const DecodeTileView& tile) override;

    [[nodiscard]] std::vector<int> detect_sample_ploidy_device(
        const DecodeTileView& tile) override;

    [[nodiscard]] CanonicalTile transpose_to_canonical(
        const SnpMajorTileView& view) override;

    [[nodiscard]] steppe::device::DeviceDecodeResult decode_af_compact_autosome(
        const DecodeTileView& tile, std::span<const int> chrom,
        std::span<const double> genpos, std::span<const double> physpos,
        int chrom_min, int chrom_max) override;

    [[nodiscard]] steppe::device::DeviceDecodeResult decode_af_compact_filter(
        const DecodeTileView& tile, std::span<const char> ref, std::span<const char> alt,
        std::span<const int> chrom, std::span<const double> genpos,
        std::span<const double> physpos, const FilterConfig& cfg,
        std::span<const std::size_t> pop_individuals, int ploidy, double maxmiss,
        long s_lo) override;

    // Per-site Weir & Cockerham 1984 FST over a population pair (`steppe fst`) — uploads
    // the packed tile, launches the per-site WC kernel, and runs the device-resident
    // (native-FP64) masked Σnum/Σden/n_valid reduction. New standalone stat seam.
    [[nodiscard]] FstPerSite fst_wc_per_site(const DecodeTileView& tile, int popA, int popB,
                                             std::span<const std::uint8_t> summary_include) override;

    // 2D joint site-frequency spectrum over a population pair (`steppe sfs`) — uploads the
    // packed tile, launches the joint-histogram kernel over the device tile, and D2H's only
    // the finished integer grid + the complete-site counter. Integer-count standalone stat.
    [[nodiscard]] SfsJoint joint_sfs_2pop(const DecodeTileView& tile, int popA, int popB,
                                          bool folded) override;

    // D-statistic block reduction — reference §6
    void dstat_block_reduce_device(const double* dQ, const double* dV, int P, long M,
                                   const int* block_id, int n_block,
                                   std::span<const int> quadruples,
                                   double* numsum, double* densum, double* cnt);

    void dstat_block_reduce(const double* Q, const double* V, int P, long M,
                            const int* block_id, int n_block,
                            std::span<const int> quadruples,
                            double* numsum, double* densum, double* cnt) override;

    void dstat_block_reduce(const steppe::device::DeviceDecodeResult& dec,
                            const int* block_id, int n_block,
                            std::span<const int> quadruples,
                            double* numsum, double* densum, double* cnt) override;

    // DATES weighted-LD engine — reference §10
    [[nodiscard]] DatesMoments dates_curve(
        const double* src1_freq, const double* src2_freq, const double* src_valid,
        const std::uint8_t* packed, std::size_t bytes_per_record, int n_target,
        const int* target_ploidy, const int* grid_cell, long M,
        const int* chrom_first, const int* chrom_last, int n_chrom,
        int numqbins, int n_bin, int diffmax, double binsize, int qbin,
        const Precision& precision) override;

    void dates_repack(const std::uint8_t* src, std::size_t src_bpr, const long* kept_src,
                      long M_kept, int n_target, std::size_t dst_bpr,
                      std::uint8_t* dst) override;

    [[nodiscard]] std::vector<DatesExpFit> dates_fit(const double* curves, int win_len,
                                                     int n_curves, double step,
                                                     bool affine) override;

    // Li-Stephens copying forward-backward (the `steppe paint` FB core) — reference §10.
    // Batched, device-resident, native-FP64 per-column-rescaled, with an always-on
    // checkpoint/recompute alpha scheme. Returns the per-column copying posterior gamma.
    [[nodiscard]] LsPosterior ls_forward_backward(
        const std::uint8_t* recipient, const std::uint8_t* donors, const double* pi,
        const double* rho, const double* mu, int K, long M,
        const Precision& precision) override;

    // Li-Stephens ChromoPainter coancestry FACE (the `steppe paint` Phase-2 output).
    // Batched over N recipients (one block each), the paint sink folds gamma into the
    // per-donor chunkcounts/chunklengths on-device; only the N*K accumulators cross
    // PCIe (the K*M posterior is never materialized). Native FP64.
    [[nodiscard]] LsCoancestry ls_paint_coancestry(
        const std::uint8_t* recipients, const std::uint8_t* donors, const double* pi,
        const double* rho, const double* mu, const double* w, int K, long M, int N,
        const Precision& precision) override;

    // Li-Stephens LOCAL-ANCESTRY FACE (the `steppe paint --face localanc` Phase-3 output).
    // Batched over N recipients (one block each), the localanc sink folds gamma per SNP
    // into the M*P per-label posterior on-device; only the N*M*P posterior crosses PCIe
    // (the K*M gamma is never materialized). Native FP64.
    [[nodiscard]] LsLocalAncestry ls_localanc(
        const std::uint8_t* recipients, const std::uint8_t* donors, const double* pi,
        const double* rho, const double* mu, const int* donor_group, int K, long M, int N,
        int P, const Precision& precision) override;

    // qpfstats smoothing — reference §9
    [[nodiscard]] QpfstatsSmooth qpfstats_smooth(std::span<const double> x,
                                                 std::span<const double> ymat,
                                                 std::span<const double> y,
                                                 int npopcomb, int npairs,
                                                 int n_block, double ridge,
                                                 const Precision& precision) override;

    [[nodiscard]] QpfstatsSmooth qpfstats_blocks_smooth(
        const double* Q, const double* V, int P, long M, const int* block_id, int n_block,
        std::span<const int> quadruples, std::span<const double> x, int npopcomb, int npairs,
        std::span<const int> block_sizes, double ridge, const Precision& precision) override;

    [[nodiscard]] QpfstatsSmooth qpfstats_blocks_smooth(
        const steppe::device::DeviceDecodeResult& dec, const int* block_id, int n_block,
        std::span<const int> quadruples, std::span<const double> x, int npopcomb, int npairs,
        std::span<const int> block_sizes, double ridge, const Precision& precision) override;

    [[nodiscard]] QpfstatsSmooth qpfstats_blocks_smooth_device(
        const double* dQ, const double* dV, int P, long M, const int* block_id, int n_block,
        std::span<const int> quadruples, std::span<const double> x, int npopcomb, int npairs,
        std::span<const int> block_sizes, double ridge, const Precision& precision);

    // Capability probing — reference §15
    [[nodiscard]] BackendCapabilities capabilities() const override;

    // Solve-precision setter (precision axes) — reference §16
    void set_solve_precision(const Precision& precision) override;

    // Batched-dispatch instrumentation — reference §11
    [[nodiscard]] std::size_t batched_dispatch_count() const override;

    // f-statistic sweeps and survivor blocks — reference §8
    [[nodiscard]] std::vector<int> device_survivor_blocks(
        const steppe::device::DeviceF2Blocks& f2, int nb, int P);

    [[nodiscard]] SweepSurvivors run_fstat_sweep_device(
        const steppe::device::DeviceF2Blocks& f2, const SweepConfig& cfg,
        const Precision& precision, int k);

    // qpAdm fit: per-block f4 assemble — reference §11
    [[nodiscard]] F4Blocks assemble_f4(const steppe::device::DeviceF2Blocks& f2,
                                       std::span<const int> left_idx,
                                       std::span<const int> right_idx,
                                       const Precision& precision) override;

    [[nodiscard]] F4Blocks assemble_f4(const F2BlockTensor& f2,
                                       std::span<const int> left_idx,
                                       std::span<const int> right_idx,
                                       const Precision& precision) override;

    [[nodiscard]] F4Blocks assemble_f4_quartets(
        const steppe::device::DeviceF2Blocks& f2,
        std::span<const int> quartets,
        const Precision& precision) override;

    // Fused ratio-jackknife producers — reference §7
    [[nodiscard]] RatioBlockJackknife f4ratio_blocks_jackknife(
        const steppe::device::DeviceF2Blocks& f2, std::span<const int> flat, int N,
        double setmiss_thresh, const Precision& precision) override;

    [[nodiscard]] RatioBlockJackknife dstat_blocks_jackknife(
        const steppe::device::DeviceDecodeResult& dec, const int* block_id, int n_block,
        std::span<const int> quadruples) override;

    [[nodiscard]] RatioBlockJackknife f4ratio_blocks_jackknife(
        const F2BlockTensor& f2, std::span<const int> flat, int N, double setmiss_thresh,
        const Precision& precision) override;

    [[nodiscard]] RatioBlockJackknife dstat_blocks_jackknife(
        const double* Q, const double* V, int P, long M, const int* block_id, int n_block,
        std::span<const int> quadruples) override;

    // qpAdm fit: standalone f4/f3 assemble (+ host-oracle doors) — reference §11
    [[nodiscard]] F4Blocks assemble_f4_quartets(const F2BlockTensor& f2,
                                                std::span<const int> quartets,
                                                const Precision& precision) override;

    [[nodiscard]] F4Blocks assemble_f3_triples(
        const steppe::device::DeviceF2Blocks& f2,
        std::span<const int> triples,
        const Precision& precision) override;

    [[nodiscard]] F4Blocks assemble_f3_triples(const F2BlockTensor& f2,
                                               std::span<const int> triples,
                                               const Precision& precision) override;

    // qpGraph optimizer fleet — reference §14
    [[nodiscard]] QpGraphFleet qpgraph_fit_fleet(const QpGraphTopoArena& topo,
                                                 std::span<const double> f_obs,
                                                 std::span<const double> qinv,
                                                 int numstart, int maxit, double tol,
                                                 const Precision& precision) override;

    [[nodiscard]] QpGraphFleetBatch qpgraph_fit_fleet_batch(
        const std::vector<QpGraphTopoArena>& topos, std::span<const double> f_obs,
        std::span<const double> qinv, int numstart, int maxit, double tol,
        const Precision& precision) override;

    // f4 / f3 sweeps — reference §8
    [[nodiscard]] SweepSurvivors f4_sweep(const steppe::device::DeviceF2Blocks& f2,
                                          const SweepConfig& cfg,
                                          const Precision& precision) override;

    [[nodiscard]] SweepSurvivors f3_sweep(const steppe::device::DeviceF2Blocks& f2,
                                          const SweepConfig& cfg,
                                          const Precision& precision) override;

    // READv2 windowed-mismatch kinship — reference §8
    [[nodiscard]] Readv2Bitmatrix readv2_alloc_bitmatrix(int n_samples, int window_snps,
                                                         long m0) override;

    void readv2_pack_chunk(Readv2Bitmatrix& bits, const std::uint8_t* chunk_packed,
                           std::size_t chunk_bytes_per_record, int n_samples, long snp0,
                           long snp_count) override;

    [[nodiscard]] Readv2Pairs readv2_mismatch(const Readv2Bitmatrix& bits, long long n_pairs,
                                              bool tiled) override;

    // GL/PL/GP likelihood tensor upload + residency-proof reduction
    [[nodiscard]] LikelihoodTensor upload_likelihood_tensor(const double* host_l,
                                                            const std::uint8_t* host_present,
                                                            long n_site, int n_sample) override;

    [[nodiscard]] double likelihood_tensor_checksum(const LikelihoodTensor& t) override;

    // qpAdm fit: jackknife covariance — reference §11
    [[nodiscard]] JackknifeCov jackknife_cov(const F4Blocks& x,
                                             std::span<const int> block_sizes,
                                             double fudge,
                                             const Precision& precision) override;

    [[nodiscard]] JackknifeDiag jackknife_diag(const F4Blocks& x,
                                               std::span<const int> block_sizes,
                                               const Precision& precision) override;

    // Large-model path helpers (cuSOLVER SVD) — reference §12
    static bool model_fits_small_path(int nl, int nr, int r);

    static bool gesvdj_applicable(int nl, int nr);

    // SvdScratchSizes (large-SVD scratch) — reference §3
    struct SvdScratchSizes {
        std::size_t s = 0, u = 0, vt = 0, a2 = 0, info = 1;
        int lwork = 0;
    };

    // Large-SVD scratch sizing and solve — reference §12
    [[nodiscard]] SvdScratchSizes large_svd_scratch_sizes(int nl, int nr);

    void large_svd_V(const double* dXmat, int nl, int nr, int r,
                     double* dVout, double* dXt,
                     double* sS, double* sU, double* sVt, double* sA2,
                     int* sInfo, double* sWork, int lwork, cudaStream_t stream);

    void large_svd_V(const double* dXmat, int nl, int nr, int r,
                     double* dVout, double* dXt, cudaStream_t stream);

    static std::size_t large_dbl_scratch(int nl, int nr, int r);
    static std::size_t large_int_scratch(int nl, int nr, int r);

    static std::size_t large_loo_dbl_refit(int nl, int nr, int r);

    void large_fit_one(const double* dXmat, const double* dQinv, int nl, int nr, int r,
                       double fudge, int als_iters, double* dA, double* dB, double* dW,
                       double* dchisq, int* dStatus, double* dVout, double* dXt,
                       double* dScratch, int* dIntScratch, cudaStream_t stream);

    // qpAdm fit: rank sweep, weights, standard error — reference §11
    [[nodiscard]] bool provides_rank_sweep() const override;

    [[nodiscard]] RankSweep rank_sweep(const F4Blocks& x,
                                       const JackknifeCov& cov,
                                       double alpha,
                                       const QpAdmOptions& opts,
                                       const Precision& precision) override;

    [[nodiscard]] GlsWeights gls_weights(const F4Blocks& x,
                                         const JackknifeCov& cov,
                                         int r,
                                         const QpAdmOptions& opts,
                                         const Precision& precision) override;

    [[nodiscard]] std::vector<double> gls_weights_loo_batched(
        const F4Blocks& x, const JackknifeCov& cov, int r,
        const QpAdmOptions& opts, const Precision& precision) override;

    [[nodiscard]] std::vector<double> se_from_wmat(
        const F4Blocks& x, const JackknifeCov& cov, int r,
        const QpAdmOptions& opts, const Precision& precision) override;

private:
    // LOO wmat producer — reference §11
    void populate_loo_wmat_resident(const F4Blocks& x, const JackknifeCov& cov, int r,
                                    const QpAdmOptions& opts, const Precision& precision,
                                    DeviceBuffer<double>& dWmat);

public:
    // Batched-fit capability query — reference §11
    [[nodiscard]] bool provides_batched_fit() const override;

    // Batched model-space rotation — reference §13
    [[nodiscard]] std::vector<QpAdmResult> fit_models_batched(
        const steppe::device::DeviceF2Blocks& f2,
        std::span<const QpAdmModel> models,
        const QpAdmOptions& opts,
        const Precision& precision) override;

private:
    // Batched-rotation helpers — reference §13
    void fit_one_bucket(const steppe::device::DeviceF2Blocks& f2,
                        std::span<const QpAdmModel> models,
                        const std::vector<std::size_t>& mem, int nl, int nr, int r,
                        int nb, const std::vector<int>& survivor_block_sizes,
                        const int* d_surv, const QpAdmOptions& opts,
                        const Precision& precision,
                        Precision::Kind tag, std::vector<QpAdmResult>& results);

    void fit_chunk(const steppe::device::DeviceF2Blocks& f2,
                   std::span<const QpAdmModel> models,
                   const std::vector<std::size_t>& mem, std::size_t off, std::size_t B,
                   int nl, int nr, int r_fit, int rmax, int m, int nb, int P, double n,
                   const int* dBlockSizes, const int* d_surv, const QpAdmOptions& opts,
                   const Precision& precision, Precision::Kind tag,
                   std::vector<QpAdmResult>& results);

    // AssembleFlags — reference §3
    struct AssembleFlags {
        bool nonspd;
        bool se_computed;
    };

    // Per-model result assembly — reference §13
    void assemble_result(const QpAdmModel& mdl, int nl, int nr, int r_fit, int rmax,
                         Precision::Kind tag, AssembleFlags flags, int fit_status,
                         const double* weight, const double* se, double chisq,
                         const double* rank_chisq, const double* pop_chisq,
                         const double* pop_wfull, double rank_alpha, QpAdmResult& res);

private:
    // Inline device-guard helpers — reference §18
    void guard_device() const { STEPPE_CUDA_CHECK(cudaSetDevice(device_id_)); }

    [[nodiscard]] static int set_and_return_device(int device_id) {
        STEPPE_CUDA_CHECK(cudaSetDevice(device_id));
        return device_id;
    }

    // Private data members; declaration order is load-bearing at teardown — reference §17
    int device_id_ = 0;

    Stream stream_{};
    CublasHandle blas_{};
    CusolverDnHandle solver_{};
    DeviceBuffer<std::byte> workspace_{steppe::kCublasWorkspaceBytes};

    Precision solve_precision_{Precision::Kind::Fp64};

    std::size_t batched_dispatch_count_ = 0;

    std::vector<double> tot_line_{};

    PinnedRegistryCache pinned_in_{};

    PinnedBuffer<double> stage_f2_{};
    PinnedBuffer<double> stage_vpair_{};

    DeviceBuffer<double> solver_work_{};
    DeviceBuffer<double> svd_s_{};
    DeviceBuffer<double> svd_u_{};
    DeviceBuffer<double> svd_vt_{};
    DeviceBuffer<double> svd_a2_{};
    DeviceBuffer<int>    svd_info_{};
};

}  // namespace steppe::device
