// src/device/backend.hpp
//
// ComputeBackend — the CUDA-free interface every compute operation routes
// through, plus the plain-data structs that cross it. Two implementations
// satisfy it: CudaBackend (the GPU deliverable) and CpuBackend (the reference
// oracle). CUDA stays private to the device layer, behind this one door.
//
// Reference: docs/reference/src_device_backend.hpp.md
#ifndef STEPPE_DEVICE_BACKEND_HPP
#define STEPPE_DEVICE_BACKEND_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"
#include "steppe/fstats.hpp"
#include "steppe/qpadm.hpp"
#include "core/internal/views.hpp"
#include "core/internal/decode_af.hpp"
#include "device/device_partial.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/device_decode_result.hpp"
#include "device/stream_f2_blocks.hpp"

namespace steppe {

// f2 matrix result — reference §6
struct F2Result {
    std::vector<double> f2;

    std::vector<double> vpair;

    int P = 0;
};

// Per-block f4 + jackknife estimate — reference §7
struct F4Blocks {
    std::vector<double> x_blocks;

    std::vector<double> x_total;

    std::vector<double> x_loo;

    int nl = 0;
    int nr = 0;
    int n_block = 0;

    std::vector<int> block_sizes;
};

// Jackknife covariance and its inverse — reference §7
struct JackknifeCov {
    std::vector<double> Q;

    std::vector<double> Qinv;

    int m = 0;

    Status status = Status::Ok;
};

// GLS weight fit — reference §7
struct GlsWeights {
    std::vector<double> w;
    std::vector<double> A;
    std::vector<double> B;
    double chisq = 0.0;
    int r = 0;

    Status status = Status::Ok;
};

// qpfstats smoothing-solve output — reference §9
struct QpfstatsSmooth {
    std::vector<double> b;

    std::vector<double> bglob;

    std::vector<double> recenter_shift;

    int npairs = 0;
    int n_block = 0;

    Status status = Status::Ok;
};

// Diagonal-only jackknife variance — reference §7
struct JackknifeDiag {
    std::vector<double> var;
    int m = 0;
    Status status = Status::Ok;
};

// Ratio block-jackknife output — reference §8
struct RatioBlockJackknife {
    std::vector<double> est;
    std::vector<double> se;
    std::vector<double> z;
    std::vector<double> p;
    int N = 0;
    Status status = Status::Ok;
};

// Ratio-jackknife input array descriptor — reference §8
struct RatioJackArray {
    const double* data = nullptr;
    long base = 0;
    long item_stride = 0;
    long block_stride = 0;
};

// qpGraph topology arena — reference §10
struct QpGraphTopoArena {
    int npop = 0, nedge_norm = 0, nadmix = 0, npair = 0, npath = 0, base_leaf = 0;
    std::vector<double> pwts0;
    std::vector<int> pe_edge, pe_leaf, pe_path;
    std::vector<int> pae_path, pae_admixedge;
    std::vector<int> cmb1, cmb2;
    bool constrained = true;
    double fudge = 1e-4;
};

// qpGraph best-of-restarts fit — reference §10
struct QpGraphFleet {
    std::vector<double> theta;
    std::vector<double> theta_lo;
    std::vector<double> theta_hi;
    std::vector<double> edge_length;
    std::vector<double> f3_fit;
    double score = 0.0;
    double restart_spread = 0.0;
    Status status = Status::Ok;
};

// qpGraph batched best scores — reference §10
struct QpGraphFleetBatch {
    std::vector<double> best_score;
    std::vector<double> restart_spread;
    Status status = Status::Ok;
};

// f-stat sweep survivors — reference §11
struct SweepSurvivors {
    std::vector<std::array<int, 4>> keys;
    std::vector<double> est;
    std::vector<double> se;
    std::vector<double> z;

    std::size_t enumerated = 0;
    bool capped = false;
    Status status = Status::Ok;
};

// Sweep filter-mode constants — reference §11
inline constexpr int kSweepFilterMinZ = 0;
inline constexpr int kSweepFilterTopK = 1;

// f-stat sweep request — reference §11
struct SweepConfig {
    int k = 4;
    int filter_mode = kSweepFilterMinZ;
    double min_z = 3.0;
    std::size_t top_k = 1000000;
    std::vector<int> pop_subset;
    bool sure = false;
};

// Rank-test sweep output — reference §7
struct RankSweep {
    std::vector<double> chisq;
    std::vector<int>    dof;
    std::vector<double> p;

    std::vector<int>    rd_f4rank;
    std::vector<int>    rd_dof;
    std::vector<double> rd_chisq;
    std::vector<double> rd_p;
    std::vector<int>    rd_dofdiff;
    std::vector<double> rd_chisqdiff;
    std::vector<double> rd_p_nested;

    int f4rank = 0;
    int rank_Q = 0;

    int svd_path = 0;

    Status status = Status::Ok;
};

// Leave-one-source-out feasibility row — reference §7
struct PopDropRow {
    std::string         pat;
    int                 wt = 0;
    int                 dof = 0;
    double              chisq = 0.0;
    double              p = 0.0;
    int                 f4rank = 0;
    std::vector<double> weight;
    bool                feasible = false;
    Status              status = Status::Ok;
};

// Genotype decode input tile view — reference §5
struct DecodeTileView {
    const std::uint8_t* packed = nullptr;
    std::size_t bytes_per_record = 0;
    std::size_t n_snp = 0;
    std::size_t n_individuals = 0;
    const std::size_t* pop_offsets = nullptr;
    int n_pop = 0;

    const int* sample_ploidy = nullptr;

    int ploidy = 2;

    bool detect_ploidy_on_device = false;
};

// Genotype decode output — reference §5
struct DecodeResult {
    std::vector<double> q;
    std::vector<double> v;
    std::vector<double> n;
    int P = 0;
    long M = 0;
};

// Native→canonical code map — reference §5
enum class TileEncoding : int {
    Identity = 0,
};

// SNP-major source tile view — reference §5
struct SnpMajorTileView {
    const std::uint8_t* snp_major = nullptr;
    std::size_t src_bytes_per_record = 0;
    std::size_t n_snp = 0;

    const std::size_t* sel_rows = nullptr;
    std::size_t n_individuals = 0;

    const std::size_t* pop_offsets = nullptr;
    int n_pop = 0;

    TileEncoding encoding = TileEncoding::Identity;
};

// Canonical individual-major tile — reference §5
struct CanonicalTile {
    std::vector<std::uint8_t> packed;
    std::size_t bytes_per_record = 0;
    std::size_t n_snp = 0;
    std::size_t n_individuals = 0;
    std::vector<std::size_t> pop_offsets;
};

// Backend capability probe — reference §13
struct BackendCapabilities {
    int device_count = 0;

    int compute_major = 0;
    int compute_minor = 0;

    std::size_t total_vram_bytes = 0;
    std::size_t free_vram_bytes  = 0;

    bool can_access_peer = false;

    bool emulated_fp64_honorable = false;
};

// DATES weighted-LD moments — reference §12
struct DatesMoments {
    int n_chrom = 0;
    int n_bin = 0;
    std::vector<double> s0;
    std::vector<double> s1;
    std::vector<double> s2;
    std::vector<double> s11;
    std::vector<double> s12;
    std::vector<double> s22;
    Status status = Status::Ok;
};

// DATES exponential-decay fit — reference §12
struct DatesExpFit {
    double date_gen = 0.0;
    double error_sd = 0.0;
    int ok = 0;
};

class ComputeBackend;

// Host helper functions — reference §15
namespace core::qpadm {
[[nodiscard]] std::vector<QpAdmResult> fit_models_batched_default(
    ComputeBackend& be, const steppe::device::DeviceF2Blocks& f2,
    std::span<const QpAdmModel> models, const QpAdmOptions& opts);

[[nodiscard]] bool model_in_small_path(const QpAdmModel& model, const QpAdmOptions& opts);

}  // namespace core::qpadm

// The ComputeBackend interface — reference §14
class ComputeBackend {
public:
    ComputeBackend() = default;
    ComputeBackend(const ComputeBackend&) = delete;
    ComputeBackend& operator=(const ComputeBackend&) = delete;
    ComputeBackend(ComputeBackend&&) = delete;
    ComputeBackend& operator=(ComputeBackend&&) = delete;
    virtual ~ComputeBackend() = default;

    [[nodiscard]] virtual F2Result compute_f2(const core::MatView& Q,
                                              const core::MatView& V,
                                              const core::MatView& N,
                                              const Precision& precision) = 0;

    [[nodiscard]] virtual F2BlockTensor compute_f2_blocks(const core::MatView& Q,
                                                          const core::MatView& V,
                                                          const core::MatView& N,
                                                          const int* block_id,
                                                          int n_block,
                                                          const Precision& precision) = 0;

    [[nodiscard]] virtual steppe::device::DeviceF2Blocks compute_f2_blocks_device(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, const Precision& precision) {
        (void)Q; (void)V; (void)N; (void)block_id; (void)n_block; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::compute_f2_blocks_device: not supported by this backend "
            "(device-resident output requires a CUDA backend)");
    }

    [[nodiscard]] virtual steppe::device::DevicePartial compute_f2_blocks_resident(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, int b0, const Precision& precision) {
        (void)Q; (void)V; (void)N; (void)block_id; (void)n_block; (void)b0; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::compute_f2_blocks_resident: not supported by this backend "
            "(device-resident combine requires a peer-capable CUDA backend; the §4 gate "
            "must have routed a non-CUDA/non-peer backend to the host-staged path)");
    }

    virtual void compute_f2_blocks_into(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, int b0,
        double* dst_f2, double* dst_vpair, int* block_sizes_dst,
        const Precision& precision) {
        (void)Q; (void)V; (void)N; (void)block_id; (void)n_block; (void)b0;
        (void)dst_f2; (void)dst_vpair; (void)block_sizes_dst; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::compute_f2_blocks_into: not supported by this backend");
    }

    virtual void compute_f2_blocks_streamed(
        const core::MatView& Q, const core::MatView& V, const core::MatView& N,
        const int* block_id, int n_block, const Precision& precision,
        steppe::device::StreamTarget& target,
        const steppe::device::RedecodeSource* redecode = nullptr) {
        (void)Q; (void)V; (void)N; (void)block_id; (void)n_block; (void)precision; (void)target;
        (void)redecode;
        throw std::runtime_error(
            "ComputeBackend::compute_f2_blocks_streamed: not supported by this backend "
            "(out-of-core block streaming requires a CUDA backend)");
    }

    [[nodiscard]] virtual DecodeResult decode_af(const DecodeTileView& tile) = 0;

    [[nodiscard]] virtual std::vector<int> detect_sample_ploidy_device(
        const DecodeTileView& tile) {
        std::vector<int> ploidy(tile.n_individuals, core::kPloidyPseudoHaploid);
        const std::size_t window =
            (static_cast<std::size_t>(core::kPloidyDetectSnps) < tile.n_snp)
                ? static_cast<std::size_t>(core::kPloidyDetectSnps)
                : tile.n_snp;
        if (window == 0 || tile.n_individuals == 0) return ploidy;
        for (std::size_t g = 0; g < tile.n_individuals; ++g) {
            const std::uint8_t* rec = tile.packed + g * tile.bytes_per_record;
            for (std::size_t s = 0; s < window; ++s) {
                const std::size_t byte_in_rec =
                    s / static_cast<std::size_t>(core::kCodesPerByte);
                const int pos_in_byte =
                    static_cast<int>(s % static_cast<std::size_t>(core::kCodesPerByte));
                if (core::genotype_code(rec[byte_in_rec], pos_in_byte) ==
                    core::kHeterozygousGenotypeCode) {
                    ploidy[g] = core::kPloidyDiploid;
                    break;
                }
            }
        }
        return ploidy;
    }

    [[nodiscard]] virtual CanonicalTile transpose_to_canonical(
        const SnpMajorTileView& view) {
        CanonicalTile out;
        out.n_snp = view.n_snp;
        out.n_individuals = view.n_individuals;
        const std::size_t cpb = static_cast<std::size_t>(core::kCodesPerByte);
        out.bytes_per_record =
            view.n_snp == 0 ? 0 : (view.n_snp + cpb - 1) / cpb;
        if (view.pop_offsets != nullptr && view.n_pop >= 0) {
            out.pop_offsets.assign(
                view.pop_offsets,
                view.pop_offsets + static_cast<std::size_t>(view.n_pop) + 1u);
        }
        const std::size_t total = view.n_individuals * out.bytes_per_record;
        out.packed.assign(total, std::uint8_t{0});
        if (total == 0) return out;

        for (std::size_t g = 0; g < view.n_individuals; ++g) {
            const std::size_t src_row = view.sel_rows[g];
            const std::size_t src_byte_in_snp = src_row / cpb;
            const int src_pos = static_cast<int>(src_row % cpb);
            for (std::size_t b = 0; b < out.bytes_per_record; ++b) {
                std::uint8_t out_byte = 0;
                const std::size_t s0 = b * cpb;
                for (int k = 0; k < core::kCodesPerByte; ++k) {
                    const std::size_t s = s0 + static_cast<std::size_t>(k);
                    if (s >= view.n_snp) break;
                    const std::uint8_t src_byte =
                        view.snp_major[s * view.src_bytes_per_record + src_byte_in_snp];
                    const std::uint8_t code = core::genotype_code(src_byte, src_pos);
                    std::uint8_t canon = code;
                    switch (view.encoding) {
                        case TileEncoding::Identity:
                        default:
                            canon = code;
                            break;
                    }
                    const int shift =
                        (core::kCodesPerByte - 1 - k) * core::kBitsPerCode;
                    out_byte = static_cast<std::uint8_t>(
                        out_byte | static_cast<std::uint8_t>(canon << shift));
                }
                out.packed[g * out.bytes_per_record + b] = out_byte;
            }
        }
        return out;
    }

    [[nodiscard]] virtual steppe::device::DeviceDecodeResult decode_af_compact_autosome(
        const DecodeTileView& tile, std::span<const int> chrom,
        std::span<const double> genpos, std::span<const double> physpos,
        int chrom_min, int chrom_max) {
        (void)tile; (void)chrom; (void)genpos; (void)physpos;
        (void)chrom_min; (void)chrom_max;
        throw std::runtime_error(
            "ComputeBackend::decode_af_compact_autosome: not implemented by this backend "
            "(the device-resident decode seam requires a CUDA backend)");
    }

    [[nodiscard]] virtual steppe::device::DeviceDecodeResult decode_af_compact_filter(
        const DecodeTileView& tile, std::span<const char> ref, std::span<const char> alt,
        std::span<const int> chrom, std::span<const double> genpos,
        std::span<const double> physpos, const FilterConfig& cfg,
        std::span<const std::size_t> pop_individuals, int ploidy, double maxmiss,
        long s_lo) {
        (void)tile; (void)ref; (void)alt; (void)chrom; (void)genpos; (void)physpos;
        (void)cfg; (void)pop_individuals; (void)ploidy; (void)maxmiss; (void)s_lo;
        throw std::runtime_error(
            "ComputeBackend::decode_af_compact_filter: not implemented by this backend "
            "(the regime-B device-resident decode seam requires a CUDA backend)");
    }

    virtual void set_solve_precision(const Precision& precision) { (void)precision; }

    [[nodiscard]] virtual F4Blocks assemble_f4(const steppe::device::DeviceF2Blocks& f2,
                                               std::span<const int> left_idx,
                                               std::span<const int> right_idx,
                                               const Precision& precision) {
        (void)f2; (void)left_idx; (void)right_idx; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::assemble_f4: not implemented by this backend");
    }

    [[nodiscard]] virtual F4Blocks assemble_f4(const F2BlockTensor& f2,
                                               std::span<const int> left_idx,
                                               std::span<const int> right_idx,
                                               const Precision& precision) {
        (void)f2; (void)left_idx; (void)right_idx; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::assemble_f4(host): not implemented by this backend");
    }

    [[nodiscard]] virtual F4Blocks assemble_f4_quartets(
        const steppe::device::DeviceF2Blocks& f2,
        std::span<const int> quartets,
        const Precision& precision) {
        (void)f2; (void)quartets; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::assemble_f4_quartets: not implemented by this backend");
    }

    [[nodiscard]] virtual F4Blocks assemble_f4_quartets(
        const F2BlockTensor& f2,
        std::span<const int> quartets,
        const Precision& precision) {
        (void)f2; (void)quartets; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::assemble_f4_quartets(host): not implemented by this backend");
    }

    [[nodiscard]] virtual F4Blocks assemble_f3_triples(
        const steppe::device::DeviceF2Blocks& f2,
        std::span<const int> triples,
        const Precision& precision) {
        (void)f2; (void)triples; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::assemble_f3_triples: not implemented by this backend");
    }

    [[nodiscard]] virtual F4Blocks assemble_f3_triples(
        const F2BlockTensor& f2,
        std::span<const int> triples,
        const Precision& precision) {
        (void)f2; (void)triples; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::assemble_f3_triples(host): not implemented by this backend");
    }

    [[nodiscard]] virtual QpGraphFleet qpgraph_fit_fleet(const QpGraphTopoArena& topo,
                                                         std::span<const double> f_obs,
                                                         std::span<const double> qinv,
                                                         int numstart, int maxit,
                                                         double tol,
                                                         const Precision& precision) {
        (void)topo; (void)f_obs; (void)qinv; (void)numstart; (void)maxit; (void)tol;
        (void)precision;
        throw std::runtime_error(
            "ComputeBackend::qpgraph_fit_fleet: not implemented by this backend");
    }

    [[nodiscard]] virtual QpGraphFleetBatch qpgraph_fit_fleet_batch(
        const std::vector<QpGraphTopoArena>& topos, std::span<const double> f_obs,
        std::span<const double> qinv, int numstart, int maxit, double tol,
        const Precision& precision) {
        (void)topos; (void)f_obs; (void)qinv; (void)numstart; (void)maxit; (void)tol;
        (void)precision;
        throw std::runtime_error(
            "ComputeBackend::qpgraph_fit_fleet_batch: not implemented by this backend");
    }

    virtual void dstat_block_reduce(const double* Q, const double* V, int P, long M,
                                    const int* block_id, int n_block,
                                    std::span<const int> quadruples,
                                    double* numsum, double* densum, double* cnt) {
        (void)Q; (void)V; (void)P; (void)M; (void)block_id; (void)n_block;
        (void)quadruples; (void)numsum; (void)densum; (void)cnt;
        throw std::runtime_error(
            "ComputeBackend::dstat_block_reduce: not implemented by this backend");
    }

    virtual void dstat_block_reduce(const steppe::device::DeviceDecodeResult& dec,
                                    const int* block_id, int n_block,
                                    std::span<const int> quadruples,
                                    double* numsum, double* densum, double* cnt) {
        (void)dec; (void)block_id; (void)n_block;
        (void)quadruples; (void)numsum; (void)densum; (void)cnt;
        throw std::runtime_error(
            "ComputeBackend::dstat_block_reduce(DeviceDecodeResult): not implemented "
            "by this backend (the device-resident decode seam requires a CUDA backend)");
    }

    [[nodiscard]] virtual DatesMoments dates_curve(
        const double* src1_freq, const double* src2_freq, const double* src_valid,
        const std::uint8_t* packed, std::size_t bytes_per_record, int n_target,
        const int* target_ploidy, const int* grid_cell, long M,
        const int* chrom_first, const int* chrom_last, int n_chrom,
        int numqbins, int n_bin, int diffmax, double binsize, int qbin,
        const Precision& precision) {
        (void)src1_freq; (void)src2_freq; (void)src_valid; (void)packed;
        (void)bytes_per_record; (void)n_target; (void)target_ploidy; (void)grid_cell;
        (void)M; (void)chrom_first; (void)chrom_last; (void)n_chrom; (void)numqbins;
        (void)n_bin; (void)diffmax; (void)binsize; (void)qbin; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::dates_curve: not implemented by this backend "
            "(the cuFFT autocorrelation LD engine requires a CUDA backend; the "
            "CpuBackend provides the FFT-free reference oracle)");
    }

    virtual void dates_repack(const std::uint8_t* src, std::size_t src_bpr,
                              const long* kept_src, long M_kept, int n_target,
                              std::size_t dst_bpr, std::uint8_t* dst) {
        (void)src; (void)src_bpr; (void)kept_src; (void)M_kept; (void)n_target;
        (void)dst_bpr; (void)dst;
        throw std::runtime_error(
            "ComputeBackend::dates_repack: not implemented by this backend "
            "(CpuBackend uses core::dates::dates_repack_default; CUDA the device gather)");
    }

    virtual std::vector<DatesExpFit> dates_fit(const double* curves, int win_len,
                                               int n_curves, double step, bool affine) {
        (void)curves; (void)win_len; (void)n_curves; (void)step; (void)affine;
        throw std::runtime_error(
            "ComputeBackend::dates_fit: not implemented by this backend "
            "(CpuBackend uses core::dates::dates_fit_default; CUDA the device fit)");
    }

    [[nodiscard]] virtual QpfstatsSmooth qpfstats_smooth(std::span<const double> x,
                                                         std::span<const double> ymat,
                                                         std::span<const double> y,
                                                         int npopcomb, int npairs,
                                                         int n_block, double ridge,
                                                         const Precision& precision) {
        (void)x; (void)ymat; (void)y; (void)npopcomb; (void)npairs; (void)n_block;
        (void)ridge; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::qpfstats_smooth: not implemented by this backend");
    }

    [[nodiscard]] virtual QpfstatsSmooth qpfstats_blocks_smooth(
        const double* Q, const double* V, int P, long M, const int* block_id, int n_block,
        std::span<const int> quadruples, std::span<const double> x, int npopcomb, int npairs,
        std::span<const int> block_sizes, double ridge, const Precision& precision) {
        (void)Q; (void)V; (void)P; (void)M; (void)block_id; (void)n_block; (void)quadruples;
        (void)x; (void)npopcomb; (void)npairs; (void)block_sizes; (void)ridge; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::qpfstats_blocks_smooth: not implemented by this backend");
    }

    [[nodiscard]] virtual QpfstatsSmooth qpfstats_blocks_smooth(
        const steppe::device::DeviceDecodeResult& dec, const int* block_id, int n_block,
        std::span<const int> quadruples, std::span<const double> x, int npopcomb, int npairs,
        std::span<const int> block_sizes, double ridge, const Precision& precision) {
        (void)dec; (void)block_id; (void)n_block; (void)quadruples;
        (void)x; (void)npopcomb; (void)npairs; (void)block_sizes; (void)ridge; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::qpfstats_blocks_smooth(DeviceDecodeResult): not implemented "
            "by this backend (the device-resident decode seam requires a CUDA backend)");
    }

    [[nodiscard]] virtual JackknifeCov jackknife_cov(const F4Blocks& x,
                                                     std::span<const int> block_sizes,
                                                     double fudge,
                                                     const Precision& precision) {
        (void)x; (void)block_sizes; (void)fudge; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::jackknife_cov: not implemented by this backend");
    }

    [[nodiscard]] virtual JackknifeDiag jackknife_diag(const F4Blocks& x,
                                                       std::span<const int> block_sizes,
                                                       const Precision& precision) {
        (void)x; (void)block_sizes; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::jackknife_diag: not implemented by this backend");
    }

    [[nodiscard]] virtual RatioBlockJackknife ratio_block_jackknife(
        const RatioJackArray& num, const RatioJackArray& den, const RatioJackArray& weight,
        const RatioJackArray& xblk_num, const RatioJackArray& xblk_den, int N, int n_block,
        int tot_mode, double setmiss_thresh, bool compute_p, const Precision& precision) {
        (void)num; (void)den; (void)weight; (void)xblk_num; (void)xblk_den;
        (void)N; (void)n_block; (void)tot_mode; (void)setmiss_thresh; (void)compute_p;
        (void)precision;
        throw std::runtime_error(
            "ComputeBackend::ratio_block_jackknife: not implemented by this backend");
    }

    [[nodiscard]] virtual RatioBlockJackknife f4ratio_blocks_jackknife(
        const steppe::device::DeviceF2Blocks& f2, std::span<const int> flat, int N,
        double setmiss_thresh, const Precision& precision) {
        (void)f2; (void)flat; (void)N; (void)setmiss_thresh; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::f4ratio_blocks_jackknife(DeviceF2Blocks): not implemented by this "
            "backend (the device-resident assemble seam requires a CUDA backend)");
    }

    [[nodiscard]] virtual RatioBlockJackknife f4ratio_blocks_jackknife(
        const F2BlockTensor& f2, std::span<const int> flat, int N, double setmiss_thresh,
        const Precision& precision) {
        (void)f2; (void)flat; (void)N; (void)setmiss_thresh; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::f4ratio_blocks_jackknife(host): not implemented by this backend");
    }

    [[nodiscard]] virtual RatioBlockJackknife dstat_blocks_jackknife(
        const steppe::device::DeviceDecodeResult& dec, const int* block_id, int n_block,
        std::span<const int> quadruples) {
        (void)dec; (void)block_id; (void)n_block; (void)quadruples;
        throw std::runtime_error(
            "ComputeBackend::dstat_blocks_jackknife(DeviceDecodeResult): not implemented by this "
            "backend (the device-resident decode seam requires a CUDA backend)");
    }

    [[nodiscard]] virtual RatioBlockJackknife dstat_blocks_jackknife(
        const double* Q, const double* V, int P, long M, const int* block_id, int n_block,
        std::span<const int> quadruples) {
        (void)Q; (void)V; (void)P; (void)M; (void)block_id; (void)n_block; (void)quadruples;
        throw std::runtime_error(
            "ComputeBackend::dstat_blocks_jackknife(host): not implemented by this backend");
    }

    [[nodiscard]] virtual SweepSurvivors f4_sweep(const steppe::device::DeviceF2Blocks& f2,
                                                  const SweepConfig& cfg,
                                                  const Precision& precision) {
        (void)f2; (void)cfg; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::f4_sweep: not implemented by this backend "
            "(the GPU-only sweep requires a CUDA backend)");
    }

    [[nodiscard]] virtual SweepSurvivors f3_sweep(const steppe::device::DeviceF2Blocks& f2,
                                                  const SweepConfig& cfg,
                                                  const Precision& precision) {
        (void)f2; (void)cfg; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::f3_sweep: not implemented by this backend "
            "(the GPU-only sweep requires a CUDA backend)");
    }

    [[nodiscard]] virtual RankSweep rank_sweep(const F4Blocks& x,
                                               const JackknifeCov& cov,
                                               double alpha,
                                               const QpAdmOptions& opts,
                                               const Precision& precision) {
        (void)x; (void)cov; (void)alpha; (void)opts; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::rank_sweep: not implemented by this backend");
    }

    [[nodiscard]] virtual bool provides_rank_sweep() const { return false; }

    [[nodiscard]] virtual GlsWeights gls_weights(const F4Blocks& x,
                                                 const JackknifeCov& cov,
                                                 int r,
                                                 const QpAdmOptions& opts,
                                                 const Precision& precision) {
        (void)x; (void)cov; (void)r; (void)opts; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::gls_weights: not implemented by this backend");
    }

    [[nodiscard]] virtual std::vector<double> gls_weights_loo_batched(
        const F4Blocks& x, const JackknifeCov& cov, int r,
        const QpAdmOptions& opts, const Precision& precision) {
        (void)x; (void)cov; (void)r; (void)opts; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::gls_weights_loo_batched: not implemented by this backend");
    }

    [[nodiscard]] virtual std::vector<double> se_from_wmat(
        const F4Blocks& x, const JackknifeCov& cov, int r,
        const QpAdmOptions& opts, const Precision& precision) {
        (void)x; (void)cov; (void)r; (void)opts; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::se_from_wmat: not implemented by this backend");
    }

    [[nodiscard]] virtual std::vector<QpAdmResult> fit_models_batched(
        const steppe::device::DeviceF2Blocks& f2,
        std::span<const QpAdmModel> models,
        const QpAdmOptions& opts,
        const Precision& precision) {
        (void)f2; (void)models; (void)opts; (void)precision;
        throw std::runtime_error(
            "ComputeBackend::fit_models_batched: this backend has no batched override "
            "(route through core::qpadm::fit_models_batched_default instead)");
    }

    [[nodiscard]] virtual bool provides_batched_fit() const { return false; }

    [[nodiscard]] virtual BackendCapabilities capabilities() const {
        return BackendCapabilities{};
    }

    [[nodiscard]] virtual std::size_t batched_dispatch_count() const { return 0; }
};

}  // namespace steppe

#endif  // STEPPE_DEVICE_BACKEND_HPP
