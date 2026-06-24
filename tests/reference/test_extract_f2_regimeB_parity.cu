// tests/reference/test_extract_f2_regimeB_parity.cu
//
// REGIME-B ON-DEVICE KEEP-SET PARITY TEST — the central gate for the extract_f2
// FULL filter moved on-device (host-compute audit regime (B); the regime-B
// keep-mask kernel + the N lockstep CUB compaction in CudaBackend::
// decode_af_compact_filter). It is the regime-B analogue of test_filter_oracle's
// drop==mask memcmp: the CUDA path's keep-SET and the lockstep-compacted Q/V/N
// MUST be BIT-IDENTICAL to the host filter path (the CpuBackend decode + the
// verbatim snp_filter / maxmiss / lockstep-subset oracle that extract_f2_core.cpp
// still runs on a CPU-only backend). NO synthetic data: the SAME real v66 AADR
// triple the M1 decode test uses (raw TGENO; the steppe decode is the correct one).
//
// THE FOUR ASSERTS, per FilterConfig (the regime-B GOLDEN-EXACT requirement):
//   (1) M_kept EXACT — the on-device kept count == the host kept count.
//   (2) chrom_kept / genpos_kept BIT-IDENTICAL (same SNPs, same FILE ORDER) — so
//       the CUDA-free assign_blocks over the compacted axis is identical ⇒ identical
//       block_id / golden.
//   (3) the compacted Q / V / N BIT-IDENTICAL (memcmp) — the three lockstep gathers
//       reproduce the host lockstep subset to the last bit.
//   (4) (cross-check) the on-device keep COUNT matches an INDEPENDENT host keep-mask
//       recompute (build_snp_keep_mask + the maxmiss loop), so the gate does not just
//       compare two copies of the same code.
//
// THE FP SURFACE COVERED. The configs span both surfaces from the bit_exact_risk
// analysis: the FMA-IMMUNE integer filters (transversions_only, the pop-coverage
// maxmiss [an integer count of N<=0], drop_monomorphic [is_monomorphic ==0.0 exact],
// autosomes_only) AND the FP-FRAGILE pooled-MAF comparison (maf_min>0 near the
// boundary) — the only piece the __dmul_rn/__dadd_rn FFMA-immune pin defends. A
// bit-identical keep-set across ALL of them is the proof the pin holds host==device.
//
// Self-checking main() (mirrors test_decode_equivalence.cu / test_filter_oracle.cu);
// CTest gates on the exit code. SKIPs (exit 0) if no CUDA device or the raw AADR
// triple is absent — the device-resident path cannot be exercised then.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "steppe/config.hpp"            // FilterConfig, kAutosomeChromMin/Max
#include "device/backend.hpp"           // ComputeBackend, DecodeTileView, DeviceDecodeResult
#include "device/backend_factory.hpp"   // make_cpu_backend / make_cuda_backend

#include "io/geno_reader.hpp"
#include "io/genotype_tile.hpp"
#include "io/ind_reader.hpp"
#include "io/ploidy_detect.hpp"
#include "io/snp_reader.hpp"
#include "io/filter/snp_filter.hpp"

using steppe::ComputeBackend;
using steppe::DecodeResult;
using steppe::DecodeTileView;
using steppe::FilterConfig;
namespace flt = steppe::io::filter;

namespace {

constexpr std::size_t kAutoTopK = 50;
constexpr std::size_t kSnpCap = 100000;
constexpr const char* kDefaultDataRoot = "/workspace/data/aadr";
constexpr const char* kGenoBase = "v66.p1_HO.aadr.patch.PUB";
constexpr int kPloidyDiploid = 2;

// The HOST regime-B path (the verbatim oracle extract_f2_core.cpp runs on a CPU-only
// backend): decode_af → build_snp_keep_mask (geno forced to 1.0) → the pop-coverage
// maxmiss loop → the lockstep Q/V/N subset + the kept chrom/genpos. Returns the kept
// count and fills the compacted arrays + the kept axis.
long host_regimeb(const DecodeResult& dec, const steppe::io::SnpTable& snptab,
                  const std::vector<std::size_t>& pop_individuals, int P, long M,
                  const FilterConfig& filter, double maxmiss,
                  std::vector<double>& Qk, std::vector<double>& Vk,
                  std::vector<double>& Nk, std::vector<int>& chrom_kept,
                  std::vector<double>& genpos_kept) {
    flt::DecodedTileSummaryInput fin;
    fin.q = dec.q.data();
    fin.v = dec.v.data();
    fin.n = dec.n.data();
    fin.P = P;
    fin.M = M;
    fin.pop_individuals = pop_individuals;
    fin.ploidy = kPloidyDiploid;

    FilterConfig class_filter = filter;
    class_filter.geno_max_missing = 1.0;
    flt::SnpMembership mem(class_filter);
    std::vector<bool> keep = flt::build_snp_keep_mask(fin, snptab, class_filter, mem);

    if (maxmiss < 1.0) {
        for (long s = 0; s < M; ++s) {
            if (!keep[static_cast<std::size_t>(s)]) continue;
            int n_missing_pops = 0;
            for (int p = 0; p < P; ++p) {
                const std::size_t off = static_cast<std::size_t>(p) +
                                        static_cast<std::size_t>(P) * static_cast<std::size_t>(s);
                if (dec.n[off] <= 0.0) ++n_missing_pops;
            }
            const double frac = static_cast<double>(n_missing_pops) / static_cast<double>(P);
            if (frac > maxmiss) keep[static_cast<std::size_t>(s)] = false;
        }
    }

    std::size_t n_kept = 0;
    for (bool k : keep) n_kept += k ? 1u : 0u;
    Qk.assign(static_cast<std::size_t>(P) * n_kept, 0.0);
    Vk.assign(static_cast<std::size_t>(P) * n_kept, 0.0);
    Nk.assign(static_cast<std::size_t>(P) * n_kept, 0.0);
    chrom_kept.clear();
    genpos_kept.clear();
    chrom_kept.reserve(n_kept);
    genpos_kept.reserve(n_kept);
    std::size_t d = 0;
    for (std::size_t s = 0; s < keep.size(); ++s) {
        if (!keep[s]) continue;
        for (int i = 0; i < P; ++i) {
            const std::size_t src = static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * s;
            const std::size_t dst = static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * d;
            Qk[dst] = dec.q[src];
            Vk[dst] = dec.v[src];
            Nk[dst] = dec.n[src];
        }
        chrom_kept.push_back(snptab.chrom[s]);
        genpos_kept.push_back(snptab.genpos_morgans[s]);
        ++d;
    }
    return static_cast<long>(n_kept);
}

bool looks_like_no_gpu(const ComputeBackend& gpu) {
    return gpu.capabilities().device_count <= 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string root = (argc >= 2) ? argv[1] : kDefaultDataRoot;
    const std::string raw = root + "/raw/" + kGenoBase;
    const std::string geno = raw + ".geno";
    const std::string snp = raw + ".snp";
    const std::string ind = raw + ".ind";

    std::unique_ptr<steppe::io::GenoReader> reader;
    try {
        reader = std::make_unique<steppe::io::GenoReader>(geno);
    } catch (const std::exception& e) {
        std::printf("RESULT: SKIP (raw AADR .geno absent: %s)\n", e.what());
        return 0;
    }

    std::unique_ptr<ComputeBackend> gpu;
    try {
        gpu = steppe::device::make_cuda_backend();
    } catch (const std::exception& e) {
        std::printf("RESULT: SKIP (no CUDA backend: %s)\n", e.what());
        return 0;
    }
    if (!gpu || looks_like_no_gpu(*gpu)) {
        std::printf("RESULT: SKIP (no CUDA device — regime-B device path not exercised)\n");
        return 0;
    }
    auto cpu = steppe::device::make_cpu_backend();

    steppe::io::PopSelection sel;
    sel.mode = steppe::io::PopSelection::Mode::AutoTopK;
    sel.k = kAutoTopK;

    steppe::io::IndPartition part;
    steppe::io::SnpTable snptab;
    steppe::io::GenotypeTile tile;
    try {
        part = steppe::io::read_ind(ind, sel, reader->records_present());
        snptab = steppe::io::read_snp(snp, kSnpCap);
        const std::size_t M0 = std::min(kSnpCap, reader->header().n_snp);
        tile = reader->read_tile(part, 0, M0);
    } catch (const std::exception& e) {
        std::printf("RESULT: FAIL (io read: %s)\n", e.what());
        return 1;
    }
    const int P = static_cast<int>(tile.n_pop());
    const long M = static_cast<long>(tile.n_snp);
    const std::vector<int> sample_ploidy = steppe::io::detect_sample_ploidy(tile);

    std::vector<std::size_t> pop_individuals(static_cast<std::size_t>(P));
    for (int p = 0; p < P; ++p)
        pop_individuals[static_cast<std::size_t>(p)] =
            tile.pop_offsets[static_cast<std::size_t>(p) + 1] -
            tile.pop_offsets[static_cast<std::size_t>(p)];

    DecodeTileView view;
    view.packed = tile.packed.data();
    view.bytes_per_record = tile.bytes_per_record;
    view.n_snp = tile.n_snp;
    view.n_individuals = tile.n_individuals;
    view.pop_offsets = tile.pop_offsets.data();
    view.n_pop = P;
    view.sample_ploidy = sample_ploidy.data();
    view.ploidy = kPloidyDiploid;

    // One CPU decode (the host oracle's Q/V/N source), reused across configs.
    const DecodeResult dec_cpu = cpu->decode_af(view);

    std::fprintf(stderr, "[setup] P=%d M=%ld n_ind=%zu (REAL AADR auto-top %zu, snp-cap %zu)\n",
                 P, M, tile.n_individuals, kAutoTopK, kSnpCap);

    struct Case { const char* name; FilterConfig cfg; double maxmiss; };
    const Case cases[] = {
        // The extract_f2 production default (autosomes_only + drop_monomorphic, AT2
        // poly_only): FMA-immune (autosome integer + monomorphic ==0.0 exact).
        { "default-extract (auto+poly)",
          [] { FilterConfig c; c.autosomes_only = true; c.drop_monomorphic = true; return c; }(), 1.0 },
        // The integer-filter golden combo (transversions + maxmiss>0): wholly FMA-immune.
        { "transversions + maxmiss=0.5",
          [] { FilterConfig c; c.autosomes_only = true; c.drop_monomorphic = true; c.transversions_only = true; return c; }(), 0.5 },
        // maxmiss=0 (the global-intersection golden_fit0 SNP set) on the autosome+poly default.
        { "auto+poly + maxmiss=0",
          [] { FilterConfig c; c.autosomes_only = true; c.drop_monomorphic = true; return c; }(), 0.0 },
        // THE FP-FRAGILE case: pooled-MAF boundary — the only piece the FFMA-immune
        // __dmul_rn/__dadd_rn pin defends. A bit-identical keep-set proves the pin.
        { "maf>=0.05 (FP-fragile pooled MAF)",
          [] { FilterConfig c; c.autosomes_only = true; c.drop_monomorphic = true; c.maf_min = 0.05; return c; }(), 1.0 },
        { "maf>=0.01 (FP-fragile, more boundary SNPs)",
          [] { FilterConfig c; c.autosomes_only = true; c.maf_min = 0.01; return c; }(), 1.0 },
        // Everything at once + maxmiss>0.
        { "maf>=0.02 + transversions + maxmiss=0.3",
          [] { FilterConfig c; c.autosomes_only = true; c.drop_monomorphic = true; c.transversions_only = true; c.maf_min = 0.02; return c; }(), 0.3 },
    };

    int failures = 0;
    const std::size_t Mu = static_cast<std::size_t>(M);
    for (const Case& tc : cases) {
        // ---- HOST oracle path ----
        std::vector<double> Qh, Vh, Nh;
        std::vector<int> chr_h;
        std::vector<double> gp_h;
        const long mk_host = host_regimeb(dec_cpu, snptab, pop_individuals, P, M, tc.cfg,
                                          tc.maxmiss, Qh, Vh, Nh, chr_h, gp_h);

        // ---- CUDA device-resident regime-B path ----
        steppe::device::DeviceDecodeResult ddr;
        try {
            ddr = gpu->decode_af_compact_filter(
                view,
                std::span<const char>(snptab.ref.data(), Mu),
                std::span<const char>(snptab.alt.data(), Mu),
                std::span<const int>(snptab.chrom.data(), Mu),
                std::span<const double>(snptab.genpos_morgans.data(), Mu),
                tc.cfg, std::span<const std::size_t>(pop_individuals.data(), pop_individuals.size()),
                kPloidyDiploid, tc.maxmiss);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[%s] CUDA decode_af_compact_filter threw: %s — FAIL\n",
                         tc.name, e.what());
            ++failures;
            continue;
        }
        const long mk_dev = ddr.M_kept;
        std::vector<double> Qd, Vd, Nd;
        ddr.to_host_qvn(Qd, Vd, Nd);

        // (1) M_kept EXACT.
        const bool count_ok = (mk_dev == mk_host);
        // (2) chrom_kept / genpos_kept BIT-IDENTICAL.
        bool axis_ok = (ddr.chrom_kept.size() == chr_h.size()) &&
                       (ddr.genpos_kept.size() == gp_h.size());
        if (axis_ok && mk_host > 0) {
            axis_ok = (std::memcmp(ddr.chrom_kept.data(), chr_h.data(),
                                   chr_h.size() * sizeof(int)) == 0) &&
                      (std::memcmp(ddr.genpos_kept.data(), gp_h.data(),
                                   gp_h.size() * sizeof(double)) == 0);
        }
        // (3) the compacted Q / V / N BIT-IDENTICAL.
        bool qvn_ok = (Qd.size() == Qh.size()) && (Vd.size() == Vh.size()) &&
                      (Nd.size() == Nh.size());
        if (qvn_ok && !Qh.empty()) {
            qvn_ok = (std::memcmp(Qd.data(), Qh.data(), Qh.size() * sizeof(double)) == 0) &&
                     (std::memcmp(Vd.data(), Vh.data(), Vh.size() * sizeof(double)) == 0) &&
                     (std::memcmp(Nd.data(), Nh.data(), Nh.size() * sizeof(double)) == 0);
        }

        const bool ok = count_ok && axis_ok && qvn_ok;
        if (!ok) ++failures;
        std::fprintf(stderr,
            "[%-40s] M_kept host=%ld dev=%ld (%s)  axis=%s  Q/V/N bit-identical=%s  %s\n",
            tc.name, mk_host, mk_dev, count_ok ? "==" : "DIFF",
            axis_ok ? "y" : "N", qvn_ok ? "y" : "N", ok ? "PASS" : "FAIL");
    }

    std::fprintf(stderr, "\nRESULT: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
