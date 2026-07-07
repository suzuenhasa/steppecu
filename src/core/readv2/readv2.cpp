// src/core/readv2/readv2.cpp — run_readv2, the READv2 windowed-mismatch kinship driver.
//
// Host-pure and CUDA-free: it reaches the GPU only through the ComputeBackend seam
// (primary_backend), exactly like fstat_sweep.cpp. It streams SNP-tile chunks into a
// resident device bit-matrix, runs the all-pairs __popc mismatch sweep, forms the
// all-pairs MEDIAN background, and turns each surviving pair into a schema row handed
// to the streaming sink. The four device-resident per-pair scalars survive the sweep;
// the formatted rows never all materialize (they stream through `sink`).
//
// Precision: INTEGER popcount on device + a SINGLE native-FP64 normalization ratio
// (P0_mean / background) on the host — never emulated-FP64 (scope locked decision).
#include "steppe/readv2.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/internal/decode_af.hpp"
#include "core/internal/primary_backend.hpp"
#include "core/readv2/readv2_classify.hpp"
#include "core/stats/read_canonical_tile.hpp"
#include "device/backend.hpp"
#include "device/readv2_bitmatrix.hpp"
#include "device/resources.hpp"
#include "io/geno_reader.hpp"
#include "io/genotype_source.hpp"
#include "io/genotype_tile.hpp"
#include "io/ind_reader.hpp"
#include "io/ploidy_detect.hpp"

namespace steppe {

namespace {

namespace rc = core::readv2;

// C(N,2) with an overflow guard (returns 0 on N<2).
[[nodiscard]] long long pair_count(long long n) {
    if (n < 2) return 0;
    return n * (n - 1) / 2;
}

// Host mirror of readv2_unrank_pair: flat rank r = C(j,2)+i (i<j) -> (i, j).
void unrank_pair_host(long long r, int& i, int& j) {
    const double rd = static_cast<double>(r);
    long long jj = static_cast<long long>((1.0 + std::sqrt(1.0 + 8.0 * rd)) * 0.5);
    while (jj * (jj - 1) / 2 > r) --jj;
    while ((jj + 1) * jj / 2 <= r) ++jj;
    j = static_cast<int>(jj);
    i = static_cast<int>(r - jj * (jj - 1) / 2);
}

}  // namespace

Readv2Result run_readv2(const std::string& geno, const std::string& snp, const std::string& ind,
                        const io::IndPartition& individuals, const Readv2Options& opts,
                        const Readv2RowSink& sink, device::Resources& resources) {
    (void)ind;  // partition already resolved from .ind by the caller
    Readv2Result res;
    res.precision_tag = Precision::Kind::Fp64;

    const int N = static_cast<int>(individuals.groups.size());
    res.n_individuals = static_cast<std::size_t>(N);
    const long long n_pairs = pair_count(N);
    res.n_pairs = static_cast<std::size_t>(n_pairs);
    if (N < 2 || opts.window_snps <= 0) {
        res.status = Status::InvalidConfig;
        return res;
    }

    ComputeBackend& be = device::primary_backend(resources);

    io::GenoReader reader(geno);
    const io::GenoFormat fmt = reader.header().format;
    const io::SnpTable snptab = io::read_snp_table(fmt, snp, SIZE_MAX);
    long m0 = static_cast<long>(std::min(reader.header().n_snp, snptab.count));
    if (m0 <= 0) {
        res.status = Status::InvalidConfig;
        return res;
    }

    // Autosome restriction (READv2 convention; on by default). The canonical-tile read
    // below only supports a byte-aligned SNP prefix beginning at 0, so the SNP axis can
    // only be trimmed to a CONTIGUOUS leading run — this drops the trailing sex/other
    // chromosomes of the standard 1240K layout (chr1-22 then chr23/X, chr24/Y). A
    // general interspersed keep-mask needs a per-SNP mask threaded into the pack kernel
    // and window tiling (scope T3); until then, any autosome that appears AFTER a
    // non-autosome is a fail-fast reject rather than silent data loss.
    if (opts.autosomes_only) {
        if (static_cast<long>(snptab.chrom.size()) < m0) {
            throw std::invalid_argument(
                "readv2 --auto-only: .snp chromosome column is shorter than the genotype "
                "SNP axis; cannot apply the autosome restriction (use --no-auto-only to "
                "run on the full SNP set).");
        }
        long prefix = 0;
        while (prefix < m0 && snptab.chrom[static_cast<std::size_t>(prefix)] >= kAutosomeChromMin &&
               snptab.chrom[static_cast<std::size_t>(prefix)] <= kAutosomeChromMax) {
            ++prefix;
        }
        for (long k = prefix; k < m0; ++k) {
            const int c = snptab.chrom[static_cast<std::size_t>(k)];
            if (c >= kAutosomeChromMin && c <= kAutosomeChromMax) {
                throw std::invalid_argument(
                    "readv2 --auto-only: autosomal SNPs are interspersed with "
                    "non-autosomes in the .snp (an autosome follows a non-autosome at "
                    "index " + std::to_string(k) + "); the contiguous-prefix autosome "
                    "restriction cannot handle this layout. Sort the panel by chromosome, "
                    "pre-filter to autosomes, or pass --no-auto-only.");
            }
        }
        m0 = prefix;
        if (m0 <= 0) {
            res.status = Status::InvalidConfig;
            return res;
        }
    }
    const int window_snps = opts.window_snps;

    // Resident bit-matrix (zeroed at alloc so padding stays valid=0).
    device::Readv2Bitmatrix bits = be.readv2_alloc_bitmatrix(N, window_snps, m0);

    // Read the SNP axis as a SINGLE whole-genome tile (snp_begin == 0). The canonical-
    // tile readers only support a byte-aligned SNP prefix beginning at 0: TGENO
    // (io::GenoReader::read_tile) and the SNP-major gather (check_snp_major_range) both
    // hard-reject any nonzero snp_begin — the nonzero-begin "M5 tile loop" is not yet
    // implemented. So chunking the SNP axis would issue a second, rejected read on any
    // dataset larger than one chunk. For the fixed ~1.24M-SNP 1240K panel the whole SNP
    // axis per sample is small and bounded, and the device bit-matrix already holds all
    // m0 SNPs resident regardless, so one tile is both correct and memory-safe. The
    // loop below therefore runs exactly once (and stays a loop so it becomes true
    // SNP-tile streaming, scope T3, the moment the reader grows a nonzero-snp_begin
    // gather — each chunk would still be a whole number of windows).
    const long chunk_snps = m0;

    bool checked_ploidy = false;
    for (long snp0 = 0; snp0 < m0; snp0 += chunk_snps) {
        const long snp_end = std::min(snp0 + chunk_snps, m0);
        io::GenotypeTile tile = core::read_canonical_tile(
            reader, individuals, be, static_cast<std::size_t>(snp0),
            static_cast<std::size_t>(snp_end));

        // Ploidy gate on the FIRST streamed chunk (covers the 1000-SNP detect window):
        // hard-reject any diploid/het sample before enumeration (scope §3 / critic fix).
        if (!checked_ploidy) {
            const std::vector<int> ploidy = io::detect_sample_ploidy(tile);
            std::string offenders;
            int n_off = 0;
            for (std::size_t g = 0; g < ploidy.size(); ++g) {
                if (ploidy[g] != core::kPloidyPseudoHaploid) {
                    if (n_off < 8) {
                        if (!offenders.empty()) offenders += ", ";
                        offenders += (g < tile.pop_labels.size()) ? tile.pop_labels[g]
                                                                  : std::to_string(g);
                    }
                    ++n_off;
                }
            }
            if (n_off > 0) {
                // std::invalid_argument marks a fail-fast INPUT reject (the command maps
                // it to kExitInvalidConfig), distinct from a device runtime error.
                throw std::invalid_argument(
                    "sample(s) " + offenders + (n_off > 8 ? ", ..." : "") +
                    " are diploid/het; READv2 v1 requires pseudo-haploid hardcalls (the "
                    "single-allele-bit layout has no encoding for a het). Filter to "
                    "pseudo-haploid samples; the pseudo-haploid down-read is out of scope "
                    "for v1.");
            }
            checked_ploidy = true;
        }

        be.readv2_pack_chunk(bits, tile.packed.data(), tile.bytes_per_record, N, snp0,
                             snp_end - snp0);
    }

    // All-pairs windowed-mismatch reduction (the four per-pair scalars).
    const device::Readv2Pairs pairs = be.readv2_mismatch(bits, n_pairs, opts.tiled);

    // Pass 1: form P0_mean, apply the min-overlap gate, and collect the finite P0_mean
    // set the background reduces. A pair with n_windows==0 (or n_overlap==0) is excluded
    // from BOTH the background and the output (never a 0/0 NaN in the median; critic fix).
    const double min_overlap_sites = opts.min_overlap * static_cast<double>(m0);
    const std::size_t np = static_cast<std::size_t>(n_pairs);
    std::vector<char> keep(np, 0);
    std::vector<double> p0mean(np, 0.0);
    std::vector<double> finite;
    finite.reserve(np);
    for (std::size_t r = 0; r < np; ++r) {
        const int nw = pairs.n_win_used[r];
        const std::int64_t tc = pairs.tot_comp[r];
        if (nw <= 0 || tc <= 0) continue;
        if (static_cast<double>(tc) < min_overlap_sites) continue;
        const double pm = pairs.sum_p0[r] / static_cast<double>(nw);
        if (!std::isfinite(pm)) continue;
        p0mean[r] = pm;
        keep[r] = 1;
        finite.push_back(pm);
    }

    if (finite.empty()) {
        // No pair cleared the gates — nothing to normalize against; a clean empty run.
        res.background = std::nan("");
        res.status = Status::Ok;
        return res;
    }

    // Background = MEDIAN (default) or MEAN of the surviving pairs' P0_mean.
    double background = 0.0;
    if (opts.norm == Readv2Norm::Mean) {
        double sum = 0.0;
        for (double v : finite) sum += v;
        background = sum / static_cast<double>(finite.size());
    } else {
        std::sort(finite.begin(), finite.end());
        const std::size_t n = finite.size();
        background = (n % 2 == 1) ? finite[n / 2]
                                  : 0.5 * (finite[n / 2 - 1] + finite[n / 2]);
    }
    res.background = background;
    if (!(background > 0.0)) {
        // Degenerate (>half the pairs perfectly match) — impossible on real AADR; guard
        // it rather than emit inf.
        res.status = Status::InvalidConfig;
        return res;
    }

    // Pass 2: turn each surviving pair into a schema row and stream it to the sink.
    std::size_t n_emitted = 0;
    for (std::size_t r = 0; r < np; ++r) {
        if (!keep[r]) continue;
        int i = 0, j = 0;
        unrank_pair_host(static_cast<long long>(r), i, j);
        const int nw = pairs.n_win_used[r];
        const double pm = p0mean[r];
        const double p0norm = pm / background;
        Readv2PairRow row;
        row.i = i;
        row.j = j;
        row.n_windows = nw;
        row.n_overlap = pairs.tot_comp[r];
        row.p0_mean = pm;
        row.p0_norm = p0norm;
        row.degree = rc::degree_from_p0norm(p0norm);
        row.z = rc::readv2_z(pm, p0norm, background, nw, pairs.sum_p0_sq[r]);
        sink(row);
        ++n_emitted;
    }
    res.n_emitted = n_emitted;
    res.status = Status::Ok;
    return res;
}

}  // namespace steppe
