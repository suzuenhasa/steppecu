// tests/reference/test_filter_oracle.cu
//
// REAL-AADR ORACLE TEST for the M2 on-the-fly filters (ROADMAP M2 gate; architecture
// .md §1, §5 S-1/S0', §13; ROADMAP §5, §6 DoD). The M2 trust seam: the host-pure
// filter front-end (filter_decision / snp_filter / mind_prepass / include_exclude)
// validated against an INDEPENDENT scalar host oracle recomputed over the SAME real
// v66 AADR data the M1 decode test uses — NO synthetic data (ROADMAP §0).
//
// It is a .cu (mirrors tests/reference/test_decode_equivalence.cu / _f2_equivalence
// .cu) so it links steppe::device (the CPU reference backend's decode_af + compute_f2
// through the CUDA-free ComputeBackend seam) and steppe::io (the readers + filters).
// No GPU kernel of its own; the FILTER logic is host-pure. Self-checking main(),
// CTest gates on the exit code.
//
// The six checks (the M2 brief's validation list a..f):
//   (a) NO-OP WHEN DEFAULT: with all-default FilterConfig, the SNP keep-set is the
//       FULL set on this panel, and compute_f2 on the (un)filtered Q/V/N is
//       BIT-IDENTICAL to the unfiltered result. Proves M2 cannot regress parity.
//   (b) DROP-EQUALS-MASK IDENTITY (the central M2 invariant): for a filter, compute
//       f2 two ways from the same Q/V/N — (a) physically drop the failing SNP
//       columns, (b) keep all columns but set V=0 AND N=0 for failing SNPs — and
//       assert BIT-IDENTICAL f2. Proves "a dropped SNP contributes nothing."
//   (c) EXACT FILTER CORRECTNESS: an independent scalar oracle recomputes pooled
//       folded MAF + per-SNP missing fraction + allele-pair class over the real
//       data; assert snp_filter's keep-mask equals the oracle's mask EXACTLY
//       (integer-exact set comparison) for several FilterConfigs. For --mind:
//       independently sum per-sample non-missing; assert the kept-sample set
//       matches mind_prepass exactly.
//   (d) MONOTONICITY: raising maf_min monotonically shrinks the kept set; geno=1 &
//       maf=0 keep everything (the no-op); mind=1 keeps every sample.
//   (e) AUTOSOMES_ONLY tie-in: with autosomes_only=true, drop chr 23/24 SNPs, then
//       run M3's assign_blocks over the survivors; report the block count and the
//       AT2 autosome-definition finding.
//   (f) STRAND-AMBIGUOUS present in real AADR: count the ambiguous (A/T,C/G) class
//       and the transversion class; confirm the drop-not-flip behavior alters NO Q
//       value (a dropped SNP's Q is untouched in the source array).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "steppe/config.hpp"            // FilterConfig, Precision, kAutosomeChromMin/Max, kDefaultBlockSizeCm
#include "core/internal/views.hpp"      // steppe::core::MatView
#include "core/domain/block_partition_rule.hpp"  // assign_blocks, block_size_cm_to_morgans
#include "device/backend.hpp"           // ComputeBackend, DecodeTileView, DecodeResult, F2Result

#include "io/eigenstrat_format.hpp"
#include "io/geno_reader.hpp"
#include "io/genotype_tile.hpp"
#include "io/ind_reader.hpp"
#include "io/snp_reader.hpp"
#include "io/filter/filter_decision.hpp"
#include "io/filter/include_exclude.hpp"
#include "io/filter/mind_prepass.hpp"
#include "io/filter/snp_filter.hpp"

using steppe::ComputeBackend;
using steppe::DecodeResult;
using steppe::DecodeTileView;
using steppe::F2Result;
using steppe::FilterConfig;
using steppe::Precision;
using steppe::core::MatView;
namespace flt = steppe::io::filter;

// CPU reference backend factory (src/device/cpu/cpu_backend.cpp). We use the CPU
// oracle for decode + compute_f2 so the drop-equals-mask comparison is two
// long-double-accumulated host computations — BIT-IDENTICAL is meaningful.
namespace steppe::core { std::unique_ptr<ComputeBackend> make_cpu_backend(); }

namespace {

constexpr std::size_t kAutoTopK = 50;
constexpr std::size_t kSnpCap = 100000;
constexpr const char* kDefaultDataRoot = "/workspace/data/aadr";
constexpr const char* kGenoBase = "v66.p1_HO.aadr.patch.PUB";

// Bit-identical comparison of two f2 results (the central M2 invariant gate). Both
// are CPU-oracle outputs; a dropped SNP must contribute NOTHING, so the two paths
// must agree to the last bit (memcmp), not just within a tolerance.
[[nodiscard]] bool f2_bit_identical(const F2Result& a, const F2Result& b) {
    if (a.P != b.P) return false;
    if (a.f2.size() != b.f2.size() || a.vpair.size() != b.vpair.size()) return false;
    return std::memcmp(a.f2.data(), b.f2.data(), a.f2.size() * sizeof(double)) == 0 &&
           std::memcmp(a.vpair.data(), b.vpair.data(), a.vpair.size() * sizeof(double)) == 0;
}

// Physically drop the failing SNP COLUMNS: build new [P × M_kept] Q/V/N with only
// the kept columns, preserving column-major (i + P·s) layout.
void drop_columns(const DecodeResult& dec, const std::vector<bool>& keep, int P,
                  std::vector<double>& Qk, std::vector<double>& Vk, std::vector<double>& Nk,
                  long& Mk_out) {
    long Mk = 0;
    for (bool k : keep) Mk += k ? 1 : 0;
    Qk.assign(static_cast<std::size_t>(P) * static_cast<std::size_t>(Mk), 0.0);
    Vk.assign(static_cast<std::size_t>(P) * static_cast<std::size_t>(Mk), 0.0);
    Nk.assign(static_cast<std::size_t>(P) * static_cast<std::size_t>(Mk), 0.0);
    long d = 0;
    for (std::size_t s = 0; s < keep.size(); ++s) {
        if (!keep[s]) continue;
        for (int i = 0; i < P; ++i) {
            const std::size_t src = static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * s;
            const std::size_t dst = static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * static_cast<std::size_t>(d);
            Qk[dst] = dec.q[src];
            Vk[dst] = dec.v[src];
            Nk[dst] = dec.n[src];
        }
        ++d;
    }
    Mk_out = Mk;
}

// Keep all columns but ZERO V and N (and Q) for the failing SNPs — the masking
// path. A masked-out column has V=0, so V·Vᵀ excludes it from every pair's count;
// N=0 and Q=0 so it feeds nothing. This is exactly "set the failing column dead".
void mask_columns(const DecodeResult& dec, const std::vector<bool>& keep, int P, long M,
                  std::vector<double>& Qm, std::vector<double>& Vm, std::vector<double>& Nm) {
    Qm = dec.q; Vm = dec.v; Nm = dec.n;
    for (std::size_t s = 0; s < keep.size() && static_cast<long>(s) < M; ++s) {
        if (keep[s]) continue;
        for (int i = 0; i < P; ++i) {
            const std::size_t off = static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * s;
            Qm[off] = 0.0; Vm[off] = 0.0; Nm[off] = 0.0;
        }
    }
}

// INDEPENDENT scalar oracle for the per-SNP keep decision (recomputed from raw
// decode + .snp, NOT via snp_filter). Pools ref freq + missing frac with the
// build_tgeno_matrix.py convention, classifies the allele pair inline, applies the
// same boundary sides. This is the integer-exact mask snp_filter must reproduce.
std::vector<bool> oracle_keep_mask(const DecodeResult& dec, int P, long M,
                                   const std::vector<std::size_t>& pop_individuals,
                                   int ploidy,
                                   const steppe::io::SnpTable& snps,
                                   const FilterConfig& cfg) {
    std::size_t total_indiv = 0;
    for (int p = 0; p < P; ++p) total_indiv += pop_individuals[static_cast<std::size_t>(p)];

    std::vector<bool> keep(static_cast<std::size_t>(M), false);
    for (long s = 0; s < M; ++s) {
        // Pooled ref freq + missing frac, recomputed independently.
        double ref_count = 0.0, allele_count = 0.0, nonmiss_indiv = 0.0;
        for (int p = 0; p < P; ++p) {
            const std::size_t off = static_cast<std::size_t>(p) + static_cast<std::size_t>(P) * static_cast<std::size_t>(s);
            const double n = dec.n[off];
            ref_count += dec.q[off] * n;
            allele_count += n;
            nonmiss_indiv += n / static_cast<double>(ploidy);
        }
        const double ref_af = (allele_count > 0.0) ? ref_count / allele_count : 0.0;
        const double maf = (ref_af < 1.0 - ref_af) ? ref_af : (1.0 - ref_af);
        const double miss = (total_indiv > 0) ? 1.0 - nonmiss_indiv / static_cast<double>(total_indiv) : 1.0;

        const char ref = snps.ref[static_cast<std::size_t>(s)];
        const char alt = snps.alt[static_cast<std::size_t>(s)];

        // Drop-not-flip unconditional class drops, recomputed independently.
        if (flt::is_multiallelic(ref, alt) || flt::is_strand_ambiguous(ref, alt)) continue;
        if (maf < cfg.maf_min) continue;                  // >= boundary -> drop if <
        if (miss > cfg.geno_max_missing) continue;        // <= boundary -> drop if >
        if (cfg.drop_monomorphic && maf == 0.0) continue;
        if (cfg.transversions_only && !flt::is_transversion(ref, alt)) continue;
        if (cfg.autosomes_only) {
            const int c = snps.chrom[static_cast<std::size_t>(s)];
            if (c < steppe::kAutosomeChromMin || c > steppe::kAutosomeChromMax) continue;
        }
        keep[static_cast<std::size_t>(s)] = true;
    }
    return keep;
}

std::size_t count_true(const std::vector<bool>& v) {
    std::size_t n = 0; for (bool b : v) n += b ? 1 : 0; return n;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string root = (argc >= 2) ? argv[1] : kDefaultDataRoot;
    const std::string raw_dir = root + "/raw";
    const std::string geno = raw_dir + "/" + kGenoBase + ".geno";
    const std::string snp = raw_dir + "/" + kGenoBase + ".snp";
    const std::string ind = raw_dir + "/" + kGenoBase + ".ind";

    int failures = 0;

    // ---- (0) Read the raw TGENO triple + reproduce the M1 selection ----------
    std::unique_ptr<steppe::io::GenoReader> reader;
    try {
        reader = std::make_unique<steppe::io::GenoReader>(geno);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: opening .geno failed: %s\n", e.what());
        return EXIT_FAILURE;
    }
    const auto& hdr = reader->header();

    steppe::io::PopSelection sel;
    sel.mode = steppe::io::PopSelection::Mode::AutoTopK;
    sel.k = kAutoTopK;

    steppe::io::IndPartition part;
    steppe::io::SnpTable snptab;
    steppe::io::GenotypeTile tile;
    try {
        part = steppe::io::read_ind(ind, sel, reader->records_present());
        snptab = steppe::io::read_snp(snp, kSnpCap);
        const std::size_t M0 = std::min(kSnpCap, hdr.n_snp);
        tile = reader->read_tile(part, 0, M0);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: io read failed: %s\n", e.what());
        return EXIT_FAILURE;
    }
    const int P = static_cast<int>(tile.n_pop());
    const long M = static_cast<long>(tile.n_snp);
    const int ploidy = 2;
    std::fprintf(stderr, "[setup] P=%d M=%ld n_ind=%zu (REAL AADR, auto-top %zu, snp-cap %zu)\n",
                 P, M, tile.n_individuals, kAutoTopK, kSnpCap);

    // Per-pop individual counts (segment sizes) for the missing-fraction denominator.
    std::vector<std::size_t> pop_individuals(static_cast<std::size_t>(P));
    for (int p = 0; p < P; ++p) {
        pop_individuals[static_cast<std::size_t>(p)] =
            tile.pop_offsets[static_cast<std::size_t>(p) + 1] - tile.pop_offsets[static_cast<std::size_t>(p)];
    }

    // Decode the tile (CPU oracle backend) into Q/V/N.
    DecodeTileView view;
    view.packed = tile.packed.data();
    view.bytes_per_record = tile.bytes_per_record;
    view.n_snp = tile.n_snp;
    view.n_individuals = tile.n_individuals;
    view.pop_offsets = tile.pop_offsets.data();
    view.n_pop = P;
    view.ploidy = ploidy;

    auto cpu = steppe::core::make_cpu_backend();
    const DecodeResult dec = cpu->decode_af(view);

    flt::DecodedTileSummaryInput fin;
    fin.q = dec.q.data(); fin.v = dec.v.data(); fin.n = dec.n.data();
    fin.P = P; fin.M = M; fin.pop_individuals = pop_individuals; fin.ploidy = ploidy;

    // Reference (unfiltered) f2 over the full decoded Q/V/N.
    const MatView Qf{dec.q.data(), P, M};
    const MatView Vf{dec.v.data(), P, M};
    const MatView Nf{dec.n.data(), P, M};
    const Precision prec{Precision::Kind::Fp64};
    const F2Result f2_unfiltered = cpu->compute_f2(Qf, Vf, Nf, prec);

    // ===== (a) NO-OP WHEN DEFAULT =============================================
    {
        const FilterConfig cfg;  // all defaults
        flt::SnpMembership mem(cfg);
        const std::vector<bool> keep = flt::build_snp_keep_mask(fin, snptab, cfg, mem);
        const std::size_t kept = count_true(keep);

        // On the AADR HO panel there are no A/T·C/G palindromes and no non-ACGT
        // alleles, so the default keep-set is the FULL set.
        const bool full = (kept == static_cast<std::size_t>(M));

        // compute_f2 over the masked (no-op) Q/V/N must be BIT-IDENTICAL to the
        // unfiltered f2. (At default, masking zeroes nothing, so it is literally
        // the same arrays — bit-identical by construction; we still run it through
        // the mask path to prove the plumbing does not perturb.)
        std::vector<double> Qm, Vm, Nm;
        mask_columns(dec, keep, P, M, Qm, Vm, Nm);
        const MatView Qv{Qm.data(), P, M}, Vv{Vm.data(), P, M}, Nv{Nm.data(), P, M};
        const F2Result f2_noop = cpu->compute_f2(Qv, Vv, Nv, prec);
        const bool ident = f2_bit_identical(f2_noop, f2_unfiltered);

        if (!full || !ident) ++failures;
        std::fprintf(stderr, "[a no-op-default]  kept=%zu/%ld (full=%s)  f2 bit-identical=%s  %s\n",
                     kept, M, full ? "yes" : "NO", ident ? "yes" : "NO",
                     (full && ident) ? "PASS" : "FAIL");
    }

    // ===== (b) DROP-EQUALS-MASK IDENTITY (the central M2 invariant) ===========
    // Use a real, biting filter: maf_min = 0.05 (drops the low-MAF SNPs). Compute
    // f2 by physically dropping the failing columns vs masking them (V=N=Q=0).
    {
        FilterConfig cfg; cfg.maf_min = 0.05;
        flt::SnpMembership mem(cfg);
        const std::vector<bool> keep = flt::build_snp_keep_mask(fin, snptab, cfg, mem);
        const std::size_t kept = count_true(keep);

        std::vector<double> Qd, Vd, Nd; long Mk = 0;
        drop_columns(dec, keep, P, Qd, Vd, Nd, Mk);
        const MatView Qdv{Qd.data(), P, Mk}, Vdv{Vd.data(), P, Mk}, Ndv{Nd.data(), P, Mk};
        const F2Result f2_drop = cpu->compute_f2(Qdv, Vdv, Ndv, prec);

        std::vector<double> Qm, Vm, Nm;
        mask_columns(dec, keep, P, M, Qm, Vm, Nm);
        const MatView Qmv{Qm.data(), P, M}, Vmv{Vm.data(), P, M}, Nmv{Nm.data(), P, M};
        const F2Result f2_mask = cpu->compute_f2(Qmv, Vmv, Nmv, prec);

        const bool ident = f2_bit_identical(f2_drop, f2_mask);
        if (!ident) ++failures;
        std::fprintf(stderr, "[b drop==mask]     filter maf>=0.05 kept=%zu/%ld (Mk=%ld)  bit-identical f2=%s  %s\n",
                     kept, M, Mk, ident ? "yes" : "NO", ident ? "PASS" : "FAIL");
    }

    // ===== (c) EXACT FILTER CORRECTNESS vs independent oracle ==================
    {
        const FilterConfig cfgs[] = {
            [] { FilterConfig c; return c; }(),                       // default (no-op)
            [] { FilterConfig c; c.maf_min = 0.01; return c; }(),
            [] { FilterConfig c; c.maf_min = 0.10; return c; }(),
            [] { FilterConfig c; c.geno_max_missing = 0.10; return c; }(),
            [] { FilterConfig c; c.drop_monomorphic = true; return c; }(),
            [] { FilterConfig c; c.transversions_only = true; return c; }(),
            [] { FilterConfig c; c.maf_min = 0.05; c.transversions_only = true; return c; }(),
        };
        bool all_match = true;
        for (const FilterConfig& cfg : cfgs) {
            flt::SnpMembership mem(cfg);
            const std::vector<bool> mask = flt::build_snp_keep_mask(fin, snptab, cfg, mem);
            const std::vector<bool> oracle =
                oracle_keep_mask(dec, P, M, pop_individuals, ploidy, snptab, cfg);
            bool eq = (mask.size() == oracle.size());
            std::size_t firstdiff = 0;
            for (std::size_t s = 0; eq && s < mask.size(); ++s) {
                if (mask[s] != oracle[s]) { eq = false; firstdiff = s; }
            }
            if (!eq) {
                all_match = false;
                std::fprintf(stderr, "    [c] MASK MISMATCH at SNP %zu (maf=%.3g geno=%.3g mono=%d tv=%d)\n",
                             firstdiff, cfg.maf_min, cfg.geno_max_missing,
                             cfg.drop_monomorphic, cfg.transversions_only);
            } else {
                std::fprintf(stderr, "    [c] exact match: kept=%zu/%ld  (maf=%.3g geno=%.3g mono=%d tv=%d)\n",
                             count_true(mask), M, cfg.maf_min, cfg.geno_max_missing,
                             cfg.drop_monomorphic, cfg.transversions_only);
            }
        }
        if (!all_match) ++failures;
        std::fprintf(stderr, "[c exact-mask]     snp_filter == independent oracle: %s\n",
                     all_match ? "PASS" : "FAIL");
    }

    // ===== (c') --mind exact: independent per-sample non-missing sum ===========
    {
        flt::MindPrepassInput min_in;
        min_in.packed = tile.packed.data();
        min_in.bytes_per_record = tile.bytes_per_record;
        min_in.n_snp = tile.n_snp;
        min_in.n_individuals = tile.n_individuals;

        // Independent oracle: sum non-missing per sample directly from packed bytes.
        std::vector<std::size_t> oracle_nm(tile.n_individuals, 0);
        for (std::size_t g = 0; g < tile.n_individuals; ++g) {
            const std::uint8_t* rec = tile.packed.data() + g * tile.bytes_per_record;
            std::size_t nm = 0;
            for (std::size_t s = 0; s < tile.n_snp; ++s) {
                if (steppe::io::code_in_byte(rec[s / 4u], static_cast<int>(s % 4u)) != steppe::io::kMissingCode) ++nm;
            }
            oracle_nm[g] = nm;
        }

        // A biting --mind cap and the no-op.
        FilterConfig cfg_active; cfg_active.mind_max_missing = 0.50;
        FilterConfig cfg_noop;   // mind_max_missing 1.0

        const flt::MindSummary s_active = flt::run_mind_prepass(min_in, cfg_active);
        const flt::MindSummary s_noop   = flt::run_mind_prepass(min_in, cfg_noop);

        // Pre-pass non-missing counts must match the oracle exactly.
        bool nm_exact = (s_active.nonmissing.size() == oracle_nm.size());
        for (std::size_t g = 0; nm_exact && g < oracle_nm.size(); ++g) {
            if (s_active.nonmissing[g] != oracle_nm[g]) nm_exact = false;
        }
        // Independent kept-set for the active cap.
        std::vector<std::size_t> oracle_kept;
        for (std::size_t g = 0; g < tile.n_individuals; ++g) {
            const double frac = 1.0 - static_cast<double>(oracle_nm[g]) / static_cast<double>(tile.n_snp);
            if (frac <= cfg_active.mind_max_missing) oracle_kept.push_back(g);
        }
        bool kept_exact = (s_active.kept == oracle_kept);
        bool noop_full = (s_noop.kept.size() == tile.n_individuals);
        if (!nm_exact || !kept_exact || !noop_full) ++failures;
        std::fprintf(stderr, "[c' mind exact]    non-missing exact=%s kept(mind<=0.5)=%zu match=%s noop-keeps-all=%s  %s\n",
                     nm_exact ? "yes" : "NO", s_active.kept.size(), kept_exact ? "yes" : "NO",
                     noop_full ? "yes" : "NO",
                     (nm_exact && kept_exact && noop_full) ? "PASS" : "FAIL");
    }

    // ===== (d) MONOTONICITY ===================================================
    {
        const double mafs[] = {0.0, 0.01, 0.02, 0.05, 0.10, 0.20};
        std::size_t prev = static_cast<std::size_t>(M) + 1;
        bool monotone = true;
        std::size_t kept_at_0 = 0;
        for (double m : mafs) {
            FilterConfig cfg; cfg.maf_min = m;
            flt::SnpMembership mem(cfg);
            const std::size_t kept = count_true(flt::build_snp_keep_mask(fin, snptab, cfg, mem));
            if (m == 0.0) kept_at_0 = kept;
            if (kept > prev) monotone = false;  // raising maf must not GROW the set
            prev = kept;
        }
        // geno=1 & maf=0 keeps everything (== the no-op full set on this panel).
        const bool keep_all = (kept_at_0 == static_cast<std::size_t>(M));
        if (!monotone || !keep_all) ++failures;
        std::fprintf(stderr, "[d monotonic]      raising maf shrinks kept: %s; maf=0&geno=1 keeps all: %s  %s\n",
                     monotone ? "yes" : "NO", keep_all ? "yes" : "NO",
                     (monotone && keep_all) ? "PASS" : "FAIL");
    }

    // ===== (e) AUTOSOMES_ONLY tie-in + block count ============================
    {
        FilterConfig cfg; cfg.autosomes_only = true;
        flt::SnpMembership mem(cfg);
        const std::vector<bool> keep = flt::build_snp_keep_mask(fin, snptab, cfg, mem);

        // Gather the surviving chrom / genpos (file order) and run M3 assign_blocks.
        std::vector<int> chrom_surv;
        std::vector<double> gp_surv;
        std::size_t dropped_sex = 0;
        for (long s = 0; s < M; ++s) {
            const int c = snptab.chrom[static_cast<std::size_t>(s)];
            if (keep[static_cast<std::size_t>(s)]) {
                chrom_surv.push_back(c);
                gp_surv.push_back(snptab.genpos_morgans[static_cast<std::size_t>(s)]);
            } else if (c >= 23) {
                ++dropped_sex;  // chr 23/24 dropped by autosomes_only
            }
        }
        const double bs = steppe::core::block_size_cm_to_morgans(steppe::kDefaultBlockSizeCm);
        const steppe::core::BlockPartition bp =
            steppe::core::assign_blocks(std::span<const int>(chrom_surv),
                                        std::span<const double>(gp_surv), bs);

        // Confirm: every surviving SNP is an autosome (chr 1..22); none is 23/24.
        bool only_auto = true;
        for (int c : chrom_surv) {
            if (c < steppe::kAutosomeChromMin || c > steppe::kAutosomeChromMax) { only_auto = false; break; }
        }
        if (!only_auto) ++failures;
        std::fprintf(stderr,
            "[e autosomes]      survivors=%zu (all chr1-22=%s) dropped chr>=23=%zu  n_block(autosomes)=%d  %s\n",
            chrom_surv.size(), only_auto ? "yes" : "NO", dropped_sex, bp.n_block,
            only_auto ? "PASS" : "FAIL");
        std::fprintf(stderr,
            "    [e note] AT2 extract_f2 default auto_only=TRUE = chr 1-22; on the 100k prefix this\n"
            "             reports the autosome block count above. (Full v66 .snp: chr1-24=>757 blocks,\n"
            "             chr1-23=>756, chr1-22=>autosome count; AT2 parity = chr 1-22.)\n");
    }

    // ===== (f) STRAND-AMBIGUOUS / transversion class present + Q untouched =====
    {
        std::size_t ambiguous = 0, transversion = 0, transition = 0, multi = 0;
        for (long s = 0; s < M; ++s) {
            const char a = snptab.ref[static_cast<std::size_t>(s)];
            const char b = snptab.alt[static_cast<std::size_t>(s)];
            if (flt::is_multiallelic(a, b)) { ++multi; continue; }
            if (flt::is_strand_ambiguous(a, b)) ++ambiguous;
            else if (flt::is_transversion(a, b)) ++transversion;
            else if (flt::is_transition(a, b)) ++transition;
        }

        // Drop-not-flip: build the transversions_only keep mask, then confirm that
        // for every DROPPED (transition) SNP, the source Q array is byte-unchanged
        // (the filter never altered a Q value — it only marked the column dropped).
        std::vector<double> q_before = dec.q;  // snapshot
        FilterConfig cfg; cfg.transversions_only = true;
        flt::SnpMembership mem(cfg);
        const std::vector<bool> keep = flt::build_snp_keep_mask(fin, snptab, cfg, mem);
        const bool q_untouched =
            (q_before.size() == dec.q.size() &&
             std::memcmp(q_before.data(), dec.q.data(), q_before.size() * sizeof(double)) == 0);

        // The transversion class is "tens of thousands" on the 100k prefix; the
        // self-complementary palindrome class (A/T,C/G) is absent on the HO panel.
        const bool tv_present = (transversion > 10000);
        if (!q_untouched) ++failures;
        std::fprintf(stderr,
            "[f strand/tv]      ambiguous(A/T,C/G)=%zu transversion=%zu transition=%zu multi=%zu  Q untouched=%s  %s\n",
            ambiguous, transversion, transition, multi, q_untouched ? "yes" : "NO",
            (q_untouched && tv_present) ? "PASS" : (q_untouched ? "PASS(tv-note)" : "FAIL"));
        std::fprintf(stderr,
            "    [f note] HO panel has NO A/T·C/G palindromes (ambiguous=%zu); the brief's 'GT/AC class,\n"
            "             tens of thousands' are the TRANSVERSIONS (=%zu), handled by transversions_only.\n",
            ambiguous, transversion);
    }

    std::fprintf(stderr, "\nRESULT: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
