// tests/reference/test_decode_equivalence.cu
//
// REFERENCE-EQUIVALENCE TEST for the M1 genotype decode → Q/V/N (ROADMAP M1 gate;
// architecture.md §13 reference-equivalence, §5 S0/S1, §12; ROADMAP §5, §6 DoD).
//
// This is the M1 trust seam: the GPU decode (`CudaBackend::decode_af`) diffed
// against the obviously-correct scalar CPU oracle (`CpuBackend::decode_af`), and
// BOTH diffed against the numpy oracle matrices Q.f64/V.f64/N.f64 in derived_acc
// — produced by the on-box build_tgeno_matrix.py from the SAME raw v66 TGENO
// triple via `--auto-top 50 --snp-cap 100000`. It is a CORRECTNESS test on REAL
// AADR data (no synthetic — ROADMAP §0).
//
// What it does (run with the data root as argv[1], default /workspace/data/aadr):
//   1. Reads the raw TGENO triple (.geno/.snp/.ind) via the `io` leaf readers and
//      reproduces derived_acc's selection: auto-top 50 populations + first 100k
//      SNPs in file order (the verified provenance — meta.json `pops` are exactly
//      the 50 largest populations, sorted; `snp-cap 100000`).
//   2. Decodes the gathered tile via BOTH backends (CPU reference + CUDA).
//   3. Asserts GPU decode == CPU-reference decode EXACTLY (zero diff — integer
//      accumulate + single FP64 divide ⇒ identical on both paths).
//   4. Asserts both == the numpy oracle Q.f64/V.f64/N.f64:
//        * N, V: EXACT (zero diff — integer-valued doubles).
//        * Q:    EXACT (max|Δ| == 0 is the goal/gate; integer AC/AN, one divide).
//   5. Property invariants: Q∈[0,1]; V∈{0,1}; N≥0; V==(N>0); N is a non-negative
//      INTEGER (the AT2 adjust_pseudohaploid N = Σ ploidy convention — pseudo-
//      haploid samples contribute 1, so N is no longer all-even).
//   6. End-to-end: feeds M1's decoded Q/V/N into compute_f2 (both backends) and
//      confirms the full S0→S2 chain reproduces the M0 f2 result on derived_acc
//      (CPU oracle vs GPU f2 within the M0 tight tier).
//   7. Records decode throughput (SNPs·samples/s) for the commit.
//
// Exits NONZERO on any failure (CTest gates on the exit code). Self-checking
// main() — NOT a GoogleTest TU (mirrors tests/reference/test_f2_equivalence.cu).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "steppe/config.hpp"            // steppe::Precision, kDefaultMantissaBits, kRelFloor, kAbsFloor
#include "core/internal/views.hpp"      // steppe::core::MatView
#include "device/backend.hpp"           // ComputeBackend, DecodeTileView, DecodeResult, F2Result
#include "device/backend_factory.hpp"   // steppe::device::make_cpu_backend / make_cuda_backend (X-9/B8)

#include "io/eigenstrat_format.hpp"
#include "io/geno_reader.hpp"
#include "io/genotype_tile.hpp"
#include "io/ind_reader.hpp"
#include "io/ploidy_detect.hpp"   // detect_sample_ploidy (AT2 adjust_pseudohaploid)
#include "io/snp_reader.hpp"

using steppe::ComputeBackend;
using steppe::DecodeResult;
using steppe::DecodeTileView;
using steppe::F2Result;
using steppe::Precision;
using steppe::core::MatView;

namespace {

// derived_acc provenance (verified): auto-top 50 pops, first 100k SNPs.
constexpr std::size_t kAutoTopK = 50;
constexpr std::size_t kSnpCap = 100000;

// Default data root on the box; argv[1] overrides. The raw TGENO triple lives in
// <root>/raw, the numpy oracle matrices in <root>/derived_acc.
constexpr const char* kDefaultDataRoot = "/workspace/data/aadr";
constexpr const char* kGenoBase = "v66.p1_HO.aadr.patch.PUB";

// f2 end-to-end tolerance: the M0 tight tier (architecture.md §12; ROADMAP §0).
constexpr double kTolNativeVsRef = 1e-9;

// ---------------------------------------------------------------------------
// SHARED BINARY FORMAT loader for the numpy oracle (mirrors
// test_f2_equivalence.cu): shape.txt "P M" + Q/V/N.f64 each P*M little-endian
// column-major [P × M] doubles. Loud on any shape/format mismatch.
// ---------------------------------------------------------------------------
void read_f64(const std::string& path, std::vector<double>& out, std::size_t count) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "ERROR: cannot open %s\n", path.c_str()); std::exit(EXIT_FAILURE); }
    out.resize(count);
    const std::size_t got = std::fread(out.data(), sizeof(double), count, f);
    std::size_t extra = 0;
    if (got == count) { double probe = 0.0; extra = std::fread(&probe, sizeof(double), 1, f); }
    std::fclose(f);
    if (got != count) {
        std::fprintf(stderr, "ERROR: %s has %zu doubles, expected %zu (P*M).\n",
                     path.c_str(), got, count);
        std::exit(EXIT_FAILURE);
    }
    if (extra != 0) {
        std::fprintf(stderr, "ERROR: %s has trailing data beyond %zu doubles.\n",
                     path.c_str(), count);
        std::exit(EXIT_FAILURE);
    }
}

void load_oracle(const std::string& dir, int& P_out, long& M_out,
                 std::vector<double>& Q, std::vector<double>& V, std::vector<double>& N) {
    const std::string shapePath = dir + "/shape.txt";
    FILE* sf = std::fopen(shapePath.c_str(), "r");
    if (!sf) { std::fprintf(stderr, "ERROR: cannot open %s\n", shapePath.c_str()); std::exit(EXIT_FAILURE); }
    int P = 0; long M = 0;
    if (std::fscanf(sf, "%d %ld", &P, &M) != 2) {
        std::fprintf(stderr, "ERROR: %s must contain 'P M'\n", shapePath.c_str());
        std::fclose(sf); std::exit(EXIT_FAILURE);
    }
    std::fclose(sf);
    const std::size_t count = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    read_f64(dir + "/Q.f64", Q, count);
    read_f64(dir + "/V.f64", V, count);
    read_f64(dir + "/N.f64", N, count);
    P_out = P; M_out = M;
}

// Max absolute element diff between two equal-length arrays.
double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1e300;
    double m = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) {
        const double d = std::fabs(a[k] - b[k]);
        if (d > m) m = d;
    }
    return m;
}

// Off-diagonal max relative error of an f2 candidate vs the f2 reference (the M0
// metric: skip near-zero refs at the floor, divide by max(|ref|, kAbsFloor)).
double f2_max_rel(const F2Result& cand, const F2Result& ref) {
    const int P = ref.P;
    double mx = 0.0;
    for (int j = 0; j < P; ++j) {
        for (int i = 0; i < P; ++i) {
            if (i == j) continue;
            const std::size_t off = static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * j;
            const double r = ref.f2[off];
            if (std::fabs(r) < steppe::kRelFloor) continue;
            const double rel = std::fabs(cand.f2[off] - r) / std::max(std::fabs(r), steppe::kAbsFloor);
            if (rel > mx) mx = rel;
        }
    }
    return mx;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string root = (argc >= 2) ? argv[1] : kDefaultDataRoot;
    const std::string raw_dir = root + "/raw";
    const std::string derived_dir = root + "/derived_acc";
    const std::string geno = raw_dir + "/" + kGenoBase + ".geno";
    const std::string snp = raw_dir + "/" + kGenoBase + ".snp";
    const std::string ind = raw_dir + "/" + kGenoBase + ".ind";

    int failures = 0;

    // ---- (0) Read the raw TGENO triple + reproduce derived_acc's selection ---
    std::unique_ptr<steppe::io::GenoReader> reader;
    try {
        reader = std::make_unique<steppe::io::GenoReader>(geno);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: opening .geno failed: %s\n", e.what());
        return EXIT_FAILURE;
    }
    const auto& hdr = reader->header();
    std::fprintf(stderr, "[geno] format=%s n_ind=%zu n_snp=%zu bytes/rec=%zu records_present=%zu\n",
                 (hdr.format == steppe::io::GenoFormat::Tgeno ? "TGENO" : "GENO"),
                 hdr.n_ind, hdr.n_snp, hdr.bytes_per_record, reader->records_present());

    steppe::io::PopSelection sel;
    sel.mode = steppe::io::PopSelection::Mode::AutoTopK;
    sel.k = kAutoTopK;

    steppe::io::IndPartition part;
    steppe::io::SnpTable snptab;
    try {
        part = steppe::io::read_ind(ind, sel, reader->records_present());
        snptab = steppe::io::read_snp(snp, kSnpCap);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: reading .ind/.snp failed: %s\n", e.what());
        return EXIT_FAILURE;
    }
    const std::size_t M = std::min(kSnpCap, hdr.n_snp);
    std::fprintf(stderr, "[select] auto-top %zu → P=%zu pops, M=%zu SNPs (snp file rows read=%zu)\n",
                 kAutoTopK, part.groups.size(), M, snptab.count);

    // Gather the packed tile for the selected populations over the SNP prefix.
    steppe::io::GenotypeTile tile;
    try {
        tile = reader->read_tile(part, 0, M);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: read_tile failed: %s\n", e.what());
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "[tile] n_individuals=%zu n_pop=%zu bytes/rec=%zu packed=%zu bytes\n",
                 tile.n_individuals, tile.n_pop(), tile.bytes_per_record, tile.packed.size());

    // PER-SAMPLE PLOIDY (AT2 adjust_pseudohaploid=TRUE; the f2 pseudo-haploid fix):
    // detect each gathered sample's ploidy from the tile (a het call in the leading
    // SNPs ⇒ diploid, none ⇒ pseudo-haploid). The oracle (build_tgeno_matrix.py,
    // --ploidy auto) uses the SAME per-sample detection + AT2 weighting, so the
    // numpy Q/V/N and BOTH backends agree exactly. The auto-top-50 set includes
    // ancient pops, so N is NOT all-even any more (PH samples contribute 1 to N).
    const std::vector<int> sample_ploidy = steppe::io::detect_sample_ploidy(tile);
    std::size_t n_ph = 0, n_dip = 0;
    for (int pl : sample_ploidy) (pl == 1 ? n_ph : n_dip)++;
    std::fprintf(stderr, "[ploidy] auto-detected %zu pseudo-haploid + %zu diploid samples\n",
                 n_ph, n_dip);

    DecodeTileView view;
    view.packed = tile.packed.data();
    view.bytes_per_record = tile.bytes_per_record;
    view.n_snp = tile.n_snp;
    view.n_individuals = tile.n_individuals;
    view.pop_offsets = tile.pop_offsets.data();
    view.n_pop = static_cast<int>(tile.n_pop());
    view.sample_ploidy = sample_ploidy.data();
    view.ploidy = 2;  // unused (sample_ploidy non-null); the diploid default

    const int P = view.n_pop;
    const std::size_t pm = static_cast<std::size_t>(P) * M;

    // ---- (1) Decode via both backends ---------------------------------------
    auto cpu = steppe::device::make_cpu_backend();
    auto gpu = steppe::device::make_cuda_backend();

    const DecodeResult dec_cpu = cpu->decode_af(view);

    // Throughput: time the GPU decode (warm one launch, then time one).
    (void)gpu->decode_af(view);  // warm-up (allocations / context)
    const auto t0 = std::chrono::high_resolution_clock::now();
    const DecodeResult dec_gpu = gpu->decode_af(view);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const double cells = static_cast<double>(M) * static_cast<double>(tile.n_individuals);
    std::fprintf(stderr, "[throughput] GPU decode %0.3f ms; %.3e SNP·sample/s (M=%zu × ind=%zu)\n",
                 secs * 1e3, cells / secs, M, tile.n_individuals);

    // ---- (2) GPU == CPU reference (EXACT) -----------------------------------
    const double dQ_gc = max_abs_diff(dec_gpu.q, dec_cpu.q);
    const double dV_gc = max_abs_diff(dec_gpu.v, dec_cpu.v);
    const double dN_gc = max_abs_diff(dec_gpu.n, dec_cpu.n);
    const bool gc_exact = (dQ_gc == 0.0 && dV_gc == 0.0 && dN_gc == 0.0);
    if (!gc_exact) ++failures;
    std::fprintf(stderr, "[gpu vs cpu]   max|Δ| Q=%.3e V=%.3e N=%.3e  %s\n",
                 dQ_gc, dV_gc, dN_gc, gc_exact ? "EXACT (PASS)" : "MISMATCH (FAIL)");

    // ---- (3) Both vs the numpy oracle ---------------------------------------
    int Po = 0; long Mo = 0;
    std::vector<double> oQ, oV, oN;
    load_oracle(derived_dir, Po, Mo, oQ, oV, oN);
    bool shape_ok = (Po == P && Mo == static_cast<long>(M));
    if (!shape_ok) {
        ++failures;
        std::fprintf(stderr, "[oracle] SHAPE MISMATCH: decoded P=%d M=%zu vs oracle P=%d M=%ld (FAIL)\n",
                     P, M, Po, Mo);
    } else {
        const double dQ = max_abs_diff(dec_gpu.q, oQ);
        const double dV = max_abs_diff(dec_gpu.v, oV);
        const double dN = max_abs_diff(dec_gpu.n, oN);
        const bool nv_exact = (dV == 0.0 && dN == 0.0);
        const bool q_exact = (dQ == 0.0);
        if (!nv_exact) ++failures;
        if (!q_exact) ++failures;  // Q exact is the gate (integer AC/AN + one divide)
        std::fprintf(stderr, "[vs oracle]    max|Δ| Q=%.3e V=%.3e N=%.3e  N/V %s, Q %s\n",
                     dQ, dV, dN,
                     nv_exact ? "EXACT (PASS)" : "NONZERO (FAIL)",
                     q_exact ? "EXACT (PASS)" : "NONZERO (FAIL)");
    }

    // ---- (4) Property invariants on the decoded Q/V/N ------------------------
    // N is Σ ploidy_i over non-missing samples (AT2 adjust_pseudohaploid), so it is a
    // non-negative INTEGER but NO LONGER always even — a pseudo-haploid sample
    // contributes 1 to N (the whole point of the pseudo-haploid fix). The pinned
    // invariant is therefore "N is a non-negative integer" (replacing the former
    // diploid-only "N even" check), plus the rest. N-integer guards against a stray
    // fractional N from a mis-applied per-sample weight.
    bool inv_q = true, inv_v = true, inv_n = true, inv_vn = true, inv_int = true;
    for (std::size_t k = 0; k < pm; ++k) {
        const double q = dec_gpu.q[k], v = dec_gpu.v[k], n = dec_gpu.n[k];
        if (q < 0.0 || q > 1.0) inv_q = false;
        if (v != 0.0 && v != 1.0) inv_v = false;
        if (n < 0.0) inv_n = false;
        if ((v != 0.0) != (n > 0.0)) inv_vn = false;
        if (n != std::floor(n)) inv_int = false;  // N is an integer haploid count
    }
    const bool inv_ok = inv_q && inv_v && inv_n && inv_vn && inv_int;
    if (!inv_ok) ++failures;
    std::fprintf(stderr, "[invariants]   Q∈[0,1]:%s V∈{0,1}:%s N≥0:%s V==(N>0):%s N-int:%s  %s\n",
                 inv_q ? "y" : "N", inv_v ? "y" : "N", inv_n ? "y" : "N",
                 inv_vn ? "y" : "N", inv_int ? "y" : "N", inv_ok ? "PASS" : "FAIL");

    // ---- (5) End-to-end: decoded Q/V/N → compute_f2 (full S0→S2) -------------
    // Feed the GPU-decoded contract into BOTH f2 backends and confirm the GPU f2
    // reproduces the CPU-oracle f2 within the M0 tight tier — i.e. the decode is a
    // drop-in replacement for the numpy producer in the existing f2 chain.
    if (shape_ok) {
        const MatView Qv{dec_gpu.q.data(), P, static_cast<long>(M)};
        const MatView Vv{dec_gpu.v.data(), P, static_cast<long>(M)};
        const MatView Nv{dec_gpu.n.data(), P, static_cast<long>(M)};
        const F2Result f2_ref = cpu->compute_f2(Qv, Vv, Nv, Precision{Precision::Kind::Fp64});
        const F2Result f2_gpu = gpu->compute_f2(Qv, Vv, Nv, Precision{Precision::Kind::Fp64});
        const double e2e_rel = f2_max_rel(f2_gpu, f2_ref);
        const bool e2e_ok = (e2e_rel < kTolNativeVsRef);
        if (!e2e_ok) ++failures;
        std::fprintf(stderr, "[end-to-end]   decoded Q/V/N → compute_f2: GPU vs CPU f2 maxRel=%.3e (tol %.0e) %s\n",
                     e2e_rel, kTolNativeVsRef, e2e_ok ? "PASS" : "FAIL");
    }

    std::fprintf(stderr, "\nRESULT: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
