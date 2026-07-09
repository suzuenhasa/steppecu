// src/app/cmd_roh.cpp
//
// The `steppe roh` handler (hapROH runs-of-homozygosity detection). Reads a target
// ancient genotype triple + a phased reference-haplotype panel triple through the shipped
// EIGENSTRAT front-end (exactly as cmd_paint), intersects them onto a common site axis
// (with the strand/polarity contract), decodes the panel to donor-major haplotype bytes +
// the target to pseudo-haploid allele bytes, builds the panel allele-frequency + the per-
// SNP ROH transition, then per chromosome runs the block-per-target (K+1)-state forward-
// backward (GPU when visible, else the CPU reference oracle), calls ROH segments, and
// writes the per-segment table + the per-individual ROH summary — mirroring hapROH.
//
// Reference: docs/planning/haproh-face-spec.md
#include "app/cmd_roh.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "app/exit_code_for_caught.hpp"
#include "core/config/exit_code.hpp"
#include "core/internal/decode_af.hpp"
#include "core/stats/genotype_front_end.hpp"
#include "core/stats/li_stephens.hpp"
#include "core/stats/roh_model.hpp"
#include "core/stats/roh_segments.hpp"
#include "device/backend.hpp"
#include "device/backend_factory.hpp"
#include "io/genotype_source.hpp"
#include "io/ind_reader.hpp"
#include "io/snp_reader.hpp"
#include "steppe/config.hpp"

namespace steppe::app {

namespace cfg = steppe::config;

namespace {

// Keep every individual (each haploid column is a haplotype) — MinN with floor 1.
[[nodiscard]] io::PopSelection all_individuals() {
    io::PopSelection sel;
    sel.mode = io::PopSelection::Mode::MinN;
    sel.min_n = 1;
    return sel;
}

// Default upper bound on host-prep worker threads. The 256-core box's raw
// hardware_concurrency() would over-shard the (few-hundred-KB per block) prep loops; a modest
// cap keeps per-block work meaningful without hurting the O(N)/O(Mp*K) parallel speedup. Only
// an upper bound — the axis size and hardware_concurrency() clamp it further.
constexpr std::size_t kRohDefaultThreadCap = 64;

// Worker-thread cap for the host-prep parallel loops. Read once from STEPPE_ROH_THREADS (a
// hidden knob, no CLI surface): a positive integer pins the cap (STEPPE_ROH_THREADS=1 forces
// the fully-serial path; a large value forces max fan-out) — used by the parity harness to
// prove the output is invariant to the thread count. Unset/invalid -> kRohDefaultThreadCap.
[[nodiscard]] std::size_t roh_thread_cap() {
    const char* e = std::getenv("STEPPE_ROH_THREADS");
    if (e != nullptr && *e != '\0') {
        char* end = nullptr;
        const long v = std::strtol(e, &end, 10);
        if (end != e && v >= 1) return static_cast<std::size_t>(v);
    }
    return kRohDefaultThreadCap;
}

// Final clamped block/thread count for an axis of size n:
// T = max(1, min(hardware_concurrency(), STEPPE_ROH_THREADS cap, n)). Capping by n means a
// small axis never spawns more threads than there are blocks; hardware_concurrency()==0 (the
// "unknowable" answer) falls back to serial. Deterministic in n, so every caller that pre-sizes
// thread-local state with this count agrees with parallel_blocks' partition.
[[nodiscard]] std::size_t roh_parallel_T(std::size_t n) {
    if (n == 0) return 0;
    const unsigned hw = std::thread::hardware_concurrency();
    const std::size_t base = (hw == 0) ? std::size_t{1} : static_cast<std::size_t>(hw);
    std::size_t T = std::min<std::size_t>({base, roh_thread_cap(), n});
    return std::max<std::size_t>(std::size_t{1}, T);
}

// Partition [0, n) into T contiguous ascending blocks (T = roh_parallel_T(n)) and run
// fn(begin, end, block_index) on one std::thread per non-empty block, then join. T==1 (or a
// tiny axis) runs fn inline with no thread spawn. Blocks are contiguous and ascending, so a
// consumer that fills a disjoint output slice per block — or a thread-local buffer merged in
// ascending block order — reproduces the serial result BYTE-for-BYTE, independent of T.
//
// Every worker's body is wrapped so a throw (e.g. a thread-local allocation's bad_alloc) is
// captured as an exception_ptr and rethrown after join — an uncaught throw crossing a
// std::thread boundary would call std::terminate. Mirrors the multi-device dispatch below.
template <typename Fn>
void parallel_blocks(std::size_t n, Fn&& fn) {
    const std::size_t T = roh_parallel_T(n);
    if (T <= 1) {
        if (n > 0) fn(std::size_t{0}, n, std::size_t{0});
        return;
    }
    const std::size_t block = (n + T - 1) / T;  // ceil, so T blocks cover [0, n)
    std::vector<std::thread> threads;
    threads.reserve(T);
    std::vector<std::exception_ptr> errs(T);
    for (std::size_t b = 0; b < T; ++b) {
        const std::size_t begin = b * block;
        if (begin >= n) break;  // ceil rounding can leave trailing blocks empty
        const std::size_t end = std::min(n, begin + block);
        threads.emplace_back([&fn, &errs, begin, end, b]() {
            try {
                fn(begin, end, b);
            } catch (...) {
                errs[b] = std::current_exception();
            }
        });
    }
    for (std::thread& th : threads) th.join();
    for (const std::exception_ptr& e : errs)
        if (e) std::rethrow_exception(e);
}

// Decode a canonical individual-major tile to a flat haplotype-major allele-byte buffer
// allele[g*M + l] in {0,1,kLsMissingAllele} (exactly cmd_paint::decode_haplotypes). Each
// individual (haploid column) is one target/donor haplotype. Parallelized over the individual
// (haplotype) axis: each block writes a disjoint set of rows [g0, g1), integer-only, so the
// output bytes are identical to the serial decode regardless of the block count.
[[nodiscard]] std::vector<std::uint8_t> decode_haplotypes(const io::GenotypeTile& tile, long M) {
    const std::size_t Ms = static_cast<std::size_t>(M);
    std::vector<std::uint8_t> out(tile.n_individuals * Ms, core::kLsMissingAllele);
    parallel_blocks(tile.n_individuals, [&](std::size_t g0, std::size_t g1, std::size_t) {
        for (std::size_t g = g0; g < g1; ++g) {
            const std::uint8_t* rec = tile.packed.data() + g * tile.bytes_per_record;
            for (long l = 0; l < M; ++l) {
                const std::size_t byte = static_cast<std::size_t>(l) / core::kCodesPerByte;
                const int pos = static_cast<int>(static_cast<std::size_t>(l) % core::kCodesPerByte);
                out[g * Ms + static_cast<std::size_t>(l)] =
                    core::haploid_allele_from_code(core::genotype_code(rec[byte], pos));
            }
        }
    });
    return out;
}

// Per-individual pop label, in tile (population-contiguous) order.
[[nodiscard]] std::vector<std::string> per_individual_labels(const io::GenotypeTile& tile) {
    std::vector<std::string> lab(tile.n_individuals, std::string("SAMPLE"));
    const std::size_t P = tile.n_pop();
    if (P > 0 && tile.pop_offsets.size() == P + 1) {
        for (std::size_t p = 0; p < P; ++p)
            for (std::size_t k = tile.pop_offsets[p]; k < tile.pop_offsets[p + 1]; ++k)
                if (k < lab.size()) lab[k] = tile.pop_labels[p];
    }
    return lab;
}

// Parse a "--device" ordinal list into a DeviceConfig (empty/"auto" -> default). A
// comma-separated list (e.g. "0,1") fills dc.devices with every ordinal — the multi-device
// dispatch reads more than one; the single path uses front(). Mirrors cmd_ingest.cpp.
[[nodiscard]] bool parse_device(const std::string& raw, steppe::DeviceConfig& dc, std::string& err) {
    std::string s;
    for (char c : raw)
        if (!std::isspace(static_cast<unsigned char>(c))) s += c;
    if (s.empty() || s == "auto") return true;
    std::vector<int> ords;
    std::size_t start = 0;
    while (start <= s.size()) {
        const std::size_t comma = s.find(',', start);
        const std::string tok =
            s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!tok.empty()) {
            try {
                ords.push_back(std::stoi(tok));
            } catch (...) {
                err = "--device ordinal '" + tok + "' is not an integer";
                return false;
            }
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    dc.devices = std::move(ords);
    return true;
}

[[nodiscard]] std::vector<double> parse_bins(const std::string& s) {
    std::vector<double> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        try {
            out.push_back(std::stod(tok));
        } catch (...) {
        }
    }
    return out;
}

[[nodiscard]] std::vector<std::string> read_list(const std::string& path) {
    std::vector<std::string> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ss(line);
        std::string tok;
        if (ss >> tok) out.push_back(tok);
    }
    return out;
}

}  // namespace

int run_roh(const RohArgs& args) {
    // --- flag validation -----------------------------------------------------
    if (args.prefix.empty()) {
        std::fprintf(stderr,
                     "steppe roh: --prefix PREFIX.{geno,snp,ind} (the TARGET ancient sample) is "
                     "required\n");
        return cfg::kExitInvalidConfig;
    }
    if (args.ref_panel.empty()) {
        std::fprintf(stderr,
                     "steppe roh: --ref-panel PREFIX.{geno,snp,ind} (the phased reference panel) "
                     "is required\n");
        return cfg::kExitInvalidConfig;
    }
    if (args.format != "tsv" && args.format != "csv") {
        std::fprintf(stderr, "steppe roh: --format must be tsv | csv (got '%s')\n",
                     args.format.c_str());
        return cfg::kExitInvalidConfig;
    }
    const char sep = (args.format == "csv") ? ',' : '\t';

    const std::vector<double> bins = parse_bins(args.roh_bins);
    if (bins.empty()) {
        std::fprintf(stderr, "steppe roh: --roh-bins must be a comma list of cM values\n");
        return cfg::kExitInvalidConfig;
    }

    core::RohParams pr;
    pr.e_rate = args.e_rate;
    pr.roh_in = args.roh_in;
    pr.roh_out = args.roh_out;
    pr.roh_jump = args.roh_jump;
    pr.in_val = args.in_val;
    pr.cutoff_post = args.cutoff_post;

    // Fail-fast: with --n-ref set, K = 2*n_ref reference haplotypes and the column-0
    // forward prior alpha_0(0) = 1 - K*in_val (roh_fb.hpp) turns NEGATIVE once
    // 2*n_ref*in_val >= 1 (n_ref >= 1/(2*in_val), ~5000 at the 1e-4 default). Reject up
    // front rather than emit hapsburg's silently-broken negative prior. The 1000G default
    // n_ref=2504 (K=5008, product 0.5008) is safely inside. Checked again below against the
    // ACTUAL selected K for the --n-ref 0 (all-panel) path.
    if (args.n_ref > 0 && !core::roh_prior_valid(2 * args.n_ref, pr.in_val)) {
        std::fprintf(stderr,
                     "steppe roh: --n-ref %d with --in-val %g makes the column-0 ROH prior "
                     "1 - 2*n_ref*in_val = %g negative (need 2*n_ref*in_val < 1, i.e. n_ref < "
                     "%g). Lower --n-ref or --in-val.\n",
                     args.n_ref, pr.in_val, 1.0 - 2.0 * static_cast<double>(args.n_ref) * pr.in_val,
                     1.0 / (2.0 * pr.in_val));
        return cfg::kExitInvalidConfig;
    }

    // --- optional per-stage host-prep wall-clock timers (STEPPE_ROH_TIMING set) --------
    // Proves the O(N) / O(Mp*K) decomposition on the box (read+transpose / panel decode /
    // target decode / intersect+p_freq) so the de-bottleneck scope is measured, not assumed.
    const bool roh_timing = std::getenv("STEPPE_ROH_TIMING") != nullptr;
    using RohClock = std::chrono::steady_clock;
    auto roh_tick = [&](const char* stage, RohClock::time_point from) {
        if (roh_timing)
            std::fprintf(stderr, "steppe roh [timing] %-18s %8.1f ms\n", stage,
                         std::chrono::duration<double, std::milli>(RohClock::now() - from).count());
    };

    // --- read target + panel front-ends (host CpuBackend as io/transpose oracle) ---
    core::GenotypeFrontEnd fe_t, fe_p;
    const RohClock::time_point t_read0 = RohClock::now();
    try {
        const io::GenotypeTriple tt = io::resolve_genotype_triple(args.prefix);
        const io::GenotypeTriple pp = io::resolve_genotype_triple(args.ref_panel);
        std::unique_ptr<ComputeBackend> be_io = device::make_cpu_backend();
        fe_t = core::read_genotype_front_end(tt.geno, tt.snp, tt.ind, all_individuals(), *be_io);
        fe_p = core::read_genotype_front_end(pp.geno, pp.snp, pp.ind, all_individuals(), *be_io);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe roh: input error: %s\n", e.what());
        return cfg::kExitIoError;
    }
    roh_tick("read+transpose", t_read0);

    const long Mt = static_cast<long>(fe_t.tile.n_snp);
    const long Mp = static_cast<long>(fe_p.tile.n_snp);
    if (Mt <= 0 || Mp <= 0) {
        std::fprintf(stderr, "steppe roh: empty target or panel (target M=%ld, panel M=%ld)\n", Mt,
                     Mp);
        return cfg::kExitInvalidConfig;
    }

    // --- select the panel haplotype columns (exclude-pops + n-ref cap) -------------
    const std::vector<std::string> panel_lab = per_individual_labels(fe_p.tile);
    std::unordered_set<std::string> excl;
    if (!args.exclude_pops.empty()) {
        std::stringstream ss(args.exclude_pops);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            // trim
            std::size_t a = tok.find_first_not_of(" \t");
            std::size_t b = tok.find_last_not_of(" \t");
            if (a != std::string::npos) excl.insert(tok.substr(a, b - a + 1));
        }
    }
    const int Kpanel_all = static_cast<int>(fe_p.tile.n_individuals);
    // K = 2*n_ref haplotype columns cap (each phased individual = 2 pseudo-haploid columns
    // in the staged panel). 0 = keep all.
    const int k_cap = (args.n_ref > 0) ? 2 * args.n_ref : Kpanel_all;
    std::vector<int> panel_cols;
    panel_cols.reserve(static_cast<std::size_t>(Kpanel_all));
    for (int k = 0; k < Kpanel_all && static_cast<int>(panel_cols.size()) < k_cap; ++k) {
        if (excl.count(panel_lab[static_cast<std::size_t>(k)])) continue;
        panel_cols.push_back(k);
    }
    const int K = static_cast<int>(panel_cols.size());
    if (K < 2) {
        std::fprintf(stderr,
                     "steppe roh: need >= 2 reference haplotype columns after exclude/n-ref (got "
                     "%d of %d)\n",
                     K, Kpanel_all);
        return cfg::kExitInvalidConfig;
    }
    // Same column-0 prior guard against the ACTUAL selected K (catches an oversized all-panel
    // run where --n-ref was left at 0 but the panel itself has >= 1/in_val haplotype columns).
    if (!core::roh_prior_valid(K, pr.in_val)) {
        std::fprintf(stderr,
                     "steppe roh: %d selected reference haplotypes with --in-val %g make the "
                     "column-0 ROH prior 1 - K*in_val = %g negative (need K*in_val < 1). Cap the "
                     "panel with --n-ref or raise --in-val.\n",
                     K, pr.in_val, 1.0 - static_cast<double>(K) * pr.in_val);
        return cfg::kExitInvalidConfig;
    }

    // --- select the target individuals (--samples pop-label subset, else all) ------
    const std::vector<std::string> tgt_lab = per_individual_labels(fe_t.tile);
    const int Ntgt_all = static_cast<int>(fe_t.tile.n_individuals);
    std::vector<int> tgt_cols;
    std::vector<std::string> tgt_iid;
    if (!args.samples.empty()) {
        std::unordered_set<std::string> want;
        for (const std::string& s : read_list(args.samples)) want.insert(s);
        for (int g = 0; g < Ntgt_all; ++g)
            if (want.count(tgt_lab[static_cast<std::size_t>(g)])) tgt_cols.push_back(g);
        if (tgt_cols.empty()) {
            std::fprintf(stderr, "steppe roh: no target individual matched --samples\n");
            return cfg::kExitInvalidConfig;
        }
    } else {
        for (int g = 0; g < Ntgt_all; ++g) tgt_cols.push_back(g);
    }
    for (int g : tgt_cols)
        tgt_iid.push_back(tgt_lab[static_cast<std::size_t>(g)] + ":" + std::to_string(g));
    const int Ntgt = static_cast<int>(tgt_cols.size());

    // --- decode the panel to haploid allele bytes ----------------------------------
    // F1: the panel decode stays full-genome (the intersect + p_freq read the whole panel by
    // (chrom,pos)); the TARGET decode is deferred to a COMPACT kept-site-only pass after the
    // intersect (below), since tgt_hap is consumed ONLY by make_cov/build over the kept sites.
    const RohClock::time_point t_pdec0 = RohClock::now();
    const std::vector<std::uint8_t> panel_hap = decode_haplotypes(fe_p.tile, Mp);
    roh_tick("panel decode", t_pdec0);

    // --- intersect target ∩ panel by (chrom, physpos); enforce the polarity contract,
    //     grouped by chromosome ------------------------------------------------------
    // Panel site index by (chrom, pos).
    std::unordered_map<long long, std::size_t> panel_by_key;
    panel_by_key.reserve(static_cast<std::size_t>(Mp) * 2);
    auto site_key = [](int ch, long long pos) -> long long {
        return static_cast<long long>(ch) * 1000000000LL + pos;
    };
    for (long l = 0; l < Mp; ++l) {
        const int ch = (static_cast<std::size_t>(l) < fe_p.snptab.chrom.size())
                           ? fe_p.snptab.chrom[static_cast<std::size_t>(l)]
                           : 0;
        const long long pos = static_cast<long long>(
            (static_cast<std::size_t>(l) < fe_p.snptab.physpos.size())
                ? fe_p.snptab.physpos[static_cast<std::size_t>(l)]
                : 0.0);
        panel_by_key.emplace(site_key(ch, pos), static_cast<std::size_t>(l));
    }

    struct KeptSite {
        int ch;
        double morgan;
        long long bp;
        double p_freq;
        std::size_t panel_l;   // panel site index
        std::size_t tgt_l;     // target site index (source row in the FULL target tile)
        bool flip;             // flip the target allele bit (consistent strand flip)
        std::size_t kept_row;  // F1: compact row of this site in the kept-only target decode
    };
    std::unordered_map<int, std::vector<KeptSite>> by_chrom;
    long long n_drop_nomatch = 0, n_drop_polarity = 0;
    const std::size_t Mps = static_cast<std::size_t>(Mp);
    const std::size_t Mts = static_cast<std::size_t>(Mt);

    // Pure per-site classifier: reads only const state (both snptabs, panel_by_key, panel_hap,
    // panel_cols, K) and either produces a KeptSite (with the SAME serial kk-loop p_freq, no
    // reassociation) or reports why the target site is dropped. No shared mutable state, so it
    // is safe to call from any worker; called in ascending lt so a contiguous lt block yields
    // KeptSites in ascending-lt order.
    enum class SiteOutcome { Keep, DropNoMatch, DropPolarity };
    auto classify_site = [&](long lt, KeptSite& out) -> SiteOutcome {
        const std::size_t lts = static_cast<std::size_t>(lt);
        const int ch = (lts < fe_t.snptab.chrom.size()) ? fe_t.snptab.chrom[lts] : 0;
        const long long pos = static_cast<long long>(
            (lts < fe_t.snptab.physpos.size()) ? fe_t.snptab.physpos[lts] : 0.0);
        const auto it = panel_by_key.find(site_key(ch, pos));
        if (it == panel_by_key.end()) return SiteOutcome::DropNoMatch;
        const std::size_t lp = it->second;
        // Polarity: REF/ALT must match or be a consistent flip; else drop.
        bool flip = false;
        const bool have_ta = lts < fe_t.snptab.ref.size() && lts < fe_t.snptab.alt.size();
        const bool have_pa = lp < fe_p.snptab.ref.size() && lp < fe_p.snptab.alt.size();
        if (have_ta && have_pa) {
            const char tR = fe_t.snptab.ref[lts], tA = fe_t.snptab.alt[lts];
            const char pR = fe_p.snptab.ref[lp], pA = fe_p.snptab.alt[lp];
            if (tR == pR && tA == pA) {
                flip = false;
            } else if (tR == pA && tA == pR) {
                flip = true;
            } else {
                return SiteOutcome::DropPolarity;
            }
        }
        // Panel allele frequency over the SELECTED K haplotype columns (non-missing).
        double sum = 0.0;
        int cnt = 0;
        for (int kk = 0; kk < K; ++kk) {
            const std::uint8_t b = panel_hap[static_cast<std::size_t>(panel_cols[static_cast<std::size_t>(kk)]) * Mps + lp];
            if (b <= 1u) {
                sum += static_cast<double>(b);
                ++cnt;
            }
        }
        const double p_freq = (cnt > 0) ? sum / static_cast<double>(cnt) : 0.5;
        const double morgan = (lp < fe_p.snptab.genpos_morgans.size())
                                  ? fe_p.snptab.genpos_morgans[lp]
                                  : 0.0;
        out = KeptSite{ch, morgan, pos, p_freq, lp, lts, flip, /*kept_row=*/0};
        return SiteOutcome::Keep;
    };

    // Parallel intersect + p_freq over the target sites, in T contiguous ascending lt blocks.
    // Each block writes its own thread-local per-chrom KeptSite vectors + drop counters — no
    // shared writes — then a SERIAL ascending-block merge appends each chrom's block-local
    // vector onto by_chrom. Because block b covers a contiguous ascending lt range iterated
    // ascending, and blocks merge in ascending b order, by_chrom[ch] receives KeptSites in
    // EXACTLY the serial ascending-lt push_back order — the subsequent morgan-sort input, and
    // thus every FB per-item input, is byte-identical and independent of the thread count.
    const RohClock::time_point t_isect0 = RohClock::now();
    {
        const std::size_t T = roh_parallel_T(Mts);
        const std::size_t slots = std::max<std::size_t>(T, std::size_t{1});
        std::vector<std::unordered_map<int, std::vector<KeptSite>>> local_by_chrom(slots);
        std::vector<long long> local_nomatch(slots, 0);
        std::vector<long long> local_polarity(slots, 0);
        parallel_blocks(Mts, [&](std::size_t lt0, std::size_t lt1, std::size_t b) {
            std::unordered_map<int, std::vector<KeptSite>>& lbc = local_by_chrom[b];
            long long nm = 0, npol = 0;
            KeptSite ks;
            for (std::size_t lt = lt0; lt < lt1; ++lt) {
                switch (classify_site(static_cast<long>(lt), ks)) {
                    case SiteOutcome::Keep:
                        lbc[ks.ch].push_back(ks);
                        break;
                    case SiteOutcome::DropNoMatch:
                        ++nm;
                        break;
                    case SiteOutcome::DropPolarity:
                        ++npol;
                        break;
                }
            }
            local_nomatch[b] = nm;
            local_polarity[b] = npol;
        });
        // Ordered merge (serial, ascending block index). Iterating a block's map over its chrom
        // keys in arbitrary order is safe: each chrom's vector is appended to its OWN by_chrom
        // slot, so no single chrom's ordering depends on the inter-chrom iteration order.
        for (std::size_t b = 0; b < slots; ++b) {
            for (auto& kv : local_by_chrom[b]) {
                std::vector<KeptSite>& dst = by_chrom[kv.first];
                dst.insert(dst.end(), std::make_move_iterator(kv.second.begin()),
                           std::make_move_iterator(kv.second.end()));
            }
            n_drop_nomatch += local_nomatch[b];
            n_drop_polarity += local_polarity[b];
        }
    }
    roh_tick("intersect+p_freq", t_isect0);

    long long n_kept = 0;
    for (auto& kv : by_chrom) n_kept += static_cast<long long>(kv.second.size());
    if (n_kept == 0) {
        std::fprintf(stderr,
                     "steppe roh: no target site intersected the panel (dropped: no_match=%lld "
                     "polarity=%lld)\n",
                     n_drop_nomatch, n_drop_polarity);
        return cfg::kExitInvalidConfig;
    }

    // --- F1: compact kept-site target decode ---------------------------------------
    // by_chrom membership is now final (the later morgan-sort only PERMUTES each chrom's vector;
    // every KeptSite carries its own kept_row, so the relabeling survives the sort). Assign each
    // kept site a dense compact row and record its source row in the FULL target tile, then
    // decode ONLY those n_kept rows (~76k chr3 sites) instead of the whole Mt genome (~1.23M).
    // Bit-identical to the old full decode_haplotypes: the SAME
    // haploid_allele_from_code(genotype_code(rec[byte], pos)) is applied at the SAME (g, tgt_l),
    // just written to the compact row instead of column tgt_l. Iteration order over by_chrom is
    // irrelevant — the compact buffer is a pure relabeling, and each consumer reads its own site
    // through s.kept_row.
    std::vector<std::size_t> kept_tgt_l;
    kept_tgt_l.reserve(static_cast<std::size_t>(n_kept));
    {
        std::size_t row = 0;
        for (auto& kv : by_chrom)
            for (KeptSite& s : kv.second) {
                s.kept_row = row++;
                kept_tgt_l.push_back(s.tgt_l);
            }
    }
    const std::size_t n_kept_rows = kept_tgt_l.size();

    // Columns = full Ntgt_all (fe_t.tile.n_individuals) so tcol (= tgt_cols[t]) indexing is
    // unchanged; rows = n_kept_rows (compact). Parallelized over the individual/haplotype axis
    // exactly like decode_haplotypes: each block writes disjoint rows [g0,g1), integer-only.
    const RohClock::time_point t_tdec0 = RohClock::now();
    std::vector<std::uint8_t> tgt_hap(static_cast<std::size_t>(fe_t.tile.n_individuals) * n_kept_rows,
                                      core::kLsMissingAllele);
    parallel_blocks(fe_t.tile.n_individuals, [&](std::size_t g0, std::size_t g1, std::size_t) {
        for (std::size_t g = g0; g < g1; ++g) {
            const std::uint8_t* rec = fe_t.tile.packed.data() + g * fe_t.tile.bytes_per_record;
            for (std::size_t r = 0; r < n_kept_rows; ++r) {
                const std::size_t l = kept_tgt_l[r];
                const std::size_t byte = l / core::kCodesPerByte;
                const int pos = static_cast<int>(l % core::kCodesPerByte);
                tgt_hap[g * n_kept_rows + r] =
                    core::haploid_allele_from_code(core::genotype_code(rec[byte], pos));
            }
        }
    });
    roh_tick("target decode", t_tdec0);

    // --- parse + validate the --device ordinal list --------------------------------
    steppe::DeviceConfig dc;
    {
        std::string derr;
        if (!parse_device(args.device, dc, derr)) {
            std::fprintf(stderr, "steppe roh: %s\n", derr.c_str());
            return cfg::kExitInvalidConfig;
        }
    }
    // Stable-dedup so a repeated ordinal (e.g. --device 0,0) collapses to a single device
    // (and never uploads the panel twice to the same GPU or double-counts a target).
    {
        std::vector<int> uniq;
        std::unordered_set<int> seen;
        for (int d : dc.devices)
            if (seen.insert(d).second) uniq.push_back(d);
        dc.devices = std::move(uniq);
    }
    const int n_visible = device::visible_device_count();
    // Ordinal range-validation applies ONLY on the GPU path. When no GPU is visible the run
    // falls back to the CpuBackend oracle, which ignores the ordinal entirely — so --device N
    // (any N) must stay reachable there exactly as before (CI / no-GPU box / tests).
    if (n_visible > 0) {
        for (int d : dc.devices) {
            if (d < 0 || d >= n_visible) {
                std::fprintf(stderr,
                             "steppe roh: --device ordinal %d is out of range (%d GPU%s visible)\n",
                             d, n_visible, n_visible == 1 ? "" : "s");
                return cfg::kExitInvalidConfig;
            }
        }
    }
    // Multi-device dispatch only when the user named >1 distinct ordinal AND >1 GPU is visible.
    // A single --device N (or the CPU-oracle fallback) stays on the single-device path below,
    // byte-identical to the pre-multi-GPU reference.
    const bool multi_device = (dc.devices.size() > 1) && (n_visible > 1);

    const steppe::Precision prec = steppe::Precision::fp64();

    // --- shared per-chromosome prep (identical inputs for BOTH device paths) --------
    std::vector<int> chroms;
    for (const auto& kv : by_chrom) chroms.push_back(kv.first);
    std::sort(chroms.begin(), chroms.end());
    // Pre-sort each chromosome's intersected sites by genetic position (shared over targets).
    for (int ch : chroms)
        std::sort(by_chrom[ch].begin(), by_chrom[ch].end(),
                  [](const KeptSite& a, const KeptSite& b) { return a.morgan < b.morgan; });
    const int nchrom = static_cast<int>(chroms.size());

    // Mmax bounds every item's M: an item's kept-site count never exceeds its chromosome's
    // intersected-site count. The SAME global Mmax is reused on every device (it is the wave
    // kernel's Mstride) — every chromosome runs on every device, so a per-device recompute
    // would coincide anyway; reuse the global to avoid ambiguity.
    long Mmax = 0;
    for (int ch : chroms)
        Mmax = std::max<long>(Mmax, static_cast<long>(by_chrom[ch].size()));

    // The target's non-missing covered sites on a chromosome (in the pre-sorted morgan order);
    // recomputed identically in the builder and the consumer (cheap O(sites-on-chrom)). Uses
    // by_chrom.at() (const lookup) so D device threads may call it concurrently without racing
    // the map — every ch is already a key, so no insertion/rehash ever occurs.
    auto make_cov = [&](int t, int ch) -> std::vector<const KeptSite*> {
        const std::size_t tcol = static_cast<std::size_t>(tgt_cols[static_cast<std::size_t>(t)]);
        const std::vector<KeptSite>& sr = by_chrom.at(ch);
        std::vector<const KeptSite*> cov;
        cov.reserve(sr.size());
        for (const KeptSite& s : sr) {
            const std::uint8_t a = tgt_hap[tcol * n_kept_rows + s.kept_row];
            if (a <= 1u) cov.push_back(&s);
        }
        return cov;
    };

    std::vector<core::RohSegment> all_segments;

    if (!multi_device) {
        // ============================ SINGLE-DEVICE PATH ============================
        // Byte-identical to the pre-multi-GPU reference binary. One backend (GPU front()
        // ordinal, or the CpuBackend oracle when no GPU is visible), one panel upload, one
        // roh_fb_batch over ALL Ntgt*nchrom items, in-order segment append to all_segments.
        std::unique_ptr<ComputeBackend> be;
        try {
            const int dev = dc.devices.empty() ? 0 : dc.devices.front();
            be = (n_visible > 0) ? device::make_cuda_backend(dev) : device::make_cpu_backend();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "steppe roh: device init failed: %s\n", e.what());
            return cfg::kExitIoError;
        }

        // Exact-hapROH parity path (only_calls=True, docs §3b option 1): run ONE target per FB
        // launch on the target's OWN covered (non-missing) sites, rebuilding the map/allele-
        // freq/transition on that subset — so a target-missing site is DROPPED, not emitted
        // uniform.
        //
        // The panel is uploaded ONCE as-is (Kpanel_all rows of Mp) with the selected-donor row
        // map panel_cols[] — the on-device gather reproduces panel_hap[panel_cols[k]*Mps +
        // panel_l] byte-for-byte (no second host copy). panel_hap and panel_cols outlive the
        // batch call.
        RohPanelHandle panel = be->roh_upload_panel(panel_hap.data(), Kpanel_all,
                                                    static_cast<long>(Mp), panel_cols.data(), K);

        // Phase P: fill work-item i's GPU inputs (ob with the polarity flip, the panel-column
        // site_map, allele-freq p, transition T). i = t*nchrom + ch-index, the SAME (t-outer,
        // ch-inner) order as the serial loop — so the consumer drains in the identical order.
        RohItemBuilder build = [&](long i, std::uint8_t* ob, int* site_map, double* p,
                                   double* T) -> long {
            const int t = static_cast<int>(i / nchrom);
            const int ch = chroms[static_cast<std::size_t>(i % nchrom)];
            const std::vector<const KeptSite*> cov = make_cov(t, ch);
            const long M = static_cast<long>(cov.size());
            if (M <= 0) return 0;  // empty item — skipped, exactly like the serial `continue`
            const std::size_t tcol =
                static_cast<std::size_t>(tgt_cols[static_cast<std::size_t>(t)]);
            std::vector<double> gpos(static_cast<std::size_t>(M), 0.0);
            for (long l = 0; l < M; ++l) {
                const KeptSite& s = *cov[static_cast<std::size_t>(l)];
                const std::size_t ls = static_cast<std::size_t>(l);
                p[ls] = s.p_freq;
                gpos[ls] = s.morgan;
                site_map[ls] = static_cast<int>(s.panel_l);
                std::uint8_t a = tgt_hap[tcol * n_kept_rows + s.kept_row];
                if (a <= 1u && s.flip) a = static_cast<std::uint8_t>(1u - a);
                ob[ls] = a;
            }
            const std::vector<double> Tv = core::roh_build_transition(gpos, pr, K);
            std::copy(Tv.begin(), Tv.end(), T);
            return M;
        };

        // Phase C: call ROH segments on item i's posterior, in STRICT item order — appended to
        // all_segments in the identical (t, ch) order as the serial path (byte-stable output).
        RohPosteriorConsumer consume = [&](long i, const double* p_roh, long M) {
            const int t = static_cast<int>(i / nchrom);
            const int ch = chroms[static_cast<std::size_t>(i % nchrom)];
            const std::vector<const KeptSite*> cov = make_cov(t, ch);
            std::vector<double> gpos(static_cast<std::size_t>(M), 0.0);
            std::vector<long long> bp(static_cast<std::size_t>(M), 0);
            for (long l = 0; l < M; ++l) {
                const KeptSite& s = *cov[static_cast<std::size_t>(l)];
                gpos[static_cast<std::size_t>(l)] = s.morgan;
                bp[static_cast<std::size_t>(l)] = s.bp;
            }
            std::vector<core::RohSegment> segs = core::roh_call_segments(
                gpos, bp, std::vector<double>(p_roh, p_roh + M), ch,
                tgt_iid[static_cast<std::size_t>(t)], pr);
            for (core::RohSegment& sg : segs) all_segments.push_back(std::move(sg));
        };

        try {
            be->roh_fb_batch(panel, static_cast<long>(Ntgt) * static_cast<long>(nchrom), Mmax,
                             pr.e_rate, pr.in_val, prec, build, consume);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "steppe roh: FB failed: %s\n", e.what());
            return exit_code_for_caught(e);
        }
    } else {
        // ============================ MULTI-DEVICE PATH =============================
        // In-process dispatch across dc.devices. DECODE-ONCE / PANEL-ONCE: everything above is
        // shared read-only state (never written after prep). Each device gets its OWN thread +
        // its OWN CudaBackend + its OWN device-resident panel copy (no P2P assumed), and runs
        // roh_fb_batch over a DISJOINT subset of targets. Each target is computed on exactly one
        // device; segments merge in global item order -> byte-identical to the single path.
        //
        // BYTE-IDENTITY CAVEAT: guaranteed only across IDENTICAL-architecture devices. The FB
        // reduction is a tree with no atomics (deterministic), but roh_call_segments hard-
        // thresholds the posterior (p_roh[l] > cutoff_post), so a single-ULP cross-arch delta
        // would flip a boundary. box5090b's 2x RTX 5090 are same sm_120/driver/toolkit ->
        // deterministic. A heterogeneous multi-GPU box is UNSUPPORTED for this path.
        const int D = static_cast<int>(dc.devices.size());

        // WORK PROXY M_i: the total FB column count target t drives = sum over chromosomes of
        // its non-missing covered-site count (== make_cov length summed) — the true work.
        std::vector<long> work(static_cast<std::size_t>(Ntgt), 0);
        for (int t = 0; t < Ntgt; ++t) {
            long m = 0;
            for (int ch : chroms) m += static_cast<long>(make_cov(t, ch).size());
            work[static_cast<std::size_t>(t)] = m;
        }

        // GREEDY LONGEST-PROCESSING-TIME: sort targets by work DESC (tie-break ascending index
        // for determinism), assign each to the least-loaded device. 4/3-optimal makespan -> both
        // GPUs finish together. Whole targets (all nchrom chrom-items) go to ONE device, which
        // is what makes the lock-free merge below correct.
        std::vector<int> order(static_cast<std::size_t>(Ntgt));
        for (int t = 0; t < Ntgt; ++t) order[static_cast<std::size_t>(t)] = t;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            const long wa = work[static_cast<std::size_t>(a)];
            const long wb = work[static_cast<std::size_t>(b)];
            if (wa != wb) return wa > wb;
            return a < b;
        });
        std::vector<std::vector<int>> Td(static_cast<std::size_t>(D));
        std::vector<long> load(static_cast<std::size_t>(D), 0);
        for (int t : order) {
            int best = 0;
            for (int d = 1; d < D; ++d)
                if (load[static_cast<std::size_t>(d)] < load[static_cast<std::size_t>(best)])
                    best = d;
            Td[static_cast<std::size_t>(best)].push_back(t);
            load[static_cast<std::size_t>(best)] += work[static_cast<std::size_t>(t)];
        }

        // ORDERED-MERGE buckets: gi = t*nchrom + ch-index is unique per (t, ch). Pre-sized to
        // Ntgt*nchrom BEFORE spawning and NEVER resized; each gi is written by exactly one
        // thread (its owning device), so the threads touch DISJOINT elements -> race-free, no
        // lock. An empty/target-missing item leaves its bucket default-empty (consume unfired),
        // exactly as the single path emits nothing for it.
        std::vector<std::vector<core::RohSegment>> seg_by_item(static_cast<std::size_t>(Ntgt) *
                                                               static_cast<std::size_t>(nchrom));

        std::vector<std::exception_ptr> errs(static_cast<std::size_t>(D));
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(D));
        for (int d = 0; d < D; ++d) {
            if (Td[static_cast<std::size_t>(d)].empty()) continue;  // nothing to do on device d
            threads.emplace_back([&, d]() {
                try {
                    // Construct the backend INSIDE the thread so the primary-context bind +
                    // workspace/stream alloc land on dev[d] on THIS host thread (CUDA's current
                    // device is per-thread; guard_device() rebinds it on every backend call).
                    const int dev = dc.devices[static_cast<std::size_t>(d)];
                    std::unique_ptr<ComputeBackend> be_d = device::make_cuda_backend(dev);
                    RohPanelHandle panel_d = be_d->roh_upload_panel(
                        panel_hap.data(), Kpanel_all, static_cast<long>(Mp), panel_cols.data(), K);

                    const std::vector<int>& mine = Td[static_cast<std::size_t>(d)];
                    const long Nlocal = static_cast<long>(mine.size()) * static_cast<long>(nchrom);

                    // Local item j -> (td = j/nchrom, ci = j%nchrom); global target t = mine[td].
                    // The per-item fill is byte-identical to the single path, just target-indexed
                    // through Td[d].
                    RohItemBuilder build_d = [&](long j, std::uint8_t* ob, int* site_map, double* p,
                                                 double* T) -> long {
                        const int td = static_cast<int>(j / nchrom);
                        const int ci = static_cast<int>(j % nchrom);
                        const int t = mine[static_cast<std::size_t>(td)];
                        const int ch = chroms[static_cast<std::size_t>(ci)];
                        const std::vector<const KeptSite*> cov = make_cov(t, ch);
                        const long M = static_cast<long>(cov.size());
                        if (M <= 0) return 0;
                        const std::size_t tcol =
                            static_cast<std::size_t>(tgt_cols[static_cast<std::size_t>(t)]);
                        std::vector<double> gpos(static_cast<std::size_t>(M), 0.0);
                        for (long l = 0; l < M; ++l) {
                            const KeptSite& s = *cov[static_cast<std::size_t>(l)];
                            const std::size_t ls = static_cast<std::size_t>(l);
                            p[ls] = s.p_freq;
                            gpos[ls] = s.morgan;
                            site_map[ls] = static_cast<int>(s.panel_l);
                            std::uint8_t a = tgt_hap[tcol * n_kept_rows + s.kept_row];
                            if (a <= 1u && s.flip) a = static_cast<std::uint8_t>(1u - a);
                            ob[ls] = a;
                        }
                        const std::vector<double> Tv = core::roh_build_transition(gpos, pr, K);
                        std::copy(Tv.begin(), Tv.end(), T);
                        return M;
                    };

                    // Recover the GLOBAL bucket gi = t*nchrom + ci and write this item's segments
                    // there. Strict per-device item order + disjoint gi across devices == the
                    // single path's ascending-i append, reconstructed by the global-order merge.
                    RohPosteriorConsumer consume_d = [&](long j, const double* p_roh, long M) {
                        const int td = static_cast<int>(j / nchrom);
                        const int ci = static_cast<int>(j % nchrom);
                        const int t = mine[static_cast<std::size_t>(td)];
                        const int ch = chroms[static_cast<std::size_t>(ci)];
                        const std::size_t gi =
                            static_cast<std::size_t>(t) * static_cast<std::size_t>(nchrom) +
                            static_cast<std::size_t>(ci);
                        const std::vector<const KeptSite*> cov = make_cov(t, ch);
                        std::vector<double> gpos(static_cast<std::size_t>(M), 0.0);
                        std::vector<long long> bp(static_cast<std::size_t>(M), 0);
                        for (long l = 0; l < M; ++l) {
                            const KeptSite& s = *cov[static_cast<std::size_t>(l)];
                            gpos[static_cast<std::size_t>(l)] = s.morgan;
                            bp[static_cast<std::size_t>(l)] = s.bp;
                        }
                        seg_by_item[gi] = core::roh_call_segments(
                            gpos, bp, std::vector<double>(p_roh, p_roh + M), ch,
                            tgt_iid[static_cast<std::size_t>(t)], pr);
                    };

                    be_d->roh_fb_batch(panel_d, Nlocal, Mmax, pr.e_rate, pr.in_val, prec, build_d,
                                       consume_d);
                } catch (...) {
                    errs[static_cast<std::size_t>(d)] = std::current_exception();
                }
            });
        }
        for (std::thread& th : threads) th.join();
        // Rethrow the first device error into the existing catch semantics.
        for (int d = 0; d < D; ++d) {
            if (errs[static_cast<std::size_t>(d)]) {
                try {
                    std::rethrow_exception(errs[static_cast<std::size_t>(d)]);
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "steppe roh: FB failed: %s\n", e.what());
                    return exit_code_for_caught(e);
                }
            }
        }
        // MERGE in GLOBAL item order -> reproduces the single-device ascending-i append exactly.
        for (std::vector<core::RohSegment>& bucket : seg_by_item)
            for (core::RohSegment& sg : bucket) all_segments.push_back(std::move(sg));
    }

    // Pass the full ordered target list so every target gets a summary row — a zero-ROH
    // individual (no called segments) emits an all-zeros row instead of being dropped.
    const std::vector<core::RohSummary> summ =
        core::roh_summarize(all_segments, bins, /*snp_cm=*/50.0, /*merge_gap_cm=*/0.5,
                            /*merge_min_len1_cm=*/2.0, /*merge_min_len2_cm=*/4.0, tgt_iid);

    // --- emit the per-segment table ------------------------------------------------
    auto write_segments = [&](std::ostream& os) {
        os << "iid" << sep << "ch" << sep << "Start" << sep << "End" << sep << "StartM" << sep
           << "EndM" << sep << "length" << sep << "lengthM" << sep << "lengthCM" << sep
           << "StartBP" << sep << "EndBP" << "\n";
        for (const core::RohSegment& s : all_segments) {
            os << s.iid << sep << s.ch << sep << s.start << sep << s.end << sep << s.startM << sep
               << s.endM << sep << s.length << sep << s.lengthM << sep << (s.lengthM * 100.0) << sep
               << s.startBP << sep << s.endBP << "\n";
        }
    };
    if (args.out.empty()) {
        write_segments(std::cout);
        std::cout.flush();
    } else {
        std::ofstream os(args.out, std::ios::trunc);
        if (!os) {
            std::fprintf(stderr, "steppe roh: cannot open --out: %s\n", args.out.c_str());
            return cfg::kExitIoError;
        }
        write_segments(os);
        if (!os) {
            std::fprintf(stderr, "steppe roh: write failed: %s\n", args.out.c_str());
            return cfg::kExitIoError;
        }
    }

    // --- emit the per-individual summary -------------------------------------------
    auto write_summary = [&](std::ostream& os) {
        os << "iid" << sep << "max_roh";
        for (double c : bins)
            os << sep << "sum_roh>" << static_cast<int>(c) << sep << "n_roh>" << static_cast<int>(c);
        os << "\n";
        for (const core::RohSummary& s : summ) {
            os << s.iid << sep << s.max_roh;
            for (std::size_t j = 0; j < bins.size(); ++j)
                os << sep << s.sum_cm[j] << sep << s.n_seg[j];
            os << "\n";
        }
    };
    std::string summary_path = args.summary;
    if (summary_path.empty() && !args.out.empty()) summary_path = args.out + ".summary";
    if (summary_path.empty()) {
        std::fprintf(stderr, "steppe roh: per-individual summary (%zu targets):\n", summ.size());
        write_summary(std::cerr);
    } else {
        std::ofstream os(summary_path, std::ios::trunc);
        if (!os) {
            std::fprintf(stderr, "steppe roh: cannot open summary: %s\n", summary_path.c_str());
            return cfg::kExitIoError;
        }
        write_summary(os);
    }

    std::fprintf(stderr,
                 "steppe roh: %d targets, K=%d ref haplotypes, %zu chromosomes | sites kept=%lld, "
                 "dropped: no_match=%lld polarity=%lld | segments=%zu\n",
                 Ntgt, K, chroms.size(), n_kept, n_drop_nomatch, n_drop_polarity,
                 all_segments.size());
    return cfg::kExitOk;
}

}  // namespace steppe::app
