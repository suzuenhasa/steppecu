// src/core/stats/bp_windows.hpp
//
// build_bp_windows: the host-side bp window layout for the `steppe fst --windowed` /
// `--pbs` selection scan. It reproduces scikit-allel's window semantics EXACTLY — the
// pairing of allel.stats.window.position_windows (edge generation) and windowed_statistic
// (site location by searchsorted) — so a steppe per-window ratio-of-averages Fst matches
// allel.windowed_weir_cockerham_fst window-for-window. Header-only, host-only, CUDA-free.
//
// The two allel functions this mirrors, term-for-term:
//
//   position_windows(pos, size, start, stop, step):
//       start defaults to pos[0], stop defaults to pos[-1], step defaults to size.
//       for window_start in range(start, stop, step):      # start, start+step, ... < stop
//           window_stop = window_start + size
//           if window_stop >= stop:  window_stop = stop; last = True     # END-WINDOW CLIP
//           else:                    window_stop -= 1
//           emit [window_start, window_stop]               # 1-based INCLUSIVE both ends
//           if last: break
//
//   windowed_statistic(pos, ..., windows):
//       i = searchsorted(pos, wstart, side='left')         # first idx with pos >= wstart
//       j = searchsorted(pos, wstop,  side='right')        # first idx with pos >  wstop
//       count = j - i                                       # ALL positions in [wstart,wstop]
//
// Windows are per-chromosome (chromosome-reset): the site axis is scanned in tile order and
// split into maximal runs of one chromosome value, each windowed independently with its own
// start = run-first physpos, stop = run-last physpos. This requires the standard sorted .snp
// layout (chromosomes contiguous, physpos non-decreasing within a chromosome); apply_snp_filter
// subsets the axis without reordering, so a filtered run preserves that layout.
#ifndef STEPPE_CORE_STATS_BP_WINDOWS_HPP
#define STEPPE_CORE_STATS_BP_WINDOWS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace steppe::core {

// One bp window: its chromosome, 1-based inclusive [start, end] bp edges (end already clipped
// to the chromosome's last physpos), and the GLOBAL site index half-open slice [lo, hi) it
// covers in tile order. n_snp = hi - lo counts ALL positions in [start, end] (validity is not
// applied here — an invalid site only zeroes its num/den contribution downstream, mirroring
// allel's nansum, never its position count).
struct BpWindow {
    int  chrom = 0;
    long start = 0;
    long end = 0;
    long lo = 0;   // first site index with physpos >= start  (searchsorted left)
    long hi = 0;   // first site index with physpos >  end     (searchsorted right)
};

// Build the per-chromosome bp windows for (size, step). `chrom` and `physpos` are the kept-order
// per-SNP chromosome and base-pair position (physpos is integer-valued but carried as double, as
// in SnpTable); they must be the same length. size and step must both be > 0 (step defaults to
// size at the call site when a bare SIZE is given). Positions are read as std::llround(physpos).
[[nodiscard]] inline std::vector<BpWindow> build_bp_windows(std::span<const int> chrom,
                                                            std::span<const double> physpos,
                                                            long size, long step) {
    std::vector<BpWindow> out;
    const std::size_t M = std::min(chrom.size(), physpos.size());
    if (M == 0 || size <= 0 || step <= 0) return out;

    std::size_t i = 0;
    while (i < M) {
        // Maximal run [i, j) of a single chromosome value (chromosome-reset).
        const int c = chrom[i];
        std::size_t j = i;
        while (j < M && chrom[j] == c) ++j;

        const long start = std::llround(physpos[i]);
        const long stop = std::llround(physpos[j - 1]);

        // range(start, stop, step): window starts start, start+step, ... strictly < stop. A
        // single-position run (start >= stop) yields no windows, exactly as allel's empty range.
        for (long ws = start; ws < stop; ws += step) {
            long we = ws + size;
            bool last = false;
            if (we >= stop) {
                we = stop;        // END-WINDOW CLIP: the last window's end is clamped to stop.
                last = true;
            } else {
                we = ws + size - 1;
            }

            // searchsorted within the chromosome run: lo = first pos >= ws (left), hi = first
            // pos > we (right). physpos is non-decreasing across [i, j), integer-valued, so the
            // exact-double comparisons are exact.
            const double* base = physpos.data();
            const double* run_lo = base + i;
            const double* run_hi = base + j;
            const double* p_lo = std::lower_bound(run_lo, run_hi, static_cast<double>(ws));
            const double* p_hi = std::upper_bound(run_lo, run_hi, static_cast<double>(we));

            BpWindow w;
            w.chrom = c;
            w.start = ws;
            w.end = we;
            w.lo = static_cast<long>(p_lo - base);
            w.hi = static_cast<long>(p_hi - base);
            out.push_back(w);

            if (last) break;
        }

        i = j;
    }
    return out;
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_STATS_BP_WINDOWS_HPP
