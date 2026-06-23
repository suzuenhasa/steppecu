// include/steppe/fstat_sweep.hpp
//
// PUBLIC, CUDA-FREE entry point for the GPU-ONLY all-combinations f-stat SWEEP — the
// production-scale sibling of run_f4 / run_f3. Instead of an explicit caller-supplied quartet
// (triple) list, a sweep enumerates EVERY C(P,4) quartet (C(P,3) triple) over a population
// set, computes its f4 (f3) point estimate + diagonal jackknife SE, FILTERS by |z| (or top-K)
// ON THE GPU, and returns ONLY the survivors. The full [N × stat] table is NEVER materialized
// (a multi-TB dump at sweep scale): the filter + a CUB stream-compaction run ON THE DEVICE, so
// only the small survivor set crosses the CUDA-free seam.
//
// THE GPU-ONLY PIPELINE (the fix for the CPU-bound host-enumeration disaster; design-verified
// against CUDA 13.x docs): per chunk of the lex index range — (1) an on-device UNRANK kernel
// maps thread t -> its quartet (combinatorial number system), writing the device quartet list
// the EXISTING batched assemble_f4_quartets gather reads (NO host enumeration); (2) the SAME
// gather + loo/total + xtau + diagonal-jackknife device kernels run verbatim (emulated-FP64
// policy inherited, native carve-out for the cancellation-sensitive combine); (3) an on-device
// |z| FILTER flags survivors; (4) cub::DeviceSelect::Flagged stream-compacts them on-device;
// (5) ONLY the compacted survivors D2H. The HOST drives ONLY the chunk loop + receives
// survivors — NO host enumeration, NO host filter, NO host per-item loop.
//
// CAP: a maxcomb cap (kFstatMaxComb, config.hpp) fires BEFORE any compute — a sweep enumerating
// more than the cap REFUSES unless `sure` is set (it guards compute TIME, since every item is
// computed to test the filter). FILTER: --min-z (on-device |z|>=min_z) or --top-k (the device
// keeps every item, the host ranks the compacted set to K). MULTI-GPU PARKED: single-GPU.
#ifndef STEPPE_FSTAT_SWEEP_HPP
#define STEPPE_FSTAT_SWEEP_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "steppe/config.hpp"  // Precision
#include "steppe/error.hpp"   // Status
#include "steppe/qpadm.hpp"   // device::DeviceF2Blocks / device::Resources fwd-decls

namespace steppe {

/// Which survivor filter the sweep applies on-device.
enum class SweepFilter {
    MinZ,  ///< keep items with |z| >= min_z (the on-device flag; the multi-TB-safe default).
    TopK,  ///< keep the K items with the largest |z| (device keeps all; host ranks to K).
};

/// One sweep request. The arity (k) is fixed by the entry point (run_f4_sweep ⇒ 4, run_f3_sweep
/// ⇒ 3). `pop_subset` (optional) restricts the sweep to a subset of the f2 P-axis indices; empty
/// ⇒ sweep the whole [0,P). `sure` lifts the maxcomb cap.
struct SweepRequest {
    SweepFilter filter = SweepFilter::MinZ;
    double min_z = 3.0;          ///< |z| threshold for MinZ (AT2-style significance cut).
    std::size_t top_k = 100;     ///< K for TopK.
    std::vector<int> pop_subset; ///< optional subset of f2 indices to sweep; empty ⇒ all P.
    bool sure = false;           ///< lift the maxcomb cap (the AT2 `sure` analogue).
};

/// Sweep result: ONLY the survivors (in no guaranteed order beyond the per-chunk lex order;
/// TopK is sorted by |z| descending). keys[r] = {p1,p2,p3,p4} P-axis indices of survivor r
/// (the 4th is 0/unused for f3). `enumerated` is the total C(P,k) the sweep WOULD enumerate
/// (reported even when capped). `capped` ⇒ the sweep refused (enumerated > kFstatMaxComb &&
/// !sure); status == InvalidConfig, no compute ran.
struct SweepResult {
    std::vector<std::array<int, 4>> keys;  ///< survivor P-axis index tuples (p4 unused for f3).
    std::vector<double> est;               ///< f4/f3 point estimate per survivor.
    std::vector<double> se;                ///< diagonal jackknife SE.
    std::vector<double> z;                 ///< est / se.
    std::vector<double> p;                 ///< 2*pnorm_upper(|z|) (two-sided normal).

    std::size_t enumerated = 0;  ///< total C(P,k) the sweep would enumerate.
    std::size_t survivors = 0;   ///< number of kept rows (== keys.size()).
    bool capped = false;         ///< refused by the maxcomb cap (no compute ran).

    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

/// GPU-ONLY f4 sweep over DEVICE-RESIDENT f2: enumerate every C(P,4) quartet (or C(|subset|,4)),
/// compute f4 + diagonal-jackknife SE + |z| ON THE GPU, filter + compact on-device, return only
/// survivors. Routes through resources.gpus[0].backend (the CUDA backend's f4_sweep virtual).
[[nodiscard]] SweepResult run_f4_sweep(const device::DeviceF2Blocks& f2,
                                       const SweepRequest& req,
                                       device::Resources& resources);

/// GPU-ONLY f3 sweep — the three-slab sibling of run_f4_sweep (every C(P,3) triple).
[[nodiscard]] SweepResult run_f3_sweep(const device::DeviceF2Blocks& f2,
                                       const SweepRequest& req,
                                       device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_FSTAT_SWEEP_HPP
