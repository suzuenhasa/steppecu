// src/core/qpadm/model_search.cpp — the S8 ROTATION orchestrator + the base
// ComputeBackend::fit_models_batched default body.

#include "core/qpadm/model_search.hpp"

#include <cstddef>
#include <exception>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include "core/qpadm/model_search_core.hpp"  // plan_model_shards, ModelShard
#include "core/qpadm/qpadm_bounds.hpp"       // model_fits_small_path — the SINGLE-SOURCE kQpMax* envelope
#include "core/qpadm/qpadm_fit.hpp"          // run_impl, left_with_target
#include "device/backend.hpp"                // ComputeBackend
#include "device/device_f2_blocks.hpp"       // DeviceF2Blocks, upload_f2_blocks_to_device
#include "device/resources.hpp"              // device::Resources
#include "steppe/config.hpp"                 // Precision, kDefaultMantissaBits
#include "steppe/error.hpp"                  // Status
#include "steppe/fstats.hpp"                 // F2BlockTensor

namespace steppe::core::qpadm {

QpAdmResult fit_one_model_device(ComputeBackend& be,
                                 const device::DeviceF2Blocks& f2,
                                 const QpAdmModel& model,
                                 const QpAdmOptions& opts) {
    // Unified default precision (= the f2 default; fit-engine.md §1.4). assemble_f4
    // stays native by carve-out; the covariance SYRK engages this (emulated{40}
    // default, auto-native fallback) inside jackknife_cov; SVD/Qinv/chi^2 native.
    const Precision prec{Precision::Kind::EmulatedFp64, steppe::kDefaultMantissaBits};
    const std::vector<int> left_idx = left_with_target(model);
    // S3 — device-resident assemble (zero D2H of the tensor on the CUDA path; the
    // CpuBackend oracle reads host memory). NOTE: assemble_f4 caches tot_line_ as a
    // per-backend member consumed by jackknife_cov inside run_impl, so the assemble
    // and the run_impl must be adjacent on the SAME backend (they are — this whole
    // function runs on one backend instance, one model at a time; the search loop
    // calls it sequentially per device worker).
    F4Blocks X = be.assemble_f4(f2, std::span<const int>(left_idx),
                                std::span<const int>(model.right), prec);
    QpAdmResult res = run_impl(be, std::move(X),
                               std::span<const int>(f2.block_sizes), model, opts);
    res.model_index = model.model_index;  // echo the caller's stable identity
    return res;
}

// ---- The per-model DEFAULT body (the LARGE/tail path + the CpuBackend oracle) ----
// Used for (a) the >32 / large tail of the rotation (one device dispatch per model is
// correct for the tail, design §5) and (b) the CpuBackend parity oracle (a host loop
// is the CORRECT shape for the oracle, NOT the deliverable). The SMALL-path rotation
// common case does NOT come here — run_qpadm_search routes it to the device-BATCHED
// virtual be.fit_models_batched. Each model is fit WHOLLY through `be`'s own device
// virtuals (assemble_f4 reading the resident f2 → run_impl). Domain outcomes ride in
// results[i].status, NEVER a throw.
std::vector<QpAdmResult> fit_models_batched_default(
    ComputeBackend& be, const device::DeviceF2Blocks& f2,
    std::span<const QpAdmModel> models, const QpAdmOptions& opts) {
    std::vector<QpAdmResult> out;
    out.reserve(models.size());
    for (const QpAdmModel& m : models) {
        out.push_back(fit_one_model_device(be, f2, m, opts));
    }
    return out;
}

// CUDA-FREE host gate for the kQpMax* bit-parity small-path envelope. Used by
// run_qpadm_search to partition the model list: small-path → the device-BATCHED
// fit_models_batched virtual; large → per-model. Delegates to the SINGLE SOURCE
// (qpadm_bounds.hpp model_fits_small_path) that ALSO sizes the kernel per-thread
// arrays and backs CudaBackend::model_fits_small_path — so the host partition gate
// cannot drift wider than the kernel arrays it routes into (a wider gate would admit
// an oversized model and overflow the fixed device local arrays — UB).
bool model_in_small_path(const QpAdmModel& model, const QpAdmOptions& opts) {
    const int nl = static_cast<int>(model.left.size());
    const int nr = static_cast<int>(model.right.size()) - 1;
    const int r = (opts.rank < 0) ? (nl - 1) : opts.rank;
    return model_fits_small_path(nl, nr, r);
}

// Fit a shard of models on `be` (one device): SMALL-path models go through the device-
// BATCHED virtual `be.fit_models_batched` (the genuine batched rotation primitive — a
// single batched dispatch per same-shape bucket, NOT a per-model loop); the LARGE/>32
// tail goes through the per-model fit_models_batched_default. Returns results aligned
// to `models` order (each result echoes its model_index for the caller's re-sort). The
// CpuBackend's fit_models_batched throws the sentinel ⇒ the oracle path skips the
// virtual and routes EVERY model through the per-model default (correct for the oracle).
std::vector<QpAdmResult> fit_shard(ComputeBackend& be, const device::DeviceF2Blocks& f2,
                                   std::span<const QpAdmModel> models,
                                   const QpAdmOptions& opts) {
    const Precision prec{Precision::Kind::EmulatedFp64, steppe::kDefaultMantissaBits};
    std::vector<QpAdmResult> out(models.size());

    // Partition into small-path (batched) and large-path (per-model) sub-lists,
    // preserving each model's identity (model_index) for the caller's re-sort.
    std::vector<QpAdmModel> small, large;
    std::vector<std::size_t> small_pos, large_pos;  // positions into `models`/`out`
    small.reserve(models.size());
    for (std::size_t i = 0; i < models.size(); ++i) {
        if (model_in_small_path(models[i], opts)) {
            small.push_back(models[i]); small_pos.push_back(i);
        } else {
            large.push_back(models[i]); large_pos.push_back(i);
        }
    }

    if (!small.empty()) {
        std::vector<QpAdmResult> small_results;
        try {
            small_results = be.fit_models_batched(f2, std::span<const QpAdmModel>(small), opts, prec);
        } catch (const std::runtime_error&) {
            // No batched override on this backend (the CpuBackend oracle sentinel) —
            // route the small-path models through the per-model default too.
            small_results = fit_models_batched_default(be, f2, std::span<const QpAdmModel>(small), opts);
        }
        for (std::size_t k = 0; k < small_results.size() && k < small_pos.size(); ++k)
            out[small_pos[k]] = std::move(small_results[k]);
    }
    if (!large.empty()) {
        std::vector<QpAdmResult> large_results =
            fit_models_batched_default(be, f2, std::span<const QpAdmModel>(large), opts);
        for (std::size_t k = 0; k < large_results.size() && k < large_pos.size(); ++k)
            out[large_pos[k]] = std::move(large_results[k]);
    }
    return out;
}

}  // namespace steppe::core::qpadm

// ---- Public entry points (include/steppe/qpadm.hpp) -------------------------
namespace steppe {

namespace {

/// One-time f2 REPLICATION broadcast (design §3.2): ensure every gpus[g] device has
/// a resident f2. The input `f2` is resident on ONE device (f2.device_id). For each
/// device that is NOT that one, materialize once to host (f2.to_host(), the existing
/// opt-in D2H) and upload (upload_f2_blocks_to_device, the existing H2D). The device
/// that already holds `f2` reuses the input handle (no copy). The returned vector is
/// indexed by g; replicas it owns are freed at scope exit (RAII). The returned
/// pointers are borrowed: replicated[g] points either at &f2 (the resident device)
/// or into `owned` (a fresh upload).
///
/// ============================ TODO(multigpu-host-bounce) ============================
/// KNOWN PROBLEM — multi-GPU rotation is bounce-capped on no-P2P cards. DEFERRED.
/// The single-GPU path (run_qpadm_search G==1 fast path) is the SUPPORTED / recommended
/// path for now; this G>=2 replication is CORRECT + deterministic (G1==G2 bit-identical)
/// but throughput-suboptimal — do NOT rely on it for speed until the bounce is removed.
///
///   * The doc once said "kB-MB scale" — WRONG. MEASURED on REAL AADR (P=600,
///     n_block=757): this replication is ~8.72 GB through host (2x D2H + 2x H2D),
///     ~3.8 s COLD. At 9086 real models G2/G1 only reached 1.21x and never hit 1.5x
///     in the swept range — the fixed host-bounce dominates until impractically large N.
///   * ROOT CAUSE: consumer RTX 5090s have GPU<->GPU P2P disabled
///     (caps.can_access_peer == false), so the device-resident cudaMemcpyPeer fast-path
///     is unavailable and we fall back to the host-staged transport (upload_f2_blocks_
///     to_device = to_host()+H2D). P2P-capable cards (e.g. RTX PRO 6000) avoid it.
///   * DEEPER WART (the "weird orchestration"): the PRECOMPUTE shards SNP tiles across
///     GPUs then COMBINES the partials down to ONE device; the rotation then
///     re-broadcasts that single f2 back OUT to the other device here — a
///     gather-then-scatter round-trip.
///   * FIX TO ELIMINATE (not reduce): give the rotation a PER-DEVICE PRECOMPUTE path —
///     each GPU builds its OWN full f2 directly from the genotype stream (zero
///     cross-GPU transfer, ~2.6 s parallel, CHEAPER than the 3.8 s bounce, works on any
///     hardware incl. no-P2P). Then no to_host(), no upload, no combine, no replicate.
///     (Avoid the band-aid of merely pinning the staging: warm pinned DMA drops a
///     *repeat* replica to ~320 ms, but a single rotation pays the COLD ~3.8 s once.)
/// ===================================================================================
struct F2Replication {
    std::vector<const device::DeviceF2Blocks*> per_device;  // borrowed, indexed by g
    std::vector<device::DeviceF2Blocks> owned;              // the fresh uploads (RAII)
};

/// Scatter a worker's batch into the pre-sized result slots by model_index ([7.1]
/// dedup). The pre-sized-slot write IS the deterministic re-sort: each model_index in
/// [0,n) is written EXACTLY once (run_qpadm_search sets model_index = i on the input).
/// FAIL-FAST on an out-of-range index (the determinism invariant the re-sort rests on)
/// — single-homed here so the G==1 fast path and the G>=2 worker enforce it IDENTICALLY
/// rather than from two divergent inline copies (the host-oracle overload keeps its own
/// loop-index FALLBACK because its semantics genuinely differ — it does not fail-fast).
void scatter_into_slots(std::vector<QpAdmResult>& results,
                        std::vector<QpAdmResult>& batch, std::size_t n) {
    for (QpAdmResult& r : batch) {
        const int mi = r.model_index;
        if (mi < 0 || static_cast<std::size_t>(mi) >= n)
            throw std::runtime_error("run_qpadm_search: model_index out of range "
                                     "(must be 0..n-1 for the deterministic re-sort)");
        results[static_cast<std::size_t>(mi)] = std::move(r);
    }
}

F2Replication replicate_f2(const device::DeviceF2Blocks& f2, device::Resources& resources) {
    const std::size_t G = resources.device_count();
    F2Replication rep;
    rep.per_device.assign(G, nullptr);
    // Classify per-g residency ONCE ([7.2] dedup): the residency rule
    // `gpus[g].device_id != f2.device_id` is the single per-device "needs an upload"
    // predicate; this one pre-pass records it per g and counts the uploads, so the rule
    // has ONE expression. The host tensor is then materialized iff n_owned > 0 (any
    // non-resident device) — replacing the three loops that each re-tested the predicate
    // and could drift if someone edited only one of them.
    std::vector<bool> needs_upload(G, false);
    std::size_t n_owned = 0;
    for (std::size_t g = 0; g < G; ++g) {
        if (resources.gpus[g].device_id != f2.device_id) {
            needs_upload[g] = true;
            ++n_owned;
        }
    }
    // Materialize the host tensor ONCE only if at least one device needs a copy.
    F2BlockTensor host;
    if (n_owned > 0) host = f2.to_host();
    // Reserve so the upload loop does not reallocate the owned vector (move-only).
    rep.owned.reserve(n_owned);
    for (std::size_t g = 0; g < G; ++g) {
        if (needs_upload[g]) {
            rep.owned.push_back(
                device::upload_f2_blocks_to_device(host, resources.gpus[g].device_id));
            rep.per_device[g] = &rep.owned.back();
        } else {
            rep.per_device[g] = &f2;  // reuse the existing resident handle
        }
    }
    return rep;
}

}  // namespace

std::vector<QpAdmResult> run_qpadm_search(const device::DeviceF2Blocks& f2,
                                          std::span<const QpAdmModel> models,
                                          const QpAdmOptions& opts,
                                          device::Resources& resources) {
    const std::size_t G = resources.device_count();
    const std::size_t n = models.size();
    if (G == 0) throw std::runtime_error("run_qpadm_search: no devices in Resources");

    // Pre-size the result vector to n; each model_index slot is written EXACTLY once
    // by exactly one worker. The pre-sized-slot write IS the deterministic re-sort
    // (no post-sort needed): results[k].model_index == the k-th input's model_index
    // for any G (the determinism gate). NOTE: this requires models[i].model_index to
    // be a valid index into [0,n); the search sets model_index = i on the input.
    std::vector<QpAdmResult> results(n);
    if (n == 0) return results;

    // ---- G == 1 fast path: no shard, no threads, no f2 replication (mirrors the f2
    // G==1 fast path) — the single backend fits all n models on its device. -------
    if (G == 1) {
        ComputeBackend& be = *resources.gpus[0].backend;
        std::vector<QpAdmResult> batch =
            core::qpadm::fit_shard(be, f2, models, opts);
        scatter_into_slots(results, batch, n);  // pre-sized-slot write = the re-sort ([7.1])
        return results;
    }

    // ---- G >= 2: replicate f2 to every device (one-time), shard the model list
    // contiguously, fan out one jthread per device, each worker fits its sub-span
    // WHOLLY on its device, writes its results into the pre-sized slots (race-free:
    // distinct model_index slots per shard), rethrow the lowest-g fault.
    // TODO(multigpu-host-bounce): replicate_f2 here is the ~3.8 s / 8.72 GB host bounce
    // (no P2P on the 5090s) that caps multi-GPU at ~1.21x on real data — see the big
    // TODO on replicate_f2 above. Multi-GPU is DEFERRED; prefer G==1 until the
    // per-device-precompute fix lands.
    const F2Replication rep = replicate_f2(f2, resources);
    const std::vector<core::qpadm::ModelShard> shards =
        core::qpadm::plan_model_shards(n, G);

    std::vector<std::exception_ptr> worker_errors(G);  // value-init to nullptr
    {
        std::vector<std::jthread> workers;
        workers.reserve(G);
        for (std::size_t g = 0; g < G; ++g) {
            workers.emplace_back([&, g]() {
                try {
                    const core::qpadm::ModelShard& sh = shards[g];
                    if (sh.lo >= sh.hi) return;  // empty shard (n < G) — nothing to do
                    ComputeBackend& be = *resources.gpus[g].backend;
                    const device::DeviceF2Blocks& f2_g = *rep.per_device[g];
                    const std::span<const QpAdmModel> sub =
                        models.subspan(sh.lo, sh.hi - sh.lo);
                    std::vector<QpAdmResult> batch =
                        core::qpadm::fit_shard(be, f2_g, sub, opts);
                    // Write each result into ITS model_index slot — distinct across
                    // shards (contiguous, non-overlapping), so no two workers touch the
                    // same slot (race-free without a lock). The bounds-check + slot-write
                    // rule is single-homed in scatter_into_slots ([7.1] dedup).
                    scatter_into_slots(results, batch, n);
                } catch (...) {
                    worker_errors[g] = std::current_exception();  // surface on join
                }
            });
        }
    }  // <-- join barrier: all workers joined here (jthread dtors)

    // Rethrow the FIRST worker failure (lowest g) deterministically (a genuine
    // device/backend FAULT — domain outcomes already rode in results[*].status).
    for (std::size_t g = 0; g < G; ++g) {
        if (worker_errors[g]) std::rethrow_exception(worker_errors[g]);
    }
    return results;
}

std::vector<QpAdmResult> run_qpadm_search(const F2BlockTensor& f2_host,
                                          std::span<const QpAdmModel> models,
                                          const QpAdmOptions& opts,
                                          device::Resources& resources) {
    // HOST-ORACLE overload: route EVERY model through resources.gpus[0].backend's
    // per-model oracle chain (assemble_f4(host) → run_impl). A host loop is the
    // CORRECT shape for the oracle (design §2.3) — it is the bit-exact reference the
    // device batched path is diffed against, NOT the deliverable.
    ComputeBackend& be = *resources.gpus.at(0).backend;
    const std::size_t n = models.size();
    std::vector<QpAdmResult> results(n);
    const Precision prec{Precision::Kind::EmulatedFp64, steppe::kDefaultMantissaBits};
    for (std::size_t i = 0; i < n; ++i) {
        const QpAdmModel& m = models[i];
        const std::vector<int> left_idx = core::qpadm::left_with_target(m);
        F4Blocks X = be.assemble_f4(f2_host, std::span<const int>(left_idx),
                                    std::span<const int>(m.right), prec);
        QpAdmResult r = core::qpadm::run_impl(be, std::move(X),
                                              std::span<const int>(f2_host.block_sizes),
                                              m, opts);
        r.model_index = m.model_index;
        const int mi = r.model_index;
        const std::size_t slot = (mi >= 0 && static_cast<std::size_t>(mi) < n)
                                     ? static_cast<std::size_t>(mi) : i;
        results[slot] = std::move(r);
    }
    return results;
}

}  // namespace steppe
