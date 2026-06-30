// bindings/internal/bind_common.hpp — shared marshalling helpers for the steppe._core
// nanobind bindings (the per-tool bind_*.cpp TUs share this one home).
//
// MARSHALLING ONLY (cli-bindings.md §5, line 388-393): numpy<->span marshalling,
// name->index resolution against the f2-dir's pops.txt, QpAdmOptions/DeviceConfig fill
// from kwargs, and QpAdmResult/QpWaveResult -> flat Python objects. NO compute logic, NO
// pandas — the pandas/dataclass shaping lives in pure-Python bindings/steppe/__init__.py
// so this compiled module has no pandas link.
//
// PLAIN C++20 host TU (architecture.md §4; cli-bindings.md §6.2): it includes ONLY the
// CUDA-FREE seams (qpadm.hpp / resources.hpp / device_f2_blocks.hpp / app/f2_dir_io.hpp /
// app/pop_resolver.hpp) — never a CUDA toolkit header. A leaked <cuda_runtime.h> /
// <cublas_v2.h> would hard-fail this compile; the layering proof is structural.
//
// THE 4 SPIKE RISKS (interop-usecases.md §4) are addressed by SCOPE:
//   #1 VRAM ownership / #2 stream sync: results are HOST-SIDE; the only device object
//      (DeviceF2Blocks) is created+destroyed INSIDE each fit call (RAII), never crosses
//      into Python. No DLPack / __cuda_array_interface__ in M(py-1).
//   #3 fp64 truncation: F2BlockTensor is FP64 in every precision mode (fstats.hpp:17);
//      to_numpy exports `double` -> numpy float64. No GPU export, no float32 footgun.
//   #4 column-major layout: the tensor is i + P*j + P*P*b (fstats.hpp); to_numpy exports
//      F-contiguous (P, P, n_block) so arr[:,:,b] is slab b with no silent transpose.
#ifndef STEPPE_BINDINGS_INTERNAL_BIND_COMMON_HPP
#define STEPPE_BINDINGS_INTERNAL_BIND_COMMON_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>  // std::memcpy (f2_to_numpy bulk export copy)
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>  // std::move (already used), std::declval (with_device_f2 trailing-return)
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "app/pop_resolver.hpp"  // steppe::app::PopResolver (CUDA-FREE)
#include "device/device_f2_blocks.hpp"  // CUDA-FREE: DeviceF2Blocks, upload_f2_blocks_to_device
#include "device/resources.hpp"         // CUDA-FREE: Resources, build_resources
#include "steppe/config.hpp"            // steppe::DeviceConfig / Precision
#include "steppe/dstat.hpp"             // DstatResult (qpDstat Part B genotype-path D)
#include "steppe/error.hpp"             // steppe::Status
#include "steppe/f3.hpp"                // F3Result
#include "steppe/f4.hpp"                // F4Result
#include "steppe/f4ratio.hpp"           // F4RatioResult
#include "steppe/fstats.hpp"            // steppe::F2BlockTensor
#include "steppe/qpadm.hpp"             // run_qpadm / run_qpwave / run_qpadm_search + value types
#include "steppe/qpgraph.hpp"           // QpGraphEdge / QpGraphResult / QpGraphOptions

namespace nb = nanobind;

namespace steppe::pybind {

using namespace nb::literals;

namespace sa = steppe::app;
namespace sd = steppe::device;

// ---- The opaque f2-dir handle exposed to Python -----------------------------------
// Holds the host F2BlockTensor + the P pop labels (the name<->index map the INDEX-ONLY
// compute seam lacks; cli-bindings.md §1). build_resources is cached ON the handle so the
// precompute-once / fit-many path (ADR-0005) reuses ONE Resources across calls. The
// device upload still happens per-fit-call (mirroring cmd_qpadm.cpp:115-116) — M(py-1)
// keeps the upload inside the call so no DeviceF2Blocks crosses into Python (spike #1).
struct F2Handle {
    steppe::F2BlockTensor tensor;            // the host f2 tensor (FP64).
    std::vector<std::string> pops;           // P labels in P-axis index order.
    int device = 0;                          // the CUDA ordinal the fits target.

    // Lazily-built, cached Resources for `device` (precompute-once/fit-many, ADR-0005).
    std::unique_ptr<sd::Resources> resources;

    [[nodiscard]] int P() const { return tensor.P; }
    [[nodiscard]] int n_block() const { return tensor.n_block; }
};

// Raise a Python ValueError carrying a steppe fault reason.
[[noreturn]] inline void raise_value(const std::string& msg) { throw nb::value_error(msg.c_str()); }

// The single home for the GPU-only fail-fast: steppe is a GPU product, so an empty
// Resources::gpus is a binding-layer FAULT (§1.3). Hand-duplicating this message across
// the entry points lets the wording drift; raise it from one place.
[[noreturn]] inline void raise_no_device() {
    raise_value(
        "no CUDA device available (steppe is a GPU product; a CUDA-capable GPU is "
        "required)");
}

// Build (or reuse the cached) Resources on the handle's device, fail-fast on no-GPU.
// Faults raise; this is a binding-layer FAULT, never a domain outcome (cli-bindings.md
// §1.3 / §5.2). Mirrors cmd_qpadm.cpp:107-113.
inline sd::Resources& ensure_resources(F2Handle& h) {
    if (h.resources) return *h.resources;
    steppe::DeviceConfig cfg;
    cfg.devices = {h.device};  // single-GPU --device <n> (multi-gpu PARKED).
    std::unique_ptr<sd::Resources> r;
    try {
        r = std::make_unique<sd::Resources>(sd::build_resources(cfg));
    } catch (const std::exception& e) {
        // build_resources fault (no device, cannot bind) — a FAULT, raise (§1.3).
        raise_value(std::string("device error: ") + e.what());
    }
    if (r->gpus.empty()) raise_no_device();
    h.resources = std::move(r);
    return *h.resources;
}

// Resolve a list of pop NAMES to P-axis indices against the handle's pops.txt map. An
// unknown name raises a clean Python KeyError naming it (binding-layer concern, §5.1).
inline std::vector<int> resolve_names(const sa::PopResolver& resolver,
                                      const std::vector<std::string>& names, const char* what) {
    const sa::ResolveListResult r = resolver.resolve_all(names);
    if (!r.ok) {
        throw nb::key_error((std::string(what) + ": " + r.error).c_str());
    }
    return r.indices;
}

// The index->NAME counterpart of resolve_names: map P-axis INDICES back to their labels
// against the handle's name<->index map, bounds-checked (an out-of-range/-1 index -> the
// empty string, the honest sentinel). The single home for the result-emitters' indices->
// names step (f4/f3/dstat/f4ratio all share it; cli-bindings.md §5.1).
inline std::vector<std::string> names_of(const std::vector<int>& idx,
                                         const std::vector<std::string>& pops) {
    std::vector<std::string> out;
    out.reserve(idx.size());
    for (int i : idx)
        out.push_back((i >= 0 && static_cast<std::size_t>(i) < pops.size())
                          ? pops[static_cast<std::size_t>(i)]
                          : std::string());
    return out;
}

// Resolve a FIXED-ARITY tuple of pop NAMES to a std::array<int,N> against the resolver,
// throwing a clean Python KeyError naming the offending label on a miss (binding-layer
// §5.1). The single home for the per-tuple resolve loop the batched stats share — f4/
// qpdstat (N=4), f3 (N=3), f4ratio (N=5), and run_dstat (N=4 via quadruples) all call it.
template <std::size_t N>
std::array<int, N> resolve_tuple(const sa::PopResolver& resolver,
                                 const std::array<std::string, N>& q, const char* what) {
    std::array<int, N> qi{};
    for (std::size_t c = 0; c < N; ++c) {
        const sa::ResolveResult rr = resolver.resolve(q[c]);
        if (!rr.ok) throw nb::key_error((std::string(what) + ": " + rr.error).c_str());
        qi[c] = rr.index;
    }
    return qi;
}

// The single home for the precision-string parse (the extract-f2 / qpfstats precision=
// kwarg). The accepted token set is kept in lockstep with the CLI --precision parse
// (config_builder.cpp) and the emitted precision_tag (precision_str), so a precision is
// spelled the same on every surface — canonical set + documented aliases, cli-bindings.md
// §4.1:
//   nullopt                              -> EmulatedFp64{40} (the matmul-heavy f2-GEMM default)
//   "emu40" / "emu" / "emulated_fp64"    -> EmulatedFp64{40}
//   "emu32" / "emulated_fp64_32"         -> EmulatedFp64{32}
//   "fp64" / "native"                    -> native FP64 (the validation oracle / fallback)
//   "tf32"                               -> Tf32
// anything else is a binding fault naming `tool` + the offending string.
inline steppe::Precision parse_precision(const std::optional<std::string>& precision,
                                         const char* tool) {
    // Default = the matmul-heavy f2-GEMM EmulatedFp64 40-bit policy; the named
    // factory is byte-identical to the old default-constructed Precision.
    steppe::Precision prec = steppe::Precision::emulated_fp64();
    if (precision.has_value()) {
        const std::string& p = *precision;
        if (p == "fp64" || p == "native") {
            prec = steppe::Precision::fp64();
        } else if (p == "emu40" || p == "emu" || p == "emulated_fp64") {
            prec = steppe::Precision::emulated_fp64(40);
        } else if (p == "emu32" || p == "emulated_fp64_32") {
            // Mirror the CLI's emu32 -> {EmulatedFp64, 32} (config_builder.cpp) so the
            // 32-bit emulation mode is selectable from the Python facade too.
            prec = steppe::Precision::emulated_fp64(32);
        } else if (p == "tf32") {
            prec = steppe::Precision::tf32();
        } else {
            raise_value(std::string(tool) +
                        ": precision must be one of 'emu40'/'emu'/'emulated_fp64', "
                        "'emu32'/'emulated_fp64_32', 'fp64'/'native', 'tf32' "
                        "(got '" + p + "')");
        }
    }
    return prec;
}

// The single home for the resident-f2 fit prologue/epilogue the F2Handle fit entries share
// (mirrors cmd_qpadm.cpp:107-117): build (cached) Resources on the handle's device, then —
// inside the device-fault try/catch — pick the bound device_id, upload the host tensor (the
// DeviceF2Blocks lives ONLY here and frees in its destructor — spike #1), and invoke `run`
// with (dev_f2, resources). A CUDA/upload/seam fault re-raises as a "device error: ..."
// Python error (§1.3 / §5.2). The differing run callable + any post-call status guard stay
// in the caller. The return type is whatever `run` yields (the per-entry result struct).
template <class Run>
auto with_device_f2(F2Handle& h, Run&& run)
    -> decltype(run(std::declval<sd::DeviceF2Blocks&>(), std::declval<sd::Resources&>())) {
    sd::Resources& resources = ensure_resources(h);
    try {
        const int device_id = resources.gpus.front().device_id;
        sd::DeviceF2Blocks dev_f2 = sd::upload_f2_blocks_to_device(h.tensor, device_id);
        return run(dev_f2, resources);
    } catch (const std::exception& e) {
        raise_value(std::string("device error: ") + e.what());
    }
}

// Map the kwargs onto QpAdmOptions with defaults matching the struct (qpadm.hpp:57-100)
// so a bare call reproduces the goldens. `jackknife` accepts "all"|"feasible_only"|"none".
inline steppe::QpAdmOptions make_options(double fudge, int als_iterations, int rank,
                                         bool allow_negative_weights, double rank_alpha,
                                         const std::string& jackknife) {
    steppe::QpAdmOptions o;
    o.fudge = fudge;
    o.als_iterations = als_iterations;
    o.rank = rank;
    o.allow_negative_weights = allow_negative_weights;
    o.rank_alpha = rank_alpha;
    if (jackknife == "all") {
        o.jackknife = steppe::JackknifePolicy::All;
    } else if (jackknife == "feasible_only") {
        o.jackknife = steppe::JackknifePolicy::FeasibleOnly;
    } else if (jackknife == "none") {
        o.jackknife = steppe::JackknifePolicy::None;
    } else {
        raise_value("jackknife must be one of: 'all', 'feasible_only', 'none' (got '" +
                    jackknife + "')");
    }
    return o;
}

// Status enum -> a stable lower-case string (the Python facade maps it to a Status enum).
inline const char* status_str(steppe::Status s) {
    switch (s) {
        case steppe::Status::Ok: return "ok";
        case steppe::Status::DeviceOom: return "device_oom";
        case steppe::Status::RankDeficient: return "rank_deficient";
        case steppe::Status::NonSpdCovariance: return "non_spd_covariance";
        case steppe::Status::ChisqUndefined: return "chisq_undefined";
        case steppe::Status::InvalidConfig: return "invalid_config";
    }
    return "ok";
}

inline const char* precision_str(steppe::Precision::Kind k) {
    switch (k) {
        case steppe::Precision::Kind::Fp64: return "fp64";
        case steppe::Precision::Kind::EmulatedFp64: return "emulated_fp64";
        case steppe::Precision::Kind::Tf32: return "tf32";
    }
    return "fp64";
}

// QpAdmResult -> a Python dict of flat fields (the facade re-shapes to DataFrames).
// popdrop_feasible is char(0/1) -> a Python list[bool]. The se.empty() "not computed"
// sentinel is preserved by emitting an EMPTY list for se/z (the facade -> NaN columns),
// NEVER a fake 0 (qpadm.hpp:127-130).
inline nb::dict result_to_dict(const steppe::QpAdmResult& r) {
    nb::dict d;
    d["weight"] = r.weight;
    d["se"] = r.se;  // empty => not computed (facade -> NaN); never a fake 0.
    d["z"] = r.z;
    d["p"] = r.p;
    d["chisq"] = r.chisq;
    d["dof"] = r.dof;
    d["est_rank"] = r.est_rank;
    d["f4rank"] = r.f4rank;

    // rankdrop (parallel arrays, AT2 res$rankdrop row order; qpadm.hpp:146-147).
    d["rankdrop_f4rank"] = r.rankdrop_f4rank;
    d["rankdrop_dof"] = r.rankdrop_dof;
    d["rankdrop_dofdiff"] = r.rankdrop_dofdiff;
    d["rankdrop_chisq"] = r.rankdrop_chisq;
    d["rankdrop_p"] = r.rankdrop_p;
    d["rankdrop_chisqdiff"] = r.rankdrop_chisqdiff;
    d["rankdrop_p_nested"] = r.rankdrop_p_nested;

    // popdrop (AT2 res$popdrop; char->bool for feasible; qpadm.hpp:149-152).
    d["popdrop_pat"] = r.popdrop_pat;
    d["popdrop_wt"] = r.popdrop_wt;
    d["popdrop_dof"] = r.popdrop_dof;
    d["popdrop_f4rank"] = r.popdrop_f4rank;
    d["popdrop_chisq"] = r.popdrop_chisq;
    d["popdrop_p"] = r.popdrop_p;
    std::vector<bool> feas(r.popdrop_feasible.size());
    for (std::size_t i = 0; i < r.popdrop_feasible.size(); ++i)
        feas[i] = (r.popdrop_feasible[i] != 0);
    d["popdrop_feasible"] = feas;

    d["status"] = status_str(r.status);
    d["precision"] = precision_str(r.precision_tag);
    d["model_index"] = r.model_index;
    return d;
}

inline nb::dict qpwave_to_dict(const steppe::QpWaveResult& r) {
    nb::dict d;
    d["rank_chisq"] = r.rank_chisq;
    d["rank_dof"] = r.rank_dof;
    d["rank_p"] = r.rank_p;
    d["rankdrop_f4rank"] = r.rankdrop_f4rank;
    d["rankdrop_dof"] = r.rankdrop_dof;
    d["rankdrop_dofdiff"] = r.rankdrop_dofdiff;
    d["rankdrop_chisq"] = r.rankdrop_chisq;
    d["rankdrop_p"] = r.rankdrop_p;
    d["rankdrop_chisqdiff"] = r.rankdrop_chisqdiff;
    d["rankdrop_p_nested"] = r.rankdrop_p_nested;
    d["f4rank"] = r.f4rank;
    d["est_rank"] = r.est_rank;
    d["status"] = status_str(r.status);
    d["precision"] = precision_str(r.precision_tag);
    return d;
}

// F4Result -> a Python dict of parallel arrays {pop1,pop2,pop3,pop4 (names),est,se,z,p}.
// The result carries P-axis INDICES; `pops` is the handle's name<->index map so the
// binding resolves them back to NAMES (the facade reshapes to a DataFrame). NaN est/se/z/p
// (a degenerate quartet) ride through as numpy NaN — the honest sentinel, never a fake 0.
inline nb::dict f4_to_dict(const steppe::F4Result& r, const std::vector<std::string>& pops) {
    nb::dict d;
    d["pop1"] = names_of(r.p1, pops);
    d["pop2"] = names_of(r.p2, pops);
    d["pop3"] = names_of(r.p3, pops);
    d["pop4"] = names_of(r.p4, pops);
    d["est"] = r.est;
    d["se"] = r.se;
    d["z"] = r.z;
    d["p"] = r.p;
    d["status"] = status_str(r.status);
    d["precision"] = precision_str(r.precision_tag);
    return d;
}

// DstatResult -> a Python dict of parallel arrays {pop1..pop4 (names),est,se,z,p}. IDENTICAL
// shape to f4_to_dict (DstatResult mirrors F4Result; the genotype-path D est/se/z/p ARE the
// AT2 D sign/Z/p convention). The result carries P-axis INDICES; `pops` is the name<->index
// map so the binding resolves them back to NAMES. NaN est/se/z/p (a degenerate quadruple)
// ride through as numpy NaN — the honest sentinel, never a fake 0.
inline nb::dict dstat_to_dict(const steppe::DstatResult& r, const std::vector<std::string>& pops) {
    nb::dict d;
    d["pop1"] = names_of(r.p1, pops);
    d["pop2"] = names_of(r.p2, pops);
    d["pop3"] = names_of(r.p3, pops);
    d["pop4"] = names_of(r.p4, pops);
    d["est"] = r.est;
    d["se"] = r.se;
    d["z"] = r.z;
    d["p"] = r.p;
    d["status"] = status_str(r.status);
    d["precision"] = precision_str(r.precision_tag);
    return d;
}

// F3Result -> a Python dict of parallel arrays {pop1,pop2,pop3 (names),est,se,z,p}. The
// THREE-column clone of f4_to_dict (drop pop4). The result carries P-axis INDICES; `pops`
// is the handle's name<->index map so the binding resolves them back to NAMES (the facade
// reshapes to a DataFrame). NaN est/se/z/p (a degenerate triple) ride through as numpy NaN.
inline nb::dict f3_to_dict(const steppe::F3Result& r, const std::vector<std::string>& pops) {
    nb::dict d;
    d["pop1"] = names_of(r.p1, pops);
    d["pop2"] = names_of(r.p2, pops);
    d["pop3"] = names_of(r.p3, pops);
    d["est"] = r.est;
    d["se"] = r.se;
    d["z"] = r.z;
    d["p"] = r.p;
    d["status"] = status_str(r.status);
    d["precision"] = precision_str(r.precision_tag);
    return d;
}

// F4RatioResult -> a Python dict of parallel arrays {pop1..pop5 (names),alpha,se,z}. The
// FIVE-column clone of f4_to_dict (add pop5; alpha replaces est, NO p column — AT2 qpf4ratio
// emits only alpha/se/z). The result carries P-axis INDICES; `pops` is the handle's
// name<->index map so the binding resolves them back to NAMES (the facade reshapes to a
// DataFrame). NaN alpha/se/z (a degenerate tuple) ride through as numpy NaN.
inline nb::dict f4ratio_to_dict(const steppe::F4RatioResult& r, const std::vector<std::string>& pops) {
    nb::dict d;
    d["pop1"] = names_of(r.p1, pops);
    d["pop2"] = names_of(r.p2, pops);
    d["pop3"] = names_of(r.p3, pops);
    d["pop4"] = names_of(r.p4, pops);
    d["pop5"] = names_of(r.p5, pops);
    d["alpha"] = r.alpha;
    d["se"] = r.se;
    d["z"] = r.z;
    d["status"] = status_str(r.status);
    d["precision"] = precision_str(r.precision_tag);
    return d;
}

// QpGraphResult -> a flat Python dict (the facade re-shapes to a tidy edge table).
inline nb::dict qpgraph_to_dict(const steppe::QpGraphResult& r) {
    nb::dict d;
    d["score"] = r.score;
    d["restart_spread"] = r.restart_spread;
    d["worst_residual_z"] = r.worst_residual_z;
    d["worst_pop2"] = r.worst_pop2;
    d["worst_pop3"] = r.worst_pop3;
    d["status"] = status_str(r.status);
    d["weight"] = r.weight;
    d["weight_lo"] = r.weight_lo;
    d["weight_hi"] = r.weight_hi;
    d["admix_from"] = r.admix_from;
    d["admix_to"] = r.admix_to;
    d["edge_length"] = r.edge_length;
    d["edge_from"] = r.edge_from;
    d["edge_to"] = r.edge_to;
    d["leaves"] = r.leaves;
    return d;
}

// ---- f2.to_numpy: the host f2 tensor -> numpy float64, F-contiguous (P, P, n_block) --
// Wraps the ALREADY-RESIDENT host F2BlockTensor.f2 (no D2H — read_f2 holds it). A COPY is
// the conservative M(py-1) choice (cli-bindings.md §5.3): own the buffer with a capsule
// deleter so there is no lifetime hazard tying the array to the handle. F-contiguous
// (P, P, n_block) so arr[:,:,b] is slab b with no silent per-slab transpose (spike #4);
// float64 from the FP64 tensor (spike #3).
inline nb::object f2_to_numpy(const F2Handle& h, const std::vector<double>& src) {
    const std::size_t P = static_cast<std::size_t>(h.tensor.P);
    const std::size_t nb_count = static_cast<std::size_t>(h.tensor.n_block);
    const std::size_t n = h.tensor.size();

    // COPY into a heap buffer owned by the capsule (frees with the numpy array).
    // src is a contiguous std::vector<double> of length n (== P*P*n_block), so a single
    // bulk std::memcpy is a bit-identical FP64 copy with no per-element loop overhead.
    assert(src.size() == n);
    auto* buf = new double[n];
    std::memcpy(buf, src.data(), n * sizeof(double));
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });

    const std::size_t shape[3] = {P, P, nb_count};
    // Column-major within a slab (i + P*j) and block-major outer (+ P*P*b) IS exactly
    // F-contiguous (P, P, n_block): the fastest-varying axis is i (stride 1), then j
    // (stride P), then b (stride P*P) — the default Fortran strides for this shape.
    return nb::cast(nb::ndarray<nb::numpy, double, nb::ndim<3>, nb::f_contig>(
        buf, 3, shape, owner));
}

// ---- per-tool registration entry points (one bind_<tool>.cpp TU each) --------------
// module.cpp's NB_MODULE shell calls these in registration order: register_f2handle
// FIRST so the F2Handle nanobind type exists before any fit entry that takes it.
void register_f2handle(nb::module_& m);
void register_qpadm(nb::module_& m);
void register_qpgraph(nb::module_& m);
void register_fstats(nb::module_& m);
void register_dates(nb::module_& m);

}  // namespace steppe::pybind

#endif  // STEPPE_BINDINGS_INTERNAL_BIND_COMMON_HPP
