// bindings/module.cpp — the steppe._core nanobind TU (M(py-1)).
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
#include <algorithm>
#include <array>
#include <cstddef>
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

#include "app/f2_dir_io.hpp"     // steppe::app::read_f2_dir / F2Dir (CUDA-FREE)
#include "app/f2_dir_writer.hpp" // steppe::app::write_f2_dir / F2DirMeta (CUDA-FREE; the extract out= path)
#include "app/pop_resolver.hpp"  // steppe::app::PopResolver (CUDA-FREE)
#include "device/device_f2_blocks.hpp"  // CUDA-FREE: DeviceF2Blocks, upload_f2_blocks_to_device
#include "device/resources.hpp"         // CUDA-FREE: Resources, build_resources
#include "steppe/config.hpp"            // steppe::DeviceConfig
#include "steppe/dstat.hpp"             // run_dstat + DstatResult (qpDstat Part B genotype-path D)
#include "steppe/dates.hpp"             // run_dates + DatesResult/DatesOptions (the DATES dating tool)
#include "steppe/extract.hpp"           // run_extract_f2 + F2ExtractResult (M(py-2) genotype->f2 extract)
#include "steppe/error.hpp"             // steppe::Status
#include "steppe/f3.hpp"                // run_f3 + F3Result (the standalone-f3 binding)
#include "steppe/f4.hpp"                // run_f4 + F4Result (the standalone-f4 binding)
#include "steppe/f4ratio.hpp"           // run_f4ratio + F4RatioResult (the standalone-f4-ratio binding)
#include "steppe/fstats.hpp"            // steppe::F2BlockTensor
#include "steppe/qpadm.hpp"             // run_qpadm / run_qpwave / run_qpadm_search + value types
#include "steppe/qpgraph.hpp"           // run_qpgraph + QpGraphEdge/Result/Options (the single-graph fit binding)
#include "steppe/qpgraph_search.hpp"    // run_qpgraph_search (the topology SEARCH v1 binding)
#include "steppe/qpfstats.hpp"          // run_qpfstats + QpfstatsResult (the genotype-path joint f2 smoother)

#include "io/geno_reader.hpp"           // io::GenoReader (qpDstat Part B P-axis order)
#include "io/ind_reader.hpp"            // io::read_ind / PopSelection / IndPartition

namespace nb = nanobind;
using namespace nb::literals;

namespace {

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
[[noreturn]] void raise_value(const std::string& msg) { throw nb::value_error(msg.c_str()); }

// The single home for the GPU-only fail-fast: steppe is a GPU product, so an empty
// Resources::gpus is a binding-layer FAULT (§1.3). Hand-duplicating this message across
// the entry points lets the wording drift; raise it from one place.
[[noreturn]] void raise_no_device() {
    raise_value(
        "no CUDA device available (steppe is a GPU product; a CUDA-capable GPU is "
        "required)");
}

// Build (or reuse the cached) Resources on the handle's device, fail-fast on no-GPU.
// Faults raise; this is a binding-layer FAULT, never a domain outcome (cli-bindings.md
// §1.3 / §5.2). Mirrors cmd_qpadm.cpp:107-113.
sd::Resources& ensure_resources(F2Handle& h) {
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
std::vector<int> resolve_names(const sa::PopResolver& resolver,
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
std::vector<std::string> names_of(const std::vector<int>& idx,
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
// kwarg): nullopt -> the EmulatedFp64 40-bit default (the matmul-heavy f2-GEMM default);
// "fp64"/"native" -> native FP64; "emulated_fp64"/"emu" -> EmulatedFp64; "tf32" -> Tf32;
// anything else is a binding fault naming `tool` + the offending string.
steppe::Precision parse_precision(const std::optional<std::string>& precision,
                                  const char* tool) {
    steppe::Precision prec;  // defaults to Kind::EmulatedFp64, kDefaultMantissaBits.
    if (precision.has_value()) {
        const std::string& p = *precision;
        if (p == "fp64" || p == "native") {
            prec.kind = steppe::Precision::Kind::Fp64;
        } else if (p == "emulated_fp64" || p == "emu") {
            prec.kind = steppe::Precision::Kind::EmulatedFp64;
        } else if (p == "tf32") {
            prec.kind = steppe::Precision::Kind::Tf32;
        } else {
            raise_value(std::string(tool) +
                        ": precision must be one of 'fp64'/'native', 'emulated_fp64'/'emu', "
                        "'tf32' (got '" + p + "')");
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
steppe::QpAdmOptions make_options(double fudge, int als_iterations, int rank,
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
const char* status_str(steppe::Status s) {
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

const char* precision_str(steppe::Precision::Kind k) {
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
nb::dict result_to_dict(const steppe::QpAdmResult& r) {
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

nb::dict qpwave_to_dict(const steppe::QpWaveResult& r) {
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
nb::dict f4_to_dict(const steppe::F4Result& r, const std::vector<std::string>& pops) {
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
nb::dict dstat_to_dict(const steppe::DstatResult& r, const std::vector<std::string>& pops) {
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
nb::dict f3_to_dict(const steppe::F3Result& r, const std::vector<std::string>& pops) {
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
nb::dict f4ratio_to_dict(const steppe::F4RatioResult& r, const std::vector<std::string>& pops) {
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

// ---- read_f2: the f2-dir loader -> the opaque F2Handle ----------------------------
// Returns a raw heap pointer; the binding uses rv_policy::take_ownership so Python owns
// the F2Handle and frees it (with its cached Resources) at GC. (F2Handle is move-only via
// the unique_ptr<Resources> member, so a by-value return is not nb-convertible; a pointer
// with take_ownership is the nanobind-idiomatic transfer for an opaque, non-copyable type.)
F2Handle* read_f2(const std::string& dir, int device) {
    const sa::F2DirResult res = sa::read_f2_dir(dir);
    if (!res.ok) raise_value(res.error);
    auto* h = new F2Handle();
    h->tensor = res.dir.f2;          // host FP64 tensor (the to_numpy source; no D2H).
    h->pops = res.dir.pop_labels;    // the name<->index map.
    h->device = device;
    return h;
}

// ---- the fit entries: resolve names -> upload -> run -> result dict ----------------
// Each mirrors cmd_qpadm.cpp:88-117 exactly: resolve against pops.txt, build (cached)
// resources, upload the host tensor to `device` (the DeviceF2Blocks lives ONLY here and
// frees in its destructor — spike #1), run the CUDA-free seam, marshal the result.
// Faults (no device, OOM, CUDA runtime) raise; per-model domain outcomes ride on the
// result's `status` field, never a traceback (cli-bindings.md §1.3 / §5.2).

nb::dict run_qpadm_py(F2Handle& h, const std::string& target,
                      const std::vector<std::string>& left,
                      const std::vector<std::string>& right, int rank, double fudge,
                      int als_iterations, double rank_alpha,
                      bool allow_negative_weights) {
    if (target.empty()) raise_value("qpadm: target is required");
    if (left.empty()) raise_value("qpadm: left needs at least one source population");
    if (right.empty()) raise_value("qpadm: right needs at least one outgroup population");

    const sa::PopResolver resolver(h.pops);
    if (!resolver.valid()) raise_value(resolver.error());

    const sa::ResolveResult t = resolver.resolve(target);
    if (!t.ok) throw nb::key_error(("target: " + t.error).c_str());

    steppe::QpAdmModel model;
    model.target = t.index;
    model.left = resolve_names(resolver, left, "left");
    model.right = resolve_names(resolver, right, "right");
    model.model_index = 0;

    const steppe::QpAdmOptions opts = make_options(
        fudge, als_iterations, rank, allow_negative_weights, rank_alpha, "all");

    const steppe::QpAdmResult result =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_qpadm(dev_f2, model, opts, resources);
        });
    return result_to_dict(result);
}

nb::dict run_qpwave_py(F2Handle& h, const std::vector<std::string>& left,
                       const std::vector<std::string>& right, double fudge,
                       double rank_alpha) {
    if (left.empty())
        raise_value("qpwave: left needs at least the reference + one population");
    if (right.empty()) raise_value("qpwave: right needs at least one outgroup population");

    const sa::PopResolver resolver(h.pops);
    if (!resolver.valid()) raise_value(resolver.error());

    const std::vector<int> left_idx = resolve_names(resolver, left, "left");
    const std::vector<int> right_idx = resolve_names(resolver, right, "right");

    steppe::QpAdmOptions opts;  // qpWave consults only fudge / rank_alpha.
    opts.fudge = fudge;
    opts.rank_alpha = rank_alpha;

    const steppe::QpWaveResult result =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_qpwave(dev_f2, std::span<const int>(left_idx),
                                      std::span<const int>(right_idx), opts, resources);
        });
    return qpwave_to_dict(result);
}

// QpGraphResult -> a flat Python dict (the facade re-shapes to a tidy edge table).
nb::dict qpgraph_to_dict(const steppe::QpGraphResult& r) {
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

// run_qpgraph: fit a FIXED admixture graph (an edge list of (parent, child) name pairs)
// to the resident f2. The leaves must be f2 populations. Mirrors run_qpwave_py: build
// (cached) resources, upload the host tensor INSIDE the call, run the CUDA-free seam,
// marshal. NOTE: qpGraph uses the AT2 afprod=FALSE f2 (DIFFERENT from qpadm's afprod=TRUE);
// the f2 dir handed to read_f2 must be the afprod=FALSE one (the facade documents this).
nb::dict run_qpgraph_py(F2Handle& h,
                        const std::vector<std::array<std::string, 2>>& edges,
                        int numstart, double fudge, double diag_f3, bool constrained) {
    if (edges.empty()) raise_value("qpgraph: the graph edge list is empty");
    std::vector<steppe::QpGraphEdge> e;
    e.reserve(edges.size());
    for (const auto& pr : edges) e.push_back({pr[0], pr[1]});

    steppe::QpGraphOptions opts;
    opts.numstart = numstart;
    opts.fudge = fudge;
    opts.diag_f3 = diag_f3;
    opts.constrained = constrained;

    const steppe::QpGraphResult result =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_qpgraph(dev_f2, e, h.pops, opts, resources);
        });
    if (result.status == steppe::Status::InvalidConfig)
        raise_value("qpgraph: the graph could not be fit (a leaf is not an f2 population, "
                    "or the topology is unrooted/cyclic/invalid)");
    return qpgraph_to_dict(result);
}

// run_qpgraph_search: the topology SEARCH v1 (oracle C). Enumerate every rooted topology on
// the bounded `pops` leaf set (nadmix in {0..max_nadmix}), fit ALL in one heterogeneous-fleet
// launch, return the deterministic global-best + the exhaustive-coverage count + the heuristic
// recovery + the wall-clock. The host does only the cheap enumeration + the argmin.
nb::dict run_qpgraph_search_py(F2Handle& h, const std::vector<std::string>& pops, int max_nadmix,
                               int numstart, double fudge, double diag_f3, bool constrained,
                               bool run_heuristic) {
    if (pops.size() < 3) raise_value("qpgraph-search: need >= 3 population labels");
    steppe::QpGraphSearchOptions opts;
    opts.pops = pops;
    opts.max_nadmix = max_nadmix;
    opts.run_heuristic = run_heuristic;
    opts.fit.numstart = numstart;
    opts.fit.fudge = fudge;
    opts.fit.diag_f3 = diag_f3;
    opts.fit.constrained = constrained;

    const steppe::QpGraphSearchResult r =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_qpgraph_search(dev_f2, h.pops, opts, resources);
        });
    if (r.status == steppe::Status::InvalidConfig)
        raise_value("qpgraph-search: invalid pop-set (a pop is not an f2 population, or < 3 leaves)");

    nb::dict d;
    d["n_trees"] = r.n_trees;
    d["n_admix1"] = r.n_admix1;
    d["n_candidates"] = r.n_candidates;
    d["best_score"] = r.best.score;
    d["second_best_score"] = r.second_best_score;
    d["best_nadmix"] = r.best.nadmix;
    d["best_hash"] = static_cast<std::uint64_t>(r.best.hash);
    d["heuristic_recovered"] = r.heuristic_recovered;
    d["fit_all_wall_ms"] = r.fit_all_wall_ms;
    d["topologies_per_s"] = r.topologies_per_s;
    nb::list be;
    for (const auto& e : r.best.edges) be.append(nb::make_tuple(e.from, e.to));
    d["best_edges"] = be;

    // The FULL per-candidate scored vector (additive marshalling of the already-computed
    // QpGraphSearchResult::candidates — the same per-topology data the global-best argmin
    // reduces over, in deterministic enumeration order; trees then admix1). Parallel arrays
    // {nadmix, hash, score, restart_spread} so the facade can look a candidate up by its
    // canonical graph_hash + the score the argmin chose. NO compute change (the score vector
    // is already in the result; this only exposes it).
    std::vector<int> cand_nadmix;
    std::vector<std::uint64_t> cand_hash;
    std::vector<double> cand_score;
    std::vector<double> cand_spread;
    cand_nadmix.reserve(r.candidates.size());
    cand_hash.reserve(r.candidates.size());
    cand_score.reserve(r.candidates.size());
    cand_spread.reserve(r.candidates.size());
    for (const auto& c : r.candidates) {
        cand_nadmix.push_back(c.nadmix);
        cand_hash.push_back(static_cast<std::uint64_t>(c.hash));
        cand_score.push_back(c.score);
        cand_spread.push_back(c.restart_spread);
    }
    d["cand_nadmix"] = cand_nadmix;
    d["cand_hash"] = cand_hash;
    d["cand_score"] = cand_score;
    d["cand_restart_spread"] = cand_spread;
    return d;
}

// run_f4: a list of (p1,p2,p3,p4) quartet NAME tuples against the SAME resident f2,
// computed BATCHED (run_f4). Returns ONE dict of parallel arrays {pop1..pop4,est,se,z,p}
// in input order. Mirrors run_qpwave_py exactly: resolve names against pops.txt, build
// (cached) resources, upload the host tensor INSIDE the call (the DeviceF2Blocks lives
// only here and frees in its destructor — spike #1), run the CUDA-free seam, marshal.
nb::dict run_f4_py(F2Handle& h,
                   const std::vector<std::array<std::string, 4>>& quartets) {
    if (quartets.empty()) raise_value("f4: needs at least one (p1,p2,p3,p4) quartet");

    const sa::PopResolver resolver(h.pops);
    if (!resolver.valid()) raise_value(resolver.error());

    std::vector<std::array<int, 4>> idx_quartets;
    idx_quartets.reserve(quartets.size());
    for (const std::array<std::string, 4>& q : quartets)
        idx_quartets.push_back(resolve_tuple<4>(resolver, q, "quartet pop"));

    steppe::QpAdmOptions opts;  // f4 uses fudge=0 internally (run_f4); opts is the default.

    const steppe::F4Result result =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_f4(
                dev_f2, std::span<const std::array<int, 4>>(idx_quartets), opts, resources);
        });
    return f4_to_dict(result, h.pops);
}

// run_qpdstat: the qpDstat Part-A binding — D-statistic / f4 over the f2-data path. A THIN
// clone of run_f4_py (the qpdstat f2-path == f4: admixtools::qpdstat(f2dir,f4mode=TRUE) is
// byte-identical to f4mode=FALSE and to f4, since f4mode is a no-op without per-SNP
// genotypes). Returns the SAME dict {pop1..pop4,est,se,z,p,status,precision} as run_f4 (z =
// est/se, p = 2*(1-Phi(|z|)) ARE the AT2 D-stat sign/Z/p convention). The normalized-D
// MAGNITUDE (per-SNP genotypes) is Part B, a separate later binding. NO new compute, NO new
// emitter, NO new result type — it REUSES run_f4 + f4_to_dict verbatim.
nb::dict run_qpdstat_py(F2Handle& h,
                        const std::vector<std::array<std::string, 4>>& quartets) {
    if (quartets.empty()) raise_value("qpdstat: needs at least one (p1,p2,p3,p4) quartet");

    const sa::PopResolver resolver(h.pops);
    if (!resolver.valid()) raise_value(resolver.error());

    std::vector<std::array<int, 4>> idx_quartets;
    idx_quartets.reserve(quartets.size());
    for (const std::array<std::string, 4>& q : quartets)
        idx_quartets.push_back(resolve_tuple<4>(resolver, q, "quartet pop"));

    steppe::QpAdmOptions opts;  // qpdstat==f4 uses fudge=0 internally (run_f4); opts default.

    const steppe::F4Result result =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_f4(
                dev_f2, std::span<const std::array<int, 4>>(idx_quartets), opts, resources);
        });
    return f4_to_dict(result, h.pops);
}

// run_dstat: the qpDstat Part-B genotype-path NORMALIZED-D. UNLIKE run_qpdstat (which reads
// the f2 cache and reports f4), this reads the GENOTYPE TRIPLE prefix.{geno,snp,ind} directly
// (the extract-f2 decode front-end + the per-SNP D kernel + the num/den block-jackknife), so
// it does NOT take an F2Handle. It builds the P-axis from read_ind(MinN,1) (every population,
// sorted ASC by label — IDENTICAL to run_dstat's internal decode), resolves the quadruple
// names to that axis, builds resources for `device`, and runs the CUDA-free seam. Returns the
// SAME dict {pop1..pop4,est,se,z,p} as run_qpdstat (the D convention). `blgsize` is MORGANS
// (AT2 default 0.05). Faults (no device, missing files, CUDA runtime) raise (§1.3 / §5.2).
nb::dict run_dstat_py(const std::string& prefix,
                      const std::vector<std::array<std::string, 4>>& quadruples,
                      double blgsize, int device) {
    if (quadruples.empty()) raise_value("qpdstat (genotype): needs at least one (p1,p2,p3,p4) quadruple");
    const std::string geno = prefix + ".geno";
    const std::string snp = prefix + ".snp";
    const std::string ind = prefix + ".ind";

    // The pop UNION (the AT2 indvec): the DISTINCT names across every quadruple. run_dstat
    // reads ONLY these (read_ind(Explicit{union}), NOT the whole prefix). The P axis is that
    // Explicit partition (sorted ASC by label); the resolver below is built over the SAME
    // read_ind(Explicit{union}) so its indices match run_dstat's decode exactly.
    std::vector<std::string> pop_union;
    for (const std::array<std::string, 4>& q : quadruples) {
        for (const std::string& nm : q) {
            if (std::find(pop_union.begin(), pop_union.end(), nm) == pop_union.end())
                pop_union.push_back(nm);
        }
    }

    std::vector<std::string> pops;  // the SORTED Explicit partition (the P axis order).
    try {
        steppe::io::GenoReader reader(geno);
        const std::size_t n_present = reader.records_present();
        steppe::io::PopSelection sel;
        sel.mode = steppe::io::PopSelection::Mode::Explicit;
        sel.labels = pop_union;
        const steppe::io::IndPartition part = steppe::io::read_ind(ind, sel, n_present);
        pops.reserve(part.groups.size());
        for (const steppe::io::PopGroup& g : part.groups) pops.push_back(g.label);
    } catch (const std::exception& e) {
        raise_value(std::string("input error: ") + e.what());
    }

    const sa::PopResolver resolver(pops);
    if (!resolver.valid()) raise_value(resolver.error());

    std::vector<std::array<int, 4>> idx_quads;
    idx_quads.reserve(quadruples.size());
    for (const std::array<std::string, 4>& q : quadruples)
        idx_quads.push_back(resolve_tuple<4>(resolver, q, "quadruple pop"));

    steppe::DeviceConfig cfg;
    cfg.devices = {device};  // single-GPU (multi-gpu PARKED).
    steppe::DstatResult result;
    try {
        sd::Resources resources = sd::build_resources(cfg);
        if (resources.gpus.empty()) raise_no_device();
        result = steppe::run_dstat(geno, snp, ind,
                                   std::span<const std::string>(pop_union),
                                   std::span<const std::array<int, 4>>(idx_quads),
                                   blgsize, resources);
    } catch (const std::exception& e) {
        raise_value(std::string("device error: ") + e.what());
    }
    return dstat_to_dict(result, pops);
}

// run_dates: the DATES admixture-dating binding. Reads the genotype TRIPLE prefix.{geno,snp,ind}
// directly (the cuFFT autocorrelation LD engine; NEVER the f2 cache, NEVER a host O(M²) SNP-pair
// loop) and returns the date in generations + the leave-one-chromosome block-jackknife SE.
// `target` is the admixed population; `source1`/`source2` are the two reference sources (the
// weight is wt = freq(source1) - freq(source2)). The .snp MUST carry a real genetic map (cM).
// Returns a dict {target, source1, source2, date_gen, se, fit_error_sd, curve_cm, curve_corr,
// status}. Faults (no device, missing files) raise (§1.3 / §5.2).
nb::dict run_dates_py(const std::string& prefix, const std::string& target,
                      const std::string& source1, const std::string& source2, int device) {
    const std::string geno = prefix + ".geno";
    const std::string snp = prefix + ".snp";
    const std::string ind = prefix + ".ind";

    steppe::DeviceConfig cfg;
    cfg.devices = {device};  // single-GPU (multi-gpu PARKED).
    steppe::DatesOptions opts;  // defaults == the DATES reference par.dates.
    steppe::DatesResult result;
    try {
        sd::Resources resources = sd::build_resources(cfg);
        if (resources.gpus.empty()) raise_no_device();
        result = steppe::run_dates(geno, snp, ind, target, source1, source2, opts, resources);
    } catch (const std::exception& e) {
        raise_value(std::string("input/device error: ") + e.what());
    }

    nb::dict d;
    d["target"] = target;
    d["source1"] = source1;
    d["source2"] = source2;
    d["date_gen"] = result.date_gen;
    d["se"] = result.se;
    d["fit_error_sd"] = result.fit_error_sd;
    d["curve_cm"] = result.curve_cm;
    d["curve_corr"] = result.curve_corr;
    d["status"] = (result.status == steppe::Status::Ok) ? "ok" : "error";
    return d;
}

// run_extract_f2: the M(py-2) genotype->f2 EXTRACT binding. Reads the genotype TRIPLE
// prefix.{geno,snp,ind} directly and builds the f2_blocks tensor (decode->filter->
// assign_blocks->tiered f2 compute->to_host) through the CUDA-free steppe::run_extract_f2
// seam (extract.hpp) — the SAME chain the CLI extract-f2 command runs (DRY). GPU-only:
// builds Resources for `device`, fail-fasts on no-GPU with the same message as the CLI.
//
// TWO RETURN MODES (capsule/path idiom, NOT a giant disk round-trip): if `out` is given,
// SERIALIZE the result to an STPF2BK1 dir via write_f2_dir and return the path STRING (the
// user can then read_f2(path)); if `out` is empty, wrap the host F2BlockTensor + labels in
// a NEW F2Handle and return it (rv_policy::take_ownership, like read_f2 — NO disk round-
// trip). The F2Handle pointer return reuses read_f2's exact ownership transfer. Faults (no
// device, unknown Explicit pop, missing file, every SNP filtered, an unwritable out dir)
// raise (the library THROWS; the binding re-raises as a Python error — §1.3 / §5.2).
//
// `pops` -> PopSelection::Explicit{pops} (the named-subset case; the P axis is that
// selection sorted ASC by label). `precision`: nullopt -> the EmulatedFp64 default (the
// f2-GEMM default, matching the CLI); "fp64"/"native" -> native FP64 oracle; "emulated_fp64"
// -> EmulatedFp64. `ploidy`: "auto" (AT2 adjust_pseudohaploid; the default), "1"/"pseudo"
// -> pseudo-haploid, "2"/"diploid" -> diploid. Defaults match the CLI/AT2 extract_f2 so a
// bare extract reproduces the golden (autosomes_only ON, maxmiss as the pop-axis semantic).
nb::object run_extract_f2_py(const std::string& prefix, const std::vector<std::string>& pops,
                             const std::string& out, int device, double blgsize,
                             double maf, double maxmiss, bool autosomes_only,
                             bool drop_monomorphic, bool transversions_only,
                             const std::string& ploidy,
                             std::optional<std::string> precision) {
    if (pops.empty()) raise_value("extract_f2: pops needs at least one population name");
    const std::string geno = prefix + ".geno";
    const std::string snp = prefix + ".snp";
    const std::string ind = prefix + ".ind";

    // pops -> the Explicit selection (the P axis is read_ind(Explicit{pops}) sorted ASC).
    steppe::io::PopSelection sel;
    sel.mode = steppe::io::PopSelection::Mode::Explicit;
    sel.labels = pops;

    // The on-the-fly QC filter (the AT2 extract_f2 defaults; maxmiss is the POP-axis
    // coverage semantic, reproduced inside run_extract_f2 — NOT the sample-axis predicate).
    steppe::FilterConfig filter;
    filter.maf_min = maf;
    filter.geno_max_missing = maxmiss;      // AT2 pop-axis maxmiss (0 = global intersection).
    filter.autosomes_only = autosomes_only; // AT2 extract_f2 default auto_only=TRUE.
    filter.drop_monomorphic = drop_monomorphic;
    filter.transversions_only = transversions_only;

    // The precision policy (default EmulatedFp64 40-bit, the f2-GEMM default = the CLI).
    const steppe::Precision prec = parse_precision(precision, "extract_f2");

    steppe::ExtractPloidy ep = steppe::ExtractPloidy::Auto;
    if (ploidy == "auto") {
        ep = steppe::ExtractPloidy::Auto;
    } else if (ploidy == "1" || ploidy == "pseudo" || ploidy == "pseudo_haploid") {
        ep = steppe::ExtractPloidy::PseudoHaploid;
    } else if (ploidy == "2" || ploidy == "diploid") {
        ep = steppe::ExtractPloidy::Diploid;
    } else {
        raise_value("extract_f2: ploidy must be one of 'auto', '1'/'pseudo', '2'/'diploid' "
                    "(got '" + ploidy + "')");
    }

    steppe::DeviceConfig cfg;
    cfg.devices = {device};  // single-GPU (multi-gpu PARKED).
    cfg.precision = prec;

    steppe::F2ExtractResult result;
    try {
        sd::Resources resources = sd::build_resources(cfg);
        if (resources.gpus.empty()) raise_no_device();
        result = steppe::run_extract_f2(geno, snp, ind, sel, filter, prec, blgsize, ep,
                                        resources);
    } catch (const std::exception& e) {
        raise_value(std::string("extract_f2: ") + e.what());
    }

    // MODE A: out= given -> serialize an STPF2BK1 dir + return the path STRING.
    if (!out.empty()) {
        sa::F2DirMeta meta;
        meta.precision_mantissa_bits = prec.mantissa_bits;
        meta.precision_tag =
            (result.precision_tag == steppe::Precision::Kind::EmulatedFp64) ? "emu"
          : (result.precision_tag == steppe::Precision::Kind::Tf32)         ? "tf32"
                                                                            : "fp64";
        meta.blgsize_cm = blgsize * steppe::kCentimorgansPerMorgan;  // Morgans -> centimorgans (meta records cM).
        meta.n_block = result.f2.n_block;
        meta.P = result.f2.P;
        meta.n_snp_total = result.n_snp_total;
        meta.n_snp_kept = result.n_snp_kept;
        meta.maf_min = filter.maf_min;
        meta.geno_max_missing = filter.geno_max_missing;
        meta.autosomes_only = filter.autosomes_only;
        meta.drop_monomorphic = filter.drop_monomorphic;
        meta.transversions_only = filter.transversions_only;
        meta.geno_path = geno;
        meta.snp_path = snp;
        meta.ind_path = ind;
        meta.hash_source_files = false;  // the Python extract does not hash the big .geno.
        const sa::F2DirWriteResult wr =
            sa::write_f2_dir(out, result.f2, result.pop_labels, meta);
        if (!wr.ok) raise_value("extract_f2: " + wr.error);
        return nb::cast(out);
    }

    // MODE B: out= empty -> wrap the host tensor + labels in a new F2Handle (no disk
    // round-trip). The pointer return uses rv_policy::take_ownership at the def site.
    auto* h = new F2Handle();
    h->tensor = std::move(result.f2);
    h->pops = std::move(result.pop_labels);
    h->device = device;
    return nb::cast(h, nb::rv_policy::take_ownership);
}

// run_qpfstats: the genotype-path JOINT f2 SMOOTHER (include/steppe/qpfstats.hpp). Reads the
// GENOTYPE TRIPLE prefix.{geno,snp,ind} directly, drives the qpDstat-B numerator engine over
// the FULL f2/f3/f4 popcomb set, runs the on-device shared-factor smoothing solve, and
// returns a SMOOTHED f2 — the SAME dual-return idiom as run_extract_f2: out= given ->
// serialize an STPF2BK1 dir + return the path STRING; out= empty -> wrap the smoothed
// F2BlockTensor + labels in a NEW F2Handle (so read_f2/run_f4/run_qpadm consume it). `pops`
// is the smoothing pop set (sorted ASC internally = the AT2 dimnames order). `precision`:
// nullopt -> EmulatedFp64 default (the matmul sub-steps); "fp64"/"native" -> native FP64.
// Faults (no device, missing files, unknown pop, CUDA runtime) raise (§1.3 / §5.2).
nb::object run_qpfstats_py(const std::string& prefix, const std::vector<std::string>& pops,
                           const std::string& out, int device, double blgsize,
                           std::optional<std::string> precision) {
    if (pops.size() < 4) raise_value("qpfstats: pops needs at least 4 populations (the f4 basis)");
    const std::string geno = prefix + ".geno";
    const std::string snp = prefix + ".snp";
    const std::string ind = prefix + ".ind";

    const steppe::Precision prec = parse_precision(precision, "qpfstats");

    steppe::DeviceConfig cfg;
    cfg.devices = {device};  // single-GPU (multi-gpu PARKED).
    cfg.precision = prec;

    steppe::QpfstatsResult result;
    try {
        sd::Resources resources = sd::build_resources(cfg);
        if (resources.gpus.empty()) raise_no_device();
        result = steppe::run_qpfstats(geno, snp, ind, std::span<const std::string>(pops),
                                      blgsize, prec, resources);
    } catch (const std::exception& e) {
        raise_value(std::string("qpfstats: ") + e.what());
    }
    if (result.status != steppe::Status::Ok)
        raise_value("qpfstats: could not build the smoothed f2 (check pops are all present)");

    // MODE A: out= given -> serialize an STPF2BK1 dir + return the path STRING.
    if (!out.empty()) {
        sa::F2DirMeta meta;
        meta.precision_mantissa_bits = prec.mantissa_bits;
        meta.precision_tag =
            (prec.kind == steppe::Precision::Kind::EmulatedFp64) ? "emu"
          : (prec.kind == steppe::Precision::Kind::Tf32)         ? "tf32"
                                                                 : "fp64";
        meta.blgsize_cm = blgsize * steppe::kCentimorgansPerMorgan;  // Morgans -> centimorgans (meta records cM).
        meta.n_block = result.f2.n_block;
        meta.P = result.f2.P;
        meta.autosomes_only = true;
        meta.geno_path = geno; meta.snp_path = snp; meta.ind_path = ind;
        meta.pop_selection = "qpfstats-smoothed";
        const sa::F2DirWriteResult wr =
            sa::write_f2_dir(out, result.f2, result.pop_labels, meta);
        if (!wr.ok) raise_value("qpfstats: " + wr.error);
        return nb::cast(out);
    }

    // MODE B: out= empty -> wrap the smoothed tensor + labels in a new F2Handle.
    auto* h = new F2Handle();
    h->tensor = std::move(result.f2);
    h->pops = std::move(result.pop_labels);
    h->device = device;
    return nb::cast(h, nb::rv_policy::take_ownership);
}

// run_f3: a list of (C,A,B) triple NAME tuples against the SAME resident f2, computed
// BATCHED (run_f3). Returns ONE dict of parallel arrays {pop1,pop2,pop3,est,se,z,p} in
// input order. The THREE-slab clone of run_f4_py: resolve names against pops.txt, build
// (cached) resources, upload the host tensor INSIDE the call (the DeviceF2Blocks lives only
// here and frees in its destructor — spike #1), run the CUDA-free seam, marshal.
nb::dict run_f3_py(F2Handle& h,
                   const std::vector<std::array<std::string, 3>>& triples) {
    if (triples.empty()) raise_value("f3: needs at least one (C,A,B) triple");

    const sa::PopResolver resolver(h.pops);
    if (!resolver.valid()) raise_value(resolver.error());

    std::vector<std::array<int, 3>> idx_triples;
    idx_triples.reserve(triples.size());
    for (const std::array<std::string, 3>& t : triples)
        idx_triples.push_back(resolve_tuple<3>(resolver, t, "triple pop"));

    steppe::QpAdmOptions opts;  // f3 uses fudge=0 internally (run_f3); opts is the default.

    const steppe::F3Result result =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_f3(
                dev_f2, std::span<const std::array<int, 3>>(idx_triples), opts, resources);
        });
    return f3_to_dict(result, h.pops);
}

// run_f4ratio: a list of (p1,p2,p3,p4,p5) 5-tuple NAME tuples against the SAME resident f2,
// computed BATCHED (run_f4ratio). Returns ONE dict of parallel arrays {pop1..pop5,alpha,se,z}
// in input order. The FIVE-column clone of run_f4_py: resolve names against pops.txt, build
// (cached) resources, upload the host tensor INSIDE the call (the DeviceF2Blocks lives only
// here and frees in its destructor — spike #1), run the CUDA-free seam, marshal.
nb::dict run_f4ratio_py(F2Handle& h,
                        const std::vector<std::array<std::string, 5>>& tuples) {
    if (tuples.empty()) raise_value("f4-ratio: needs at least one (p1,p2,p3,p4,p5) tuple");

    const sa::PopResolver resolver(h.pops);
    if (!resolver.valid()) raise_value(resolver.error());

    std::vector<std::array<int, 5>> idx_tuples;
    idx_tuples.reserve(tuples.size());
    for (const std::array<std::string, 5>& t : tuples)
        idx_tuples.push_back(resolve_tuple<5>(resolver, t, "f4-ratio pop"));

    steppe::QpAdmOptions opts;  // f4-ratio uses fudge=0 internally (run_f4ratio); opts default.

    const steppe::F4RatioResult result =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_f4ratio(
                dev_f2, std::span<const std::array<int, 5>>(idx_tuples), opts, resources);
        });
    return f4ratio_to_dict(result, h.pops);
}

// qpadm_search: a list of explicit (left, right) models against the SAME resident f2,
// fit BATCHED (run_qpadm_search). Returns one dict per model in INPUT order
// (results[k].model_index resolves the k-th input; qpadm.hpp:190-191).
nb::list run_qpadm_search_py(F2Handle& h, const std::string& target,
                             const std::vector<std::vector<std::string>>& lefts,
                             const std::vector<std::string>& right, int rank, double fudge,
                             int als_iterations, double rank_alpha,
                             bool allow_negative_weights, const std::string& jackknife) {
    if (target.empty()) raise_value("qpadm_search: target is required");
    if (lefts.empty()) raise_value("qpadm_search: models needs at least one (left) entry");
    if (right.empty())
        raise_value("qpadm_search: right needs at least one outgroup population");

    const sa::PopResolver resolver(h.pops);
    if (!resolver.valid()) raise_value(resolver.error());

    const sa::ResolveResult t = resolver.resolve(target);
    if (!t.ok) throw nb::key_error(("target: " + t.error).c_str());
    const std::vector<int> right_idx = resolve_names(resolver, right, "right");

    std::vector<steppe::QpAdmModel> models;
    models.reserve(lefts.size());
    for (std::size_t i = 0; i < lefts.size(); ++i) {
        if (lefts[i].empty())
            raise_value("qpadm_search: model " + std::to_string(i) +
                        " has an empty left source list");
        steppe::QpAdmModel m;
        m.target = t.index;
        m.left = resolve_names(resolver, lefts[i], "left");
        m.right = right_idx;
        m.model_index = static_cast<int>(i);
        models.push_back(std::move(m));
    }

    const steppe::QpAdmOptions opts = make_options(
        fudge, als_iterations, rank, allow_negative_weights, rank_alpha, jackknife);

    const std::vector<steppe::QpAdmResult> results =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_qpadm_search(
                dev_f2, std::span<const steppe::QpAdmModel>(models), opts, resources);
        });

    nb::list out;
    for (const steppe::QpAdmResult& r : results) out.append(result_to_dict(r));
    return out;
}

// ---- f2.to_numpy: the host f2 tensor -> numpy float64, F-contiguous (P, P, n_block) --
// Wraps the ALREADY-RESIDENT host F2BlockTensor.f2 (no D2H — read_f2 holds it). A COPY is
// the conservative M(py-1) choice (cli-bindings.md §5.3): own the buffer with a capsule
// deleter so there is no lifetime hazard tying the array to the handle. F-contiguous
// (P, P, n_block) so arr[:,:,b] is slab b with no silent per-slab transpose (spike #4);
// float64 from the FP64 tensor (spike #3).
nb::object f2_to_numpy(const F2Handle& h, const std::vector<double>& src) {
    const std::size_t P = static_cast<std::size_t>(h.tensor.P);
    const std::size_t nb_count = static_cast<std::size_t>(h.tensor.n_block);
    const std::size_t n = h.tensor.size();

    // COPY into a heap buffer owned by the capsule (frees with the numpy array).
    auto* buf = new double[n];
    for (std::size_t i = 0; i < n; ++i) buf[i] = src[i];
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });

    const std::size_t shape[3] = {P, P, nb_count};
    // Column-major within a slab (i + P*j) and block-major outer (+ P*P*b) IS exactly
    // F-contiguous (P, P, n_block): the fastest-varying axis is i (stride 1), then j
    // (stride P), then b (stride P*P) — the default Fortran strides for this shape.
    return nb::cast(nb::ndarray<nb::numpy, double, nb::ndim<3>, nb::f_contig>(
        buf, 3, shape, owner));
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "steppe._core — GPU bindings for the qpAdm/qpWave fit plus the f-stat / "
              "graph / dating tools (f3/f4/f4ratio, qpdstat/dstat, qpfstats, qpgraph + "
              "qpgraph-search, dates, extract-f2/read-f2); M(py-2)+, marshalling only.";

    nb::class_<F2Handle>(m, "F2Handle", "Opaque f2-dir handle: host f2 tensor + pops.")
        .def_prop_ro("pops", [](const F2Handle& h) { return h.pops; },
                     "The P population labels in P-axis index order (the name<->index map).")
        .def_prop_ro("P", &F2Handle::P, "Population count P.")
        .def_prop_ro("n_block", &F2Handle::n_block, "Number of jackknife blocks.")
        .def_prop_ro("block_sizes",
                     [](const F2Handle& h) { return h.tensor.block_sizes; },
                     "Per-block SNP counts (length n_block).")
        .def_prop_ro("device", [](const F2Handle& h) { return h.device; },
                     "The CUDA ordinal the fits target.")
        .def("_f2_numpy",
             [](const F2Handle& h) { return f2_to_numpy(h, h.tensor.f2); },
             "The f2 tensor as numpy float64, F-contiguous (P, P, n_block). COPY.")
        .def("_vpair_numpy",
             [](const F2Handle& h) { return f2_to_numpy(h, h.tensor.vpair); },
             "The vpair tensor as numpy float64, F-contiguous (P, P, n_block). COPY.");

    m.def("read_f2", &read_f2, "dir"_a, "device"_a = 0,
          nb::rv_policy::take_ownership,
          "Load an f2-dir (f2.bin STPF2BK1 + pops.txt) into an opaque F2Handle.");

    m.def("run_qpadm", &run_qpadm_py, "f2"_a, "target"_a, "left"_a, "right"_a,
          "rank"_a = -1, "fudge"_a = 1e-4, "als_iterations"_a = 20, "rank_alpha"_a = 0.05,
          "allow_negative_weights"_a = true,
          "Single-model qpAdm fit (GPU). Returns a flat dict of result fields.");

    m.def("run_qpwave", &run_qpwave_py, "f2"_a, "left"_a, "right"_a, "fudge"_a = 1e-4,
          "rank_alpha"_a = 0.05,
          "qpWave rank-sufficiency sweep (GPU; left[0] is the reference). Flat dict.");

    m.def("run_qpgraph", &run_qpgraph_py, "f2"_a, "edges"_a, "numstart"_a = 10,
          "fudge"_a = 1e-4, "diag_f3"_a = 1e-5, "constrained"_a = true,
          "Single-graph qpGraph fit (GPU; the IDEA-1 fleet on-device). `edges` is a list "
          "of (parent, child) name pairs; the leaves must be f2 populations. NOTE: qpGraph "
          "uses the AT2 afprod=FALSE f2 (read_f2 of an afprod=FALSE dir). Returns a flat "
          "dict {score, weight, admix_from/to, edge_length, edge_from/to, ...}.");

    m.def("run_qpgraph_search", &run_qpgraph_search_py, "f2"_a, "pops"_a, "max_nadmix"_a = 1,
          "numstart"_a = 10, "fudge"_a = 1e-4, "diag_f3"_a = 1e-5, "constrained"_a = true,
          "run_heuristic"_a = true,
          "qpGraph TOPOLOGY SEARCH v1 (GPU; the heterogeneous-topology fleet, ONE launch fits "
          "ALL candidates). Exhaustively enumerates every rooted topology on the bounded `pops` "
          "leaf set (nadmix in {0..max_nadmix}; reproduces admixtools generate_all_graphs 1:1) "
          "and returns the deterministic global-best. Flat dict {n_trees, n_admix1, "
          "n_candidates, best_score, second_best_score, best_nadmix, best_hash, best_edges, "
          "heuristic_recovered, fit_all_wall_ms, topologies_per_s}.");

    m.def("run_f4", &run_f4_py, "f2"_a, "quartets"_a,
          "Standalone f4(p1,p2;p3,p4) (GPU). `quartets` is a list of (p1,p2,p3,p4) name "
          "tuples; returns a dict of parallel arrays {pop1,pop2,pop3,pop4,est,se,z,p}.");

    m.def("run_qpdstat", &run_qpdstat_py, "f2"_a, "quartets"_a,
          "D-statistic / f4 over the f2-data path (GPU; qpDstat Part A). The --f2-dir "
          "qpdstat path reports f4 (the AT2 f2-path convention: qpdstat(f2dir,f4mode) is "
          "byte-identical to f4, f4mode being a no-op without per-SNP genotypes). `quartets` "
          "is a list of (p1,p2,p3,p4) name tuples; returns a dict of parallel arrays "
          "{pop1,pop2,pop3,pop4,est,se,z,p}. The normalized-D magnitude needs a genotype "
          "prefix (Part B, not yet implemented).");

    m.def("run_dstat", &run_dstat_py, "prefix"_a, "quadruples"_a, "blgsize"_a = 0.05,
          "device"_a = 0,
          "Genotype-path NORMALIZED-D (GPU; qpDstat Part B). Reads the genotype triple "
          "<prefix>.{geno,snp,ind} directly (NOT the f2 cache): D = mean_snp(num)/mean_snp(den), "
          "num=(a-b)(c-d), den=(a+b-2ab)(c+d-2cd) over per-SNP allele freqs, block-jackknifed. "
          "`quadruples` is a list of (p1,p2,p3,p4) name tuples; `blgsize` is the block size in "
          "MORGANS (AT2 default 0.05). Returns a dict of parallel arrays {pop1,pop2,pop3,pop4,"
          "est,se,z,p} (the AT2 D sign/Z/p convention). Forced diploid + allsnps=TRUE + "
          "autosomes-only are pinned (AT2 qpdstat_geno parity).");

    m.def("run_dates", &run_dates_py, "prefix"_a, "target"_a, "source1"_a, "source2"_a,
          "device"_a = 0,
          "Admixture DATING (GPU; the DATES tool). Reads the genotype triple "
          "<prefix>.{geno,snp,ind} directly (NOT the f2 cache) and infers the time since "
          "admixture from the weighted ancestry-covariance decay vs genetic distance (the cuFFT "
          "autocorrelation LD engine; NO host O(M^2) SNP-pair loop). `target` is the admixed "
          "population; `source1`/`source2` are the two reference sources (weight = "
          "freq(source1)-freq(source2)). The .snp must carry a real cM genetic map. Returns a "
          "dict {target,source1,source2,date_gen,se,fit_error_sd,curve_cm,curve_corr,status}; "
          "date_gen is generations since admixture, se the leave-one-chromosome block jackknife.");

    m.def("run_extract_f2", &run_extract_f2_py, "prefix"_a, "pops"_a, "out"_a = "",
          "device"_a = 0, "blgsize"_a = 0.05, "maf"_a = 0.0, "maxmiss"_a = 0.0,
          "autosomes_only"_a = true, "drop_monomorphic"_a = false,
          "transversions_only"_a = false, "ploidy"_a = "auto",
          "precision"_a = nb::none(),
          "Build an f2_blocks tensor from a genotype prefix (GPU; M(py-2) extract-f2). "
          "Reads <prefix>.{geno,snp,ind} directly and runs decode->filter->assign_blocks->"
          "tiered f2 compute->to_host (the SAME chain as the CLI extract-f2). `pops` is the "
          "Explicit population subset (the P axis is that selection sorted ASC by label); "
          "`blgsize` is MORGANS (AT2 default 0.05); `maxmiss` is the AT2 POP-axis coverage "
          "(0 = global intersection). If `out` is given, writes an STPF2BK1 dir there and "
          "returns the path string; else returns a new F2Handle (no disk round-trip). "
          "GPU-only: no CUDA device raises.");

    m.def("run_qpfstats", &run_qpfstats_py, "prefix"_a, "pops"_a, "out"_a = "",
          "device"_a = 0, "blgsize"_a = 0.05, "precision"_a = nb::none(),
          "Genotype-path JOINT f2 SMOOTHER (GPU; admixtools::qpfstats). Reads "
          "<prefix>.{geno,snp,ind} directly, drives the qpDstat-B numerator engine over the "
          "FULL f2/f3/f4 popcomb set, runs the on-device shared-factor smoothing regression, "
          "and returns a SMOOTHED per-block f2 tensor. `pops` is the smoothing pop set (sorted "
          "ASC internally = the AT2 dimnames order); `blgsize` is MORGANS (AT2 default 0.05). "
          "If `out` is given, writes an STPF2BK1 dir there and returns the path string; else "
          "returns a new F2Handle (read_f2/run_f4/run_qpadm consume it). GPU-only: no CUDA "
          "device raises.");

    m.def("run_f3", &run_f3_py, "f2"_a, "triples"_a,
          "Standalone f3(C;A,B) (GPU). `triples` is a list of (C,A,B) name tuples; "
          "returns a dict of parallel arrays {pop1,pop2,pop3,est,se,z,p}.");

    m.def("run_f4ratio", &run_f4ratio_py, "f2"_a, "tuples"_a,
          "Standalone f4-ratio alpha = f4(p1,p2;p3,p4)/f4(p1,p2;p5,p4) (GPU). `tuples` is a "
          "list of (p1,p2,p3,p4,p5) name tuples; returns a dict of parallel arrays "
          "{pop1,pop2,pop3,pop4,pop5,alpha,se,z}.");

    m.def("run_qpadm_search", &run_qpadm_search_py, "f2"_a, "target"_a, "lefts"_a,
          "right"_a, "rank"_a = -1, "fudge"_a = 1e-4, "als_iterations"_a = 20,
          "rank_alpha"_a = 0.05, "allow_negative_weights"_a = true,
          "jackknife"_a = "all",
          "Batched qpAdm over a list of left-source sets (shared target/right). "
          "Returns a list of per-model dicts in input order.");
}
