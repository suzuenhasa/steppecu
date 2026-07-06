// bindings/internal/bind_common.hpp
//
// Shared marshalling helpers for the compiled steppe._core nanobind bindings; the
// per-tool bind_*.cpp translation units all pull their common helpers from here.
// Marshalling only (numpy<->span, name<->index, kwargs->options, result-structs->
// Python dicts) — no compute, no pandas — and a CUDA-free host TU that includes
// only the library's CUDA-free seams.
//
// Reference: docs/reference/bindings_internal_bind_common.hpp.md
#ifndef STEPPE_BINDINGS_INTERNAL_BIND_COMMON_HPP
#define STEPPE_BINDINGS_INTERNAL_BIND_COMMON_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "app/pop_resolver.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/resources.hpp"
#include "steppe/config.hpp"
#include "steppe/dstat.hpp"
#include "steppe/error.hpp"
#include "steppe/f3.hpp"
#include "steppe/f4.hpp"
#include "steppe/f4ratio.hpp"
#include "steppe/fstats.hpp"
#include "steppe/qpadm.hpp"
#include "steppe/qpgraph.hpp"

namespace nb = nanobind;

namespace steppe::pybind {

using namespace nb::literals;

namespace sa = steppe::app;
namespace sd = steppe::device;

// F2 directory handle — reference §3
struct F2Handle {
    steppe::F2BlockTensor tensor;
    std::vector<std::string> pops;
    int device = 0;

    std::unique_ptr<sd::Resources> resources;

    [[nodiscard]] int P() const { return tensor.P; }
    [[nodiscard]] int n_block() const { return tensor.n_block; }
};

// Fault helpers — reference §4
[[noreturn]] inline void raise_value(const std::string& msg) { throw nb::value_error(msg.c_str()); }

[[noreturn]] inline void raise_no_device() {
    raise_value(
        "no CUDA device available (steppe is a GPU product; a CUDA-capable GPU is "
        "required)");
}

inline sd::Resources& ensure_resources(F2Handle& h) {
    if (h.resources) return *h.resources;
    steppe::DeviceConfig cfg;
    cfg.devices = {h.device};
    std::unique_ptr<sd::Resources> r;
    try {
        r = std::make_unique<sd::Resources>(sd::build_resources(cfg));
    } catch (const std::exception& e) {
        raise_value(std::string("device error: ") + e.what());
    }
    if (r->gpus.empty()) raise_no_device();
    h.resources = std::move(r);
    return *h.resources;
}

// Population name/index resolution — reference §5
inline std::vector<int> resolve_names(const sa::PopResolver& resolver,
                                      const std::vector<std::string>& names, const char* what) {
    const sa::ResolveListResult r = resolver.resolve_all(names);
    if (!r.ok) {
        throw nb::key_error((std::string(what) + ": " + r.error).c_str());
    }
    return r.indices;
}

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

// Precision-string parse — reference §6
inline steppe::Precision parse_precision(const std::optional<std::string>& precision,
                                         const char* tool) {
    steppe::Precision prec = steppe::Precision::emulated_fp64();
    if (precision.has_value()) {
        const std::string& p = *precision;
        if (p == "fp64" || p == "native") {
            prec = steppe::Precision::fp64();
        } else if (p == "emu40" || p == "emu" || p == "emulated_fp64") {
            prec = steppe::Precision::emulated_fp64(40);
        } else if (p == "emu32" || p == "emulated_fp64_32") {
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

// Resident-f2 fit wrapper — reference §7
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

// Kwargs to fit options — reference §8
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

// Result structs to Python dicts — reference §9
inline const char* precision_str(steppe::Precision::Kind k) {
    switch (k) {
        case steppe::Precision::Kind::Fp64: return "fp64";
        case steppe::Precision::Kind::EmulatedFp64: return "emulated_fp64";
        case steppe::Precision::Kind::Tf32: return "tf32";
    }
    return "fp64";
}

// Shared status/precision tail: the two string fields every quartet/fit dict carries
template <class R>
void put_status_precision(nb::dict& d, const R& r) {
    d["status"] = steppe::status_str(r.status);
    d["precision"] = precision_str(r.precision_tag);
}

inline nb::dict result_to_dict(const steppe::QpAdmResult& r) {
    nb::dict d;
    d["weight"] = r.weight;
    d["se"] = r.se;
    d["z"] = r.z;
    d["p"] = r.p;
    d["chisq"] = r.chisq;
    d["dof"] = r.dof;
    d["est_rank"] = r.est_rank;
    d["f4rank"] = r.f4rank;

    d["rankdrop_f4rank"] = r.rankdrop_f4rank;
    d["rankdrop_dof"] = r.rankdrop_dof;
    d["rankdrop_dofdiff"] = r.rankdrop_dofdiff;
    d["rankdrop_chisq"] = r.rankdrop_chisq;
    d["rankdrop_p"] = r.rankdrop_p;
    d["rankdrop_chisqdiff"] = r.rankdrop_chisqdiff;
    d["rankdrop_p_nested"] = r.rankdrop_p_nested;

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

    put_status_precision(d, r);
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
    put_status_precision(d, r);
    return d;
}

// Shared p1..p4 quartet builder (est/se/z/p + status/precision) — F4Result ≡ DstatResult
template <class R>
nb::dict quartet_to_dict(const R& r, const std::vector<std::string>& pops) {
    nb::dict d;
    d["pop1"] = names_of(r.p1, pops);
    d["pop2"] = names_of(r.p2, pops);
    d["pop3"] = names_of(r.p3, pops);
    d["pop4"] = names_of(r.p4, pops);
    d["est"] = r.est;
    d["se"] = r.se;
    d["z"] = r.z;
    d["p"] = r.p;
    put_status_precision(d, r);
    return d;
}

inline nb::dict f4_to_dict(const steppe::F4Result& r, const std::vector<std::string>& pops) {
    return quartet_to_dict(r, pops);
}

inline nb::dict dstat_to_dict(const steppe::DstatResult& r, const std::vector<std::string>& pops) {
    return quartet_to_dict(r, pops);
}

inline nb::dict f3_to_dict(const steppe::F3Result& r, const std::vector<std::string>& pops) {
    nb::dict d;
    d["pop1"] = names_of(r.p1, pops);
    d["pop2"] = names_of(r.p2, pops);
    d["pop3"] = names_of(r.p3, pops);
    d["est"] = r.est;
    d["se"] = r.se;
    d["z"] = r.z;
    d["p"] = r.p;
    put_status_precision(d, r);
    return d;
}

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
    put_status_precision(d, r);
    return d;
}

inline nb::dict qpgraph_to_dict(const steppe::QpGraphResult& r) {
    nb::dict d;
    d["score"] = r.score;
    d["restart_spread"] = r.restart_spread;
    d["worst_residual_z"] = r.worst_residual_z;
    d["worst_pop2"] = r.worst_pop2;
    d["worst_pop3"] = r.worst_pop3;
    d["status"] = steppe::status_str(r.status);
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

// f2 tensor to numpy export — reference §10
inline nb::object f2_to_numpy(const F2Handle& h, const std::vector<double>& src) {
    const std::size_t P = static_cast<std::size_t>(h.tensor.P);
    const std::size_t nb_count = static_cast<std::size_t>(h.tensor.n_block);
    const std::size_t n = h.tensor.size();

    assert(src.size() == n);
    auto* buf = new double[n];
    std::memcpy(buf, src.data(), n * sizeof(double));
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });

    const std::size_t shape[3] = {P, P, nb_count};
    return nb::cast(nb::ndarray<nb::numpy, double, nb::ndim<3>, nb::f_contig>(
        buf, 3, shape, owner));
}

// Per-tool registration entry points — reference §11
void register_f2handle(nb::module_& m);
void register_qpadm(nb::module_& m);
void register_qpgraph(nb::module_& m);
void register_fstats(nb::module_& m);
void register_dates(nb::module_& m);

}  // namespace steppe::pybind

#endif  // STEPPE_BINDINGS_INTERNAL_BIND_COMMON_HPP
