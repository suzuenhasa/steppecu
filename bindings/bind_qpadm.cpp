// bindings/bind_qpadm.cpp — the qpAdm / qpWave fit entries for the steppe._core module.
//
// Each entry resolves names, builds resources, uploads the host f2 tensor inside the call,
// and runs the CUDA-free fit seam; faults raise, per-model domain outcomes ride on `status`.
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "internal/bind_common.hpp"

namespace steppe::pybind {
namespace {

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

    steppe::QpAdmOptions opts;
    opts.fudge = fudge;
    opts.rank_alpha = rank_alpha;

    const steppe::QpWaveResult result =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_qpwave(dev_f2, std::span<const int>(left_idx),
                                      std::span<const int>(right_idx), opts, resources);
        });
    return qpwave_to_dict(result);
}

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

}  // namespace

void register_qpadm(nb::module_& m) {
    m.def("run_qpadm", &run_qpadm_py, "f2"_a, "target"_a, "left"_a, "right"_a,
          "rank"_a = -1, "fudge"_a = 1e-4, "als_iterations"_a = 20, "rank_alpha"_a = 0.05,
          "allow_negative_weights"_a = true,
          "Single-model qpAdm fit (GPU). Returns a flat dict of result fields.");

    m.def("run_qpwave", &run_qpwave_py, "f2"_a, "left"_a, "right"_a, "fudge"_a = 1e-4,
          "rank_alpha"_a = 0.05,
          "qpWave rank-sufficiency sweep (GPU; left[0] is the reference). Flat dict.");

    m.def("run_qpadm_search", &run_qpadm_search_py, "f2"_a, "target"_a, "lefts"_a,
          "right"_a, "rank"_a = -1, "fudge"_a = 1e-4, "als_iterations"_a = 20,
          "rank_alpha"_a = 0.05, "allow_negative_weights"_a = true,
          "jackknife"_a = "all",
          "Batched qpAdm over a list of left-source sets (shared target/right). "
          "Returns a list of per-model dicts in input order.");
}

}  // namespace steppe::pybind
