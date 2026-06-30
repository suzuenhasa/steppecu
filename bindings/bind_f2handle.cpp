// bindings/bind_f2handle.cpp — the F2Handle opaque type + read_f2 loader (steppe._core).
//
// register_f2handle() runs FIRST in module.cpp's NB_MODULE shell so the F2Handle nanobind
// type is registered before any fit entry (bind_qpadm/bind_qpgraph/bind_fstats) that takes
// or returns it. Marshalling only (cli-bindings.md §5); CUDA-FREE seams (§6.2).
#include <string>

#include "internal/bind_common.hpp"

#include "app/f2_dir_io.hpp"  // steppe::app::read_f2_dir / F2Dir (CUDA-FREE)

namespace steppe::pybind {
namespace {

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

}  // namespace

void register_f2handle(nb::module_& m) {
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
}

}  // namespace steppe::pybind
