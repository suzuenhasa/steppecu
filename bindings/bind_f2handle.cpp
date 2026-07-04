// bindings/bind_f2handle.cpp — the F2Handle opaque type + read_f2 loader (steppe._core).
//
// register_f2handle() must run first in module.cpp's NB_MODULE shell, so F2Handle is
// registered before the fit bindings that take or return it.
#include <string>

#include "internal/bind_common.hpp"

#include "app/f2_dir_io.hpp"

namespace steppe::pybind {
namespace {

F2Handle* read_f2(const std::string& dir, int device) {
    const sa::F2DirResult res = sa::read_f2_dir(dir);
    if (!res.ok) raise_value(res.error);
    auto* h = new F2Handle();
    h->tensor = res.dir.f2;
    h->pops = res.dir.pop_labels;
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
