// bindings/module.cpp — the steppe._core nanobind module assembler.
//
// Thin assembler: just the NB_MODULE shell that calls each register_<tool> in
// order. register_f2handle runs FIRST so the F2Handle nanobind type is
// registered before any fit entry that takes it.
#include <nanobind/nanobind.h>

#include "internal/bind_common.hpp"

NB_MODULE(_core, m) {
    m.doc() = "steppe._core — GPU bindings for the qpAdm/qpWave fit plus the f-stat / "
              "graph / dating tools (f3/f4/f4ratio, qpdstat/dstat, qpfstats, qpgraph + "
              "qpgraph-search, dates, extract-f2/read-f2); M(py-2)+, marshalling only.";

    steppe::pybind::register_f2handle(m);
    steppe::pybind::register_qpadm(m);
    steppe::pybind::register_qpgraph(m);
    steppe::pybind::register_fstats(m);
    steppe::pybind::register_dates(m);
}
