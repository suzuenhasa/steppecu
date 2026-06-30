// bindings/module.cpp — the steppe._core nanobind module assembler (M(py-1)+).
//
// THIN ASSEMBLER: this TU is just the NB_MODULE shell. The per-tool binding code lives in
// the bind_<tool>.cpp TUs (bind_f2handle / bind_qpadm / bind_qpgraph / bind_fstats /
// bind_dates), each exposing a `void register_<tool>(nb::module_&)`; the shared marshalling
// helpers (parse_precision / f2_to_numpy / the result-struct converters / the F2Handle
// type) live in internal/bind_common.hpp. The body below calls each register_* in
// registration order — register_f2handle FIRST so the F2Handle nanobind type is registered
// before any fit entry that takes it.
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
#include <nanobind/nanobind.h>

#include "internal/bind_common.hpp"

NB_MODULE(_core, m) {
    m.doc() = "steppe._core — GPU bindings for the qpAdm/qpWave fit plus the f-stat / "
              "graph / dating tools (f3/f4/f4ratio, qpdstat/dstat, qpfstats, qpgraph + "
              "qpgraph-search, dates, extract-f2/read-f2); M(py-2)+, marshalling only.";

    // Registration order preserved: F2Handle (+ read_f2) FIRST so its nanobind type exists
    // before any fit entry (qpadm / qpgraph / fstats) takes or returns it.
    steppe::pybind::register_f2handle(m);
    steppe::pybind::register_qpadm(m);
    steppe::pybind::register_qpgraph(m);
    steppe::pybind::register_fstats(m);
    steppe::pybind::register_dates(m);
}
