// src/core/qpadm/qpgraph_arena.hpp
//
// Reference: docs/reference/src_core_qpadm_qpgraph_arena.hpp.md
//
// fill_topo_arena_common() — copies the topology-invariant QpGraphTopoArena
// fields shared by the qpGraph fit and search paths. Each caller still sets the
// pair layout itself (npair, cmb1, cmb2).
#ifndef STEPPE_CORE_QPADM_QPGRAPH_ARENA_HPP
#define STEPPE_CORE_QPADM_QPGRAPH_ARENA_HPP

#include "core/qpadm/qpgraph_model.hpp"
#include "device/backend.hpp"
#include "steppe/qpgraph.hpp"

namespace steppe::core::qpadm {

// The 13 arena fields that don't depend on the pair layout — reference
// qpgraph_fit §5 / qpgraph_search §5. npair/cmb1/cmb2 stay with the caller.
inline void fill_topo_arena_common(QpGraphTopoArena& out, const QpGraphModel& m,
                                   const QpGraphOptions& opts) {
    out.npop = m.npop;
    out.nedge_norm = m.nedge_norm;
    out.nadmix = m.nadmix;
    out.npath = m.npath;
    out.base_leaf = m.base_leaf;
    out.pwts0 = m.pwts0;
    out.pe_edge = m.pe_edge;
    out.pe_leaf = m.pe_leaf;
    out.pe_path = m.pe_path;
    out.pae_path = m.pae_path;
    out.pae_admixedge = m.pae_admixedge;
    out.constrained = opts.constrained;
    out.fudge = opts.fudge;
}

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_QPGRAPH_ARENA_HPP
