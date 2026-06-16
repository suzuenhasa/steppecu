// src/io/ind_reader.hpp
//
// .ind parse → per-individual population labels + the pop→individual-row-index
// map and the per-pop sample list, in the DETERMINISTIC oracle ordering
// (architecture.md §4 `io` LEAF, §5 S−2/S0; ROADMAP M1).
//
// The EIGENSTRAT .ind file has one whitespace-separated record per individual:
//   <sample-id>  <sex>  <population-label>
// Row order is the individual axis of the genotype matrix (TGENO record order),
// so the row index of a .ind line IS the genotype record index — this reader is
// what binds a population to the set of genotype records it owns.
//
// POPULATION SELECTION (matches the on-box oracle build_tgeno_matrix.py exactly,
// so the decoder reproduces derived_acc bit-for-bit):
//   * Explicit list  — keep exactly the requested labels that are present.
//   * auto-top K     — the K populations with the MOST individuals (Counter
//                      .most_common(K): count descending, ties broken by FIRST
//                      appearance in .ind row order).
//   * min-n N        — every population with >= N individuals.
// In ALL modes the FINAL selected set is sorted ASCENDING by label (the oracle's
// `sel = sorted(sel)`), which fixes the population (row) ordering of Q/V/N. The
// derived_acc validation set is `auto-top 50` on the real v66 .ind.
//
// LAYERING: `io`-leaf header (architecture.md §4) — pure host C++20, depends on
// nothing in core/device.
#ifndef STEPPE_IO_IND_READER_HPP
#define STEPPE_IO_IND_READER_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace steppe::io {

/// How to select which populations become the rows of the decoded Q/V/N. Mirrors
/// the on-box oracle's three selection modes so the decoder reproduces the
/// validation matrices exactly (ROADMAP M1 oracle provenance).
struct PopSelection {
    enum class Mode {
        Explicit,  ///< keep exactly `labels` that are present (any order; result sorted)
        AutoTopK,  ///< the `k` largest populations by individual count (oracle most_common)
        MinN       ///< every population with >= `min_n` individuals
    };

    Mode mode = Mode::AutoTopK;

    /// auto-top K — number of largest populations to keep (Mode::AutoTopK).
    std::size_t k = 0;

    /// min-n threshold — keep pops with >= this many individuals (Mode::MinN).
    std::size_t min_n = 1;

    /// Explicit label list (Mode::Explicit). The present subset is kept, sorted.
    std::vector<std::string> labels;
};

/// One selected population: its label and the genotype-record (individual) row
/// indices that belong to it, in ascending row order (file order). The row
/// indices index the .ind rows == TGENO individual records.
struct PopGroup {
    std::string label;                  ///< the .ind column-3 population label
    std::vector<std::size_t> rows;      ///< individual-record indices (ascending)
};

/// The parsed .ind, partitioned into the SELECTED populations. `groups` is in the
/// final population (row) order of Q/V/N — sorted ascending by label, matching the
/// oracle. `n_individuals_total` is the .ind row count (the genotype individual
/// axis), independent of selection.
struct IndPartition {
    std::vector<PopGroup> groups;       ///< selected pops, in Q/V/N row order
    std::size_t n_individuals_total = 0;  ///< total .ind rows == TGENO records
};

/// Parse the .ind at `path` and apply the population selection. Reads the
/// population label (column 3) of every row, groups individual-record indices by
/// label, then selects per `sel` and orders the result for the Q/V/N row axis.
///
/// `n_records_present` caps the individual axis to the records actually present
/// in the .geno (the oracle uses the same `pops_all[:n_records]` cap to handle a
/// partial file); rows at index >= `n_records_present` are ignored. Pass the
/// genotype header's `n_records` for TGENO. Use SIZE_MAX to use every .ind row.
///
/// Throws std::runtime_error on a missing/unreadable file or an empty selection
/// (the `io` leaf surfaces I/O failures as exceptions to its `app`/test caller;
/// it does not depend on core/device error types — architecture.md §4).
[[nodiscard]] IndPartition read_ind(const std::string& path,
                                    const PopSelection& sel,
                                    std::size_t n_records_present);

}  // namespace steppe::io

#endif  // STEPPE_IO_IND_READER_HPP
