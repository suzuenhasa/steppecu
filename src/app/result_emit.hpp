// src/app/result_emit.hpp
//
// QpAdmResult -> tidy CSV / TSV / JSON serialization (cli-bindings.md §4.4). The CLI
// owns stdout (architecture.md §10: no printf in the library; printing is app-only);
// this is the app-side serializer. PLAIN C++20, app-only, NO CUDA header.
//
// FORMAT (cli-bindings.md §4.4):
//   * CSV/TSV (default csv): admixr-shaped SECTIONS in one stream — a `weights` table
//     (target,left,weight,se,z; one row per left source; se/z = literal "NA" when the
//     se sentinel is empty), a `summary` row (p,chisq,dof,f4rank,est_rank,feasible,
//     status,precision), the `rankdrop` table (AT2 res$rankdrop order), and the
//     `popdrop` table (AT2 res$popdrop). Column names mirror the committed golden CSVs
//     so AT2 downstream scripts read them unchanged. Each section is prefixed by a
//     `# section: NAME` comment line so a parser can split the stream.
//   * JSON: a single object mirroring the golden_fit0.json shape (weights/rankdrop/
//     popdrop blocks of parallel arrays + a summary block of scalars) so a run is
//     diff-able against a golden file.
//
// STATUS handling (cli-bindings.md §4.4): `status` is a per-row string; on
// ChisqUndefined `p` is the NaN sentinel -> emitted as "NA"; filter on status, not p.
// Domain outcomes are rows, never serialization errors.
#ifndef STEPPE_APP_RESULT_EMIT_HPP
#define STEPPE_APP_RESULT_EMIT_HPP

#include <ostream>
#include <string>
#include <vector>

#include "steppe/qpadm.hpp"  // steppe::QpAdmResult

namespace steppe::app {

/// The on-the-wire output format selector (parsed from --format; default Csv).
enum class OutputFormat { Csv, Tsv, Json };

/// Map a --format token to OutputFormat ("csv"/"tsv"/"json"). Returns false on an
/// unknown token (the caller fail-fasts; ConfigBuilder also validates it, but the app
/// re-maps here at emit time so this serializer is self-contained).
[[nodiscard]] bool parse_output_format(const std::string& token, OutputFormat& out);

/// Serialize one qpAdm result to `os` in `fmt`. `target_label` is the resolved target
/// name; `left_labels` are the resolved left-source names (len == result.weight when
/// the fit produced weights), used to label the per-source rows / popdrop columns so
/// the output is human- and AT2-readable. The serialization is total: a domain-outcome
/// result (empty weights/se) emits NA sentinels, never throws.
void emit_qpadm_result(std::ostream& os, OutputFormat fmt,
                       const QpAdmResult& result,
                       const std::string& target_label,
                       const std::vector<std::string>& left_labels);

}  // namespace steppe::app

#endif  // STEPPE_APP_RESULT_EMIT_HPP
