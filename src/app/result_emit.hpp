// src/app/result_emit.hpp
//
// QpAdm and standalone f-stat results -> tidy CSV / TSV / JSON serialization. App-side
// only: the CLI owns stdout, the library never prints. Plain C++20, no CUDA header.
//
// Column names and JSON shapes mirror ADMIXTOOLS 2 (AT2), so downstream AT2 scripts read
// the output unchanged and a run diffs directly against a golden file. CSV/TSV output is
// sectioned — each block is prefixed by a `# section: NAME` comment line so a parser can
// split the stream. Serialization is total: a domain outcome (empty weights, NaN estimate,
// undefined chi-square) emits "NA"/null sentinels rather than throwing, so filter on the
// per-row `status` column, not on `p`.
#ifndef STEPPE_APP_RESULT_EMIT_HPP
#define STEPPE_APP_RESULT_EMIT_HPP

#include <ostream>
#include <span>
#include <string>
#include <vector>

#include "steppe/f3.hpp"      // steppe::F3Result (the standalone-f3 emitter)
#include "steppe/f4.hpp"      // steppe::F4Result (the standalone-f4 emitter)
#include "steppe/f4ratio.hpp" // steppe::F4RatioResult (the standalone-f4-ratio emitter)
#include "steppe/qpadm.hpp"   // steppe::QpAdmResult

namespace steppe::app {

/// The on-the-wire output format selector (parsed from --format; default Csv).
enum class OutputFormat { Csv, Tsv, Json };

/// Map a --format token ("csv"/"tsv"/"json") to OutputFormat; false on an unknown token.
/// Re-mapped here at emit time so this serializer is self-contained.
[[nodiscard]] bool parse_output_format(const std::string& token, OutputFormat& out);

/// Conditionally quote a CSV field per RFC 4180: returns `s` unchanged unless it contains
/// a separator, quote, or newline, in which case it wraps `s` in double quotes and doubles
/// any embedded quote. Distinct from the always-quote form the qpadm/f4 golden CSVs need:
/// the dates / qpgraph / fstat-sweep emitters write bare labels whose CLI gates split on
/// bare tokens, so conditional quoting keeps every real population name byte-identical while
/// still escaping a pathological one.
[[nodiscard]] std::string csv_field(const std::string& s, char sep);

/// Minimal JSON string escaping for a label (quotes, backslash, and the \n/\r/\t controls),
/// returning the value WITH its surrounding double quotes. Shared so the JSON emitters
/// (dates / qpgraph / fstat-sweep) escape labels rather than concatenating a raw `"` +
/// label + `"`. For a label with no quote/backslash/control this is byte-identical to that
/// manual concatenation.
[[nodiscard]] std::string json_quote(const std::string& s);

/// Serialize one qpAdm result to `os` in `fmt`. `target_label` is the resolved target name;
/// `left_labels` are the resolved left-source names (len == result.weight when the fit
/// produced weights), used to label the per-source rows and popdrop columns. Total: a
/// domain-outcome result (empty weights/se) emits NA sentinels, never throws.
void emit_qpadm_result(std::ostream& os, OutputFormat fmt,
                       const QpAdmResult& result,
                       const std::string& target_label,
                       const std::vector<std::string>& left_labels);

/// Serialize a qpAdm ROTATION result set to `os` in `fmt` — one row per model, in input
/// (model_index) order. Sibling of emit_qpadm_result, not the single-model four-section
/// layout: the rotation reports the per-model feasibility table run_qpadm_search returns.
///
/// `results[i]` is the fit of the i-th input model (engine returns them in input order).
/// `target_label` is the shared resolved target name; `left_labels_per_model[i]` are model
/// i's resolved left-source names (len == results.size(); used for the row's `left` column
/// and the per-weight headers). `right_n` is the convention nr (right.size()-1) per row.
///
/// CSV/TSV: a `# section: rotation` header, a column-header row, and one data row per model.
/// JSON: a `{ "models": [ {model_index,left,weight,se,z,p,chisq,dof,f4rank,feasible,
/// status}, ... ] }` array that diffs directly against the golden. Total: a domain-outcome
/// row emits NA/null sentinels.
void emit_rotation_table(std::ostream& os, OutputFormat fmt,
                         std::span<const QpAdmResult> results,
                         const std::string& target_label,
                         const std::vector<std::vector<std::string>>& left_labels_per_model,
                         int right_n);

/// Serialize one qpWave rank-sweep result to `os` in `fmt`. qpWave has no target and
/// produces no admixture weights/popdrop; the result is the per-rank sufficiency sweep, the
/// AT2-shaped rankdrop table, and the f4rank/est_rank/status summary. `left_labels` are the
/// resolved left-source names (left_labels[0] is the reference row), echoed into the header;
/// `right_n` is the convention nr (right.size()-1). Reuses emit_qpadm_result's format
/// primitives, so the rankdrop block is byte-shaped exactly like the golden.
///
/// CSV/TSV: three `# section: NAME` sections — `rankdrop` (f4rank-descending, the AT2
/// rankdrop table: f4rank,dof,chisq,p,dofdiff,chisqdiff,p_nested), `per_rank` (ascending-r
/// sweep: rank,chisq,dof,p), and `summary` (f4rank,est_rank,status,precision). JSON: a
/// single object { left[], right_n, rankdrop{}, per_rank{}, summary{} }. Total: NA/null
/// sentinels for the last-row nested-diff NA (dofdiff==INT_MIN, chisqdiff/p_nested==NaN),
/// never throws.
void emit_qpwave_result(std::ostream& os, OutputFormat fmt,
                        const QpWaveResult& result,
                        const std::vector<std::string>& left_labels,
                        int right_n);

/// Serialize a standalone f4 result table (the `steppe f4` command) to `os` in `fmt` — one
/// row per quartet, in input order, columns pop1,pop2,pop3,pop4,est,se,z,p so the parity
/// test diffs row-for-row. `p1..p4 labels` are the resolved population names of each quartet
/// (len == result.est; the result carries the indices, the app resolves them back to names).
/// Reuses the same format primitives as the emitters above (no compute or format
/// duplicated). Total: a domain-outcome row (NaN est/se) emits NA/null sentinels, never
/// throws.
void emit_f4_result(std::ostream& os, OutputFormat fmt,
                    const F4Result& result,
                    const std::vector<std::string>& p1_labels,
                    const std::vector<std::string>& p2_labels,
                    const std::vector<std::string>& p3_labels,
                    const std::vector<std::string>& p4_labels);

/// Serialize a standalone f3 result table (the `steppe f3` command) to `os` in `fmt` — one
/// row per triple, in input order, columns pop1,pop2,pop3,est,se,z,p so the parity test
/// diffs row-for-row. The three-slab form of emit_f4_result (drop the pop4 column). `p1..p3
/// labels` are the resolved population names of each triple (len == result.est). Reuses the
/// same format primitives (no compute or format duplicated). A domain-outcome row (NaN
/// est/se) emits NA/null sentinels, never throws.
void emit_f3_result(std::ostream& os, OutputFormat fmt,
                    const F3Result& result,
                    const std::vector<std::string>& p1_labels,
                    const std::vector<std::string>& p2_labels,
                    const std::vector<std::string>& p3_labels);

/// Serialize a standalone f4-ratio result table (the `steppe f4-ratio` command) to `os` in
/// `fmt` — one row per 5-tuple, in input order, columns pop1..pop5,alpha,se,z (no p column:
/// AT2 qpf4ratio emits only alpha/se/z). The five-column form of emit_f4_result. `p1..p5
/// labels` are the resolved population names of each 5-tuple (len == result.alpha). Reuses
/// the same format primitives (no compute or format duplicated). A domain-outcome row (NaN
/// alpha/se) emits NA/null sentinels, never throws.
void emit_f4ratio_result(std::ostream& os, OutputFormat fmt,
                         const F4RatioResult& result,
                         const std::vector<std::string>& p1_labels,
                         const std::vector<std::string>& p2_labels,
                         const std::vector<std::string>& p3_labels,
                         const std::vector<std::string>& p4_labels,
                         const std::vector<std::string>& p5_labels);

}  // namespace steppe::app

#endif  // STEPPE_APP_RESULT_EMIT_HPP
