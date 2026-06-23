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

/// Serialize a qpAdm ROTATION result set (M(cli-3); cli-bindings.md §4.4 `qpadm-rotate`)
/// to `os` in `fmt` — ONE ROW PER MODEL, in input (model_index) order. This is a sibling
/// of emit_qpadm_result (NOT the single-model four-section layout): the rotation reports
/// the per-model feasibility table the batched run_qpadm_search returns.
///
/// `results` are the per-model fits (results[i] resolves the i-th input model — the
/// engine returns them in input order, qpadm.hpp:190). `target_label` is the shared
/// resolved target name; `left_labels_per_model[i]` are the resolved left-source names of
/// model i (len == results.size(); used for the row's `left` column and the per-weight
/// label headers). `right_n` is the convention nr (right.size()-1) reported per row.
///
/// CSV/TSV: a `# section: rotation` header + a column-header row + one data row per model.
/// JSON: a `{ "models": [ {model_index,left,weight,se,z,p,chisq,dof,f4rank,feasible,
/// status}, ... ] }` array mirroring golden_rot.json's models[] shape so a run diffs
/// directly against the golden. Total: a domain-outcome row emits NA/null sentinels.
void emit_rotation_table(std::ostream& os, OutputFormat fmt,
                         std::span<const QpAdmResult> results,
                         const std::string& target_label,
                         const std::vector<std::vector<std::string>>& left_labels_per_model,
                         int right_n);

/// Serialize one qpWave rank-sweep result (M(cli-2); cli-bindings.md §4.1 row `qpwave`)
/// to `os` in `fmt`. qpWave has NO target and produces NO admixture weights/popdrop; the
/// result is the per-rank rank-sufficiency sweep + the AT2-shaped rankdrop table + the
/// f4rank/est_rank/status summary (QpWaveResult, qpadm.hpp:212). `left_labels` are the
/// resolved left-source names (left_labels[0] is the qpWave reference row), echoed into
/// the output header so a run is human-readable; `right_n` is the convention nr
/// (right.size()-1). The serialization REUSES emit_qpadm_result's format primitives
/// (fmt_double/json_double/fmt_dofdiff/quote/status + the parallel-array lambdas), so the
/// rankdrop block is byte-shaped exactly like golden_qpwave.json's per-model rankdrop{}.
///
/// CSV/TSV: three `# section: NAME` sections — `rankdrop` (f4rank-DESCENDING, the AT2
/// res$rankdrop table: f4rank,dof,chisq,p,dofdiff,chisqdiff,p_nested), `per_rank` (the
/// ASCENDING-r sweep: rank,chisq,dof,p), and `summary` (f4rank,est_rank,status,precision).
/// JSON: a single object { left[], right_n, rankdrop{}, per_rank{}, summary{} } mirroring
/// golden_qpwave.json's rankdrop parallel-array shape. Total: NA/null sentinels for the
/// last-row nested-diff NA (dofdiff==INT_MIN, chisqdiff/p_nested==NaN), never throws.
void emit_qpwave_result(std::ostream& os, OutputFormat fmt,
                        const QpWaveResult& result,
                        const std::vector<std::string>& left_labels,
                        int right_n);

/// Serialize a STANDALONE f4 result table (the `steppe f4` command) to `os` in `fmt` —
/// ONE ROW PER QUARTET, in input order, exactly the regenerated golden schema
/// (golden_fit0_f4_readf2.csv): columns pop1,pop2,pop3,pop4,est,se,z,p so the parity test
/// diffs row-for-row. `p1..p4 labels` are the resolved population names of each quartet
/// (len == result.est; the result carries the indices, the app resolves them back to
/// names). This REUSES the SAME OutputFormat / fmt_double / json_double / csv_quote /
/// json_quote / status_str format primitives emit_qpwave_result reuses (no compute / no
/// format duplicated). Total: a domain-outcome row (NaN est/se) emits NA/null sentinels,
/// never throws.
void emit_f4_result(std::ostream& os, OutputFormat fmt,
                    const F4Result& result,
                    const std::vector<std::string>& p1_labels,
                    const std::vector<std::string>& p2_labels,
                    const std::vector<std::string>& p3_labels,
                    const std::vector<std::string>& p4_labels);

/// Serialize a STANDALONE f3 result table (the `steppe f3` command) to `os` in `fmt` —
/// ONE ROW PER TRIPLE, in input order, exactly the fixture-matched golden schema
/// (golden_fit0_f3_readf2.csv): columns pop1,pop2,pop3,est,se,z,p so the parity test diffs
/// row-for-row. The THREE-slab clone of emit_f4_result (drop the pop4 column). `p1..p3
/// labels` are the resolved population names of each triple (len == result.est; the result
/// carries the indices, the app resolves them back to names). This REUSES the SAME
/// OutputFormat / fmt_double / json_double / csv_quote / json_quote / status_str format
/// primitives emit_f4_result reuses (no compute / no format duplicated). A domain-outcome
/// row (NaN est/se) emits NA/null sentinels, never throws.
void emit_f3_result(std::ostream& os, OutputFormat fmt,
                    const F3Result& result,
                    const std::vector<std::string>& p1_labels,
                    const std::vector<std::string>& p2_labels,
                    const std::vector<std::string>& p3_labels);

/// Serialize a STANDALONE f4-ratio result table (the `steppe f4-ratio` command) to `os` in
/// `fmt` — ONE ROW PER 5-TUPLE, in input order, exactly the fixture-matched golden schema
/// (golden_fit0_f4ratio_readf2.csv): columns pop1,pop2,pop3,pop4,pop5,alpha,se,z (NO p
/// column — AT2 qpf4ratio emits only alpha/se/z). The FIVE-column clone of emit_f4_result
/// (drop the est/p, add pop5 + alpha). `p1..p5 labels` are the resolved population names of
/// each 5-tuple (len == result.alpha). REUSES the SAME OutputFormat / fmt_double / json_double
/// / csv_quote / json_quote / status_str format primitives (no compute / no format
/// duplicated). A domain-outcome row (NaN alpha/se) emits NA/null sentinels, never throws.
void emit_f4ratio_result(std::ostream& os, OutputFormat fmt,
                         const F4RatioResult& result,
                         const std::vector<std::string>& p1_labels,
                         const std::vector<std::string>& p2_labels,
                         const std::vector<std::string>& p3_labels,
                         const std::vector<std::string>& p4_labels,
                         const std::vector<std::string>& p5_labels);

}  // namespace steppe::app

#endif  // STEPPE_APP_RESULT_EMIT_HPP
