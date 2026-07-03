// src/app/result_emit.hpp
//
// The command-line tool's result serializer: turns the compute engine's result
// structs (qpAdm fits, qpWave rank sweeps, standalone f4/f3/f4-ratio) into CSV,
// TSV, or JSON on an output stream. App-only and CUDA-free — the library never
// prints; only the CLI writes stdout.
//
// Reference: docs/reference/src_app_result_emit.hpp.md
#ifndef STEPPE_APP_RESULT_EMIT_HPP
#define STEPPE_APP_RESULT_EMIT_HPP

#include <ostream>
#include <span>
#include <string>
#include <vector>

#include "steppe/f3.hpp"
#include "steppe/f4.hpp"
#include "steppe/f4ratio.hpp"
#include "steppe/qpadm.hpp"

namespace steppe::app {

// Output format selector — reference §2
enum class OutputFormat { Csv, Tsv, Json };

// Format helpers — reference §3
[[nodiscard]] bool parse_output_format(const std::string& token, OutputFormat& out);

[[nodiscard]] std::string csv_field(const std::string& s, char sep);

[[nodiscard]] std::string json_quote(const std::string& s);

// Serialize one qpAdm result — reference §4
void emit_qpadm_result(std::ostream& os, OutputFormat fmt,
                       const QpAdmResult& result,
                       const std::string& target_label,
                       const std::vector<std::string>& left_labels);

// Serialize a qpAdm rotation table — reference §5
void emit_rotation_table(std::ostream& os, OutputFormat fmt,
                         std::span<const QpAdmResult> results,
                         const std::string& target_label,
                         const std::vector<std::vector<std::string>>& left_labels_per_model,
                         int right_n);

// Serialize a qpWave rank sweep — reference §6
void emit_qpwave_result(std::ostream& os, OutputFormat fmt,
                        const QpWaveResult& result,
                        const std::vector<std::string>& left_labels,
                        int right_n);

// Standalone f-statistic emitters (f4, f3, f4-ratio) — reference §7
void emit_f4_result(std::ostream& os, OutputFormat fmt,
                    const F4Result& result,
                    const std::vector<std::string>& p1_labels,
                    const std::vector<std::string>& p2_labels,
                    const std::vector<std::string>& p3_labels,
                    const std::vector<std::string>& p4_labels);

void emit_f3_result(std::ostream& os, OutputFormat fmt,
                    const F3Result& result,
                    const std::vector<std::string>& p1_labels,
                    const std::vector<std::string>& p2_labels,
                    const std::vector<std::string>& p3_labels);

void emit_f4ratio_result(std::ostream& os, OutputFormat fmt,
                         const F4RatioResult& result,
                         const std::vector<std::string>& p1_labels,
                         const std::vector<std::string>& p2_labels,
                         const std::vector<std::string>& p3_labels,
                         const std::vector<std::string>& p4_labels,
                         const std::vector<std::string>& p5_labels);

}  // namespace steppe::app

#endif  // STEPPE_APP_RESULT_EMIT_HPP
