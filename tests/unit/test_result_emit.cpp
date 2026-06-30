// tests/unit/test_result_emit.cpp
//
// C2 unit gate for the shared output-escaping primitives in result_emit:
//   * app::csv_field  — RFC-4180 CONDITIONAL quoting (bare unless the field carries the
//                       active separator, a double quote, or a CR/LF).
//   * app::json_quote — minimal JSON string escaping (quotes + backslash + \n/\r/\t).
//
// These are the seam the dates / qpgraph / fstat-sweep emitters route every label cell
// through. The acceptance contract is TWO-SIDED:
//   (a) a NORMAL pop label is byte-for-byte BARE in CSV and byte-identical to the old
//       manual `"` + label + `"` in JSON — so NO committed golden / CLI test moves; and
//   (b) a pathological label with a comma / quote / newline is correctly quoted+escaped.
//
// PLAIN C++ host TU (NO CUDA, NO GPU, NO data): it compiles result_emit.cpp directly and
// calls the two pure string primitives. Self-checking main() (returns non-zero on failure;
// CTest gates on the exit code), so it does NOT link gtest_main.
#include <cstdio>
#include <string>

#include "app/result_emit.hpp"

namespace {

int g_failures = 0;

void check(const char* what, const std::string& got, const std::string& want) {
    const bool ok = (got == want);
    std::printf("  [%s] %-28s got=[%s] want=[%s]\n", ok ? "PASS" : "FAIL", what, got.c_str(),
                want.c_str());
    if (!ok) ++g_failures;
}

}  // namespace

int main() {
    using steppe::app::csv_field;
    using steppe::app::json_quote;
    std::printf("=== result_emit: csv_field (RFC-4180 conditional) + json_quote (C2) ===\n");

    // --- csv_field: a real pop label stays BARE (byte-identical to today's output) ------
    check("csv normal bare",   csv_field("England_BellBeaker", ','), "England_BellBeaker");
    check("csv empty bare",    csv_field("", ','),                    "");
    // A tab is NOT special for a comma field — the rule is keyed on the ACTIVE separator.
    check("csv tab not sep",   csv_field("a\tb", ','),                "a\tb");

    // --- csv_field: special chars force quoting (and embedded quotes DOUBLE) -------------
    check("csv comma quoted",  csv_field("a,b", ','),                 "\"a,b\"");
    check("csv quote doubled", csv_field("a\"b", ','),                "\"a\"\"b\"");
    check("csv newline quoted",csv_field("a\nb", ','),                "\"a\nb\"");
    check("csv cr quoted",     csv_field("a\rb", ','),                "\"a\rb\"");
    // The separator IS the tab for a TSV field, so a tab now forces quoting.
    check("csv tab is sep",    csv_field("a\tb", '\t'),               "\"a\tb\"");

    // --- json_quote: a real label == the old manual `"` + label + `"` --------------------
    check("json normal",       json_quote("England_BellBeaker"),      "\"England_BellBeaker\"");
    check("json empty",        json_quote(""),                        "\"\"");

    // --- json_quote: escape quote / backslash / control ---------------------------------
    check("json quote escaped",     json_quote("a\"b"),  "\"a\\\"b\"");
    check("json backslash escaped", json_quote("a\\b"),  "\"a\\\\b\"");
    check("json newline escaped",   json_quote("a\nb"),  "\"a\\nb\"");

    std::printf("\nRESULT: %s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
