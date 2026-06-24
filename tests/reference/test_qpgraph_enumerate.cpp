// tests/reference/test_qpgraph_enumerate.cpp
//
// qpGraph topology ENUMERATOR — the EXHAUSTIVE-COVERAGE PROOF (oracle C leg 2). The host
// enumerator (enumerate_trees / enumerate_admix1) reproduces admixtools' OWN enumerator
// (generate_all_trees / generate_all_graphs) EXACTLY: the COUNT (105 trees + 1485 non-iso
// nadmix=1 graphs) AND the canonical graph_hash SET == the committed AT2 fixture
// (goldens/at2/fixtures/at2_enum_5pop.txt, dumped from admixtools 2.0.10 on box5090). So
// AT2 is the exhaustive-coverage oracle and steppe's enumeration is proven exhaustive 1:1.
//
// Also: every enumerated topology PARSES through parse_qpgraph (the single-graph data model
// the fleet consumes), and the hill-climb neighbor move set is non-empty + connected across
// the nadmix boundary (so the heuristic can reach the nadmix=1 optimum from a tree seed).
// HOST-ONLY (no CUDA). Self-checking main(); CTest gates the exit code.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/qpadm/qpgraph_enumerate.hpp"
#include "core/qpadm/qpgraph_model.hpp"

namespace {

int g_failures = 0;
void check_true(const char* what, bool ok) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

std::vector<steppe::QpGraphEdge> parse_edge_str(const std::string& s) {
    std::vector<steppe::QpGraphEdge> e;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ';')) {
        const auto gt = tok.find('>');
        if (gt == std::string::npos) continue;
        e.push_back({tok.substr(0, gt), tok.substr(gt + 1)});
    }
    return e;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace steppe;
    using namespace steppe::core::qpadm;
    const std::string golden_dir = (argc > 1) ? argv[1] : "tests/reference/goldens/at2";
    std::printf("=== qpGraph enumerator exhaustive-coverage proof (vs AT2 generate_all_graphs) ===\n");

    const std::vector<std::string> leaves = {"Mbuti", "Han", "Iran_GanjDareh_N",
                                             "Israel_Natufian", "Czechia_EBA_CordedWare"};

    // ---- load the AT2 canonical enumeration (hash via steppe's graph_hash) -----------
    std::unordered_set<std::uint64_t> at2_trees, at2_admix;
    {
        std::ifstream f(golden_dir + "/fixtures/at2_enum_5pop.txt");
        if (!f) { std::printf("  [FAIL] cannot open AT2 enum fixture\n"); std::printf("\nRESULT: FAIL\n"); return 1; }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line);
            std::string k, idx, es;
            std::getline(ss, k, '\t'); std::getline(ss, idx, '\t'); std::getline(ss, es, '\t');
            const std::uint64_t h = graph_hash(parse_edge_str(es));
            (k == "T" ? at2_trees : at2_admix).insert(h);
        }
    }
    std::printf("  AT2: trees=%zu admix=%zu\n", at2_trees.size(), at2_admix.size());
    check_true("AT2 fixture has 105 trees", at2_trees.size() == 105);
    check_true("AT2 fixture has 1485 non-iso admix-1 graphs", at2_admix.size() == 1485);

    // ---- steppe trees ----------------------------------------------------------------
    const std::vector<EnumeratedTopology> trees = enumerate_trees(leaves);
    std::unordered_set<std::uint64_t> st;
    for (const auto& t : trees) st.insert(t.hash);
    std::printf("  steppe trees: %zu (distinct hashes %zu)\n", trees.size(), st.size());
    check_true("steppe trees count == 105", trees.size() == 105);
    check_true("steppe tree hash-set == AT2 (exhaustive + non-iso)", st == at2_trees);

    // ---- steppe admix-1 --------------------------------------------------------------
    const std::vector<EnumeratedTopology> adm = enumerate_admix1(leaves);
    std::unordered_set<std::uint64_t> sa;
    for (const auto& a : adm) sa.insert(a.hash);
    std::printf("  steppe admix-1: %zu (distinct hashes %zu)\n", adm.size(), sa.size());
    check_true("steppe admix-1 count == 1485", adm.size() == 1485);
    check_true("steppe admix-1 hash-set == AT2 (exhaustive + non-iso)", sa == at2_admix);

    // ---- the bounded space + parseability --------------------------------------------
    const std::vector<EnumeratedTopology> all = enumerate_bounded_space(leaves, 1);
    check_true("bounded space == 1590 candidates", all.size() == 1590);
    int parsed = 0, parse_bad = 0, nadmix_mismatch = 0;
    for (const auto& c : all) {
        const QpGraphModel m = parse_qpgraph(c.edges, leaves, leaves.front());
        if (!m.ok()) { ++parse_bad; continue; }
        if (m.nadmix != c.nadmix) { ++nadmix_mismatch; continue; }
        ++parsed;
    }
    std::printf("  parse_qpgraph: ok=%d bad=%d nadmix_mismatch=%d\n", parsed, parse_bad, nadmix_mismatch);
    check_true("every enumerated topology parses (parse_qpgraph)", parse_bad == 0);
    check_true("every parsed nadmix matches the enumerated nadmix", nadmix_mismatch == 0);

    // ---- neighbor move set is non-empty + crosses the nadmix boundary ----------------
    // from a tree seed: there must be admix-1 neighbors (add-admix), so the hill-climb can
    // reach the nadmix=1 optimum (the global best is a nadmix=1 graph).
    const std::vector<EnumeratedTopology> nb_tree = topology_neighbors(trees.front(), leaves, 1);
    int tree_nb = 0, admix_nb = 0;
    for (const auto& n : nb_tree) (n.nadmix == 0 ? tree_nb : admix_nb) += 1;
    std::printf("  tree neighbors: %zu (tree=%d admix=%d)\n", nb_tree.size(), tree_nb, admix_nb);
    check_true("a tree has add-admix (nadmix=1) neighbors (boundary-crossing move)", admix_nb > 0);
    check_true("a tree has tree (NNI) neighbors", tree_nb > 0);
    // from an admix-1 seed: there must be drop-admix (tree) neighbors.
    const std::vector<EnumeratedTopology> nb_adm = topology_neighbors(adm.front(), leaves, 1);
    int admin_tree = 0; for (const auto& n : nb_adm) if (n.nadmix == 0) ++admin_tree;
    check_true("an admix-1 graph has drop-admix (tree) neighbors", admin_tree > 0);

    // ---- determinism: re-enumeration is identical (the (C) property find_graphs lacks) -
    const std::vector<EnumeratedTopology> adm2 = enumerate_admix1(leaves);
    bool det = adm2.size() == adm.size();
    for (std::size_t i = 0; det && i < adm.size(); ++i) det = (adm2[i].hash == adm[i].hash);
    check_true("re-enumeration is bit-stable (deterministic order + hashes)", det);

    std::printf("\nRESULT: %s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
