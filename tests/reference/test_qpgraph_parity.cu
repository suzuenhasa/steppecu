// tests/reference/test_qpgraph_parity.cu
//
// qpGraph single-graph-fit ACCEPTANCE: the steppe qpGraph fit (the productized IDEA-1
// optimizer-spike fleet + the path-algebra objective + the resident f3 basis) validated
// against the admixtools 2.0.10 golden (tests/reference/goldens/at2/golden_qpgraph_*,
// R 4.3.3) on BOTH backends — the CpuBackend reference (the bit-exact diff oracle + the
// no-GPU fallback) AND the CUDA backend (the PRODUCTION GPU path, f3 basis RESIDENT in
// VRAM, the fleet on-device).
//
// f2-SOURCE: the SAME committed 9-pop f2 fixture (fixtures/f2_fit0_9pop.bin) the qpAdm
// parity gate uses — the f2 admixtools::qpgraph(read_f2(...)) read internally. The
// topology is the committed WELL-IDENTIFIED golden topology (golden_qpgraph_topology.csv):
// base=Mbuti, the single admixture node aCW (=Czechia_EBA_CordedWare) mixing pSteppe (off
// the East-Eurasian clade) and pIran. The GOLDEN is the CONVERGED OPTIMUM (score
// 80.0674246076, the interior admix weight pSteppe->aCW=0.153484, all 10 restarts + 5
// seeds agree to ~1e-11) — steppe's fleet reaching the SAME minimum is parity, NOT step-
// identical L-BFGS-B.
//
// GATE: score (tight), the admix weight keyed by parent NAME (pSteppe->aCW), the restart
// spread (the convergence/identifiability witness), and CpuBackend == CudaBackend. The
// GPU block SKIPs cleanly when no CUDA device is visible. Self-checking main() (exits non-
// zero on failure; CTest gates on the exit code).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "device/backend.hpp"
#include "device/backend_factory.hpp"   // make_cpu_backend, make_cuda_backend, visible_device_count
#include "device/device_f2_blocks.hpp"  // DeviceF2Blocks, upload_f2_blocks_to_device
#include "device/resources.hpp"         // Resources / PerGpuResources
#include "steppe/error.hpp"
#include "steppe/fstats.hpp"            // F2BlockTensor
#include "steppe/qpgraph.hpp"           // run_qpgraph + QpGraphEdge/Result/Options

namespace {

int g_failures = 0;

void check_close(const char* what, double got, double want, double rtol, double atol) {
    const double tol = atol + rtol * std::fabs(want);
    const double diff = std::fabs(got - want);
    const bool ok = diff <= tol;
    std::printf("  [%s] %-28s got=% .12e want=% .12e |d|=% .3e tol=% .3e\n",
                ok ? "PASS" : "FAIL", what, got, want, diff, tol);
    if (!ok) ++g_failures;
}
void check_true(const char* what, bool ok) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

bool read_fixture(const std::string& path, steppe::F2BlockTensor& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::printf("  [FAIL] cannot open fixture: %s\n", path.c_str()); return false; }
    std::int32_t P = 0, nb = 0;
    f.read(reinterpret_cast<char*>(&P), sizeof(P));
    f.read(reinterpret_cast<char*>(&nb), sizeof(nb));
    if (!f || P <= 0 || nb <= 0) { std::printf("  [FAIL] bad fixture header\n"); return false; }
    out.P = P; out.n_block = nb;
    out.block_sizes.resize(static_cast<std::size_t>(nb));
    f.read(reinterpret_cast<char*>(out.block_sizes.data()),
           static_cast<std::streamsize>(sizeof(std::int32_t) * static_cast<std::size_t>(nb)));
    const std::size_t n = static_cast<std::size_t>(P) * static_cast<std::size_t>(P) * static_cast<std::size_t>(nb);
    out.f2.resize(n);
    f.read(reinterpret_cast<char*>(out.f2.data()), static_cast<std::streamsize>(sizeof(double) * n));
    if (!f) { std::printf("  [FAIL] fixture truncated\n"); return false; }
    std::printf("  fixture: P=%d n_block=%d\n", P, nb);
    return true;
}

// The committed golden topology (golden_qpgraph_topology.csv).
std::vector<steppe::QpGraphEdge> golden_topology() {
    return {
        {"R", "Mbuti"}, {"R", "nOOA"}, {"nOOA", "Papuan"}, {"nOOA", "nEAS"},
        {"nEAS", "Han"}, {"nEAS", "Karitiana"}, {"nOOA", "nWE"}, {"nWE", "Israel_Natufian"},
        {"nWE", "nAnat"}, {"nAnat", "Turkey_N"}, {"nAnat", "England_BellBeaker"},
        {"nWE", "pIran"}, {"pIran", "Iran_GanjDareh_N"}, {"nEAS", "pSteppe"},
        {"pSteppe", "aCW"}, {"pIran", "aCW"}, {"aCW", "Czechia_EBA_CordedWare"},
    };
}

// The fixture P-axis pop order (the qpadm parity test's index map).
std::vector<std::string> fixture_pops() {
    return {"England_BellBeaker", "Czechia_EBA_CordedWare", "Turkey_N", "Mbuti",
            "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana"};
}

// Golden constants (golden_qpgraph_score.csv / golden_qpgraph_weights.csv).
constexpr double kGoldenScore = 80.0674246076313;
constexpr double kGoldenWeightSteppeToCW = 0.153483829987482;  // pSteppe->aCW
constexpr double kGoldenWeightIranToCW = 0.846516170012518;    // pIran->aCW

// Pull the fitted weight on a named parent->child admix edge from the result.
double admix_weight(const steppe::QpGraphResult& r, const std::string& from, const std::string& to) {
    for (std::size_t j = 0; j < r.weight.size(); ++j)
        if (r.admix_from[j] == from && r.admix_to[j] == to) return r.weight[j];
    // The other parent of the same admix node carries 1 - weight.
    for (std::size_t j = 0; j < r.weight.size(); ++j)
        if (r.admix_to[j] == to) return 1.0 - r.weight[j];
    return std::nan("");
}

void gate_result(const char* label, const steppe::QpGraphResult& r) {
    std::printf("--- %s ---\n", label);
    check_true("status Ok", r.status == steppe::Status::Ok);
    std::printf("  score=%.12f  restart_spread=%.3e  nadmix=%zu  nedge=%zu\n",
                r.score, r.restart_spread, r.weight.size(), r.edge_length.size());
    check_close("score", r.score, kGoldenScore, 0.0, 1e-4);
    const double w_steppe = admix_weight(r, "pSteppe", "aCW");
    const double w_iran = admix_weight(r, "pIran", "aCW");
    std::printf("  weight pSteppe->aCW=%.9f  pIran->aCW=%.9f\n", w_steppe, w_iran);
    check_close("weight pSteppe->aCW", w_steppe, kGoldenWeightSteppeToCW, 0.0, 1e-5);
    check_close("weight pIran->aCW", w_iran, kGoldenWeightIranToCW, 0.0, 1e-5);
    // identifiability witness: all restarts reach the same optimum (a TIGHT spread).
    check_true("restart_spread tight (unique optimum)", r.restart_spread < 1e-4);
    std::printf("  worst f3 residual z=%.4f (Mbuti;%s,%s)\n", r.worst_residual_z,
                r.worst_pop2.c_str(), r.worst_pop3.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    const std::string golden_dir = (argc > 1) ? argv[1] : "tests/reference/goldens/at2";
    std::printf("=== qpGraph single-graph-fit parity test ===\n");
    std::printf("golden dir: %s\n", golden_dir.c_str());

    // The qpGraph f3 basis uses the AT2 afprod=FALSE f2 (qpgraph_precompute_f3 calls
    // get_f2(data, pops) with the afprod=FALSE default — DIFFERENT from qpadm's afprod=TRUE
    // f2_fit0_9pop.bin). f2_qpgraph_9pop.bin is the afprod=FALSE f2 of the SAME f2 dir
    // (f2_fit0_FINAL) the golden was generated from, in steppe [P x P x nb] layout.
    steppe::F2BlockTensor f2;
    if (!read_fixture(golden_dir + "/fixtures/f2_qpgraph_9pop.bin", f2)) {
        std::printf("RESULT: FAIL (fixture read)\n");
        return 1;
    }

    const std::vector<steppe::QpGraphEdge> edges = golden_topology();
    const std::vector<std::string> pops = fixture_pops();
    steppe::QpGraphOptions opts;  // defaults == the golden's (fudge 1e-4, diag_f3 1e-5, numstart 10, constrained)

    int gpu_count = 0;
    try { gpu_count = steppe::device::visible_device_count(); }
    catch (const std::exception& e) { std::printf("  [INFO] visible_device_count threw: %s\n", e.what()); gpu_count = 0; }

    // ---- Block A: CpuBackend ORACLE (always runs) -------------------------------
    steppe::QpGraphResult cpu_res;
    {
        steppe::device::Resources res;
        steppe::device::PerGpuResources cpu;
        cpu.backend = steppe::device::make_cpu_backend();
        res.gpus.push_back(std::move(cpu));
        cpu_res = steppe::run_qpgraph(f2, edges, pops, opts, res);
        gate_result("CpuBackend (oracle, native FP64)", cpu_res);
    }

    // ---- Block B: CudaBackend PRODUCTION path (f3 basis RESIDENT) ---------------
    if (gpu_count > 0) {
        try {
            steppe::device::Resources res;
            steppe::device::PerGpuResources g0;
            g0.backend = steppe::device::make_cuda_backend(0);
            g0.device_id = 0;
            res.gpus.push_back(std::move(g0));
            steppe::F2BlockTensor f2_up = f2;
            f2_up.vpair.assign(f2.f2.size(), 0.0);
            steppe::device::DeviceF2Blocks dev = steppe::device::upload_f2_blocks_to_device(f2_up, 0);
            check_true("f2 resident (f2_device != null)", dev.f2_device() != nullptr);
            const steppe::QpGraphResult gpu_res = steppe::run_qpgraph(dev, edges, pops, opts, res);
            gate_result("CudaBackend (GPU, f3 basis resident, fleet on-device)", gpu_res);
            // CpuBackend == CudaBackend localization.
            check_close("GPU vs CPU score", gpu_res.score, cpu_res.score, 0.0, 1e-7);
            if (!gpu_res.weight.empty() && !cpu_res.weight.empty())
                check_close("GPU vs CPU weight[0]", gpu_res.weight[0], cpu_res.weight[0], 0.0, 1e-6);
        } catch (const std::exception& e) {
            std::printf("  [FAIL] GPU path threw: %s\n", e.what());
            ++g_failures;
        }
    } else {
        std::printf("--- CudaBackend: SKIP (no CUDA device visible) ---\n");
    }

    std::printf("\nRESULT: %s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
