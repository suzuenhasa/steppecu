// examples/cpp/quickstart_qpadm.cpp
//
// QUICK-START: read an f2_blocks dir -> qpAdm GLS fit -> print weights + p, on the GPU.
//
// This is a LIVING API CANARY for the C++ surface: it walks the SAME CUDA-free seams the
// CLI `qpadm` command uses (src/app/cmd_qpadm.cpp `run_qpadm_command`) and adds NO new
// compute, so it stops compiling/running the moment that surface drifts. It is a plain
// C++20 host program (no CUDA header): the GPU is reached only through the seams
//   read_f2_dir (steppe::app, steppe::access)        — STPF2BK1 f2.bin + pops.txt reader
//   PopResolver (steppe::app, steppe::access)         — pop NAME -> P-axis INDEX
//   build_resources / upload_f2_blocks_to_device      — steppe::device CUDA-free builders
//   run_qpadm                                          — steppe::api fit entry
//
// USAGE
//   quickstart_qpadm <f2-dir>
//     <f2-dir>  a directory holding f2.bin (STPF2BK1) + pops.txt (one pop per line, in
//               P-axis index order). See examples/README.md for how to stage the committed
//               golden fixture into such a dir (the Python quickstart writes one), or point
//               at a real extract-f2 dir (e.g. the box's /workspace/data/aadr/f2_fit0_corrected).
//
// The fit MODEL is fixed to the real-AADR golden_fit0 9-pop model (England_BellBeaker =
// CordedWare + Turkey_N) so a reader can self-check the printed weights/p against the value
// pinned in examples/README.md. Edit kTarget / kLeft / kRight below for a different model.
//
// BUILD: configure with -DSTEPPE_BUILD_EXAMPLES=ON -DSTEPPE_BUILD_CLI=ON on a CUDA box
// (sm_120 / CUDA 13). The local RTX 2070 / CUDA 11.8 is the WRONG arch and CANNOT build or
// run this — that is expected (see the README). The Python quickstart needs no build.
//
// PRINT-ONLY: like the CLI, a per-model DOMAIN outcome (rank-deficient / non-SPD) is a
// printed status, NOT an exception; only a genuine device/IO fault returns nonzero. This
// example asserts NOTHING — it just prints, so it can never "fail" the goldens.
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "app/f2_dir_io.hpp"             // steppe::app::read_f2_dir, F2DirResult
#include "app/pop_resolver.hpp"          // steppe::app::PopResolver, ResolveResult/ListResult
#include "device/device_f2_blocks.hpp"   // CUDA-FREE: DeviceF2Blocks, upload_f2_blocks_to_device
#include "device/resources.hpp"          // CUDA-FREE: Resources, build_resources
#include "steppe/config.hpp"             // steppe::DeviceConfig
#include "steppe/qpadm.hpp"              // steppe::run_qpadm + model/result/options

namespace {

// The real-AADR golden_fit0 model (tests/reference/goldens/at2/golden_fit0.json metadata).
// These names must exist in the f2-dir's pops.txt; the staged golden_fit0 fixture carries
// them in this exact order (see examples/README.md).
constexpr const char* kTarget = "England_BellBeaker";
const std::vector<std::string> kLeft = {"Czechia_EBA_CordedWare", "Turkey_N"};
const std::vector<std::string> kRight = {
    "Mbuti", "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana"};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: quickstart_qpadm <f2-dir>\n"
                     "  <f2-dir> holds f2.bin (STPF2BK1) + pops.txt. Stage the committed "
                     "golden\n  fixture with examples/python/quickstart_qpadm.py --stage "
                     "<f2-dir>,\n  or point at a real extract-f2 dir. See examples/README.md.\n");
        return 2;
    }
    const std::string f2_dir = argv[1];

    // ---- 1. Read the f2_blocks dir (f2.bin + pops.txt) — steppe::access -----------
    const steppe::app::F2DirResult dir = steppe::app::read_f2_dir(f2_dir);
    if (!dir.ok) {
        std::fprintf(stderr, "quickstart_qpadm: %s\n", dir.error.c_str());
        return 1;
    }

    // ---- 2. Resolve the model NAMES -> P-axis INDICES against pops.txt -------------
    const steppe::app::PopResolver resolver(dir.dir.pop_labels);
    if (!resolver.valid()) {
        std::fprintf(stderr, "quickstart_qpadm: %s\n", resolver.error().c_str());
        return 1;
    }
    const steppe::app::ResolveResult target = resolver.resolve(kTarget);
    const steppe::app::ResolveListResult left = resolver.resolve_all(kLeft);
    const steppe::app::ResolveListResult right = resolver.resolve_all(kRight);
    if (!target.ok) { std::fprintf(stderr, "quickstart_qpadm: %s\n", target.error.c_str()); return 1; }
    if (!left.ok)   { std::fprintf(stderr, "quickstart_qpadm: %s\n", left.error.c_str());   return 1; }
    if (!right.ok)  { std::fprintf(stderr, "quickstart_qpadm: %s\n", right.error.c_str());  return 1; }

    steppe::QpAdmModel model;
    model.target = target.index;
    model.left = left.indices;
    model.right = right.indices;
    model.model_index = 0;

    // ---- 3/4. build_resources -> upload f2 to VRAM -> run_qpadm (the GPU path) ------
    // The same CUDA-free seams cmd_qpadm.cpp uses; a default DeviceConfig auto-enumerates
    // (empty `devices` => every visible CUDA device, first ordinal is the fit device). The
    // try/catch mirrors the CLI idiom: build/upload/run FAULTS (no device, OOM) are caught
    // and exit nonzero; a per-model DOMAIN outcome never throws — it arrives as result.status.
    const steppe::QpAdmOptions opts;  // defaults (fudge 1e-4, als_iterations 20, rank -1) = the goldens
    steppe::QpAdmResult result;
    try {
        steppe::device::Resources resources =
            steppe::device::build_resources(steppe::DeviceConfig{});
        if (resources.gpus.empty()) {
            std::fprintf(stderr,
                         "quickstart_qpadm: no CUDA device available (steppe is a GPU "
                         "product; a CUDA-capable GPU is required)\n");
            return 1;
        }
        const int device_id = resources.gpus.front().device_id;
        steppe::device::DeviceF2Blocks dev_f2 =
            steppe::device::upload_f2_blocks_to_device(dir.dir.f2, device_id);
        result = steppe::run_qpadm(dev_f2, model, opts, resources);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "quickstart_qpadm: device error: %s\n", e.what());
        return 1;
    }

    // ---- 5. PRINT the weights table + the tail p (no asserts — a print-only canary) -
    std::printf("qpadm  target=%s  f2-dir=%s\n", kTarget, f2_dir.c_str());
    std::printf("  %-26s %12s\n", "left", "weight");
    for (std::size_t i = 0; i < model.left.size(); ++i) {
        const std::string& name = resolver.label_at(model.left[i]);
        const double w = (i < result.weight.size()) ? result.weight[i] : 0.0;
        std::printf("  %-26s %12.6f\n", name.c_str(), w);
    }
    std::printf("  p=%.6g  chisq=%.6g  dof=%d  f4rank=%d  status=%d\n",
                result.p, result.chisq, result.dof, result.f4rank,
                static_cast<int>(result.status));
    return 0;
}
