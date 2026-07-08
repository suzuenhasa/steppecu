// src/app/cmd_pcangsd.cpp
//
// The `steppe pcangsd` handler (PCAngsd GL-PCA). Reads a beagle GL file, builds a
// backend (GPU when visible, else the CpuBackend reference oracle), runs the IAF EM
// + GL-weighted covariance + top-e PCA via steppe::run_pcangsd, and writes the
// PCAngsd-convention outputs: PREFIX.cov (N x N), PREFIX.eigenvec (N x e),
// PREFIX.eigenval (e), and — under --emit-freq / --emit-iaf — PREFIX.freq (M) /
// PREFIX.pi (M x N). Mirrors the cmd_ibd / cmd_roh self-contained shape.
#include "app/cmd_pcangsd.hpp"

#include <cctype>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "app/exit_code_for_caught.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "device/backend.hpp"
#include "device/backend_factory.hpp"
#include "steppe/config.hpp"
#include "steppe/pcangsd.hpp"

namespace steppe::app {

namespace cfg = steppe::config;

namespace {

[[nodiscard]] bool parse_device(const std::string& raw, int& dev, std::string& err) {
    std::string s;
    for (char c : raw)
        if (!std::isspace(static_cast<unsigned char>(c))) s += c;
    if (s.empty() || s == "auto") { dev = 0; return true; }
    try {
        dev = std::stoi(s);
    } catch (...) {
        err = "--device ordinal '" + raw + "' is not an integer";
        return false;
    }
    return true;
}

// Write a row-major R x C matrix as a whitespace/sep table (PCAngsd .cov/.eigenvec/
// .pi convention — no header, one row per line). Returns false on a write failure.
[[nodiscard]] bool write_matrix(const std::string& path, const std::vector<double>& m, long R,
                                long C, char sep) {
    std::ofstream os(path, std::ios::trunc);
    if (!os) {
        std::fprintf(stderr, "steppe pcangsd: cannot open output: %s\n", path.c_str());
        return false;
    }
    for (long r = 0; r < R; ++r) {
        for (long c = 0; c < C; ++c) {
            if (c) os << sep;
            os << fmt_double(m[static_cast<std::size_t>(r) * static_cast<std::size_t>(C) +
                               static_cast<std::size_t>(c)]);
        }
        os << "\n";
    }
    os.flush();
    if (!os.good()) {
        std::fprintf(stderr, "steppe pcangsd: write failed: %s\n", path.c_str());
        return false;
    }
    return true;
}

// Write a length-n vector, one value per line.
[[nodiscard]] bool write_vector(const std::string& path, const std::vector<double>& v, long n) {
    std::ofstream os(path, std::ios::trunc);
    if (!os) {
        std::fprintf(stderr, "steppe pcangsd: cannot open output: %s\n", path.c_str());
        return false;
    }
    for (long i = 0; i < n; ++i) os << fmt_double(v[static_cast<std::size_t>(i)]) << "\n";
    os.flush();
    if (!os.good()) {
        std::fprintf(stderr, "steppe pcangsd: write failed: %s\n", path.c_str());
        return false;
    }
    return true;
}

const char* status_text(steppe::Status s) {
    switch (s) {
        case steppe::Status::Ok: return "ok";
        case steppe::Status::RankDeficient: return "rank_deficient";
        case steppe::Status::NonSpdCovariance: return "non_spd_covariance";
        case steppe::Status::ChisqUndefined: return "chisq_undefined";
        case steppe::Status::DeviceOom: return "device_oom";
        case steppe::Status::InvalidConfig: return "invalid_config";
    }
    return "unknown";
}

}  // namespace

int run_pcangsd_cmd(const PcangsdArgs& args) {
    if (args.beagle.empty()) {
        std::fprintf(stderr, "steppe pcangsd: --beagle <file> is required\n");
        return cfg::kExitInvalidConfig;
    }
    if (args.eig < 1) {
        std::fprintf(stderr,
                     "steppe pcangsd: -e/--eig must be >= 1 (number of PCs / IAF rank; the auto "
                     "MAP-test default is deferred in v1)\n");
        return cfg::kExitInvalidConfig;
    }
    if (args.out.empty()) {
        std::fprintf(stderr, "steppe pcangsd: --out <PREFIX> is required\n");
        return cfg::kExitInvalidConfig;
    }
    if (args.precision != "emu" && args.precision != "fp64") {
        std::fprintf(stderr, "steppe pcangsd: --precision must be emu | fp64 (got '%s')\n",
                     args.precision.c_str());
        return cfg::kExitInvalidConfig;
    }
    const char sep = (args.format == "csv") ? ',' : '\t';

    steppe::PcangsdParams params;
    params.e = args.eig;
    params.iter = args.iter;
    params.tol = args.tole;
    params.maf = args.maf;
    params.maf_iter = args.maf_iter;
    params.maf_tol = args.maf_tole;
    params.want_pi = args.emit_iaf;
    params.precision =
        (args.precision == "fp64") ? steppe::Precision::fp64() : steppe::Precision::emulated_fp64();

    // Backend: GPU when visible, else the CpuBackend reference oracle.
    std::unique_ptr<ComputeBackend> be;
    try {
        int dev = 0;
        std::string derr;
        if (!parse_device(args.device, dev, derr)) {
            std::fprintf(stderr, "steppe pcangsd: %s\n", derr.c_str());
            return cfg::kExitInvalidConfig;
        }
        be = (device::visible_device_count() > 0) ? device::make_cuda_backend(dev)
                                                  : device::make_cpu_backend();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe pcangsd: device init failed: %s\n", e.what());
        return cfg::kExitRuntimeError;
    }

    steppe::PcangsdResult r;
    try {
        r = steppe::run_pcangsd(args.beagle, params, *be);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe pcangsd: input/device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }
    if (r.status != Status::Ok) {
        std::fprintf(stderr,
                     "steppe pcangsd: %s (check --beagle, and that -e <= min(samples, kept "
                     "sites))\n",
                     status_text(r.status));
        return cfg::exit_code_for(r.status);
    }

    // Outputs (PCAngsd conventions): PREFIX.cov (N x N), .eigenvec (N x e), .eigenval (e).
    if (!write_matrix(args.out + ".cov", r.cov, r.N, r.N, sep)) return cfg::kExitIoError;
    if (!write_matrix(args.out + ".eigenvec", r.coords, r.N, r.e, sep)) return cfg::kExitIoError;
    if (!write_vector(args.out + ".eigenval", r.eigenvalues, r.e)) return cfg::kExitIoError;
    if (args.emit_freq) {
        if (!write_vector(args.out + ".freq", r.freq, r.M_used)) return cfg::kExitIoError;
    }
    if (args.emit_iaf) {
        if (!write_matrix(args.out + ".pi", r.pi, r.M_used, r.N, sep)) return cfg::kExitIoError;
    }

    const double v1 = r.e > 0 ? r.var_explained[0] * 100.0 : 0.0;
    // Honest convergence wording: the main loop is a plain fixed-point EM (linear
    // convergence), so at a tight --tole it commonly runs to the --iter cap without
    // rmse dipping below tol. Report which happened rather than always claiming
    // "converged" (both reach the same fixed point; see cuda_backend_pcangsd.cu).
    const bool converged = r.final_rmse < args.tole;
    std::fprintf(stderr,
                 "steppe pcangsd: %d samples x %ld/%ld sites (MAF %.3g), %d PCs; %s %d "
                 "iters (rmse=%.3e, tol=%.3e), PC1 var=%.2f%% -> "
                 "%s.{cov,eigenvec,eigenval}%s%s\n",
                 r.N, r.M_used, r.M_total, args.maf, r.e,
                 converged ? "converged in" : "stopped at the --iter cap after", r.iters_run,
                 r.final_rmse, args.tole, v1, args.out.c_str(), args.emit_freq ? " .freq" : "",
                 args.emit_iaf ? " .pi" : "");
    return cfg::kExitOk;
}

}  // namespace steppe::app
