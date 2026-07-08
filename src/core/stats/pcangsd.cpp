// src/core/stats/pcangsd.cpp
//
// run_pcangsd: the PCAngsd GL-PCA driver. Reads a beagle GL file into the shipped
// LikelihoodTile, then hands the host tile to the backend's pcangsd_fit (which
// uploads the resident GL tensor + runs the IAF EM device-resident on a CUDA
// backend, or the reference oracle on the CpuBackend), and maps the device result
// to the public PcangsdResult with the beagle sample IDs attached. Mirrors pca.cpp's
// run_pca shape (thin host driver over a backend seam).
#include "steppe/pcangsd.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "device/backend.hpp"
#include "io/beagle_reader.hpp"

namespace steppe {

PcangsdResult run_pcangsd(const std::string& beagle, const PcangsdParams& params,
                          ComputeBackend& be) {
    PcangsdResult res;
    res.precision_tag = params.precision.kind;

    if (params.e < 1) {
        res.status = Status::InvalidConfig;
        return res;
    }

    const io::BeagleReadResult br = io::read_beagle_gl(beagle);
    const io::LikelihoodTile& tile = br.tile;
    res.sample_ids = tile.sample_ids;

    if (br.n_site <= 0 || br.n_sample <= 0) {
        res.status = Status::InvalidConfig;
        return res;
    }

    const PcangsdFit fit =
        be.pcangsd_fit(tile.l.data(), tile.present.data(), br.n_site, br.n_sample, params.e,
                       params.iter, params.tol, params.maf, params.maf_iter, params.maf_tol,
                       params.want_pi, params.precision);
    if (fit.status != Status::Ok) {
        res.status = fit.status;
        return res;
    }

    res.cov = fit.cov;
    res.coords = fit.coords;
    res.eigenvalues = fit.eigenvalues;
    res.var_explained = fit.var_explained;
    res.freq = fit.freq;
    res.pi = fit.pi;
    res.N = fit.N;
    res.e = fit.e;
    res.M_used = fit.M_used;
    res.M_total = fit.M_total;
    res.iters_run = fit.iters_run;
    res.final_rmse = fit.final_rmse;
    res.precision_tag = fit.precision_tag;
    res.status = Status::Ok;
    return res;
}

}  // namespace steppe
