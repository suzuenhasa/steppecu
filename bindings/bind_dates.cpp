// bindings/bind_dates.cpp — the DATES admixture-dating binding (steppe._core).
//
// run_dates reads the genotype triple prefix.{geno,snp,ind} directly (never the f2 cache)
// and returns the date in generations plus the block-jackknife SE.
#include <string>

#include "internal/bind_common.hpp"

#include "steppe/dates.hpp"

namespace steppe::pybind {
namespace {

nb::dict run_dates_py(const std::string& prefix, const std::string& target,
                      const std::string& source1, const std::string& source2, int device) {
    const std::string geno = prefix + ".geno";
    const std::string snp = prefix + ".snp";
    const std::string ind = prefix + ".ind";

    steppe::DeviceConfig cfg;
    cfg.devices = {device};
    steppe::DatesOptions opts;
    steppe::DatesResult result;
    try {
        sd::Resources resources = sd::build_resources(cfg);
        if (resources.gpus.empty()) raise_no_device();
        result = steppe::run_dates(geno, snp, ind, target, source1, source2, opts, resources);
    } catch (const std::exception& e) {
        raise_value(std::string("input/device error: ") + e.what());
    }

    nb::dict d;
    d["target"] = target;
    d["source1"] = source1;
    d["source2"] = source2;
    d["date_gen"] = result.date_gen;
    d["se"] = result.se;
    d["fit_error_sd"] = result.fit_error_sd;
    d["curve_cm"] = result.curve_cm;
    d["curve_corr"] = result.curve_corr;
    d["status"] = (result.status == steppe::Status::Ok) ? "ok" : "error";
    return d;
}

}  // namespace

void register_dates(nb::module_& m) {
    m.def("run_dates", &run_dates_py, "prefix"_a, "target"_a, "source1"_a, "source2"_a,
          "device"_a = 0,
          "Admixture DATING (GPU; the DATES tool). Reads the genotype triple "
          "<prefix>.{geno,snp,ind} directly (NOT the f2 cache) and infers the time since "
          "admixture from the weighted ancestry-covariance decay vs genetic distance (the cuFFT "
          "autocorrelation LD engine; NO host O(M^2) SNP-pair loop). `target` is the admixed "
          "population; `source1`/`source2` are the two reference sources (weight = "
          "freq(source1)-freq(source2)). The .snp must carry a real cM genetic map. Returns a "
          "dict {target,source1,source2,date_gen,se,fit_error_sd,curve_cm,curve_corr,status}; "
          "date_gen is generations since admixture, se the leave-one-chromosome block jackknife.");
}

}  // namespace steppe::pybind
