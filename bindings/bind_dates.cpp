// bindings/bind_dates.cpp — the DATES admixture-dating entry (steppe._core).
//
// run_dates reads the genotype TRIPLE prefix.{geno,snp,ind} directly (the cuFFT
// autocorrelation LD engine; NEVER the f2 cache, NEVER a host O(M^2) SNP-pair loop) and
// returns the date in generations + the leave-one-chromosome block-jackknife SE. Faults (no
// device, missing files) raise (cli-bindings.md §1.3 / §5.2).
#include <string>

#include "internal/bind_common.hpp"

#include "steppe/dates.hpp"  // run_dates + DatesResult/DatesOptions (the DATES dating tool)

namespace steppe::pybind {
namespace {

// run_dates: the DATES admixture-dating binding. Reads the genotype TRIPLE prefix.{geno,snp,ind}
// directly (the cuFFT autocorrelation LD engine; NEVER the f2 cache, NEVER a host O(M²) SNP-pair
// loop) and returns the date in generations + the leave-one-chromosome block-jackknife SE.
// `target` is the admixed population; `source1`/`source2` are the two reference sources (the
// weight is wt = freq(source1) - freq(source2)). The .snp MUST carry a real genetic map (cM).
// Returns a dict {target, source1, source2, date_gen, se, fit_error_sd, curve_cm, curve_corr,
// status}. Faults (no device, missing files) raise (§1.3 / §5.2).
nb::dict run_dates_py(const std::string& prefix, const std::string& target,
                      const std::string& source1, const std::string& source2, int device) {
    const std::string geno = prefix + ".geno";
    const std::string snp = prefix + ".snp";
    const std::string ind = prefix + ".ind";

    steppe::DeviceConfig cfg;
    cfg.devices = {device};  // single-GPU (multi-gpu PARKED).
    steppe::DatesOptions opts;  // defaults == the DATES reference par.dates.
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
