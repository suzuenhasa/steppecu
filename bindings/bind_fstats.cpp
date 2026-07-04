// bindings/bind_fstats.cpp
//
// Registers the f-statistic / extract / smoother entries into steppe._core: a thin
// marshalling layer that resolves pop names to indices and calls the CUDA-free library
// seam. Every statistic reads from one of two data paths — a pre-computed f2 cache or the
// raw genotype triple.
//
// Reference: docs/reference/bindings_bind_fstats.cpp.md
#include <algorithm>
#include <array>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "internal/bind_common.hpp"

#include "app/f2_dir_writer.hpp"
#include "steppe/extract.hpp"
#include "steppe/qpfstats.hpp"

#include "io/geno_reader.hpp"
#include "io/ind_reader.hpp"

namespace steppe::pybind {
namespace {

// run_f4 — standalone f4 (f2-cache path) — reference §3
nb::dict run_f4_py(F2Handle& h,
                   const std::vector<std::array<std::string, 4>>& quartets) {
    if (quartets.empty()) raise_value("f4: needs at least one (p1,p2,p3,p4) quartet");

    const sa::PopResolver resolver(h.pops);
    if (!resolver.valid()) raise_value(resolver.error());

    std::vector<std::array<int, 4>> idx_quartets;
    idx_quartets.reserve(quartets.size());
    for (const std::array<std::string, 4>& q : quartets)
        idx_quartets.push_back(resolve_tuple<4>(resolver, q, "quartet pop"));

    steppe::QpAdmOptions opts;

    const steppe::F4Result result =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_f4(
                dev_f2, std::span<const std::array<int, 4>>(idx_quartets), opts, resources);
        });
    return f4_to_dict(result, h.pops);
}

// run_qpdstat — the f2-path D (== f4) — reference §3
nb::dict run_qpdstat_py(F2Handle& h,
                        const std::vector<std::array<std::string, 4>>& quartets) {
    if (quartets.empty()) raise_value("qpdstat: needs at least one (p1,p2,p3,p4) quartet");

    const sa::PopResolver resolver(h.pops);
    if (!resolver.valid()) raise_value(resolver.error());

    std::vector<std::array<int, 4>> idx_quartets;
    idx_quartets.reserve(quartets.size());
    for (const std::array<std::string, 4>& q : quartets)
        idx_quartets.push_back(resolve_tuple<4>(resolver, q, "quartet pop"));

    steppe::QpAdmOptions opts;

    const steppe::F4Result result =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_f4(
                dev_f2, std::span<const std::array<int, 4>>(idx_quartets), opts, resources);
        });
    return f4_to_dict(result, h.pops);
}

// run_dstat — the genotype-path normalized D — reference §4
nb::dict run_dstat_py(const std::string& prefix,
                      const std::vector<std::array<std::string, 4>>& quadruples,
                      double blgsize, int device) {
    if (quadruples.empty()) raise_value("qpdstat (genotype): needs at least one (p1,p2,p3,p4) quadruple");
    const std::string geno = prefix + ".geno";
    const std::string snp = prefix + ".snp";
    const std::string ind = prefix + ".ind";

    std::vector<std::string> pop_union;
    for (const std::array<std::string, 4>& q : quadruples) {
        for (const std::string& nm : q) {
            if (std::find(pop_union.begin(), pop_union.end(), nm) == pop_union.end())
                pop_union.push_back(nm);
        }
    }

    std::vector<std::string> pops;
    try {
        steppe::io::GenoReader reader(geno);
        const std::size_t n_present = reader.records_present();
        steppe::io::PopSelection sel;
        sel.mode = steppe::io::PopSelection::Mode::Explicit;
        sel.labels = pop_union;
        const steppe::io::IndPartition part = steppe::io::read_ind(ind, sel, n_present);
        pops.reserve(part.groups.size());
        for (const steppe::io::PopGroup& g : part.groups) pops.push_back(g.label);
    } catch (const std::exception& e) {
        raise_value(std::string("input error: ") + e.what());
    }

    const sa::PopResolver resolver(pops);
    if (!resolver.valid()) raise_value(resolver.error());

    std::vector<std::array<int, 4>> idx_quads;
    idx_quads.reserve(quadruples.size());
    for (const std::array<std::string, 4>& q : quadruples)
        idx_quads.push_back(resolve_tuple<4>(resolver, q, "quadruple pop"));

    steppe::DeviceConfig cfg;
    cfg.devices = {device};
    steppe::DstatResult result;
    try {
        sd::Resources resources = sd::build_resources(cfg);
        if (resources.gpus.empty()) raise_no_device();
        result = steppe::run_dstat(geno, snp, ind,
                                   std::span<const std::string>(pop_union),
                                   std::span<const std::array<int, 4>>(idx_quads),
                                   blgsize, resources);
    } catch (const std::exception& e) {
        raise_value(std::string("device error: ") + e.what());
    }
    return dstat_to_dict(result, pops);
}

// run_extract_f2 — build an f2 tensor from genotypes — reference §5
nb::object run_extract_f2_py(const std::string& prefix, const std::vector<std::string>& pops,
                             const std::string& out, int device, double blgsize,
                             double maf, double maxmiss, bool autosomes_only,
                             bool drop_monomorphic, bool transversions_only,
                             const std::string& ploidy, const std::string& strand_mode,
                             std::optional<std::string> precision) {
    if (pops.empty()) raise_value("extract_f2: pops needs at least one population name");
    const std::string geno = prefix + ".geno";
    const std::string snp = prefix + ".snp";
    const std::string ind = prefix + ".ind";

    steppe::io::PopSelection sel;
    sel.mode = steppe::io::PopSelection::Mode::Explicit;
    sel.labels = pops;

    steppe::FilterConfig filter;
    filter.maf_min = maf;
    filter.geno_max_missing = maxmiss;
    filter.autosomes_only = autosomes_only;
    filter.drop_monomorphic = drop_monomorphic;
    filter.transversions_only = transversions_only;

    if (strand_mode == "drop") {
        filter.strand_mode = steppe::StrandMode::Drop;
    } else if (strand_mode == "keep") {
        filter.strand_mode = steppe::StrandMode::Keep;
    } else if (strand_mode == "flip") {
        filter.strand_mode = steppe::StrandMode::Flip;
    } else {
        raise_value("extract_f2: strand_mode must be one of 'drop', 'keep', 'flip' "
                    "(got '" + strand_mode + "')");
    }

    const steppe::Precision prec = parse_precision(precision, "extract_f2");

    steppe::ExtractPloidy ep = steppe::ExtractPloidy::Auto;
    if (ploidy == "auto") {
        ep = steppe::ExtractPloidy::Auto;
    } else if (ploidy == "1" || ploidy == "pseudo" || ploidy == "pseudo_haploid") {
        ep = steppe::ExtractPloidy::PseudoHaploid;
    } else if (ploidy == "2" || ploidy == "diploid") {
        ep = steppe::ExtractPloidy::Diploid;
    } else {
        raise_value("extract_f2: ploidy must be one of 'auto', '1'/'pseudo', '2'/'diploid' "
                    "(got '" + ploidy + "')");
    }

    steppe::DeviceConfig cfg;
    cfg.devices = {device};
    cfg.precision = prec;

    steppe::F2ExtractResult result;
    try {
        sd::Resources resources = sd::build_resources(cfg);
        if (resources.gpus.empty()) raise_no_device();
        result = steppe::run_extract_f2(geno, snp, ind, sel, filter, prec, blgsize, ep,
                                        resources);
    } catch (const std::exception& e) {
        raise_value(std::string("extract_f2: ") + e.what());
    }

    if (!out.empty()) {
        sa::F2DirMeta meta;
        meta.precision_mantissa_bits = prec.mantissa_bits;
        meta.precision_tag =
            (result.precision_tag == steppe::Precision::Kind::EmulatedFp64) ? "emu"
          : (result.precision_tag == steppe::Precision::Kind::Tf32)         ? "tf32"
                                                                            : "fp64";
        meta.blgsize_cm = blgsize * steppe::kCentimorgansPerMorgan;
        meta.n_block = result.f2.n_block;
        meta.P = result.f2.P;
        meta.n_snp_total = result.n_snp_total;
        meta.n_snp_kept = result.n_snp_kept;
        meta.maf_min = filter.maf_min;
        meta.geno_max_missing = filter.geno_max_missing;
        meta.autosomes_only = filter.autosomes_only;
        meta.drop_monomorphic = filter.drop_monomorphic;
        meta.transversions_only = filter.transversions_only;
        meta.geno_path = geno;
        meta.snp_path = snp;
        meta.ind_path = ind;
        meta.hash_source_files = false;
        const sa::F2DirWriteResult wr =
            sa::write_f2_dir(out, result.f2, result.pop_labels, meta);
        if (!wr.ok) raise_value("extract_f2: " + wr.error);
        return nb::cast(out);
    }

    auto* h = new F2Handle();
    h->tensor = std::move(result.f2);
    h->pops = std::move(result.pop_labels);
    h->device = device;
    return nb::cast(h, nb::rv_policy::take_ownership);
}

// run_qpfstats — the genotype-path joint f2 smoother — reference §6
nb::object run_qpfstats_py(const std::string& prefix, const std::vector<std::string>& pops,
                           const std::string& out, int device, double blgsize,
                           std::optional<std::string> precision) {
    if (pops.size() < 4) raise_value("qpfstats: pops needs at least 4 populations (the f4 basis)");
    const std::string geno = prefix + ".geno";
    const std::string snp = prefix + ".snp";
    const std::string ind = prefix + ".ind";

    const steppe::Precision prec = parse_precision(precision, "qpfstats");

    steppe::DeviceConfig cfg;
    cfg.devices = {device};
    cfg.precision = prec;

    steppe::QpfstatsResult result;
    try {
        sd::Resources resources = sd::build_resources(cfg);
        if (resources.gpus.empty()) raise_no_device();
        result = steppe::run_qpfstats(geno, snp, ind, std::span<const std::string>(pops),
                                      blgsize, prec, resources);
    } catch (const std::exception& e) {
        raise_value(std::string("qpfstats: ") + e.what());
    }
    if (result.status != steppe::Status::Ok)
        raise_value("qpfstats: could not build the smoothed f2 (check pops are all present)");

    if (!out.empty()) {
        sa::F2DirMeta meta;
        meta.precision_mantissa_bits = prec.mantissa_bits;
        meta.precision_tag =
            (prec.kind == steppe::Precision::Kind::EmulatedFp64) ? "emu"
          : (prec.kind == steppe::Precision::Kind::Tf32)         ? "tf32"
                                                                 : "fp64";
        meta.blgsize_cm = blgsize * steppe::kCentimorgansPerMorgan;
        meta.n_block = result.f2.n_block;
        meta.P = result.f2.P;
        meta.autosomes_only = true;
        meta.geno_path = geno; meta.snp_path = snp; meta.ind_path = ind;
        meta.pop_selection = "qpfstats-smoothed";
        const sa::F2DirWriteResult wr =
            sa::write_f2_dir(out, result.f2, result.pop_labels, meta);
        if (!wr.ok) raise_value("qpfstats: " + wr.error);
        return nb::cast(out);
    }

    auto* h = new F2Handle();
    h->tensor = std::move(result.f2);
    h->pops = std::move(result.pop_labels);
    h->device = device;
    return nb::cast(h, nb::rv_policy::take_ownership);
}

// run_f3 — standalone f3 (f2-cache path) — reference §3
nb::dict run_f3_py(F2Handle& h,
                   const std::vector<std::array<std::string, 3>>& triples) {
    if (triples.empty()) raise_value("f3: needs at least one (C,A,B) triple");

    const sa::PopResolver resolver(h.pops);
    if (!resolver.valid()) raise_value(resolver.error());

    std::vector<std::array<int, 3>> idx_triples;
    idx_triples.reserve(triples.size());
    for (const std::array<std::string, 3>& t : triples)
        idx_triples.push_back(resolve_tuple<3>(resolver, t, "triple pop"));

    steppe::QpAdmOptions opts;

    const steppe::F3Result result =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_f3(
                dev_f2, std::span<const std::array<int, 3>>(idx_triples), opts, resources);
        });
    return f3_to_dict(result, h.pops);
}

// run_f4ratio — standalone f4-ratio (f2-cache path) — reference §3
nb::dict run_f4ratio_py(F2Handle& h,
                        const std::vector<std::array<std::string, 5>>& tuples) {
    if (tuples.empty()) raise_value("f4-ratio: needs at least one (p1,p2,p3,p4,p5) tuple");

    const sa::PopResolver resolver(h.pops);
    if (!resolver.valid()) raise_value(resolver.error());

    std::vector<std::array<int, 5>> idx_tuples;
    idx_tuples.reserve(tuples.size());
    for (const std::array<std::string, 5>& t : tuples)
        idx_tuples.push_back(resolve_tuple<5>(resolver, t, "f4-ratio pop"));

    steppe::QpAdmOptions opts;

    const steppe::F4RatioResult result =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_f4ratio(
                dev_f2, std::span<const std::array<int, 5>>(idx_tuples), opts, resources);
        });
    return f4ratio_to_dict(result, h.pops);
}

}  // namespace

// register_fstats — the registered Python entries + defaults — reference §9
void register_fstats(nb::module_& m) {
    m.def("run_f4", &run_f4_py, "f2"_a, "quartets"_a,
          "Standalone f4(p1,p2;p3,p4) (GPU). `quartets` is a list of (p1,p2,p3,p4) name "
          "tuples; returns a dict of parallel arrays {pop1,pop2,pop3,pop4,est,se,z,p}.");

    m.def("run_qpdstat", &run_qpdstat_py, "f2"_a, "quartets"_a,
          "D-statistic / f4 over the f2-data path (GPU; qpDstat Part A). The --f2-dir "
          "qpdstat path reports f4 (the AT2 f2-path convention: qpdstat(f2dir,f4mode) is "
          "byte-identical to f4, f4mode being a no-op without per-SNP genotypes). `quartets` "
          "is a list of (p1,p2,p3,p4) name tuples; returns a dict of parallel arrays "
          "{pop1,pop2,pop3,pop4,est,se,z,p}. The normalized-D magnitude needs the genotype "
          "path: call steppe.dstat(prefix, quadruples) (the run_dstat binding below), or the "
          "CLI qpdstat --prefix PREFIX.{geno,snp,ind}.");

    m.def("run_dstat", &run_dstat_py, "prefix"_a, "quadruples"_a, "blgsize"_a = 0.05,
          "device"_a = 0,
          "Genotype-path NORMALIZED-D (GPU; qpDstat Part B). Reads the genotype triple "
          "<prefix>.{geno,snp,ind} directly (NOT the f2 cache): D = mean_snp(num)/mean_snp(den), "
          "num=(a-b)(c-d), den=(a+b-2ab)(c+d-2cd) over per-SNP allele freqs, block-jackknifed. "
          "`quadruples` is a list of (p1,p2,p3,p4) name tuples; `blgsize` is the block size in "
          "MORGANS (AT2 default 0.05). Returns a dict of parallel arrays {pop1,pop2,pop3,pop4,"
          "est,se,z,p} (the AT2 D sign/Z/p convention). Forced diploid + allsnps=TRUE + "
          "autosomes-only are pinned (AT2 qpdstat_geno parity).");

    m.def("run_extract_f2", &run_extract_f2_py, "prefix"_a, "pops"_a, "out"_a = "",
          "device"_a = 0, "blgsize"_a = 0.05, "maf"_a = 0.0, "maxmiss"_a = 0.0,
          "autosomes_only"_a = true, "drop_monomorphic"_a = false,
          "transversions_only"_a = false, "ploidy"_a = "auto",
          "strand_mode"_a = "drop", "precision"_a = nb::none(),
          "Build an f2_blocks tensor from a genotype prefix (GPU; M(py-2) extract-f2). "
          "Reads <prefix>.{geno,snp,ind} directly and runs decode->filter->assign_blocks->"
          "tiered f2 compute->to_host (the SAME chain as the CLI extract-f2). `pops` is the "
          "Explicit population subset (the P axis is that selection sorted ASC by label); "
          "`blgsize` is MORGANS (AT2 default 0.05); `maxmiss` is the AT2 POP-axis coverage "
          "(0 = global intersection). If `out` is given, writes an STPF2BK1 dir there and "
          "returns the path string; else returns a new F2Handle (no disk round-trip). "
          "GPU-only: no CUDA device raises.");

    m.def("run_qpfstats", &run_qpfstats_py, "prefix"_a, "pops"_a, "out"_a = "",
          "device"_a = 0, "blgsize"_a = 0.05, "precision"_a = nb::none(),
          "Genotype-path JOINT f2 SMOOTHER (GPU; admixtools::qpfstats). Reads "
          "<prefix>.{geno,snp,ind} directly, drives the qpDstat-B numerator engine over the "
          "FULL f2/f3/f4 popcomb set, runs the on-device shared-factor smoothing regression, "
          "and returns a SMOOTHED per-block f2 tensor. `pops` is the smoothing pop set (sorted "
          "ASC internally = the AT2 dimnames order); `blgsize` is MORGANS (AT2 default 0.05). "
          "If `out` is given, writes an STPF2BK1 dir there and returns the path string; else "
          "returns a new F2Handle (read_f2/run_f4/run_qpadm consume it). GPU-only: no CUDA "
          "device raises.");

    m.def("run_f3", &run_f3_py, "f2"_a, "triples"_a,
          "Standalone f3(C;A,B) (GPU). `triples` is a list of (C,A,B) name tuples; "
          "returns a dict of parallel arrays {pop1,pop2,pop3,est,se,z,p}.");

    m.def("run_f4ratio", &run_f4ratio_py, "f2"_a, "tuples"_a,
          "Standalone f4-ratio alpha = f4(p1,p2;p3,p4)/f4(p1,p2;p5,p4) (GPU). `tuples` is a "
          "list of (p1,p2,p3,p4,p5) name tuples; returns a dict of parallel arrays "
          "{pop1,pop2,pop3,pop4,pop5,alpha,se,z}.");
}

}  // namespace steppe::pybind
