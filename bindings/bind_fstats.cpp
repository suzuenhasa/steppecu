// bindings/bind_fstats.cpp — the f-statistic / extract / smoother entries (steppe._core).
//
// run_f4 / run_qpdstat (the f2-path D == f4) / run_f3 / run_f4ratio over the SAME resident
// f2; run_dstat (the genotype-path NORMALIZED-D, NO F2Handle); run_extract_f2 (genotype->f2
// EXTRACT) and run_qpfstats (genotype-path JOINT f2 SMOOTHER), both the dual capsule/path
// return idiom. Faults raise; per-item domain outcomes ride on the result's `status`
// (cli-bindings.md §1.3 / §5.2).
#include <algorithm>
#include <array>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "internal/bind_common.hpp"

#include "app/f2_dir_writer.hpp"  // steppe::app::write_f2_dir / F2DirMeta (CUDA-FREE; the extract out= path)
#include "steppe/extract.hpp"     // run_extract_f2 + F2ExtractResult (M(py-2) genotype->f2 extract)
#include "steppe/qpfstats.hpp"    // run_qpfstats + QpfstatsResult (the genotype-path joint f2 smoother)

#include "io/geno_reader.hpp"  // io::GenoReader (qpDstat Part B P-axis order)
#include "io/ind_reader.hpp"   // io::read_ind / PopSelection / IndPartition

namespace steppe::pybind {
namespace {

// run_f4: a list of (p1,p2,p3,p4) quartet NAME tuples against the SAME resident f2,
// computed BATCHED (run_f4). Returns ONE dict of parallel arrays {pop1..pop4,est,se,z,p}
// in input order. Mirrors run_qpwave_py exactly: resolve names against pops.txt, build
// (cached) resources, upload the host tensor INSIDE the call (the DeviceF2Blocks lives
// only here and frees in its destructor — spike #1), run the CUDA-free seam, marshal.
nb::dict run_f4_py(F2Handle& h,
                   const std::vector<std::array<std::string, 4>>& quartets) {
    if (quartets.empty()) raise_value("f4: needs at least one (p1,p2,p3,p4) quartet");

    const sa::PopResolver resolver(h.pops);
    if (!resolver.valid()) raise_value(resolver.error());

    std::vector<std::array<int, 4>> idx_quartets;
    idx_quartets.reserve(quartets.size());
    for (const std::array<std::string, 4>& q : quartets)
        idx_quartets.push_back(resolve_tuple<4>(resolver, q, "quartet pop"));

    steppe::QpAdmOptions opts;  // f4 uses fudge=0 internally (run_f4); opts is the default.

    const steppe::F4Result result =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_f4(
                dev_f2, std::span<const std::array<int, 4>>(idx_quartets), opts, resources);
        });
    return f4_to_dict(result, h.pops);
}

// run_qpdstat: the qpDstat Part-A binding — D-statistic / f4 over the f2-data path. A THIN
// clone of run_f4_py (the qpdstat f2-path == f4: admixtools::qpdstat(f2dir,f4mode=TRUE) is
// byte-identical to f4mode=FALSE and to f4, since f4mode is a no-op without per-SNP
// genotypes). Returns the SAME dict {pop1..pop4,est,se,z,p,status,precision} as run_f4 (z =
// est/se, p = 2*(1-Phi(|z|)) ARE the AT2 D-stat sign/Z/p convention). The normalized-D
// MAGNITUDE (per-SNP genotypes) is Part B, a separate later binding. NO new compute, NO new
// emitter, NO new result type — it REUSES run_f4 + f4_to_dict verbatim.
nb::dict run_qpdstat_py(F2Handle& h,
                        const std::vector<std::array<std::string, 4>>& quartets) {
    if (quartets.empty()) raise_value("qpdstat: needs at least one (p1,p2,p3,p4) quartet");

    const sa::PopResolver resolver(h.pops);
    if (!resolver.valid()) raise_value(resolver.error());

    std::vector<std::array<int, 4>> idx_quartets;
    idx_quartets.reserve(quartets.size());
    for (const std::array<std::string, 4>& q : quartets)
        idx_quartets.push_back(resolve_tuple<4>(resolver, q, "quartet pop"));

    steppe::QpAdmOptions opts;  // qpdstat==f4 uses fudge=0 internally (run_f4); opts default.

    const steppe::F4Result result =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_f4(
                dev_f2, std::span<const std::array<int, 4>>(idx_quartets), opts, resources);
        });
    return f4_to_dict(result, h.pops);
}

// run_dstat: the qpDstat Part-B genotype-path NORMALIZED-D. UNLIKE run_qpdstat (which reads
// the f2 cache and reports f4), this reads the GENOTYPE TRIPLE prefix.{geno,snp,ind} directly
// (the extract-f2 decode front-end + the per-SNP D kernel + the num/den block-jackknife), so
// it does NOT take an F2Handle. It builds the P-axis from read_ind(MinN,1) (every population,
// sorted ASC by label — IDENTICAL to run_dstat's internal decode), resolves the quadruple
// names to that axis, builds resources for `device`, and runs the CUDA-free seam. Returns the
// SAME dict {pop1..pop4,est,se,z,p} as run_qpdstat (the D convention). `blgsize` is MORGANS
// (AT2 default 0.05). Faults (no device, missing files, CUDA runtime) raise (§1.3 / §5.2).
nb::dict run_dstat_py(const std::string& prefix,
                      const std::vector<std::array<std::string, 4>>& quadruples,
                      double blgsize, int device) {
    if (quadruples.empty()) raise_value("qpdstat (genotype): needs at least one (p1,p2,p3,p4) quadruple");
    const std::string geno = prefix + ".geno";
    const std::string snp = prefix + ".snp";
    const std::string ind = prefix + ".ind";

    // The pop UNION (the AT2 indvec): the DISTINCT names across every quadruple. run_dstat
    // reads ONLY these (read_ind(Explicit{union}), NOT the whole prefix). The P axis is that
    // Explicit partition (sorted ASC by label); the resolver below is built over the SAME
    // read_ind(Explicit{union}) so its indices match run_dstat's decode exactly.
    std::vector<std::string> pop_union;
    for (const std::array<std::string, 4>& q : quadruples) {
        for (const std::string& nm : q) {
            if (std::find(pop_union.begin(), pop_union.end(), nm) == pop_union.end())
                pop_union.push_back(nm);
        }
    }

    std::vector<std::string> pops;  // the SORTED Explicit partition (the P axis order).
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
    cfg.devices = {device};  // single-GPU (multi-gpu PARKED).
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

// run_extract_f2: the M(py-2) genotype->f2 EXTRACT binding. Reads the genotype TRIPLE
// prefix.{geno,snp,ind} directly and builds the f2_blocks tensor (decode->filter->
// assign_blocks->tiered f2 compute->to_host) through the CUDA-free steppe::run_extract_f2
// seam (extract.hpp) — the SAME chain the CLI extract-f2 command runs (DRY). GPU-only:
// builds Resources for `device`, fail-fasts on no-GPU with the same message as the CLI.
//
// TWO RETURN MODES (capsule/path idiom, NOT a giant disk round-trip): if `out` is given,
// SERIALIZE the result to an STPF2BK1 dir via write_f2_dir and return the path STRING (the
// user can then read_f2(path)); if `out` is empty, wrap the host F2BlockTensor + labels in
// a NEW F2Handle and return it (rv_policy::take_ownership, like read_f2 — NO disk round-
// trip). The F2Handle pointer return reuses read_f2's exact ownership transfer. Faults (no
// device, unknown Explicit pop, missing file, every SNP filtered, an unwritable out dir)
// raise (the library THROWS; the binding re-raises as a Python error — §1.3 / §5.2).
//
// `pops` -> PopSelection::Explicit{pops} (the named-subset case; the P axis is that
// selection sorted ASC by label). `precision`: nullopt -> the EmulatedFp64 default (the
// f2-GEMM default, matching the CLI); "fp64"/"native" -> native FP64 oracle; "emulated_fp64"
// -> EmulatedFp64. `ploidy`: "auto" (AT2 adjust_pseudohaploid; the default), "1"/"pseudo"
// -> pseudo-haploid, "2"/"diploid" -> diploid. Defaults match the CLI/AT2 extract_f2 so a
// bare extract reproduces the golden (autosomes_only ON, maxmiss as the pop-axis semantic).
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

    // pops -> the Explicit selection (the P axis is read_ind(Explicit{pops}) sorted ASC).
    steppe::io::PopSelection sel;
    sel.mode = steppe::io::PopSelection::Mode::Explicit;
    sel.labels = pops;

    // The on-the-fly QC filter (the AT2 extract_f2 defaults; maxmiss is the POP-axis
    // coverage semantic, reproduced inside run_extract_f2 — NOT the sample-axis predicate).
    steppe::FilterConfig filter;
    filter.maf_min = maf;
    filter.geno_max_missing = maxmiss;      // AT2 pop-axis maxmiss (0 = global intersection).
    filter.autosomes_only = autosomes_only; // AT2 extract_f2 default auto_only=TRUE.
    filter.drop_monomorphic = drop_monomorphic;
    filter.transversions_only = transversions_only;

    // --strand-mode drop|keep|flip: the strand-ambiguous (palindromic A/T, C/G) SNP
    // policy. "drop" (DEFAULT) reproduces the frozen behavior bit-identically (merge-safe);
    // "keep" retains ambiguous SNPs (reproduces AT2's default); "flip" is a documented
    // not-yet-implemented token (currently == keep, no freq-based reorientation).
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

    // The precision policy (default EmulatedFp64 40-bit, the f2-GEMM default = the CLI).
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
    cfg.devices = {device};  // single-GPU (multi-gpu PARKED).
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

    // MODE A: out= given -> serialize an STPF2BK1 dir + return the path STRING.
    if (!out.empty()) {
        sa::F2DirMeta meta;
        meta.precision_mantissa_bits = prec.mantissa_bits;
        meta.precision_tag =
            (result.precision_tag == steppe::Precision::Kind::EmulatedFp64) ? "emu"
          : (result.precision_tag == steppe::Precision::Kind::Tf32)         ? "tf32"
                                                                            : "fp64";
        meta.blgsize_cm = blgsize * steppe::kCentimorgansPerMorgan;  // Morgans -> centimorgans (meta records cM).
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
        meta.hash_source_files = false;  // the Python extract does not hash the big .geno.
        const sa::F2DirWriteResult wr =
            sa::write_f2_dir(out, result.f2, result.pop_labels, meta);
        if (!wr.ok) raise_value("extract_f2: " + wr.error);
        return nb::cast(out);
    }

    // MODE B: out= empty -> wrap the host tensor + labels in a new F2Handle (no disk
    // round-trip). The pointer return uses rv_policy::take_ownership at the def site.
    auto* h = new F2Handle();
    h->tensor = std::move(result.f2);
    h->pops = std::move(result.pop_labels);
    h->device = device;
    return nb::cast(h, nb::rv_policy::take_ownership);
}

// run_qpfstats: the genotype-path JOINT f2 SMOOTHER (include/steppe/qpfstats.hpp). Reads the
// GENOTYPE TRIPLE prefix.{geno,snp,ind} directly, drives the qpDstat-B numerator engine over
// the FULL f2/f3/f4 popcomb set, runs the on-device shared-factor smoothing solve, and
// returns a SMOOTHED f2 — the SAME dual-return idiom as run_extract_f2: out= given ->
// serialize an STPF2BK1 dir + return the path STRING; out= empty -> wrap the smoothed
// F2BlockTensor + labels in a NEW F2Handle (so read_f2/run_f4/run_qpadm consume it). `pops`
// is the smoothing pop set (sorted ASC internally = the AT2 dimnames order). `precision`:
// nullopt -> EmulatedFp64 default (the matmul sub-steps); "fp64"/"native" -> native FP64.
// Faults (no device, missing files, unknown pop, CUDA runtime) raise (§1.3 / §5.2).
nb::object run_qpfstats_py(const std::string& prefix, const std::vector<std::string>& pops,
                           const std::string& out, int device, double blgsize,
                           std::optional<std::string> precision) {
    if (pops.size() < 4) raise_value("qpfstats: pops needs at least 4 populations (the f4 basis)");
    const std::string geno = prefix + ".geno";
    const std::string snp = prefix + ".snp";
    const std::string ind = prefix + ".ind";

    const steppe::Precision prec = parse_precision(precision, "qpfstats");

    steppe::DeviceConfig cfg;
    cfg.devices = {device};  // single-GPU (multi-gpu PARKED).
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

    // MODE A: out= given -> serialize an STPF2BK1 dir + return the path STRING.
    if (!out.empty()) {
        sa::F2DirMeta meta;
        meta.precision_mantissa_bits = prec.mantissa_bits;
        meta.precision_tag =
            (prec.kind == steppe::Precision::Kind::EmulatedFp64) ? "emu"
          : (prec.kind == steppe::Precision::Kind::Tf32)         ? "tf32"
                                                                 : "fp64";
        meta.blgsize_cm = blgsize * steppe::kCentimorgansPerMorgan;  // Morgans -> centimorgans (meta records cM).
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

    // MODE B: out= empty -> wrap the smoothed tensor + labels in a new F2Handle.
    auto* h = new F2Handle();
    h->tensor = std::move(result.f2);
    h->pops = std::move(result.pop_labels);
    h->device = device;
    return nb::cast(h, nb::rv_policy::take_ownership);
}

// run_f3: a list of (C,A,B) triple NAME tuples against the SAME resident f2, computed
// BATCHED (run_f3). Returns ONE dict of parallel arrays {pop1,pop2,pop3,est,se,z,p} in
// input order. The THREE-slab clone of run_f4_py: resolve names against pops.txt, build
// (cached) resources, upload the host tensor INSIDE the call (the DeviceF2Blocks lives only
// here and frees in its destructor — spike #1), run the CUDA-free seam, marshal.
nb::dict run_f3_py(F2Handle& h,
                   const std::vector<std::array<std::string, 3>>& triples) {
    if (triples.empty()) raise_value("f3: needs at least one (C,A,B) triple");

    const sa::PopResolver resolver(h.pops);
    if (!resolver.valid()) raise_value(resolver.error());

    std::vector<std::array<int, 3>> idx_triples;
    idx_triples.reserve(triples.size());
    for (const std::array<std::string, 3>& t : triples)
        idx_triples.push_back(resolve_tuple<3>(resolver, t, "triple pop"));

    steppe::QpAdmOptions opts;  // f3 uses fudge=0 internally (run_f3); opts is the default.

    const steppe::F3Result result =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_f3(
                dev_f2, std::span<const std::array<int, 3>>(idx_triples), opts, resources);
        });
    return f3_to_dict(result, h.pops);
}

// run_f4ratio: a list of (p1,p2,p3,p4,p5) 5-tuple NAME tuples against the SAME resident f2,
// computed BATCHED (run_f4ratio). Returns ONE dict of parallel arrays {pop1..pop5,alpha,se,z}
// in input order. The FIVE-column clone of run_f4_py: resolve names against pops.txt, build
// (cached) resources, upload the host tensor INSIDE the call (the DeviceF2Blocks lives only
// here and frees in its destructor — spike #1), run the CUDA-free seam, marshal.
nb::dict run_f4ratio_py(F2Handle& h,
                        const std::vector<std::array<std::string, 5>>& tuples) {
    if (tuples.empty()) raise_value("f4-ratio: needs at least one (p1,p2,p3,p4,p5) tuple");

    const sa::PopResolver resolver(h.pops);
    if (!resolver.valid()) raise_value(resolver.error());

    std::vector<std::array<int, 5>> idx_tuples;
    idx_tuples.reserve(tuples.size());
    for (const std::array<std::string, 5>& t : tuples)
        idx_tuples.push_back(resolve_tuple<5>(resolver, t, "f4-ratio pop"));

    steppe::QpAdmOptions opts;  // f4-ratio uses fudge=0 internally (run_f4ratio); opts default.

    const steppe::F4RatioResult result =
        with_device_f2(h, [&](sd::DeviceF2Blocks& dev_f2, sd::Resources& resources) {
            return steppe::run_f4ratio(
                dev_f2, std::span<const std::array<int, 5>>(idx_tuples), opts, resources);
        });
    return f4ratio_to_dict(result, h.pops);
}

}  // namespace

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
