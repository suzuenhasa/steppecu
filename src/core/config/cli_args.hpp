// src/core/config/cli_args.hpp
//
// CliArgs — the flat, parse-time command-line surface: one raw field per flag,
// nothing resolved or validated. The CLI parser fills it in and ConfigBuilder
// folds it over the lower config layers (compiled defaults < TOML < env < CLI),
// then build() freezes an immutable RunConfig. CUDA-free by contract. Each field
// is a std::optional "was this flag set?" sentinel — an unset field leaves the
// lower layer intact; a few vector/string fields use emptiness as that sentinel.
//
// Reference: docs/reference/src_core_config_cli_args.hpp.md
#ifndef STEPPE_CORE_CONFIG_CLI_ARGS_HPP
#define STEPPE_CORE_CONFIG_CLI_ARGS_HPP

#include <optional>
#include <string>
#include <vector>

namespace steppe::config {

// Command — the selected subcommand — reference §3
enum class Command {
    None,
    ExtractF2,
    QpAdm,
    QpWave,
    QpAdmRotate,
    F4,
    F3,
    F4Ratio,
    Qpdstat,
    F4Sweep,
    F3Sweep,
    Qpfstats,
    QpGraph,
    QpGraphSearch,
    Dates,
    Scan,
    Readv2,
    Paint,
    Fst,
    Sfs,
    Pca,
    Kinship,
};

// PloidyMode — extract-f2 ploidy policy — reference §4
enum class PloidyMode {
    Auto,
    Diploid,
    PseudoHaploid,
};

struct CliArgs {
    Command command = Command::None;

    // Global resource and precision knobs — reference §5
    std::optional<std::string> device;
    std::optional<std::string> precision;
    std::optional<std::string> config_path;

    // qpAdm and qpwave inputs — reference §6
    std::optional<std::string> f2_dir;
    std::optional<std::string> target;
    std::vector<std::string> left;
    std::vector<std::string> right;
    std::vector<std::string> pool;

    // Standalone f-statistic inputs — reference §7
    std::vector<std::string> pop1, pop2, pop3, pop4;

    // qpGraph and qpgraph-search options — reference §8
    std::optional<std::string> graph;
    std::optional<int> numstart;
    std::optional<double> diag_f3;
    std::optional<bool> constrained;
    std::optional<int> max_nadmix;

    // qpdstat magnitude prefix and f4-ratio 5th column — reference §7
    std::optional<std::string> qpdstat_prefix;
    std::vector<std::string> pop5;

    // READv2 windowed-mismatch kinship inputs
    std::optional<std::string> samples;       // FILE of Genetic IDs (one per line)
    std::optional<int>         window_snps;   // non-overlapping SNP-count window
    std::optional<std::string> norm;          // background: median | mean
    std::optional<double>      min_overlap;   // drop pairs below this comparable fraction

    // Li-Stephens `paint` inputs (recipients reuse the --prefix / qpdstat_prefix seam)
    std::optional<std::string> donors;        // phased DONOR panel triple PREFIX (haploid columns)
    std::optional<std::string> labels;        // donor haplotype -> ancestry/pop label FILE
    std::optional<std::string> face;          // paint | localanc | impute | roh | contam (v1: paint)
    std::optional<double>      ls_ne;         // effective pop size -> recombination scale
    std::optional<std::string> ls_theta;      // mutation/emission scale, or "auto" (Watterson over K)
    std::optional<bool>        self_copy;     // allow a haplotype to copy itself (off = leave-one-out)
    std::optional<int>         recip_batch;   // #recipients resident per wave (the batch/VRAM knob)
    std::optional<bool>        bp_fallback;   // opt into the bp cM-fallback when genpos is absent
    std::optional<bool>        paint_full;    // emit the per-DONOR coancestry matrix (default per-label)
    // Phased-VCF-native paint inputs (no .geno/.snp/.ind sidecars): read the recipient
    // and donor haplotype panels INLINE from phased .vcf.gz files via the dedicated
    // read_vcf_panel_front_end. When both are set, paint takes the VCF path.
    std::optional<std::string> recip_vcf;     // phased RECIPIENT .vcf.gz -> inline haplotype panel
    std::optional<std::string> donor_vcf;     // phased DONOR .vcf.gz -> inline haplotype panel
    std::optional<std::string> vcf_map;       // plink/HapMap genetic map (chrom id cM bp) -> genpos Morgans
    std::optional<std::string> vcf_region;    // bounded POS filter "CHROM:START-END" (inclusive)
    std::optional<double>      vcf_unphased_max;  // fail if the unphased-het fraction exceeds this

    // `fst` (per-SNP Weir-Cockerham FST) controls
    std::optional<std::string> fst_method;    // wc (Weir-Cockerham 1984; default). hudson is a follow-up.
    std::optional<bool>        fst_per_snp;   // emit the per-SNP FST table (else the summary row)
    std::optional<bool>        fst_all_pairs; // compute the all-pairs (P x P) WC FST matrix

    // `kinship` (KING-robust between-family kinship) controls
    std::optional<bool>        kinship_all_pairs;  // enumerate all C(N,2) pairs (else --pairs)
    std::optional<std::string> pairs;              // FILE of explicit id1<ws>id2 pairs
    std::optional<double>      min_kinship;        // emit only pairs with phi >= this

    // `sfs` (2D joint site-frequency spectrum) controls
    std::optional<bool>        sfs_fold;      // folded (per-pop minor) SFS; default unfolded (A1 copies)

    // `pca` (standalone genotype PCA) controls
    std::optional<int>         pca_k;         // number of principal components (default 10)
    std::optional<bool>        pca_eigenvalues;  // emit the scree table instead of the coord table
    std::optional<std::string> pca_emit_html;    // also write a self-contained interactive scatter HTML here
    std::vector<std::string>   project_pops;     // populations placed by lsqproject only (union with project_samples)
    std::optional<std::string> project_samples;  // FILE of Genetic IDs, each projected-only
    std::optional<std::string> project_mode;     // lsq (default) | scaled

    // qpAdm option overrides — reference §9
    std::optional<double> fudge;
    std::optional<int>    als_iterations;
    std::optional<int>    rank;
    std::optional<double> rank_alpha;
    std::optional<bool>   allow_negative_weights;
    std::optional<int>    jackknife;
    std::optional<double> p_se_threshold;
    std::optional<bool>   se_require_p;

    // qpadm-rotate enumeration bounds — reference §10
    std::optional<int> min_sources;
    std::optional<int> max_sources;

    // steppe scan (proxy/model scanner) objective — hard-gate p cutoff (α), default 0.05
    std::optional<double> scan_p_min;
    // steppe scan: may a 1-source (clade) model be crowned the winner? default true
    std::optional<bool> scan_allow_clade;
    // steppe scan Phase-1 search: strategy (greedy|beam|exhaustive), beam width, optional seed
    std::optional<std::string> scan_strategy;
    std::optional<int>         scan_beam_width;
    std::vector<std::string>   scan_base;
    // steppe scan Phase-2: relatedness shortlist mode, and drop/replace swap suggestions
    std::optional<bool> scan_prerank;
    std::optional<bool> scan_suggest_swaps;
    // steppe scan Phase-3: right-set (outgroup) optimization mode + the curated right-pool
    std::optional<std::string> scan_right_search;   // none | check | add-drop
    std::vector<std::string>   scan_right_pool;

    // f4-sweep and f3-sweep controls — reference §11
    std::optional<bool>   sweep_all_combinations;
    std::optional<double> sweep_min_z;
    std::optional<int>    sweep_top_k;
    std::optional<bool>   sweep_sure;
    std::optional<std::string> shard_dir;

    // extract-f2 inputs — reference §12
    std::optional<std::string> prefix;
    std::optional<std::string> geno;
    std::optional<std::string> snp;
    std::optional<std::string> ind;
    std::optional<std::string> out_dir;
    std::vector<std::string>   pops;
    std::optional<int>         auto_top_k;
    std::optional<int>         min_n;
    std::optional<double>      blgsize;
    std::optional<PloidyMode>  ploidy;

    // Filter overrides — reference §13
    std::optional<double> maf;
    std::optional<double> geno_max_missing;
    std::optional<double> mind_max_missing;
    std::optional<bool>   autosomes_only;
    std::optional<bool>   drop_monomorphic;
    std::optional<bool>   transversions_only;
    std::optional<std::string> strand_mode;
    // Membership + ascertainment-guard filter surface (shared by pca/fst/kinship/admixture).
    std::optional<std::string> keep_snps;    // FILE of SNP ids to restrict to (prune.in-style)
    std::optional<std::string> exclude_snps; // FILE of SNP ids to drop
    std::optional<bool>   allow_mixed_ascertainment;  // override the same-ascertainment guard
    std::optional<std::string> emit_kept_snps;        // write the retained SNP ids here
    std::optional<std::string> ld_prune;              // windowed-r2 LD prune "WIN:STEP:R2"

    // extract-f2 run controls — reference §14
    std::optional<std::string> tier;
    std::optional<bool>   dry_run;
    std::optional<bool>   hash_source;

    // Output — reference §15
    std::optional<std::string> out_file;
    std::optional<std::string> format;
};

}  // namespace steppe::config

#endif  // STEPPE_CORE_CONFIG_CLI_ARGS_HPP
