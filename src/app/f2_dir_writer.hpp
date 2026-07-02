// src/app/f2_dir_writer.hpp
//
// The f2_blocks DIRECTORY writer (cli-bindings.md §4.3, §3, §8) — the WRITE side of
// the AT2-shaped interchange the precompute-once / fit-many seam consumes (ADR-0005).
// This is the inverse of read_f2_dir (f2_dir_io.hpp); M(cli-1) shipped ONLY the
// reader (the M(cli-1) test wrote a dir by hand), so this is the missing "M7" piece
// the cli-bindings.md §3 "Honest cache gap" calls out, built in M(cli-4).
//
// A <dir> is (cli-bindings.md §4.3):
//   <dir>/f2.bin     the STPF2BK1 numeric payload (src/device/f2_disk_format.hpp):
//                    header(64B) | f2[P*P*nb] | vpair[P*P*nb] | block_sizes[nb int32],
//                    block-major outer, column-major i+P*j within — BYTE-IDENTICAL to
//                    F2BlockTensor's i + P*j + P*P*b layout, so read_f2_dir round-trips
//                    it by construction.
//   <dir>/pops.txt   the P population labels, ONE PER LINE, in P-axis index order —
//                    the name<->index map (IndPartition.groups[].label order).
//   <dir>/meta.json  provenance: meta_schema_version (the SIDECAR schema, kF2MetaSchemaVersion
//                    — NOT the f2.bin binary version kF2DiskVersion), steppe version+SHA,
//                    ENGAGED precision tag, blgsize cM, n_block, filter flags, source dataset
//                    sha256s, pop selection, pops_sha256 (sha256 of pops.txt — the name<->index
//                    map, so a swapped/corrupted pops.txt is detectable), f2_cache_id (sha256 of f2.bin).
//
// THE REAL-vpair RULE (cli-bindings.md §4.3, the M(cli-4) load-bearing decision): the
// vpair region carries the REAL per-block pairwise-valid SNP counts from the precompute
// (F2BlockTensor.vpair), NOT zeros. The M(cli-1) test helper wrote zeros because the
// FIT reads block_sizes, not vpair — but the F1 missing-block / NA path detects a
// dropped pair block by `vpair == 0`, so REAL vpair is what drives the maxmiss>0 drop.
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate): it reaches
// the CUDA-FREE f2_disk_format.hpp for the on-disk header struct + magic/version
// stamps (the single home so the writer and reader cannot drift), nothing more.
#ifndef STEPPE_APP_F2_DIR_WRITER_HPP
#define STEPPE_APP_F2_DIR_WRITER_HPP

#include <filesystem>
#include <string>
#include <vector>

#include "steppe/error.hpp"   // steppe::Status
#include "steppe/fstats.hpp"  // steppe::F2BlockTensor

namespace steppe::app {

/// The meta.json SIDECAR schema version (cli-bindings.md §4.3). This versions the JSON
/// provenance SHAPE (the field set the writer emits / a reader expects), and is the
/// value stamped as meta.json's `"meta_schema_version"`. It is DISTINCT from
/// `device::kF2DiskVersion` (src/device/f2_disk_format.hpp), which versions the f2.bin
/// BINARY payload bytes — the two evolve independently and must NOT be conflated: a
/// provenance-field change bumps this, an on-disk-layout change bumps that. The reader
/// gates on kF2DiskVersion (f2.bin); meta.json stays advisory / non-load-bearing.
inline constexpr int kF2MetaSchemaVersion = 1;

/// The provenance recorded in <dir>/meta.json (cli-bindings.md §4.3; architecture.md
/// §12 reproducibility block). All fields are app-resolved strings/values so the
/// writer is a pure serializer with no compute dependency.
struct F2DirMeta {
    std::string steppe_version;     ///< STEPPE_VERSION (project VERSION).
    std::string precision_tag;      ///< ENGAGED precision (emu/fp64/tf32 — from the result/Resources, not just requested).
    int precision_mantissa_bits = 0;///< the engaged mantissa-bit count (meaningful for emu).
    double blgsize_cm = 0.0;        ///< jackknife block size, centimorgans.
    int n_block = 0;                ///< number of jackknife blocks in f2.bin.
    int P = 0;                      ///< population count (the f2.bin P axis).
    long n_snp_total = 0;           ///< SNPs READ from the .snp (pre-filter).
    long n_snp_kept = 0;            ///< SNPs kept after the filters (the precompute SNP set).
    // Filter flags (the resolved FilterConfig, recorded so a run is reproducible).
    double maf_min = 0.0;
    double geno_max_missing = 1.0;  ///< AT2 maxmiss.
    double mind_max_missing = 1.0;
    bool autosomes_only = false;
    bool drop_monomorphic = false;
    bool transversions_only = false;
    // Source dataset shas (sha256 of the geno/snp/ind files; empty if not computed).
    std::string geno_sha256;
    std::string snp_sha256;
    std::string ind_sha256;
    std::string geno_path, snp_path, ind_path;
    // Source-provenance hash policy (the extract-f2 --hash opt-in; default OFF). The
    // whole-.geno SHA is the bottleneck (~37s of ~41s on the 6.7 GB 1240K .geno — a
    // provenance value, NOT correctness), so it is SKIPPED by default. When false the
    // writer does NOT hash any source file whose sha is still empty; meta.json records
    // geno/snp/ind_sha256 = "" + "source_hash_computed": false so a consumer knows the
    // absence is DELIBERATE (not a failed/forgotten hash). When true the writer fills
    // any empty source sha from its path (the caller may PRE-fill geno_sha256 from a
    // background thread to overlap the big hash with the GPU pipeline).
    bool hash_source_files = false;
    // Pop selection echo (mode + the resolved labels are in pops.txt; this records the request).
    std::string pop_selection;      ///< human string, e.g. "explicit:England_BellBeaker,..." | "auto-top:9" | "min-n:30".
};

/// Result of write_f2_dir: ok + the f2_cache_id (sha256 of f2.bin) on success, or a
/// fault Status + a human-readable reason on failure (the app prints the reason; the
/// library never prints — architecture.md §10). A write/IO failure is Status::IoError-
/// shaped but carried as InvalidConfig here (an unwritable --out is a config-level
/// fault the user must fix, matching the reader's fault taxonomy).
struct F2DirWriteResult {
    bool ok = false;
    Status status = Status::Ok;
    std::string error;            ///< empty on success; the reason on failure.
    std::string f2_cache_id;      ///< "sha256:<hex>" of f2.bin (valid only when ok).
};

/// Write an f2_blocks dir at `dir`: f2.bin (STPF2BK1, REAL vpair) + pops.txt (the P
/// labels in index order) + meta.json (provenance). Creates `dir` if absent. The
/// f2.bin byte layout is exactly read_f2_dir's expectation (it round-trips by
/// construction). `meta.n_block`/`meta.P` are stamped from `f2` (the writer trusts the
/// tensor's shape, not meta). Fails (ok=false + reason) on a shape mismatch
/// (pop_labels.size() != f2.P), an unwritable path, or a degenerate tensor.
[[nodiscard]] F2DirWriteResult write_f2_dir(const std::filesystem::path& dir,
                                            const F2BlockTensor& f2,
                                            const std::vector<std::string>& pop_labels,
                                            const F2DirMeta& meta);

/// SHA-256 of a whole file as a lowercase hex digest (64 chars), or "" if the file
/// cannot be opened. This is the SAME bulk/block SHA-256 the writer uses for the
/// f2_cache_id and the source-dataset shas (the single SHA home), exposed so the
/// extract-f2 command can hash the (large) source .geno on a BACKGROUND THREAD,
/// overlapping the ~tens-of-seconds whole-file hash with the GPU decode+f2 pipeline,
/// then PRE-fill F2DirMeta.geno_sha256 before write_f2_dir (which then skips re-hashing
/// it). The digest is byte-for-byte `sha256sum`-compatible (FIPS 180-4).
[[nodiscard]] std::string sha256_file(const std::filesystem::path& path);

}  // namespace steppe::app

#endif  // STEPPE_APP_F2_DIR_WRITER_HPP
