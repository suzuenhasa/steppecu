// src/app/f2_dir_writer.hpp
//
// Writes an f2-blocks directory (f2.bin + pops.txt + meta.json) to disk: the
// write side of the compute-once / fit-many artifact and the exact byte-inverse
// of the read_f2_dir reader, so a directory round-trips by construction. Plain
// C++20, CUDA-free (app-only).
//
// Reference: docs/reference/src_app_f2_dir_writer.hpp.md
#ifndef STEPPE_APP_F2_DIR_WRITER_HPP
#define STEPPE_APP_F2_DIR_WRITER_HPP

#include <filesystem>
#include <string>
#include <vector>

#include "steppe/error.hpp"
#include "steppe/fstats.hpp"

namespace steppe::app {

// meta.json schema version — reference §3
inline constexpr int kF2MetaSchemaVersion = 1;

// F2DirMeta — provenance record — reference §5
struct F2DirMeta {
    std::string steppe_version;
    std::string precision_tag;
    int precision_mantissa_bits = 0;
    double blgsize_cm = 0.0;
    int n_block = 0;
    int P = 0;
    long n_snp_total = 0;
    long n_snp_kept = 0;
    double maf_min = 0.0;
    double geno_max_missing = 1.0;
    double mind_max_missing = 1.0;
    bool autosomes_only = false;
    bool drop_monomorphic = false;
    bool transversions_only = false;
    std::string geno_sha256;
    std::string snp_sha256;
    std::string ind_sha256;
    std::string geno_path, snp_path, ind_path;
    bool hash_source_files = false;
    std::string pop_selection;
};

// F2DirWriteResult — write outcome — reference §6
struct F2DirWriteResult {
    bool ok = false;
    Status status = Status::Ok;
    std::string error;
    std::string f2_cache_id;
};

// write_f2_dir — write entry point — reference §7
[[nodiscard]] F2DirWriteResult write_f2_dir(const std::filesystem::path& dir,
                                            const F2BlockTensor& f2,
                                            const std::vector<std::string>& pop_labels,
                                            const F2DirMeta& meta);

// sha256_file — whole-file hashing — reference §8
[[nodiscard]] std::string sha256_file(const std::filesystem::path& path);

}  // namespace steppe::app

#endif  // STEPPE_APP_F2_DIR_WRITER_HPP
