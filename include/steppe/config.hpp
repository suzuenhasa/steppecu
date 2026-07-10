// include/steppe/config.hpp
//
// The single place the steppe library's configuration knobs — precision,
// device/resource, and QC-filter settings — are defined. Host-only and
// CUDA-free, so it compiles into the core, CLI, and Python bindings without
// pulling in the device layer.
//
// Reference: docs/reference/include_steppe_config.hpp.md
#ifndef STEPPE_CONFIG_HPP
#define STEPPE_CONFIG_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace steppe {

// Named constants (values + rationale) — reference §2
inline constexpr int kDefaultMantissaBits = 40;
inline constexpr unsigned long long kFstatMaxComb = 100000000ULL;
inline constexpr std::size_t kFstatDefaultSweepTopK = 1000000;
inline constexpr int kCdivBlock = 16;
inline constexpr double kRelFloor = 1e-12;
inline constexpr double kAbsFloor = 1e-300;
inline constexpr double kHetCorrDenomFloor = 1.0;
inline constexpr int kBlockGroupPadBase = 2;
inline constexpr int kGesvdjMaxDim = 32;
inline constexpr std::size_t kCublasWorkspaceBytes = 64u * 1024u * 1024u;
inline constexpr std::size_t kFitBudgetFreeVramFallbackBytes = static_cast<std::size_t>(4) << 30;
inline constexpr std::size_t kFitBudgetHeadroomBytes = static_cast<std::size_t>(512) << 20;
inline constexpr unsigned kFeederRawBufsPerPop = 3u;
inline constexpr unsigned kFeederOutBufsPerPop = 4u;
inline constexpr std::size_t kResidentTensorCount = 2;
// Dense host Q/V/N stacks the Resident tier materializes (extract-f2). The Resident
// output engine reads the whole P×M_kept input from one host buffer of this many
// double stacks; the tier-select host-input clamp keeps it off the host-RAM wall.
inline constexpr std::size_t kResidentHostInputStacks = 3;
inline constexpr std::size_t kChunkInputStacks = 4;
inline constexpr std::size_t kChunkOutputStacks = 4;
inline constexpr int kInvalidDeviceId = -1;
inline constexpr int kStreamDeviceChunks = 2;
inline constexpr std::size_t kDefaultSearchStreams = 4;
inline constexpr double kMaxVramUtilizationFraction = 0.80;
inline constexpr double kResidentTierVramFraction = 0.70;
inline constexpr double kHostTierRamFraction = 0.60;

static_assert(kResidentTierVramFraction > 0.0 && kResidentTierVramFraction <= kMaxVramUtilizationFraction,
              "kResidentTierVramFraction must be in (0, kMaxVramUtilizationFraction].");
static_assert(kHostTierRamFraction > 0.0 && kHostTierRamFraction <= 1.0,
              "kHostTierRamFraction must lie in (0, 1].");

inline constexpr double kStreamTileBudgetFraction = 0.25;
inline constexpr double kDefaultBlockSizeCm = 5.0;
inline constexpr double kCentimorgansPerMorgan = 100.0;
inline constexpr double kBpFallbackWindow = 2.0e6;

// Li-Stephens haplotype-copying engine (`steppe paint`) — the up-front cost guard.
// The forward-backward core does O(N·K·M) work over N recipient haplotypes, K donor
// states, and M streamed SNPs; the per-wave forward table is recip_batch·K doubles.
// Both are capped up front, the same --sure posture the f-stat sweep uses (§3.5):
// a run past the work cap refuses without an explicit override, and a per-wave α
// footprint past its cap is a hard error asking for a smaller --recip-batch. The
// values pick a "minutes, not hours" work envelope and a few-GB resident wave.
inline constexpr double kLsMaxWorkStates = 1.0e12;
inline constexpr std::size_t kLsMaxAlphaFootprintBytes = static_cast<std::size_t>(4) << 30;
inline constexpr double kLsDefaultNe = 20000.0;
inline constexpr int kLsDefaultRecipBatch = 256;
inline constexpr int kAutosomeChromMin = 1;
inline constexpr int kAutosomeChromMax = 22;
inline constexpr double kMindFilterInactiveThreshold = 1.0;
inline constexpr char kDefaultDiskCachePath[] = "./steppe_f2_blocks.cache";

// Precision policy — reference §3
struct Precision {
    enum class Kind {
        Fp64,
        EmulatedFp64,
        Tf32
    };

    Kind kind = Kind::EmulatedFp64;
    int mantissa_bits = kDefaultMantissaBits;

    [[nodiscard]] static constexpr Precision fp64() {
        return Precision{Kind::Fp64, kDefaultMantissaBits};
    }

    [[nodiscard]] static constexpr Precision emulated_fp64(int bits = kDefaultMantissaBits) {
        return Precision{Kind::EmulatedFp64, bits};
    }

    [[nodiscard]] static constexpr Precision tf32() {
        return Precision{Kind::Tf32, kDefaultMantissaBits};
    }
};

static_assert(Precision::emulated_fp64().kind == Precision::Kind::EmulatedFp64 &&
                  Precision::emulated_fp64().mantissa_bits == kDefaultMantissaBits,
              "Precision::emulated_fp64() must equal {EmulatedFp64, kDefaultMantissaBits}.");
static_assert(Precision::fp64().kind == Precision::Kind::Fp64 &&
                  Precision::fp64().mantissa_bits == kDefaultMantissaBits,
              "Precision::fp64() must equal {Fp64, kDefaultMantissaBits}.");
static_assert(Precision::tf32().kind == Precision::Kind::Tf32 &&
                  Precision::tf32().mantissa_bits == kDefaultMantissaBits,
              "Precision::tf32() must equal {Tf32, kDefaultMantissaBits}.");
static_assert(Precision{}.kind == Precision::Kind::EmulatedFp64 &&
                  Precision{}.mantissa_bits == kDefaultMantissaBits,
              "Precision must stay a default-constructible aggregate equal to emulated_fp64().");

// Device & runtime settings — reference §4
struct DeviceConfig {
    std::vector<int> devices;
    Precision precision;
    std::size_t stream_count = 1;
    std::size_t search_streams = kDefaultSearchStreams;
    bool use_mem_pool = true;
    bool enable_peer_access = true;
    bool prefer_p2p_combine = true;
    bool deterministic = true;

    enum class ForceTier { Auto, Resident, HostRam, Disk };
    ForceTier force_tier = ForceTier::Auto;

    std::string disk_cache_path;
};

// Strand-ambiguous SNP policy — reference §5
enum class StrandMode { Drop, Keep, Flip };

// QC filters — reference §6
struct FilterConfig {
    double maf_min = 0.0;
    double geno_max_missing = 1.0;
    double mind_max_missing = kMindFilterInactiveThreshold;
    bool autosomes_only = false;
    bool drop_monomorphic = false;
    bool transversions_only = false;
    StrandMode strand_mode = StrandMode::Drop;
    std::vector<std::string> include_snp_ids;
    std::vector<std::string> exclude_snp_ids;
    std::string prune_in_path;

    // Same-ascertainment guard override. When an external SNP-keyed input (a --keep-snps /
    // prune.in list, or include ids) is combined with the target and the two carry a different
    // ascertainment fingerprint (the list is largely absent from the target — a cross-panel
    // intersection, the consumer-DNA f4-bias failure), the front-end refuses. Setting this
    // true acknowledges the mix and proceeds anyway. Reference §6.
    bool allow_mixed_ascertainment = false;

    // Windowed-r2 LD pruning (--ld-prune WIN:STEP:R2). A variant-count sliding window (WIN
    // variants wide, sliding STEP variants each shift) that greedily removes the higher-major-
    // allele-frequency (== lower-MAF) variant of every within-window same-chromosome pair whose
    // genotypic r^2 exceeds ld_prune_r2 — plink2 --indep-pairwise, default --indep-order 2 (the
    // backward within-window scan). Runs AFTER the QC keep mask (on the survivors), on the GPU
    // (device-resident pairwise r^2 + a host greedy selection). Inactive when any of the three is
    // non-positive. Reference §6.
    int ld_prune_window = 0;
    int ld_prune_step = 0;
    double ld_prune_r2 = 0.0;

    [[nodiscard]] bool ld_prune_active() const noexcept {
        return ld_prune_window > 0 && ld_prune_step > 0 && ld_prune_r2 > 0.0;
    }
};

}  // namespace steppe

#endif  // STEPPE_CONFIG_HPP
