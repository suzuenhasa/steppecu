I read this carefully. It is **not slop** — this is a deliberate, architecture-first CUDA-free seam header written by someone who understands why the boundary matters. A senior C++ reviewer would see real competence, but also a file that has grown organically and carries some convention drift, API bloat, and an inconsistent error model.

## What's genuinely good

- **The CUDA-free seam is real and well motivated.** The header includes no CUDA types, no `cuBLAS`/`cuSOLVER` handles, and no device code. The comment block at the top (lines 1–24) explains exactly why this matters: `core` stays pure host C++20 and the device toolkit is private to `steppe_device`. That is the right way to separate a GPU project.
- **Value-typed result structs with explicit status fields.** `F2Result`, `JackknifeCov`, `GlsWeights`, `QpfstatsSmooth`, etc. use `std::vector` and carry a `Status status = Status::Ok` for domain outcomes (e.g., lines 132–143, 145–155, 157–188). The "value, not throw" intent for numerical degeneracy is correct.
- **The interface is non-copyable / non-movable and virtual-destructor protected.** Lines 672–677 make `ComputeBackend` impossible to slice accidentally, and the destructor is virtual so `std::unique_ptr<ComputeBackend>` in `Resources` is safe.
- **`[[nodiscard]]` is applied consistently** on every virtual that returns a value. That catches ignored results at the orchestration layer.
- **Default host-oracle implementations for the CPU/fake path.** `detect_sample_ploidy_device` (lines 873–896) and `transpose_to_canonical` (lines 931–987) provide a correct CUDA-free fallback so a GPU-free backend does not have to reimplement the oracle math. That is thoughtful API design.
- **The documentation explains *why*, not just *what*.** Comments cite parity gates, AT2 semantics, row/column-major conventions, and backend-drift prevention (e.g., the `F4Blocks` vectorization order contract, lines 87–93; the diagonal convention discussion, lines 56–65). A reviewer can see the design reasoning.

## What a senior developer would flag

**Raw-pointer overloads in a `std::span`/C++20 codebase.** The project clearly knows modern C++ — it uses `std::span<const int>` in many places — but several key virtuals fall back to raw pointers and separate size arguments:

```cpp
// line 1271
virtual void dstat_block_reduce(const double* Q, const double* V, int P, long M,
                                const int* block_id, int n_block, ...);

// lines 1345–1351
[[nodiscard]] virtual DatesMoments dates_curve(
    const double* src1_freq, const double* src2_freq, const double* src_valid,
    const std::uint8_t* packed, std::size_t bytes_per_record, int n_target,
    const int* target_ploidy, const int* grid_cell, long M, ...);
```

The raw pointers are convenient for device dispatch, but they lose size safety and read older than the `std::span` conventions used elsewhere. A `std::span<const double>` plus leading dimensions would be just as device-friendly and far harder to misuse.

**Inconsistent signed/unsigned size types.** One header uses `int P`, `long M`, `std::size_t n_snp`, `int n_block`, and `std::size_t top_k` interchangeably:

```cpp
int P = 0;       // F2Result, line 78
long M = 0;      // DecodeResult, line 446
std::size_t n_snp = 0;        // DecodeTileView, line 390
std::size_t top_k = 1000000;  // SweepConfig, line 316
```

`long` is 32-bit on LLP64 (Windows) and is a risky choice for a genomics column count. Either commit to `std::size_t` everywhere sizes are unsigned, or use a fixed `int64_t` for signed dimensions.

**Domain-error model inconsistency: status values vs. exceptions.** Result PODs carry `Status status` and the comments emphasize "value, not throw" (e.g., line 142, line 153). Yet the default unimplemented base methods throw `std::runtime_error`:

```cpp
// lines 751–753
throw std::runtime_error(
    "ComputeBackend::compute_f2_blocks_device: not supported by this backend ...");
```

If the seam philosophy is "record domain outcomes as values," then an unimplemented or unsupported backend path should probably return a `Status` (or these should be pure virtual). Mixing the two models forces callers to handle both exceptions and per-result statuses.

**Combinatorial API bloat around f2-block residency.** There are five closely related entry points for essentially the same computation:

```cpp
compute_f2_blocks           // line 728
compute_f2_blocks_device    // line 747
compute_f2_blocks_resident  // line 771
compute_f2_blocks_into      // line 804
compute_f2_blocks_streamed  // line 828
```

Each milestone added a new variant. A senior reviewer would push to collapse these into one `compute_f2_blocks` taking a policy struct or enum for {host, device-resident, peer-partial, pinned-direct, streamed}. The current explosion makes the interface harder to implement and harder to call correctly.

**Default algorithm implementations inside the interface header.** `detect_sample_ploidy_device` and `transpose_to_canonical` contain real nested-loop/bit-manipulation code (lines 873–987). They are correct, but they bloat a header whose job is to define a contract. Putting implementation logic in the interface makes the header depend on more details and slows compile times. Better to provide a `protected` helper or a free function in `steppe_core` and have the CPU backend delegate to it.

**Dead scaffolding in `TileEncoding`.** The enum has only one enumerator and the switch duplicates it:

```cpp
// lines 456–458
enum class TileEncoding : int {
    Identity = 0,
};

// lines 972–977
switch (view.encoding) {
    case TileEncoding::Identity:
    default:
        canon = code;
        break;
}
```

This is harmless today, but the `default` branch silently maps any future enumerator to `Identity`. If a new format is added and someone forgets to update the switch, the compiler will not warn because of the `default`. Remove the `default` and let `-Wswitch` catch new enumerators.

**Staggering header size and comment density.** At 1857 lines, this is not a thin interface. The comments are mostly high-quality, but the file is so large that finding the actual virtual contract requires scrolling through dozens of paragraphs. Some comments reference roadmap sections (`M1`, `M4.5`, `M(fit-4)`, `S8`) that may not exist in a reader's checkout, making the header feel like a design document that drifted into source control.

**Minor: `ComputeBackend` deletes move as well as copy.** Lines 673–676 delete both copy and move. Since the object is normally held by `std::unique_ptr`, immobility is fine in practice, but deleting move makes every derived backend immobile too. More idiomatic for a polymorphic base is to make copying protected/deleted and leave moving defaulted (or protected), because `unique_ptr` does not need the object to be movable anyway.

**Minor: magic-number policy defaults embedded in struct definitions.** `SweepConfig::min_z = 3.0` and `top_k = 1000000` (lines 315–316), `QpGraphTopoArena::fudge = 1e-4` (line 246), `kSweepFilterMinZ = 0` / `kSweepFilterTopK = 1` (lines 297–298). They are documented, but a senior reviewer would still ask whether these belong in a defaults header/config object rather than hard-coded in the backend contract.

## The "slop" test

**Not slop.** Slop would be unexplained magic numbers, copy-pasted code with stale comments, missing error handling, or obviously wrong math that happens to pass. This file has none of that. The comments are dense but they are *accurate* and explain architectural invariants. The interface is intentional, even where it is imperfect.

## What it actually looks like

This looks like **a well-above-average systems/architecture header written by someone who thinks hard about boundaries and parity**, but who has been adding milestones to the same file for a while without refactoring the API surface. It is the work of a domain-knowledgeable engineer who values traceability over terseness: every struct has a mini-essay attached, every default is justified, and every backend path is named. A senior teammate would trust the design intent but would immediately schedule a cleanup pass to shrink the interface, unify the f2-block variants, and replace raw-pointer seams with `std::span`.

## Verdict

**B+.** Solid, production-worthy seam design with real architectural competence, but the API has accreted too many overlapping entry points and the header is bloated by implementation logic and milestone scaffolding. In a job-application showcase, the design reasoning would impress; the raw-pointer seams, inconsistent status-vs-exception model, and `TileEncoding` scaffolding would be the talking points in a follow-up interview.
