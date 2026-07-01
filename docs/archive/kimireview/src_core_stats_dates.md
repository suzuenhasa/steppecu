I read through `dates.cpp` carefully. It is **not slop** — it's clearly written by someone who understands both the DATES algorithm and the project's frontend/backend split — but a senior C++ developer would flag a handful of real issues, mostly around validation style and C-isms.

## What's genuinely good

- **The architecture is sound.** The host file deliberately avoids the classic O(M²) SNP-pair trap by delegating the autocorrelation to the backend's FFT engine (lines 275–283). The comments explicitly map every step to the reference DATES C source (`dates.c`, `fitexp.c`, `statsubs.c`) and to the project's milestone markers (M5, M6, M-FR-2). That is careful, domain-expert work.
- **No manual memory management.** Everything is `std::vector`, `std::string`, or value types passed by reference. No raw `new`/`delete`, no `FILE*`, no `printf`/`std::cout` logging noise. Output is routed through `DatesResult`.
- **Numeric care in the jackknife.** `weight_jack` accumulates in `long double` (lines 83–104), filters non-finite leave-one-out values, and guards the square root. The caller falls back gracefully when the jackknife estimate is non-finite (lines 395–398).
- **Defensive dimension sizing.** The `n_bin`/`diffmax` calculation uses `std::lround` and checks against `std::numeric_limits<int>::max()` before narrowing (lines 264–272). The intent — preventing pathological options from overflowing the grid indices — is exactly right.
- **Consistent error vocabulary.** Every early return sets a `Status` code (`InvalidConfig`, `RankDeficient`) rather than throwing, logging, or asserting its way out. That fits a systems-style API.

## What a senior developer would flag

**`STEPPE_ASSERT` used for user-input validation (lines 267–270):**

```cpp
STEPPE_ASSERT(n_bin_l <= static_cast<long>(std::numeric_limits<int>::max()),
              "DATES n_bin grid dimension overflows int");
```

The comment admits these values come from user options. `STEPPE_ASSERT` is documented as debug-only fail-fast (line 28). In a release build, a malicious or pathological config can silently overflow `int`. This should return `Status::InvalidConfig`, not assert.

**Unchecked resource dereference (line 141):**

```cpp
ComputeBackend& be = *resources.gpus.at(kPrimaryGpu).backend;
```

If `resources.gpus` is empty or `.backend` is null, this is UB. There is no guard. A senior reviewer would expect a status check or at least an explicit precondition.

**C-style `std::nan("")` instead of idiomatic C++ NaN (lines 73, 397–398):**

```cpp
est = std::nan(""); sig = std::nan("");
```

It works, but `std::numeric_limits<double>::quiet_NaN()` is the modern C++ spelling. Small, but it signals "writes a lot of C."

**Output parameters in `weight_jack` (lines 71–72, 396):**

```cpp
void weight_jack(..., double mean, double& est, double& sig) { ... }
```

A C++ purist would return a small struct or `std::pair`. Output parameters are a C idiom and make call sites harder to read.

**Raw pointer view structs with implicit lifetimes (lines 151–160):**

```cpp
DecodeTileView view;
view.packed = tile.packed.data();
...
const DecodeResult dec = be.decode_af(view);
```

The view holds raw pointers into `tile` and `sample_ploidy`. It is synchronous here, so it is safe, but the contract is invisible to the type system. A `std::span` or at least a documented RAII wrapper would make the lifetime contract explicit.

**Quadratic-in-n-chrom SNP counting (lines 376–380):**

```cpp
for (long ks = 0; ks < M_kept; ++ks) {
    const int chr = chrom_kept[...];
    for (int kc = 0; kc < n_chrom; ++kc)
        if (chrom_present[...] == chr) { ++snp_count[...]; break; }
}
```

With `n_chrom ≤ 22` this is fine, but the inner linear search is unnecessary and reads like a last-minute fix. A small `chrom -> index` map would be cleaner and self-documenting.

**The `s_valid` mask stored as `double` (line 197):**

```cpp
s_valid.push_back((v1 != 0.0 && v2 != 0.0) ? 1.0 : 0.0);
```

It is passed to the backend as a `double*` mask, so the encoding is intentional, but a reader has to trust the contract. A `uint8_t` mask with a documented backend interpretation would be more honest.

**Minor readability nits:**
- `n_emit = n_bin` at line 321 is redundant but harmless.
- `run_dates` is ~280 lines and does a dozen distinct things. The section comments help, but a senior would still nudge it toward smaller named helpers.
- `const Precision precision;` at line 279 relies on the default constructor's policy. Fine if the default is documented, but a reviewer will pause.

## The "slop" test

**Not slop.** Slop is unexplained magic numbers, copy-pasted drift, silent error swallowing, and algorithms that only pass because the test data is small. This file has none of that. The comments are dense but accurate, the constants are named and sourced, and the numeric paths are guarded.

## What it actually looks like

This looks like **solid research/engineering code written by a domain expert who is competent in modern C++ but not a language purist.** The hard algorithmic decisions are correct, the host/backend split is clean, and the code avoids the classic genomics-compute pitfalls. A senior C++ reviewer would say: "Good bones, but tighten the input validation and lose the C-isms." A CUDA reviewer would mostly be happy because the host stays out of the GPU's way.

## Verdict

**B+, ship after hardening the validation paths.** The core math and architecture are production-quality; the `STEPPE_ASSERT`-for-input-validation pattern and the unchecked backend pointer are the only issues that would genuinely embarrass in a code review.