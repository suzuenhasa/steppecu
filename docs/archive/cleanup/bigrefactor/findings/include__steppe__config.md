# Review findings — include__steppe__config

Files: /home/suzunik/steppe/include/steppe/config.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

## Group 5 — Hardcoded values / magic numbers

This header IS the project's magic-number-promotion home (every device/algorithm
knob is an `inline constexpr` named constant with a documented value, ROADMAP §4),
so it is exemplary by design. The few remaining raw literals are struct-field
defaults; the named-constant homes (kCdivBlock, kAutosomeChromMax, the three VRAM/RAM
fractions guarded by static_asserts at :127-130, kCublasWorkspaceBytes, kCentimorgansPerMorgan)
are all single-source and drift-protected. No HIGH/MED issues.

- [5.1][LOW] config.hpp:241 — `search_streams = 4` is a bare literal default with no named-constant home, unlike every other tuned knob in this header (mantissa bits, block edge, fractions, workspace bytes were all promoted to `k...` constants per ROADMAP §4). The throughput-lane stream count is the same kind of tunable policy number. Suggested: promote to a named `kDefaultSearchStreams = 4` constant alongside the others for consistency (LOW only — it is parity-neutral throughput-only and documented in the field comment).
- [5.4][LOW] config.hpp:354-356 — the Disk-tier default cache path `"./steppe_f2_blocks.cache"` is described as "the frozen default" in the comment but the actual literal is NOT homed here (the field defaults to empty; the literal lives in the consuming resolver). For a "frozen default" string that two paths (config + STEPPE_F2_CACHE_PATH fallback) must agree on, the single-source rule this header otherwise enforces argues for naming it here. Suggested: add a named `kDefaultDiskCachePath` constant in this header so the frozen default has the same single home as the other promoted values (LOW — hygiene/consistency; no current drift observed since only one consumer holds the literal).

## Group 6 — Naming

No Group 6 issues found.

Header is exemplary on naming. Constants are uniform `kPascalCase` (`kDefaultMantissaBits`, `kCdivBlock`, `kMaxVramUtilizationFraction`, ...); struct fields are uniform `snake_case` (`mantissa_bits`, `stream_count`, `search_streams`, `maf_min`); enum types/enumerators are uniform `PascalCase` (`Precision::Kind` {`Fp64`,`EmulatedFp64`,`Tf32`}, `DeviceConfig::ForceTier` {`Auto`,`Resident`,`HostRam`,`Disk`}). No conflict with 6.3. No cryptic/opaque members — no `tmp`/`arr`/`flag`/single-letter fields (6.1). Names are accurate to semantics — `devices` is an id vector, `stream_count`/`search_streams` are counts, `include_snp_ids`/`exclude_snp_ids` are id lists (comment at :420 explicitly disambiguates "SNP IDs ... not indices"), `prefer_p2p_combine`/`enable_peer_access` match their documented MAY-WE vs WHICH-PATH split — so no misleading names (6.2). Abbreviations are all domain-standard or documented inline: `Cm`/centimorgan, `maf`, `geno`/`mind` (PLINK flag names, intentional), `P2P`, `Gds`, `Vram`/`Ram`, and `Cdiv` (ceil-divide, cross-referenced to launch_config.hpp at :50) — none nonstandard for this codebase (6.4).

## Group 7 — Duplication

No Group 7 issues found.

This unit is a pure declarative config header — `inline constexpr` constants, three plain structs (`Precision`, `DeviceConfig`, `FilterConfig`) of field declarations, two `static_assert`s (:127-130), and documentation. No function bodies, loops, casts, or `sizeof` usage exist, so 7.2/7.3 have no surface. 7.1: the two `static_assert`s check distinct invariants against distinct bounds (`kResidentTierVramFraction <= kMaxVramUtilizationFraction` vs `kHostTierRamFraction <= 1.0`), not one block differing by a constant. 7.4: the repeated `inline constexpr double` + doc-comment shape of the three VRAM/RAM fractions (:111/:119/:125) is idiomatic single-source constant definition, not foldable boilerplate — collapsing them behind a macro/helper would erase the per-value documentation that is this header's whole purpose (ROADMAP §4 magic-number-promotion home).

## Group 8 — Comments

- [8.2][LOW] config.hpp:18-19 vs :181-182 — the EmulatedFp64@40 speedup figure is stated inconsistently: the top banner says "7–13× faster" (line 19) but the `EmulatedFp64` enum doc says "MEASURED 7–17× over native FP64 on real AADR" (line 181-182). Both describe the same default mantissa-40 emulated path, so one upper bound (13× vs 17×) is stale relative to the measured number. Suggested: reconcile both to the single measured range so the two doc sites cannot drift (parity-neutral; documentation only).

## Group 9 — Constants & configuration

No Group 9 issues found.

9.1 (should-be-const left mutable): every numeric knob is already `inline constexpr` (kDefaultMantissaBits :44, kCdivBlock :51, kRelFloor :57, kAbsFloor :63, kHetCorrDenomFloor :71, kBlockGroupPadBase :82, kCublasWorkspaceBytes :88, the three VRAM/RAM fractions :111/:119/:125, kDefaultBlockSizeCm :136, kCentimorgansPerMorgan :140, kAutosomeChromMin/Max :156-157). The mutable members are the `Precision`/`DeviceConfig`/`FilterConfig` struct fields — these MUST stay mutable: they are the user-facing config populated before `build()` validates them, so `const`-ing them would break the contract. Correct as written. 9.2 (tangled config): this header IS the project's surfaced config home (ROADMAP §4) — constants at file top, knobs as documented struct fields; there is no logic in which a knob could be buried. 9.3 (positional booleans): declaration-only header, no function calls, so no positional-bool call sites exist. The several `bool` fields (use_mem_pool :246, enable_peer_access :284, prefer_p2p_combine :316, deterministic :339, autosomes_only :398, drop_monomorphic :404, transversions_only :411) are named struct fields set by name, not positional args. The correlated enable_peer_access × prefer_p2p_combine pair is a deliberate, documented MAY-WE/WHICH-PATH split (the comment at :248-316 explicitly addresses the conflation risk and the §4 combine gate ANDs both) — intentional design, not a flaggable two-bool tangle.

## Group 10 — Initialization

No Group 10 issues found.

10.1 (late/distant or uninitialized-then-assigned): this unit is a pure declarative config header — `inline constexpr` constants (:44-157), three plain structs (`Precision` :168-215, `DeviceConfig` :221-357, `FilterConfig` :365-432), and two `static_assert`s (:127-130). There are no function bodies, no local variables, and no executable statements, so there is no declaration that can be "far from first use" or "uninitialized-then-assigned." 10.2 (zero-init assumptions that do not hold): every SCALAR struct field carries an explicit in-class default initializer rather than relying on implicit zero-init — `Precision::kind`/`mantissa_bits` (:207, :214); `DeviceConfig::stream_count=1` (:237), `search_streams=4` (:241), `use_mem_pool=true` (:246), `enable_peer_access=true` (:284), `prefer_p2p_combine=true` (:316), `deterministic=true` (:339), `force_tier=Auto` (:349); `FilterConfig::maf_min=0.0` (:372), `geno_max_missing=1.0` (:380), `mind_max_missing=1.0` (:386), `autosomes_only/drop_monomorphic/transversions_only=false` (:398, :404, :411). The members without an explicit initializer are class types whose default constructors run automatically and are correctly empty-by-construction — `devices` (std::vector, :228, empty default is the documented "auto-enumerate" sentinel at :226), `precision` (Precision, :232, inherits its own member defaults), `disk_cache_path` (std::string, :356), `include_snp_ids`/`exclude_snp_ids` (std::vector, :421/:425), `prune_in_path` (std::string, :431). No scalar relies on implicit zero-init, and the explicit defaults match the documented semantics (the no-op `1.0`/`false`/`0.0` filter defaults and the capable-first `true` defaults). Correct as written.

