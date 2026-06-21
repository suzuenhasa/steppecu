# steppe access layer — CLI + Python bindings (design + milestone build-order)

Status: DESIGN, for approval. Scope: step 2 of the build sequence (productize the
FINISHED qpAdm fit backend). No implementation in this doc; it is the contract the
implementer builds against. Every API claim below cites the real header/section.

---

## 1. Goal + scope

The Phase-2 fit backend is finished and golden-gated on real AADR (GPU + CpuBackend):
precompute (`DeviceF2Blocks` / `F2BlockTensor`), `run_qpadm`, `run_qpwave`,
`run_qpadm_search`, the rank test / popdrop / rankdrop tables, jackknife SE +
`JackknifePolicy`, and the domain outcomes (`Status`, incl. `ChisqUndefined`). Step 2
exposes exactly that surface to users through a **CLI** and **Python bindings** — no new
statistics, no new math.

### What exists today (confirmed, read-only)

- **No `src/app/`, no `bindings/`, no `python/`, no `pyproject.toml`, no `main()`.** The
  access layer is greenfield. Top-level `CMakeLists.txt:60-67` adds only
  `include`, `src/io`, `src/core`, `src/device`, `tests`.
- **The CMake gates DO already exist** as OFF stubs:
  `option(STEPPE_BUILD_PYTHON ... OFF)` and `option(STEPPE_BUILD_CLI ... OFF)`
  (`cmake/SteppeOptions.cmake:22-23`). Step 2 wires subdirs behind them; it does not
  invent the option vocabulary.
- **The architecture §9 config plumbing — `ConfigBuilder`, `RunConfig`, `CliArgs`,
  `GenotypeDataset` — does NOT exist in code.** `grep` across `src/`+`include/` returns
  nothing. The real public surface is the four installed headers
  (`include/steppe/{qpadm,fstats,config,error}.hpp`) plus the CUDA-free device handles
  (`src/device/{device_f2_blocks,resources,backend_factory,f2_blocks_out,f2_disk_format}.hpp`).
  The §9 `RunConfig run_qpadm(const GenotypeDataset&, …)` signature shown in the
  architecture is design intent; the **actual** entry takes
  `DeviceF2Blocks`/`QpAdmModel`/`QpAdmOptions`/`Resources` directly. Decision below
  (§9.1): for step 2 the CLI/bindings map flags **straight onto the real structs**
  (`DeviceConfig`/`QpAdmOptions`/`PopSelection`/`FilterConfig`); a minimal
  `RunConfig`/`ConfigBuilder` for TOML/env/CLI precedence is a contained add (M(cli-0)),
  not a prerequisite.

### Layering / ADR constraints the new targets MUST obey

Enforced by the compiler via CMake link visibility (architecture.md §4 "how CMake
enforces it") + the CI architecture-grep test — not by convention:

1. **Allowed edges only: `app/bindings → api → core → device`; `io` is a sibling leaf
   `app`/`bindings` is the ONLY layer that may wire `io` output into compute**
   (architecture.md §4, line 239). The new targets link
   `steppe::core steppe::io steppe::api steppe::device`.
2. **CUDA is PRIVATE to `steppe_device`.** `app`/`bindings` are **plain C++20 CXX
   targets, never CUDA targets** (no `LANGUAGES CUDA`). They reach the GPU only through
   the CUDA-free seams: `device::build_resources(DeviceConfig)` (`resources.hpp:229`),
   the `compute_f2_blocks_multigpu_*` decls (`f2_blocks_multigpu.hpp:48,72,120`), the
   `run_qpadm*`/`run_qpwave` entries (`qpadm.hpp`), and the CUDA-free opaque handles
   (`DeviceF2Blocks`, `F2BlocksOut`, `Resources` — `Impl` lives in `.cu`). Any leaked
   `<cuda_runtime.h>`/`<cublas_v2.h>` include is a hard build failure on a host-compiled
   TU — the layering proof is structural and free.
3. **Domain outcomes are `Status`, never exceptions** (architecture.md §10, ADR-0008).
   `RankDeficient` / `NonSpdCovariance` / `ChisqUndefined` are per-model results
   (`error.hpp:22-49`). The CLI emits a `status` column and exits 0 on a model-level
   domain outcome (record-and-continue, critical in a search); only faults
   (`InvalidConfig`, `DeviceOom`, file/format errors) are non-zero exit. The binding
   maps domain `Status` to a result field, raising only on faults.
4. **No `printf`/`std::cout` in library code** (architecture.md §10). The CLI owns
   stdout; the binding installs an spdlog→Python `logging` sink (architecture.md §15).
5. **The installed cross-toolchain boundary is a C ABI** (ADR-0008, architecture.md §16:
   `steppe_*_t` opaque handles, `steppe_status_t`). But §16 explicitly permits the
   in-process binding to link the C++ convenience layer directly:
   *"The binding (§15) links the C++ layer in-process, so it is unaffected."* The CLI is
   in-tree too. **Decision (§9.4): step 2 binds the existing C++ headers in-process; the
   cross-toolchain C ABI is a separate, deferrable milestone** (M(abi-1), optional).

### ADRs invoked
- **ADR-0002** — nanobind over pybind11 (architecture.md §15).
- **ADR-0005** — precompute-once / fit-many (the `extract-f2` dir → many fits seam).
- **ADR-0008** — C ABI at the public/installed boundary (deferred for step 2; §9.4).

### The one load-bearing fact that shapes everything

**The compute seam is INDEX-ONLY; names are an app concern.** `QpAdmModel{target:int,
left:vector<int>, right:vector<int>}` references populations by **index into the f2
P axis** (`qpadm.hpp:106-119`; header comment lines 5-7: *"name→index resolution is an
app/binding concern"*). But **no f2 handle carries pop names**: `F2BlockTensor` has only
`P/n_block/block_sizes` (`fstats.hpp:67-71`), `DeviceF2Blocks` only
`P/n_block/device_id/block_sizes` (`device_f2_blocks.hpp:40-45`), and the on-disk
`STPF2BK1` header is purely numeric (`f2_disk_format.hpp:27-37`). Names are born upstream
in `io`: `read_ind` returns `IndPartition.groups[]` **sorted ascending by label**, and
that order IS the P-axis row order (`ind_reader.hpp:65-72`); `GenotypeTile.pop_labels` is
the same list (`genotype_tile.hpp:68-69`). The existing fit test proves the gap — it
loads a raw `.bin` and **hardcodes** the name→index map
(`tests/reference/test_qpadm_parity.cu`).

**Consequence:** the single biggest design decision is that the access layer must
**persist a pop-name sidecar next to the f2 cache** and own all name↔index resolution.
This is also what makes the cache AT2-ergonomic (a populations list beside the blocks).
See §4.3.

---

## 2. The public API surface the CLI/bindings wrap

All `namespace steppe` unless noted.

### 2.1 Fit entries — `include/steppe/qpadm.hpp`

| Entry | Signature (real) | Returns |
|---|---|---|
| qpAdm, device-resident (primary) | `run_qpadm(const device::DeviceF2Blocks&, const QpAdmModel&, const QpAdmOptions&, device::Resources&)` `qpadm.hpp:170` | `QpAdmResult` |
| qpAdm, host-oracle overload | `run_qpadm(const F2BlockTensor&, …)` `qpadm.hpp:178` | `QpAdmResult` |
| Rotation / model-space search | `run_qpadm_search(const device::DeviceF2Blocks&, std::span<const QpAdmModel>, const QpAdmOptions&, device::Resources&)` `qpadm.hpp:192` | `std::vector<QpAdmResult>` (input order) |
| Rotation, host-oracle overload | `run_qpadm_search(const F2BlockTensor&, …)` `qpadm.hpp:201` | `std::vector<QpAdmResult>` |
| qpWave (no target) | `run_qpwave(const device::DeviceF2Blocks&, std::span<const int> left, std::span<const int> right, const QpAdmOptions&, device::Resources&)` `qpadm.hpp:226` | `QpWaveResult` |
| qpWave, host-oracle overload | `run_qpwave(const F2BlockTensor&, …)` `qpadm.hpp:233` | `QpWaveResult` |

- `QpAdmModel{ int target; vector<int> left; vector<int> right; int model_index }`
  (`qpadm.hpp:106-119`). `target` is prepended to `left`; `right[0]` == R0.
- `QpAdmOptions{ double fudge=1e-4; int als_iterations=20; int rank=-1; bool
  allow_negative_weights=true; double rank_alpha=0.05; JackknifePolicy jackknife=All;
  double p_se_threshold=0.05; bool se_require_p=false }` (`qpadm.hpp:57-100`). The CLI/py
  defaults MUST match these exactly so a bare invocation reproduces the goldens.
- `enum class JackknifePolicy : int { None=0, FeasibleOnly=1, All=2 }` (`qpadm.hpp:48-52`)
  — the frozen `--jackknife=0/1/2` mapping. **Point estimate is identical across all
  three; the policy governs only which models pay the LOO SE. `run_qpadm`/`run_qpwave`
  IGNORE it** (they always compute SE); **only `run_qpadm_search` reads it**
  (`qpadm.hpp:46-47,84-87`).
- `QpAdmResult` (`qpadm.hpp:124-163`): `weight[]/se[]/z[]`, `p, chisq, dof`,
  `rank_chisq[]/rank_dof[]/rank_p[]`, `est_rank`, `f4rank`, the 7 `rankdrop_*[]` vectors,
  the `popdrop_*[]` table (incl. `popdrop_pat` strings and `popdrop_feasible` as `char`
  0/1), `Status status`, `Precision::Kind precision_tag`, `model_index`. **`se`/`z` EMPTY
  ⇒ "not computed" sentinel — never a fake 0/NaN** (`qpadm.hpp:127-130`).
- `QpWaveResult` (`qpadm.hpp:212-222`): same rank/rankdrop shape minus weights, plus
  `est_rank`, `f4rank`, `status`, `precision_tag`.

### 2.2 Interchange tensor — `include/steppe/fstats.hpp`

`struct F2BlockTensor{ vector<double> f2; vector<double> vpair; vector<int> block_sizes;
int P; int n_block; size() }` (`fstats.hpp:47-78`). Layout: a stack of `n_block`
column-major [P×P] slabs, flat `i + P·j + P·P·b` (block-major outer, column-major
within). **FP64 in every precision mode** (`fstats.hpp:17`). This is the cacheable,
AT2-compatible interchange artifact; the bindings expose it as a Fortran-contiguous numpy
view `(P, P, n_block)` (§5.3).

### 2.3 Config — `include/steppe/config.hpp`

- `struct Precision{ enum class Kind{Fp64, EmulatedFp64, Tf32}; Kind kind=EmulatedFp64;
  int mantissa_bits=40 }` (`config.hpp:266-313`). **The real type is this struct, not the
  §9 bare enum.**
- `struct DeviceConfig{ vector<int> devices; Precision precision; size_t stream_count=1;
  size_t search_streams=4; bool use_mem_pool; bool enable_peer_access;
  bool prefer_p2p_combine; bool deterministic; enum class
  ForceTier{Auto,Resident,HostRam,Disk} force_tier=Auto; string disk_cache_path }`
  (`config.hpp:319-455`). Empty `devices` ⇒ auto-enumerate; size 1 ⇒ single-GPU.
- `struct FilterConfig{ double maf_min; double geno_max_missing; double mind_max_missing;
  bool autosomes_only; bool drop_monomorphic; bool transversions_only;
  vector<string> include_snp_ids; vector<string> exclude_snp_ids; string prune_in_path }`
  (`config.hpp:463-531`). Defaults are no-ops (keep everything).
- Named constants to surface as defaults: `kDefaultMantissaBits=40` (`config.hpp:44`),
  `kDefaultBlockSizeCm=5.0` (`config.hpp:217`), `kAutosomeChromMin/Max=1/22`
  (`config.hpp:237-238`), `kDefaultSearchStreams=4` (`config.hpp:165`),
  `kDefaultDiskCachePath="./steppe_f2_blocks.cache"` (`config.hpp:255`).

### 2.4 Status — `include/steppe/error.hpp`

`enum class Status{ Ok, DeviceOom, RankDeficient, NonSpdCovariance, ChisqUndefined,
InvalidConfig }` (`error.hpp:22-49`). The three middle values are recoverable per-model
domain outcomes (§1 constraint 3).

### 2.5 Precompute + device handles (CUDA-free seams in `src/device/`, `src/core/fstats/`)

| Entry | Signature (real) | Returns |
|---|---|---|
| Build resources | `device::build_resources(const DeviceConfig&)` `resources.hpp:229` | `device::Resources` (one backend+probe per device; fail-fast on bad/dup ordinal) |
| CPU backend | `device::make_cpu_backend()` `backend_factory.hpp:35` | `unique_ptr<ComputeBackend>` (always exists ⇒ GPU-free path) |
| CUDA backend | `device::make_cuda_backend(int=0)` `backend_factory.hpp:53` | `unique_ptr<ComputeBackend>` |
| Precompute, device-resident (PRIMARY) | `core::compute_f2_blocks_multigpu_device(Resources&, MatView Q,V,N, BlockPartition, Precision)` `f2_blocks_multigpu.hpp:72` | `device::DeviceF2Blocks` (resident; zero D2H) |
| Precompute, tiered (Resident/HostRam/Disk) | `core::compute_f2_blocks_multigpu_tiered(Resources&, Q,V,N, partition, precision)` `f2_blocks_multigpu.hpp:48` | `device::F2BlocksOut` |
| Precompute, host wrapper | `core::compute_f2_blocks_multigpu(Resources&, …)` `f2_blocks_multigpu.hpp:120` | `F2BlockTensor` (= device + `to_host()`) |
| Single-backend core compute | `core::compute_f2_blocks(ComputeBackend&, const MatView& Q, …)` `f2_from_blocks.hpp:52` | `F2BlockTensor` |
| Materialize (THE only D2H) | `DeviceF2Blocks::to_host()` `device_f2_blocks.hpp:74` | `F2BlockTensor` |
| Reload to device (H2D inverse) | `device::upload_f2_blocks_to_device(const F2BlockTensor&, int device_id)` `device_f2_blocks.hpp:98` | `DeviceF2Blocks` (the disk-cache reload path) |

### 2.6 IO leaf (`src/io/`) — what feeds precompute (the `extract-f2` chain)

Precompute does **not** take genotype files; it takes `MatView Q/V/N`. The full chain the
CLI's `extract-f2` must wire (today only in reference tests, never a shipped driver):

1. `io::read_ind(path, PopSelection, n_records_present)` → `IndPartition`
   (`ind_reader.hpp:86`). `PopSelection::Mode::{Explicit,AutoTopK,MinN}`
   (`ind_reader.hpp:38-55`). `groups[]` is sorted ascending by label = the P-axis order
   AND the pop-name sidecar source (`ind_reader.hpp:65-72`).
2. `io::read_snp(path, max_snps)` → `SnpTable` (id, chrom, genpos_morgans, ref, alt).
3. `io::GenoReader(geno).read_tile(IndPartition, snp_begin, snp_end)` → `GenotypeTile`
   (packed bytes, no decode; carries `pop_labels` `genotype_tile.hpp:68-69`).
4. Optional `io::filter/*` per `FilterConfig`.
5. `backend->decode_af(DecodeTileView)` → `DecodeResult{q,v,n,P,M}` (the device decodes).
6. `core::assign_blocks(chrom, genpos_morgans, block_size_morgans)` → `BlockPartition`
   (cM→Morgan via `kDefaultBlockSizeCm` / `kCentimorgansPerMorgan`).
7. `compute_f2_blocks_multigpu_device(resources, Q,V,N, partition, precision)` →
   `DeviceF2Blocks` (or the tiered/disk variant for huge P).

**Gap to flag:** there is no single host function that does steps 1-7; the reference tests
wire it ad hoc. `extract-f2` IS that orchestration and belongs in `app/` (the §4 rule).
Factor a shared `app`-level helper so the CLI and bindings share it (DRY); it must link
`io`, which only `app`/`bindings` may.

---

## 3. The precompute → fit data flow (the seam)

```
EIGENSTRAT / PACKEDANCESTRYMAP files
  .ind ──read_ind(PopSelection)──▶ IndPartition (pops sorted ASC by label = P-axis order; the name sidecar)
  .snp ──read_snp───────────────▶ SnpTable (chrom, genpos_morgans, ref/alt)
  .geno─GenoReader.read_tile────▶ GenotypeTile (packed bytes; io does NOT decode)
                                      │  + FilterConfig (snp_filter / mind_prepass / include_exclude)
                                      ▼
            backend.decode_af(DecodeTileView) ─▶ DecodeResult Q/V/N [P×M] column-major
                                      │
            assign_blocks(chrom, genpos_morgans, blgsize) ─▶ BlockPartition
                                      ▼
   compute_f2_blocks_multigpu_device(Resources, Q,V,N, partition, precision)
                                      ▼
              DeviceF2Blocks  ◀── RESIDENT in VRAM (PRIMARY handoff; zero D2H)
                 │   │
   to_host()─────┘   └──────────────────────────────┐
        ▼                                            ▼
  F2BlockTensor (host) ──write f2.bin (STPF2BK1)     run_qpadm / run_qpadm_search / run_qpwave
        │              + pops.txt + meta.json        (read f2 IN VRAM → QpAdmResult / QpWaveResult)
        ▼
   <dir>/  ◀── the AT2-style f2_blocks dir (precompute-once / fit-many; ADR-0005)
        │
   upload_f2_blocks_to_device(host, device_id) ─▶ DeviceF2Blocks  (fit-many reload)
```

The on-disk format `STPF2BK1` (`f2_disk_format.hpp`): a 64-byte fixed header
(`F2DiskHeader{magic,version,dtype,P,n_block,f2_offset,vpair_offset,block_sizes_offset}`,
`f2_disk_format.hpp:27-37`), then f2 region | vpair region | block_sizes trailer,
block-major [P²] slabs column-major within — **byte-identical to `F2BlockTensor`** and
**AT2-compatible in ordering as a GOAL** (the 64-byte prefix is the only delta;
`f2_disk_format.hpp:8-12`).

**Honest cache gap:** the streaming **Disk tier** of `compute_f2_blocks_multigpu_tiered`
(via `F2BlocksOut` / `DeviceConfig::disk_cache_path` / `STEPPE_F2_CACHE_PATH`) writes
`STPF2BK1`, but a **standalone "write `F2BlockTensor` → file + read file →
`F2BlockTensor`" round-trip outside the streaming tier does not exist yet** (the "M7
reader" is referenced in the format header prose but not built). The `extract-f2 →
f2_blocks dir` deliverable needs that round-trip — it is a milestone item (M(cli-4)), the
single largest piece of step 2, not an existing call.

---

## 4. The CLI

Single binary `steppe`, CLI11 subcommands matching AT2/admixr `extract_f2` / `qpadm` /
`qpwave` shapes so AT2 users are at home.

### 4.1 Command set + arg → real-API mapping

```
steppe extract-f2  --geno/--snp/--ind PREFIX   (or --geno F --snp F --ind F)
                   --out DIR
                   [--pops a,b,… | --auto-top-k K | --min-n N]    # PopSelection
                   [--blgsize 0.05]                               # MORGANS (AT2 convention; 0.05 = 5 cM)
                   [--maf 0 --geno-max-miss 1 --mind-max-miss 1]  # FilterConfig
                   [--auto-only --drop-mono --transversions]      # FilterConfig flags (auto-only/drop-mono default ON for extract-f2)
                   [--extract F --exclude F --prune-in F]         # FilterConfig id sets
                   [--precision emu40|emu32|fp64|tf32]            # Precision
                   [--device auto|0,1]                            # DeviceConfig.devices (GPU only)
                   [--dry-run]                                    # T_max / tiles / largest-fitting

steppe qpadm       --f2-dir DIR  --target T  --left a,b  --right r0,r1,…
                   [--rank -1 --fudge 1e-4 --als-iters 20 --rank-alpha 0.05
                    --allow-neg/--no-allow-neg]
                   [--out FILE --format csv|tsv|json]
                   [--device auto|0,1 --precision …]

steppe qpwave      --f2-dir DIR  --left a,b,c  --right r0,…       # NO target; left[0]=ref
                   [--rank-alpha --fudge … --out --format …]

steppe qpadm-rotate --f2-dir DIR  --target T  --pool a,b,c,…  --right r0,…
                   [--min-sources 1 --max-sources N]             # app-side subset enumeration
                   [--jackknife 0|1|2]                           # JackknifePolicy
                   [--p-se-threshold 0.05 --se-require-p]         # only when jackknife=1
                   [--rank-alpha --fudge --als-iters --out --format …]
                   # SINGLE-GPU default; warns on ≥2 devices (multi-GPU host-bounce cap)
```

| Command | Real entry | Notes |
|---|---|---|
| `extract-f2` | `read_ind`/`read_snp`/`GenoReader` → `decode_af` → `assign_blocks` → `compute_f2_blocks_multigpu_device` (or `_tiered` for huge P) → write `<dir>` | the only `io`→compute wiring; §2.6 chain |
| `qpadm` | `upload_f2_blocks_to_device` → `run_qpadm` (GPU) | names resolved against `<dir>/pops.txt` |
| `qpwave` | `run_qpwave(left, right, …)` | `--left` is the full left set, `left[0]` is the reference; no target |
| `qpadm-rotate` | app builds `vector<QpAdmModel>` (pool subset enumeration) → `run_qpadm_search` | one row per model, **input order** via `model_index` |

Flag names mirror `QpAdmOptions` defaults exactly so the goldens reproduce with **zero
overrides**. `--jackknife` on `qpadm`/`qpwave` is accepted but documented as **ignored**
(only `run_qpadm_search` consults it; §2.1) so AT2 users are not surprised.

### 4.2 Inputs: f2_blocks dir OR raw genotypes

All three fit commands resolve input two ways: `--f2-dir DIR` reads the cache (§4.3) and
uploads via `upload_f2_blocks_to_device` for the GPU entry (or stages `F2BlockTensor` for
GPU device); OR raw `--geno/--snp/--ind` runs `extract-f2` in-process first
(extract-then-fit). An unknown pop name is `Status::InvalidConfig` fail-fast with the
offending label.

### 4.3 The f2_blocks dir (the AT2-shaped interchange + the mandatory name sidecar)

Adopt a **directory**, not the bare blob, because the engine carries no names (§1):

```
<dir>/
  f2.bin          # the STPF2BK1 payload (f2_disk_format.hpp): header(64B) | f2 | vpair | block_sizes
  pops.txt        # the P pop labels, one per line, in P-axis index order (the name↔index map)
  meta.json       # provenance: steppe version+SHA, precision_tag (engaged, not just requested),
                  #   blgsize cM, n_block, filter flags, source dataset sha256s, f2_cache_id (sha256 of f2.bin)
```

`pops.txt` is the missing piece that makes `qpadm --left England_BellBeaker,…` resolvable
(it is `IndPartition.groups[].label` in order, `ind_reader.hpp:65-72`). `f2.bin` mmaps/
preads straight into an `F2BlockTensor` (byte-identical layout, `f2_disk_format.hpp:7-9`).
`meta.json` records the architecture §12 reproducibility block (R/AT2 fields N/A for a
steppe-native run; the steppe equivalents are recorded), so `extract-f2 → qpadm` is a
content-addressed, reproducible pipeline.

### 4.4 Output: tidy CSV (default) + JSON

`--format {csv|tsv|json}` (default `csv`), `--out FILE` (stdout if omitted). Columns mirror
the committed golden CSVs so AT2 downstream scripts work unchanged. `QpAdmResult` →

- **weights** (one row per left source): `target, left, weight, se, z` (`qpadm.hpp:125-130`).
  `se`/`z` emitted as literal `NA` when `se.empty()` (the sentinel) — never fake 0/NaN.
- **summary** (row/sidecar): `p, chisq, dof, f4rank, est_rank, feasible, status,
  precision_tag` (`qpadm.hpp:131-162`).
- **rankdrop** (AT2 `res$rankdrop` order): `f4rank, dof, chisq, p, dofdiff, chisqdiff,
  p_nested` ← `rankdrop_*` (`qpadm.hpp:146-147`).
- **popdrop** (AT2 `res$popdrop`): `pat, wt, dof, chisq, p, f4rank, feasible` ←
  `popdrop_*` (`qpadm.hpp:149-152`; `popdrop_feasible` `char`→bool).
- **qpwave**: per-rank `rank, chisq, dof, p` + `rankdrop_*` + `f4rank/est_rank/status`.
- **qpadm-rotate**: one row per model in input order
  (`model_index, target, left, right_n, p, chisq, dof, f4rank, feasible, status` + optional
  per-weight columns; SE columns present only for models that got SE under the policy).

JSON mirrors the golden JSON schema so a run is diff-able against a golden file.

**Status handling.** `status` is a per-row string. On `ChisqUndefined`, `p` is the NaN
sentinel — emit `NA` and **filter on `status`, not `p`** (`qpadm.hpp:155-157`). Domain
outcomes are rows, never CLI errors. Exit 0 on a completed run even with per-model domain
outcomes; non-zero only for faults (`InvalidConfig`, IO/format, CUDA runtime).

### 4.5 Config precedence + single-GPU default

Precedence per architecture §9: **compiled defaults < TOML < env `STEPPE_*` < CLI**. For
step 2 the CLI parses into a `CliArgs` struct and a minimal `ConfigBuilder`
(`with_defaults().merge_file(--config).merge_env().merge_cli(args).build()`) produces a
validated, immutable config that maps onto `DeviceConfig`/`QpAdmOptions`/`PopSelection`/
`FilterConfig`. `build()` is where over-VRAM / unhonorable-precision / bad-arch fail fast
(`Status::InvalidConfig`); the VRAM probe routes through the CUDA-free
`build_resources`/`BackendCapabilities` seam, never a direct CUDA call.

**`qpadm-rotate` defaults to single-GPU** and warns when ≥2 devices are passed, citing
`TODO(multigpu-host-bounce)`: the rotation is host-bounce-capped at ~1.21× on no-P2P
consumer cards (`device_f2_blocks.hpp:92-97`; ~8.72 GB / ~3.8 s replication at P=600).
Do not advertise a multi-GPU rotation speedup.

---

## 5. The Python API (nanobind, thin)

Compiled module `steppe._core` (ADR-0002 / architecture §15), pure-Python `steppe`
package over it. `bindings/module.cpp` does ONLY: numpy↔span marshalling, name→index
resolution against the handle registry, `QpAdmOptions`/`DeviceConfig` fill from kwargs,
and `QpAdmResult`→Python objects. **No compute logic.** The pandas/dataclass shaping lives
in pure-Python `bindings/steppe/__init__.py` (so the compiled module has no pandas link).

### 5.1 Functions (mirror the C++ entries 1:1)

```python
import steppe   # GPU product: import is light, but the ops require a CUDA device

f2 = steppe.extract_f2(prefix="v66.1_HO", pops=None, out="my_f2/",
                       blgsize=0.05, maf=0.0, maxmiss=0.0, auto_only=True, drop_mono=True,
                       precision="emu40", devices=None)   # blgsize in MORGANS (AT2 convention);
                                                          # -> F2Blocks handle (resident) + pops + meta
f2 = steppe.read_f2("my_f2/")                              # reload the AT2-style dir

res  = steppe.qpadm(f2, target="Mbuti", left=["French","Han"], right=[...],
                    rank=-1, fudge=1e-4, als_iterations=20, rank_alpha=0.05)   # -> run_qpadm
wave = steppe.qpwave(f2, left=["French","Han","Sardinian"], right=[...])       # -> run_qpwave (left[0]=ref)
tab  = steppe.qpadm_rotate(f2, target="Mbuti", pool=[...], right=[...],
                           jackknife="feasible_only")                          # -> run_qpadm_search
```

- kwargs map 1:1 onto `QpAdmOptions` (`jackknife` accepts `"all"|"feasible_only"|"none"`
  → `JackknifePolicy`); defaults match the struct so a bare call reproduces AT2.
- Names→indices resolved Python-side against `f2.pops`; unknown name ⇒ clean
  `KeyError`/`ValueError` listing the available pops (binding-layer concern, not compute).
- The binding builds `device::Resources` once from the resolved `DeviceConfig` and passes
  `&resources` into each call; one resident `f2` is fit many times (ADR-0005).

### 5.2 Results — pandas frames + a typed object (admixr parity)

`QpAdmResult` is tabular; mirror AT2's `$weights`/`$rankdrop`/`$popdrop`:
- `res.weights` → DataFrame `[left, weight, se, z]` (NA where `se.empty()`).
- `res.rankdrop` → DataFrame `[f4rank, dof, chisq, p, dofdiff, chisqdiff, p_nested]`.
- `res.popdrop` → DataFrame `[pat, wt, dof, f4rank, chisq, p, feasible]` (`feasible` bool).
- scalars: `res.p, res.chisq, res.dof, res.f4rank, res.feasible, res.status` (a Python
  enum mirroring `Status`), `res.precision` (the `precision_tag`).
- **Domain outcomes are values, not exceptions** (`error.hpp:11-15`): a rotation row with
  `status=ChisqUndefined`, `p=NaN` is data, never a traceback. The layer raises only on
  faults (`InvalidConfig`, fatal `DeviceOom`). Critical for `qpadm_rotate` over thousands.
- `qpadm_rotate` returns ONE tidy DataFrame, one row per model in input order
  (`results[k].model_index`): `[model_index, target, left, right, p, f4rank, feasible,
  status, weights…]` — the admixr ranked-models screen.

### 5.3 The f2 interchange — opaque handle, opt-in numpy

`extract_f2`/`read_f2` return an **opaque `F2Blocks` facade** = `{DeviceF2Blocks (or
F2BlockTensor) + pop_labels[] + block_sizes[] + meta}`, **default device-resident** — so
the zero-D2H precompute-once/fit-many win is preserved (`to_host()` is "THE ONLY D2H",
`device_f2_blocks.hpp:64`, multi-GB at production P). Explicit opt-in NumPy:
- `f2.to_numpy()` → wraps `to_host()` → `nb::ndarray<double, nb::f_contig>` shape
  `(P, P, n_block)` (Fortran-contiguous == the `i + P·j + P·P·b` column-major layout). A
  **copy** is the safe default (the D2H temporary's lifetime is fragile for a zero-copy
  view).
- `f2.vpair_to_numpy()` likewise.
- *Deferred:* an on-device DLPack / `nb::device::cuda` export (cuPy/torch/JAX zero-copy) is
  the nanobind payoff but not needed for step-2 parity.

### 5.4 GPU-only (CPU is dev/test-only — RESOLVED)

**steppe is a GPU product. There is NO user-facing CPU runtime.** (User decision
2026-06-21, memory `cpu-is-test-only`: *"the whole point of this isn't for CPU. we don't
really have anything to do with the CPU except for testing purposes during development."*)
The `CpuBackend` is the dev/test parity ORACLE only (golden-diffed under `STEPPE_THOROUGH`),
NEVER a deliverable runtime. So the bindings target the GPU; `steppe.qpadm/qpwave/
qpadm_rotate/extract_f2` use the device path. NO CPU fallback at the API, NO `devices=None`
CPU-oracle path, NO `has_gpu()` graceful-degrade as a feature. A machine without a
CUDA-capable GPU is simply not a target; a GPU op invoked with no device surfaces a clear
"no CUDA device" error. (The host-oracle `run_qpadm(const F2BlockTensor&, …)` overloads stay
in the C++ test harness as the parity oracle — never exposed as a Python runtime mode.)

---

## 6. Packaging + layering (CMake wiring)

### 6.1 Targets

```
src/app/CMakeLists.txt    -> steppe_app    (CXX exe, OUTPUT_NAME steppe, CLI11)   [if STEPPE_BUILD_CLI]
bindings/CMakeLists.txt   -> steppe._core  (nanobind module)                       [if STEPPE_BUILD_PYTHON]
```

Top-level `CMakeLists.txt` (after line 63, the existing `add_subdirectory` block), behind
the **already-present** options (`SteppeOptions.cmake:22-23`):

```cmake
if(STEPPE_BUILD_CLI)     add_subdirectory(src/app) endif()
if(STEPPE_BUILD_PYTHON)  add_subdirectory(bindings) endif()
```

`steppe_app`:
```cmake
CPMAddPackage("gh:CLIUtils/CLI11@2.4.2")     # architecture.md §7 dependency table
add_executable(steppe_app main.cpp config_builder.cpp
    cmd_extract_f2.cpp cmd_qpadm.cpp cmd_qpwave.cpp cmd_rotate.cpp
    f2_dir_io.cpp pop_resolver.cpp run_metadata.cpp)
set_target_properties(steppe_app PROPERTIES OUTPUT_NAME steppe)
target_link_libraries(steppe_app
    PRIVATE steppe::core steppe::io steppe::api steppe::device CLI11::CLI11 steppe::warnings)
```
- **No `.cu` sources, no `LANGUAGES CUDA`.** It links `steppe::device` only to resolve the
  CUDA-free factory/`build_resources`/handle symbols (the same path `steppe_core` uses).
  `steppe_core` sets `CUDA_RESOLVE_DEVICE_SYMBOLS OFF`, so a consumer linking BOTH
  `steppe_core` and `steppe_device` lets `steppe_device` device-link the RDC kernels — the
  exact path the CUDA tests use. `steppe_app` device-links nothing itself.

`steppe._core`:
```cmake
find_package(nanobind CONFIG REQUIRED)        # ADR-0002; nanobind >= 2.x
nanobind_add_module(_core NB_STATIC module.cpp)
target_link_libraries(_core
    PRIVATE steppe::core steppe::io steppe::api steppe::device steppe::warnings)
install(TARGETS _core LIBRARY DESTINATION steppe)     # -> steppe/_core*.so
```

### 6.2 The PRIVATE-CUDA rule is verifiable for free

Both targets host-compile, so any leaked `<cuda_runtime.h>`/`<cublas_v2.h>` include is a
hard build failure. Add the two targets to the CI architecture-grep test
(architecture.md §4) for the cross-layer-include assertion too.

### 6.3 Wheel (scikit-build-core)

Root `pyproject.toml` per architecture §15: `scikit-build-core >= 0.10` +
`nanobind >= 2` backend, `STEPPE_BUILD_PYTHON=ON STEPPE_BUILD_CLI=OFF`, `cmake >= 3.30`,
`wheel.packages = ["bindings/steppe"]`, `build.targets = ["steppe._core"]`. Wheels use the
CUDA-13 **redistributable** model (architecture §14): do NOT bundle the CUDA runtime;
declare unsuffixed `nvidia-cuda-runtime/cublas/cusolver/curand >=13,<14` deps;
`auditwheel repair --exclude libcublas.so.13 …`. **ONE GPU wheel** (the `nvidia-*` deps) —
steppe is a GPU product, so there is NO CPU-only wheel variant (memory `cpu-is-test-only`;
the earlier "two variants" idea is dropped). The binding installs the spdlog→Python
`logging` sink.

---

## 7. Testing (golden-validated on real AADR; NO synthetic, NO smoke)

Follow `tests/reference/` exactly: a self-checking executable / pytest module reads the
**same committed real-AADR AT2 goldens**, runs through the **CLI/binding entry**, and exits
non-zero on any tolerance miss. Tolerances are the established two-tier (tight `rtol` on
weights/est/chisq, loose on se/z/p/rank — architecture §12/§13). **The existing fit
goldens already ARE an f2-dir input in disguise** — the `.bin` fixtures + the JSON pop
names are exactly what a `<dir>` contains — so the CLI/py tests reuse them with no new data.

| Test | Entry | Golden | Asserts |
|---|---|---|---|
| `tests/cli/test_cli_qpadm` | `steppe qpadm --f2-dir <fixture-as-dir> --target … --left … --right …` | `golden_fit0` + `golden_fit1_NRBIG` | parsed CSV/JSON weights/chisq/dof/p/rankdrop/popdrop == golden; name→index resolution correct; exit 0 |
| `tests/cli/test_cli_qpwave` | `steppe qpwave` | `golden_qpwave` | est_rank/f4rank/rankdrop/chisq == golden |
| `tests/cli/test_cli_rotate` | `steppe qpadm-rotate` | `golden_rot` (84 models) | per-model rows in input order match (weights rtol 1e-6, f4rank exact, p/feasible loose) |
| `tests/cli/test_cli_extract_qpadm` | `steppe extract-f2 && steppe qpadm` end-to-end | `golden_fitNA` | the one real dropped block (`maxmiss=0.99`) reproduced; proves the full genotype→f2→fit path |
| `tests/python/test_py_qpadm.py` | `steppe.qpadm(...)` → DataFrame | `golden_fit0`/`_qpwave`/`_rot` | DataFrame values == golden; GPU-vs-CPU equality where a device is present; `import steppe` succeeds GPU-free |

Wiring mirrors the fit tests: `add_test` gated by `STEPPE_BUILD_CLI`; pytest under
`STEPPE_BUILD_PYTHON`. CLI tests run on the CudaBackend (GPU) against the real-AADR goldens, with a clean SKIP
when no device is visible — identical to every
existing reference test. The `extract-f2` end-to-end test needs the real AADR genotype
triple (`STEPPE_AADR_ROOT` cache var already exists) and skips cleanly when absent, like
`f2_blocks_equivalence`. The two NEW failure surfaces the access layer introduces —
**name↔index resolution** and **CSV/JSON serialization** — are both golden-pinned by
parsing the CLI's own output back and diffing the committed golden values.

---

## 8. Milestone build-order (smallest-first, each golden-gated)

Ordered by ascending dependency on yet-unbuilt plumbing. M(cli-1) consumes an f2 dir that,
until M(cli-4), can only be the committed test fixtures — which is exactly what the goldens
provide, so the fit-CLI is testable before the genotype→f2 path is wired (matching ADR-0005
precompute-once/fit-many).

| Milestone | Deliverable | New plumbing it must build | Golden gate |
|---|---|---|---|
| **M(cli-0)** scaffold + config | `steppe` skeleton: CLI11 parse, `--version`/`--help`, subcommand stubs, the minimal `ConfigBuilder`/`RunConfig`/`CliArgs` (compiled<TOML<env<CLI), `build()` validation, `--device 0,1`, `--precision`, `Status`→exit-code map. **No compute.** Wire `src/app` behind `STEPPE_BUILD_CLI`. | `ConfigBuilder`/`RunConfig`/`CliArgs` (§9.1) | unit: precedence order + `build()` rejects (reuse the `test_config` pattern); arch-grep still passes |
| **M(cli-1)** `qpadm` on a dir | `steppe qpadm` over an existing f2_blocks dir: read dir, resolve names→indices via `pops.txt`, `upload_f2_blocks_to_device`, `run_qpadm` (GPU), tidy CSV/JSON. | `f2_dir_io` reader + `pop_resolver` + CSV/JSON emitter + `run_metadata` | `golden_fit0` + `golden_fit1_NRBIG` reproduced THROUGH the CLI |
| **M(cli-2)** `qpwave` | `steppe qpwave` (no target, `left[0]`=ref) + `--rank` surface, same dir input. | rankdrop CSV emitter | `golden_qpwave` through the CLI |
| **M(cli-3)** `qpadm-rotate` | `steppe qpadm-rotate`: pool→`vector<QpAdmModel>` enumeration, `--jackknife 0\|1\|2`, single-GPU default + ≥2-device warning. | models-pool enumerator + search CSV emitter | `golden_rot` (84 models) through the CLI, input order |
| **M(cli-4)** `extract-f2` + cache | `steppe extract-f2`: `io`→`decode_af`→`assign_blocks`→`compute_f2_blocks_multigpu_device`/`_tiered`→write `<dir>`; filters; `--dry-run`. **Builds the standalone STPF2BK1 write/read round-trip (the missing M7 piece) + the pops/meta sidecars.** Produces the dir the earlier milestones consumed. | EIGENSTRAT→f2 wiring + the f2-dir WRITER + filter wiring | `golden_fitNA` end-to-end (`extract-f2` then `qpadm`); cache round-trip byte-identical to `to_host()` |
| **M(py-1)** bindings (fit) | nanobind `steppe._core` + `steppe/__init__.py`: `qpadm`/`qpwave`/`qpadm_rotate` from a dir/handle → pandas frames; status enum + NA sentinels; numpy f2 view; log sink; `pyproject.toml`. | `bindings/module.cpp`, pure-Python sugar, wheel CI | pytest reproduces `golden_fit0`/`_qpwave`/`_rot`; GPU-vs-CPU where a device is present; `import steppe` GPU-free |
| **M(py-2)** bindings (extract) | `steppe.extract_f2(...)` from Python; (§5.4: GPU only). | extract binding | `golden_fitNA` end-to-end from Python |
| **M(abi-1)** *(optional, defer)* C ABI | The installed cross-toolchain `steppe_c.h` opaque-handle shim (`steppe_*_t`, `steppe_status_t`) for `find_package(steppe)` from Rust/Julia. **Not needed for step 2** (CLI + nanobind both link in-process). | the C ABI surface (ADR-0008, architecture §16) | go/no-go decision; not gated on a golden |

Largest single piece is **M(cli-4)** (it pulls in the `io` leaf, filters, VRAM tiering,
and the disk write/read round-trip) — heavier than wrapping `run_qpadm` (a near-trivial
wrap). Python tracks the CLI 1:1 but only after the CSV/JSON + name-resolution contracts
are frozen by M(cli-1..3).

---

## 9. Open questions for approval

1. **`ConfigBuilder`/`RunConfig`/`CliArgs` — build now or map flags directly?** They do
   not exist (architecture §9 is design intent). **Recommend:** build a minimal
   `ConfigBuilder` (TOML/env/CLI precedence + validating `build()`) in M(cli-0), CUDA-free
   in `core` so the bindings reuse it, with the TOML/env/CLI adapters in `app`. Where the
   validating `build()` lives (core vs app) is the one structural call — recommend
   `RunConfig`/`ConfigBuilder` in `core` (host-testable like `test_config`), adapters in
   `app`; the VRAM-budget check inside `build()` routes through the existing CUDA-free
   `build_resources`/`BackendCapabilities` seam, never a direct CUDA call. **Approve?**

2. **Output format default + container.** Recommend **tidy CSV default** (admixr-shaped,
   one table per result file: `weights.csv` / `rankdrop.csv` / `popdrop.csv`; rotation →
   one `models.csv`), with `--format json|tsv` and `--out`. JSON mirrors the golden schema
   for diffability. **Confirm CSV-default, and whether Parquet is wanted now or later.**

3. **Pop-name vs index resolution + the cache container.** The engine is index-only;
   **recommend the AT2-shaped `<dir>` with a `pops.txt` sidecar** as the name↔index map
   (the single most important schema decision; the alternative is forcing every consumer
   to re-hardcode the map as the test does). Note: the *numeric layout* is
   AT2-compatible (a GOAL per `f2_disk_format.hpp:10`), but the *container* (one blob +
   sidecar) is not drop-in with AT2's per-pair files. **Full per-pair AT2-dir interop is a
   bigger compatibility item — defer to step 3, or do strict interchange now?**

4. **Bindings wrap the C++ layer or the C ABI?** §16 permits the in-process binding to
   link the C++ convenience layer directly. **Recommend binding the C++ headers in-process
   for step 2** (CLI + nanobind are same-toolchain in one wheel); the cross-toolchain C
   ABI (`steppe_*_t`) is a separate, deferrable milestone (M(abi-1)). Consequence to
   accept: the Python ABI is not the SemVer-frozen surface §16 promises for the C boundary
   (fine and intended, §16). **Approve the deferral?**

5. **GPU-only wheel — RESOLVED (user, 2026-06-21).** steppe is a GPU product; the CpuBackend
   is the dev/test parity oracle ONLY, never a user-facing runtime (memory
   `cpu-is-test-only`). **ONE GPU wheel** (requires CUDA); NO CPU-only variant, NO
   `--device cpu` supported mode, NO `has_gpu()` CPU graceful-degrade. (The CLI/bindings
   target the GPU; a no-GPU box is not a target.) The §4/§5/§6 text above is updated to
   match. *No further decision needed.*

6. **`qpadm-rotate` pool-enumeration semantics.** Scoped a generic subset-of-pool
   enumerator (`--min/max-sources`). AT2's `qpadm_rotate` has specific left/right
   rotation/swap semantics. **Match AT2's exact rotation semantics, or ship the generic
   subset enumerator first?**

7. **`extract-f2` first format target.** EIGENSTRAT + PACKEDANCESTRYMAP (+ PLINK) readers
   exist in `io`. **Which is the priority smoke path for M(cli-4)?**

8. **Multi-GPU rotation** stays single-GPU default with a ≥2-device warning
   (`device_f2_blocks.hpp:92-97`, ~1.21× host-bounce cap). **Confirm we do not advertise a
   multi-GPU rotation speedup.**

---

### Key files cited (absolute)
- Public API: `/home/suzunik/steppe/include/steppe/{qpadm,fstats,config,error}.hpp`
- Precompute + handles: `/home/suzunik/steppe/src/core/fstats/{f2_blocks_multigpu,f2_from_blocks}.hpp`;
  `/home/suzunik/steppe/src/device/{device_f2_blocks,resources,backend_factory,f2_blocks_out,f2_disk_format,backend}.hpp`
- IO leaf: `/home/suzunik/steppe/src/io/{ind_reader,snp_reader,geno_reader,genotype_tile}.hpp`
- CMake: `/home/suzunik/steppe/CMakeLists.txt` (lines 60-67), `/home/suzunik/steppe/cmake/SteppeOptions.cmake` (22-23), `/home/suzunik/steppe/{include,src/io,src/core,src/device}/CMakeLists.txt`
- Architecture/ADRs: `/home/suzunik/steppe/docs/architecture.md` §4 (layering 239/241), §9 (config 589-672), §10 (status/sink), §11.4 (multi-GPU 280), §12 (metadata/parity), §14 (CUDA-13 redist), §15 (bindings/packaging 865-900, ADR-0002), §16 (ADR-0008 C ABI), ADR-0005
- Golden/test pattern: `/home/suzunik/steppe/tests/reference/goldens/at2/` (`golden_fit0/fit1_NRBIG/qpwave/rot/fitNA.json`, `csv/`, `fixtures/*.bin`); `/home/suzunik/steppe/tests/reference/test_qpadm_parity.cu` (the name→index hardcode the CLI replaces)
