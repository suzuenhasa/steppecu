# Precision, memory tiers, and device selection

Three flags that appear on almost every steppe command and control *how* the math runs, *where* the big results live, and *which* GPU does the work: `--precision`, `--tier`, and `--device`.

## What it does — plain English

These are cross-cutting flags, not a command of their own. Nearly every subcommand (`extract-f2`, `qpadm`, `qpwave`, `qpgraph`, the f-statistic tools, `dates`, `qpfstats`) accepts `--precision` and `--device`. The `--tier` flag lives on `extract-f2`, because that is the step that produces the large f2_blocks result whose home the tier decides.

You reach for these when the defaults don't fit your situation:

- **`--precision`** picks the flavor of floating-point arithmetic used in the heavy matrix multiplications. The default (`emu40`) is a fast, emulated form of double precision that stays about as accurate as the real thing. You'd change it only to get the exact native-FP64 reference numbers, to trade a little accuracy for more speed, or to do a rough screening pass.
- **`--tier`** decides where the f2_blocks result is stored while it's computed: entirely in GPU memory (fastest), spilled to host RAM, or streamed to a file on disk. The default (`auto`) figures this out for you so a big model doesn't run out of memory. You'd pin it only for testing or to force a specific path.
- **`--device`** picks which CUDA GPU runs the job. steppe is GPU-only; there is no CPU mode. Multi-GPU is parked, so in practice you run on a single device.

All three are **parity-neutral where it counts**: `--tier` and `--device` only change where and how bytes move, never a reported number. `--precision` is the one that can change the numbers, and only `fp64` is the bit-exact reference.

## Flags

### `--precision` — matmul arithmetic mode

| value | what it does | default |
|---|---|---|
| `emu40` | Emulated double precision keeping 40 mantissa bits. Fast (measured 7–17x faster than native FP64 on real data) at essentially native accuracy (worst-case error around 2.2e-11). This is the default and what you want for almost everything. | ✅ default |
| `emu32` | Same emulated math but only 32 mantissa bits. Faster still, less accurate (worst-case error around 8.6e-9). Use when you want a bit more speed and can tolerate slightly looser numbers. | |
| `fp64` | Native double precision. The gold-standard reference every other mode is validated against. Slower, but bit-for-bit reproducible run to run. Use this when you need the exact reference numbers. | |
| `tf32` | TF32 tensor-core arithmetic — lower precision, for quickly screening or ranking a space of models. Results are approximate and should be recomputed in a higher-precision mode before you report any final estimate. | |

`--precision` takes exactly one of these four tokens (default `emu40`).

Worth knowing: the delicate, cancellation-prone parts of the math (the f2 numerator, the linear solve, and the singular-value decomposition) always run in native FP64 regardless of the mode you pick — the mode only governs the big, well-conditioned matrix multiplications. So `emu40` is safe as a default even for the fit tools.

### `--tier` — f2_blocks output tier (on `extract-f2`)

| value | what it does | default |
|---|---|---|
| `auto` | steppe measures free GPU memory and free host RAM at runtime and picks the fastest tier the result fits in. This is what you almost always want. | ✅ default |
| `resident` | Force the result to stay entirely in GPU memory. Fastest, but only works when it fits. | |
| `host` | Force the spill-to-host-RAM path. Blocks are streamed off the GPU into a host tensor. The SNP-tile input is re-decoded on the fly, so this frees the GPU memory wall that caps very high-population runs. | |
| `disk` | Force the spill-to-disk path. Blocks are streamed to a file with a tiny staging buffer, so it works even on a machine with little RAM. | |

`--tier` takes one of these four tokens (default `auto`). The `host` and `disk` tiers stream the SNP-tile input, which is what lets high-population runs that would OOM in `resident` complete.

What `auto` actually does, in order: it picks `resident` if the result plus its working set fits in about 70% of free GPU memory *and* the dense decode input fits in about 60% of free host RAM; otherwise `host` if the result alone fits in about 60% of free host RAM; otherwise `disk`. The choice moves no bits — it never changes a reported f2 value. (You can also pin the tier via the `STEPPE_FORCE_TIER` environment variable set to `resident`, `host`, or `disk`; the `--tier` flag / config field takes precedence over it.)

### `--device` — which GPU(s)

| value | what it does | default |
|---|---|---|
| `auto` | Auto-detect and use the visible GPU(s) in enumeration order. | |
| `<ordinal>` | Pin one GPU by its CUDA ordinal, e.g. `0` or `1`. This is the recommended form. | (effectively `0`) |
| `<ordinal>,<ordinal>` | A comma-separated list pins a specific set and ordering of GPUs. Multi-GPU is currently parked, so this is not the path to use for normal runs. | |

`--device` is GPU-only — there is no `cpu` option. steppe requires a CUDA GPU. Because multi-GPU is parked, use a single ordinal (`--device 0`) for all real runs and benchmarks.

## Examples

Build an f2_blocks directory on GPU 0 with the default emulated precision and automatic tiering (the normal case — you don't pass `--precision` or `--tier` at all):

```
steppe extract-f2 --prefix v66.p1_1240K.aadr.patch.PUB --auto-top-k 700 --maxmiss 0.5 --device 0 --out-dir f2_700
```

Check what tier and precision a big extract would use before committing to the compute, with `--dry-run`:

```
steppe extract-f2 --prefix v66.p1_1240K.aadr.patch.PUB --auto-top-k 700 --maxmiss 0.5 --device 0 --dry-run
```

Expect a printed report of sizes, the chosen tier, and the precision — no GPU compute runs.

Run a qpAdm fit in native double precision to get the exact reference numbers (any subcommand accepts `--precision`):

```
steppe qpadm --f2-dir f2_700 --target Czechia_EBA_CordedWare --left Russia_Samara_EBA_Yamnaya,Turkey_N --right Mbuti,Han,Papuan,Karitiana,Iran_GanjDareh_N,Israel_Natufian --precision fp64 --device 0
```

Expect the same estimates as the default `emu40` to within the emulated-precision tolerance, but bit-for-bit reproducible run to run.

## Gotchas

- **`--tier` is only on `extract-f2`.** It governs where the f2_blocks result lives, so it appears on the command that produces that result. The fit and f-statistic tools read an already-built f2_blocks directory and don't take `--tier`.
- **Leave `--tier auto` alone unless you have a reason.** The automatic policy reads free memory at runtime and won't OOM. Pinning `resident` on a model that doesn't fit will fail; the auto path would have streamed it instead.
- **`emu40` is the default, not native FP64.** If a downstream comparison expects bit-exact native-double numbers, pass `--precision fp64`. The emulated modes are accuracy-*approximate* and not bit-identical (and not standards-compliant on special values like infinities).
- **`tf32` is a screening mode, not a final answer.** Never report an estimate, standard error, z-score, or p-value straight from `tf32` — recompute it in `emu40` or `fp64` first.
- **There is no CPU device.** `--device cpu` is not supported. steppe needs a CUDA GPU.
- **Use a single GPU.** Multi-GPU is parked, so pass `--device 0` (or a single ordinal). Passing a two-ordinal list is not the path for normal runs.
- **`STEPPE_FORCE_TIER` can silently override `auto`.** If a run picks an unexpected tier, check whether that environment variable is set — a config/flag value beats it, but a bare `auto` default does not.

## See also

- [./extract-f2.md](./extract-f2.md) — the command that builds the f2_blocks directory these tiers apply to, and the one place `--tier` lives.
- [docs/reference/src_device_tier_select.hpp.md](../reference/src_device_tier_select.hpp.md) — the exact tier-selection policy (thresholds, host-RAM probe, override precedence).
- [docs/reference/include_steppe_config.hpp.md](../reference/include_steppe_config.hpp.md) — the precision policy, the mantissa-bit constants, and the memory-tier fractions.
