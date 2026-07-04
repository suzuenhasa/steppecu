# f4-sweep and f3-sweep

Score *every* quartet (`f4-sweep`) or *every* triple (`f3-sweep`) over a population set on the GPU, and keep only the ones that look significant.

## What it does

An ordinary `f4` or `f3` call scores a handful of combinations you name by hand. A **sweep** names nothing: it enumerates every group of 4 populations (`f4-sweep`) or every group of 3 (`f3-sweep`) over an f2 directory, computes each one's estimate, block-jackknife standard error, and z-score, and returns only the combinations that clear a significance cut. This is how you go looking for signal when you don't yet know which populations are interesting — for example, scanning a few hundred populations for the strongest admixture (`f3 < 0`) or tree-violation (large `|f4|`) signals.

The catch is scale. `C(500,4)` is about 2.6 billion quartets; the full result table would be terabytes. steppe never builds that table on the host. Enumeration, scoring, the significance filter, and compaction all happen on the GPU, and only the small surviving set is copied back. You still pay to *score* every combination (the filter limits output, not work), which is why there is a safety cap and a `--sure` flag to lift it. These are GPU-only commands — there is no CPU fallback.

You need an f2 directory first; build one with [`extract-f2`](./extract-f2.md).

## Flags

Both commands take the same flags.

| flag | what it does | default |
|---|---|---|
| `--f2-dir` | The `f2_blocks` directory to sweep (built by [`extract-f2`](./extract-f2.md)). | — (required) |
| `--pops` | A population *subset* to sweep all combinations of, given as names. Empty means sweep the whole f2 directory. Use this to narrow an otherwise over-cap sweep to a set of interest. | empty (whole dir) |
| `--min-z` | The on-device filter: keep every combination whose `\|z\|` is at least this. Mutually exclusive with `--top-k`. The default of 3.0 matches the parity significance cut. | `3.0` |
| `--top-k` | Instead of a threshold, keep the K combinations with the largest `\|z\|` (a bounded device-side reservoir, ~K resident). Returned sorted by `\|z\|` descending. Mutually exclusive with `--min-z`. | — |
| `--sure` | Lift the maxcomb safety cap. Without it, a sweep whose combination count exceeds the cap refuses up front and does no work. Set this to force a large sweep to run. | off |
| `--out` | Write the survivor table to this FILE. Omit for stdout. | stdout |
| `--format` | Output format: `csv`, `tsv`, or `json`. | `csv` |
| `--device` | CUDA device: `auto`, a single ordinal, or two ordinals. GPU-only; `cpu` is not accepted. Multi-GPU is parked — use one device. | `auto` |
| `--precision` | Matmul precision: `emu40`, `emu32`, `fp64`, or `tf32`. The default emulated-double math is the matrix-heavy path; the cancellation-sensitive combine step uses a native-double carve-out regardless. | `emu40` |
| `--config` | **Reserved — not yet supported** (passing one currently errors). | — |

**`--min-z` XOR `--top-k`:** pick one. `--min-z` is a threshold (how significant), `--top-k` is a ranking (how many). `--min-z` is the mode that stays safe at any scale because the filter runs on the device and the full table never exists; `--top-k` keeps everything through compaction and ranks the compacted set down to K on the host.

Each survivor row carries the population indices, the f4/f3 point estimate, the block-jackknife SE, the z-score, and the two-sided p-value. The sweep also reports how many combinations it *would* have enumerated, so if it refuses on the cap you can see how far over you were.

## Examples

Sweep every quartet over a 500-population f2 dir, keeping the million strongest, written to a CSV. `C(500,4)` is ~2.6B quartets, so the cap must be lifted with `--sure`:

```
steppe f4-sweep --f2-dir /workspace/data/f2_500 --top-k 1000000 --sure --out /tmp/f4sweep.csv --device 0
```

Sweep every triple over the same dir, keeping only very strong signals (`|z| >= 8`). `C(500,3)` is ~20.7M triples — under the cap, so no `--sure` needed:

```
steppe f3-sweep --f2-dir /workspace/data/f2_500 --min-z 8 --out /tmp/f3sweep.csv --device 0
```

Narrow a sweep to a subset of interest with `--pops` (enumerates combinations only over those names):

```
steppe f4-sweep --f2-dir /workspace/data/f2_500 --pops Mbuti,Han,Sardinian,Yamnaya,Anatolia_N,WHG --min-z 3 --out /tmp/subset.csv --device 0
```

## Gotchas

- **`--min-z` and `--top-k` are mutually exclusive.** Pass one or the other, not both. With neither, the default is `--min-z 3.0`.
- **The cap guards compute time, not memory.** The filter limits how much *output* survives, but every combination is still scored on the GPU. A billion-combination sweep is real GPU time (minutes) even if only a handful of rows survive. The maxcomb cap exists to stop you kicking off hours of work by accident — a sweep over the cap refuses with no work done until you add `--sure`.
- **`--sure` doesn't make it fast, it makes it run.** It only lifts the cap. Check the reported enumeration count first so you know what you're committing to.
- **GPU-only.** `--device cpu` is not a thing here; there is no CPU sweep path. Multi-GPU is parked, so run on a single device.
- **`--top-k` is a global top-K by `|z|`,** not per-population — a few dominant populations can fill the whole list. If you want coverage across a set, a `--min-z` threshold is often more useful.
- **The same sweep is also reachable from `f4 --all-quartets` / `f3 --all-triples`.** Those forms take an extra `--shard-dir` flag that writes the survivor table as sharded CSV under a directory (created if absent) instead of a single `--out` file — handy when `--top-k` is in the millions. The dedicated `f4-sweep` / `f3-sweep` subcommands write a single file via `--out`.

## See also

- [`extract-f2`](./extract-f2.md) — build the f2 directory a sweep reads.
- [`f4`](./f-statistics.md) and [`f3`](./f-statistics.md) — score named combinations, and (via `--all-quartets` / `--all-triples` plus `--shard-dir`) the same sweep engine with sharded output.
- `docs/reference/include_steppe_fstat_sweep.hpp.md` — the sweep pipeline internals: unrank, score, filter, compact, copy-back, and the maxcomb cap.
