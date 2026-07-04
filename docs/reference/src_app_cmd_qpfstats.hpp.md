# `cmd_qpfstats.hpp` reference

## 1. Purpose

`src/app/cmd_qpfstats.hpp` declares the entry point for the `steppe qpfstats`
command-line subcommand. `qpfstats` is the one statistic in steppe that runs
in the opposite direction from the others.

Most of the fit and statistic commands — `f4`, `f3`, `qpdstat` with a
`--f2-dir` — *read* a precomputed f2 cache and *report* a number. `qpfstats`
does neither of those. It **reads raw genotypes** (the `.geno` / `.snp` /
`.ind` file triple) and **writes a smoothed f2 directory**. That output
directory is a drop-in f2 cache: `qpAdm`, `f4`, and `qpGraph` consume it exactly
the way they consume the cache produced by the `extract-f2` command.

The "smoothed" part is what makes this command distinct from plain
`extract-f2`. Rather than just estimating each pairwise f2 value on its own,
`qpfstats` fits all of the related statistics jointly and stabilizes them
against each other before writing them out. The numbers it produces are meant to
match what the `qpfstats()` function produces for the same population set[^at2].

The header itself is tiny — one function declaration plus its contract. The
substance is what that command does and the rules around it, described below.

---

## 2. What the command does

Running `steppe qpfstats` walks a fixed pipeline that mirrors how the
`extract-f2` command is built (decode the genotypes, compute, then write an f2
directory), with a joint-smoothing stage in the middle:

1. **Decode the genotypes.** The command reads the genotype file triple named
   by `--prefix` and decodes it on the GPU, the same way `extract-f2` does.
2. **Compute the full set of related statistics.** It drives the same
   genotype-based f4-numerator engine used by `qpdstat` over the *entire*
   combined set of two-, three-, and four-population comparisons (the f2, f3,
   and f4 combinations) for the requested populations, not just the pairwise f2
   values.
3. **Run the joint smoothing regression on the GPU.** All of those statistics
   are fed into a single shared-factor regression that runs on the device. This
   is the step that ties the individual estimates together and smooths them,
   rather than leaving each pairwise f2 to stand alone.
4. **Scatter the smoothed results into an f2 block tensor.** The smoothed
   coefficients are written into a `P × P × n_block` tensor — `P` populations
   by `P` populations by the number of jackknife blocks — which is exactly the
   shape and meaning of the reference `qpfstats()` output[^at2].
5. **Write the f2 directory.** That tensor is written to `--out-dir` through the
   same directory-writer the `extract-f2` command uses, so the result is a
   normal, reusable f2 cache.

The value of doing this is the usual "compute once, fit many times" pattern: a
single `qpfstats` run produces a smoothed cache that any number of later
`qpAdm` / `f4` / `qpGraph` runs can read without re-touching the genotypes.

---

## 3. Inputs and options

The command takes the following inputs:

| Flag | Meaning |
|---|---|
| `--prefix PATH` | The genotype file-triple prefix — the shared path stem for the `.geno`, `.snp`, and `.ind` files to read. This is the raw data the smoothed cache is built from. |
| `--pops A,B,C,...` | The set of populations to smooth over. Internally the list is **sorted into ascending order**, which is what fixes the row/column ordering (the dimension names) of the output tensor to match the reference ordering[^at2]. |
| `--out-dir DIR` | Where to write the smoothed f2 directory. This becomes the reusable cache that later commands read. |
| `--blgsize` | The jackknife block size, in Morgans. Default `0.05` Morgans (equivalently 5 centimorgans), which matches the parity default block size[^at2]. |
| `--precision` | Which arithmetic mode the heavy matrix-multiply sub-steps run in. Default is emulated double precision at 40 mantissa bits — steppe's standard precision policy, fast and about as accurate as native double precision. |

The reason the population list is sorted rather than left in the order the user
typed it is reproducibility: two runs that name the same populations in a
different order must produce byte-identical caches with identically-ordered
rows and columns. Sorting up front guarantees that.

---

## 4. Layering: plain C++20, no CUDA in this file

This header — and the command implementation behind it — is deliberately
written in plain C++20 with no CUDA header included. It is "app-only" code: it
reaches the GPU **only** through CUDA-free seams (the resource builder and the
`run_qpfstats` compute entry point), and it writes the output directory through
the app's shared f2-directory writer. That is the same shape the `extract-f2`
command uses to compose its decode → compute → write chain.

Keeping CUDA out of the app layer is an enforced boundary in the codebase, and
this file stays on the correct side of it by never touching a GPU type
directly. Because of that shared shape, the command reuses the existing
decode, compute, and directory-writing machinery rather than reimplementing it;
the genuinely new logic is the joint-smoothing driver behind the CUDA-free
compute call.

---

## 5. The `run_qpfstats_command` contract

```cpp
[[nodiscard]] int run_qpfstats_command(const config::RunConfig& config);
```

This is the single function the header declares. It runs `qpfstats` over an
already-frozen `RunConfig` and returns the process exit code.

- **It owns its own console output.** The command prints to stdout and stderr
  itself; the underlying library never prints. This keeps all user-facing text
  in the command layer.
- **The return value is a process exit code**, drawn from steppe's standard
  exit-code set. `[[nodiscard]]` marks it so a caller can't accidentally ignore
  whether the command succeeded.

### Which outcomes exit zero, and which don't

- **Success exits 0.** Once a smoothed f2 directory has been written to
  `--out-dir`, the command exits with code 0.
- **Faults return a nonzero code.** A genuine fault produces a nonzero exit:
  a bad `--prefix`, a bad population set, or a bad `--out-dir`; no usable GPU
  device; or a file, format, or CUDA-runtime error along the way.

The practical rule for a caller: a nonzero exit means the smoothed cache was not
written and should not be relied on, while a zero exit means `--out-dir` now
holds a complete, reusable f2 cache.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
