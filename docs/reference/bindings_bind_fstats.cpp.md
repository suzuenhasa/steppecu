# `bind_fstats.cpp` reference

## 1. Purpose

`bindings/bind_fstats.cpp` registers the f-statistic entry points into the
`steppe._core` Python module. It exposes seven functions to Python:

- `run_f4` — standalone f4(p1,p2;p3,p4)
- `run_qpdstat` — the D-statistic / f4 over the f2-cache path
- `run_dstat` — the genotype-path normalized-D magnitude
- `run_extract_f2` — build an f2 tensor from a genotype prefix
- `run_qpfstats` — the genotype-path joint f2 smoother
- `run_f3` — standalone f3(C;A,B)
- `run_f4ratio` — standalone f4-ratio

This is a thin marshalling layer, not a place where any statistic is actually
computed. Each function does the same small job: check the arguments, turn
population *names* into integer indices, build the GPU resources, call into the
CUDA-free library seam that does the real work, and convert whatever comes back
into a Python dict or a handle object. All of the numeric compute lives behind
that seam in the core library; this file only translates between Python and that
library.

Two facts shape almost everything in the file: statistics come from one of **two
data paths** (a pre-computed f2 cache, or the raw genotype files on disk — see
§2), and results are returned in one of **two shapes** (a dict of parallel
arrays, or a dual "path string / handle" return — see §7). Errors are split
across two channels as well: hard failures raise a Python exception, while
per-row outcomes ride along inside the returned data (see §8).

---

## 2. The two data paths

Every function here reads its input from exactly one of two sources.

**The f2-cache path.** `run_f4`, `run_qpdstat`, `run_f3`, and `run_f4ratio` all
work off an already-computed f2 tensor. Python passes in an `F2Handle` (the
object returned by `read_f2`, `run_extract_f2`, or `run_qpfstats`), which carries
the host-side f2 tensor and its list of population names. These functions resolve
their name tuples against that handle's population list and never touch the
genotype files.

**The genotype-triple path.** `run_dstat`, `run_extract_f2`, and `run_qpfstats`
read the three genotype files `<prefix>.geno`, `<prefix>.snp`, and `<prefix>.ind`
directly and decode them on the GPU. They do **not** take an `F2Handle` — they
build everything they need from the files. Because they read the raw genotypes,
they can compute things the f2 cache alone cannot (the normalized-D magnitude, a
freshly extracted f2, or a smoothed f2).

Keeping the two paths straight explains an otherwise confusing pair: `run_qpdstat`
and `run_dstat` both concern the D-statistic, but the first reads the f2 cache and
the second reads the genotypes, and they report different quantities (see §3
and §4).

---

## 3. The f2-cache statistics: `run_f4`, `run_qpdstat`, `run_f3`, `run_f4ratio`

These four functions share one pattern and differ only in how many populations
each item names and which columns come back.

### The shared pattern

Every one of them does the same five steps:

1. **Require at least one tuple.** An empty list raises a Python `ValueError`
   with a message naming the statistic.
2. **Resolve names to indices.** It builds a resolver over the handle's
   population list and turns each name tuple (for example `(p1,p2,p3,p4)`) into a
   tuple of integer indices into that list. An unknown name raises.
3. **Use the default options.** Each passes a default options object; f4, f3,
   f4-ratio, and the f2-path D all use a zero fudge factor internally, so the
   default is exactly what's wanted.
4. **Upload the tensor for the duration of the call, then free it.** A helper
   uploads the handle's host f2 tensor onto the GPU into a device tensor that
   lives only inside this one call and is released as soon as it returns. Holding
   the device copy no longer than necessary keeps GPU memory use low when many
   separate calls are made against the same handle.
5. **Run batched, then marshal.** The whole list of tuples is computed in one
   batched library call, and the result is converted into a single dict of
   parallel arrays, one entry per input tuple, in the original input order.

The return dicts differ only by column set:

| Function | Item shape | Returned dict columns |
|---|---|---|
| `run_f4` | `(p1,p2,p3,p4)` | `pop1,pop2,pop3,pop4,est,se,z,p` |
| `run_f3` | `(C,A,B)` | `pop1,pop2,pop3,est,se,z,p` |
| `run_f4ratio` | `(p1,p2,p3,p4,p5)` | `pop1..pop5,alpha,se,z` |
| `run_qpdstat` | `(p1,p2,p3,p4)` | `pop1,pop2,pop3,pop4,est,se,z,p` |

`run_f3` and `run_f4ratio` are simply the three-column and five-column versions
of the same pattern; there is nothing new in them beyond the tuple width.

### `run_qpdstat` and why the f2-path D equals f4

`run_qpdstat` is a deliberately thin clone of `run_f4`, and it is worth knowing
why it is allowed to be. On the f2-cache path, the D-statistic **is** f4. In the
reference implementation, running qpdstat over an f2 directory produces output
that is byte-for-byte identical whether the "f4 mode" flag is on or off — that
flag only changes anything when per-SNP genotypes are present, and an f2 cache has
none. So `run_qpdstat` introduces no new compute, no new emitter, and no new
result type; it reuses `run_f4`'s computation and its dict emitter unchanged, and
returns the same eight columns. In that dict, `z` is `est/se` and `p` is the
two-sided normal tail probability of `|z|`, which is the reference sign / Z / p
convention for the D-statistic.

What `run_qpdstat` does **not** give you is the normalized-D *magnitude*. That
number needs the per-SNP genotypes and lives in a separate function on the other
data path, `run_dstat` (see §4).

---

## 4. `run_dstat` — the genotype-path normalized D

`run_dstat` reads the genotype triple directly rather than the f2 cache, so it can
report the normalized D magnitude that `run_qpdstat` cannot. It takes a file
`prefix`, a list of `(p1,p2,p3,p4)` name quadruples, a block size `blgsize`, and a
`device` index; it has no `F2Handle` parameter.

The statistic it computes is D as a ratio of two per-SNP means, block-jackknifed
for the standard error: the numerator per SNP is `(a-b)(c-d)` and the denominator
per SNP is `(a+b-2ab)(c+d-2cd)`, where `a,b,c,d` are the per-SNP allele
frequencies of the four populations, and D is `mean_snp(numerator) /
mean_snp(denominator)`. It returns the same dict shape as `run_qpdstat`
(`pop1..pop4,est,se,z,p`) using the same D convention.

### The population-ordering invariant

The most important detail in this function is that the name-to-index resolution
must agree exactly with how the decoder orders populations, or every result would
be silently mislabeled. It guarantees that agreement as follows:

1. It forms the **population union** — the distinct names appearing across all the
   quadruples, in first-seen order. Only these populations are read from the
   files; the rest of the prefix is ignored.
2. It reads those populations as an explicit selection and takes the resulting
   partition's labels as the **P axis**, which comes back **sorted ascending by
   label**.
3. It builds the name resolver over that same sorted axis, so the indices it hands
   the compute call are exactly the indices the internal decode uses.

Because both the resolver and the decode are built from the same explicit
selection sorted the same way, the labels on the returned rows always match the
data behind them.

### Pinned parity settings

Three settings are fixed to match the reference's genotype-path qpdstat: forced
diploid genotypes, "all SNPs" mode, and autosomes-only. `blgsize` is a block size
in **Morgans** (reference default `0.05`).

---

## 5. `run_extract_f2` — building an f2 tensor from genotypes

`run_extract_f2` builds an f2-blocks tensor from a genotype prefix. It reads
`<prefix>.{geno,snp,ind}` and runs the full chain — decode, filter, assign SNPs to
jackknife blocks, compute f2 in the memory tier that fits, copy back to the host —
which is the same chain the command-line `extract-f2` runs. It returns its result
in one of the two return modes described in §7.

### Parameters and their defaults

The defaults are chosen so that a bare `run_extract_f2(prefix, pops)` reproduces
the reference `extract_f2` output (the golden), so most of them are "off / match
the reference" rather than arbitrary.

| Parameter | Default | Meaning |
|---|---|---|
| `pops` | (required) | The explicit population subset to extract. The P axis is that selection sorted ascending by label. Must be non-empty. |
| `out` | `""` | If non-empty, serialize to that directory and return the path; if empty, return a new `F2Handle` (see §7). |
| `device` | `0` | Which GPU to use (single-GPU). |
| `blgsize` | `0.05` | Jackknife block size in **Morgans** (reference default). |
| `maf` | `0.0` | Minimum minor-allele frequency; `0.0` is no filter. |
| `maxmiss` | `0.0` | The reference's **population-axis** coverage threshold, where `0` means a global intersection. This is the coverage semantic used inside the extract, not the sample-axis missing-data predicate. |
| `autosomes_only` | `true` | Keep only chromosomes 1–22. On by default because the reference `extract_f2` default is autosomes-only. |
| `drop_monomorphic` | `false` | Drop SNPs with no variation. |
| `transversions_only` | `false` | Keep only transversion SNPs. |
| `ploidy` | `"auto"` | `"auto"` applies the reference's pseudo-haploid auto-detection (the default); `"1"`/`"pseudo"`/`"pseudo_haploid"` forces pseudo-haploid; `"2"`/`"diploid"` forces diploid. Any other value raises. |
| `strand_mode` | `"drop"` | The strand-ambiguous-SNP policy (see below). |
| `precision` | `None` | `None` selects the emulated-FP64 default (the same default the f2 matrix-multiply and the CLI use); `"fp64"`/`"native"` selects native double precision; `"emulated_fp64"` selects emulated FP64. |

### `strand_mode`

The `strand_mode` string chooses what to do with strand-ambiguous (palindromic
A/T and C/G) SNPs:

- `"drop"` (the default) drops them, which reproduces steppe's original,
  merge-safe behavior bit-for-bit.
- `"keep"` retains them, which reproduces the reference's default behavior.
- `"flip"` is accepted as a documented token but is **not yet implemented**: it
  currently behaves like `"keep"` and performs no frequency-based reorientation.

Any other value raises a `ValueError`.

---

## 6. `run_qpfstats` — the genotype-path joint f2 smoother

`run_qpfstats` reads the genotype triple and produces a *smoothed* f2 tensor. It
drives the same per-SNP D-numerator engine used by `run_dstat` over the full set
of f2/f3/f4 population combinations, runs an on-device shared-factor smoothing
regression, and returns a smoothed per-block f2 tensor. It uses the same dual
return modes as `run_extract_f2` (see §7), so its output can be fed straight back
into `read_f2`, `run_f4`, or `run_qpadm`.

Parameters: a `prefix`, the smoothing population set `pops`, an `out` directory, a
`device`, a `blgsize` (Morgans, reference default `0.05`), and a `precision`
(`None` selects emulated FP64; `"fp64"`/`"native"` selects native double
precision). The `pops` set must contain **at least four** populations — that is
the minimum f4 basis — and an empty or too-small set raises. Internally the pops
are sorted ascending, which matches the reference's dimension-name order.

After the compute returns, the function checks the result's status; if it is not
OK it raises a `ValueError` suggesting the caller confirm all populations are
present.

---

## 7. The dual return idiom (path string vs `F2Handle`)

`run_extract_f2` and `run_qpfstats` both produce a full f2 tensor, and both hand it
back the same way, chosen by whether the `out` argument is empty. This avoids a
needless round-trip through disk when the caller just wants to keep working in
memory.

**Mode A — `out` is given.** The tensor is serialized to an on-disk f2 directory
(the STPF2BK1 format) via the directory writer, together with a metadata record,
and the function returns the directory **path as a string**. The caller can then
`read_f2(path)` later. The metadata captures the precision tag and mantissa-bit
count, the block count, the population-axis size `P`, the filter settings, the
source file paths, and the block size — with `blgsize` converted from Morgans to
centimorgans, because the metadata records centimorgans. The extract writer does
**not** hash the (large) `.geno` file into the metadata. If the write fails, the
function raises.

**Mode B — `out` is empty.** The tensor and its labels are moved into a
newly-allocated `F2Handle` on the heap, and ownership of that object is handed to
Python (the return-value policy takes ownership, exactly as `read_f2` does). No
disk directory is written. The returned handle can be consumed by any of the
f2-cache functions in §3, by `read_f2`, or by `run_qpadm`.

---

## 8. Faults versus per-item status

Errors are reported through two separate channels, and the split is intentional.

**Faults raise.** Anything that means the whole call cannot proceed becomes a
Python exception: no CUDA device present, a missing input file, an unknown
population name, a filter that removes every SNP, an unwritable output directory,
or a CUDA runtime error. In these cases the library throws and the binding
re-raises it as a Python error (a `ValueError`, with a message prefixed by the
operation name). The genotype-path functions build their GPU resources inside a
try block and explicitly raise a "no device" error when no GPU is found.

**Per-item outcomes ride on the result.** A domain outcome that affects only one
row — for example a single quartet whose estimate could not be formed — does not
raise. It is reported inside the returned dict on that row's own `status` (and
`precision`) field, so the surrounding rows still come back normally. Callers
inspect those per-row fields rather than catching an exception.

---

## 9. The registered Python entries and their defaults

`register_fstats` is the function that actually attaches all seven entries to the
module. Each `m.def` names the Python-visible argument names and their defaults
and carries the docstring the user sees from `help()`. The argument defaults are:

| Python function | Arguments (with defaults) |
|---|---|
| `run_f4` | `f2`, `quartets` |
| `run_qpdstat` | `f2`, `quartets` |
| `run_dstat` | `prefix`, `quadruples`, `blgsize=0.05`, `device=0` |
| `run_extract_f2` | `prefix`, `pops`, `out=""`, `device=0`, `blgsize=0.05`, `maf=0.0`, `maxmiss=0.0`, `autosomes_only=true`, `drop_monomorphic=false`, `transversions_only=false`, `ploidy="auto"`, `strand_mode="drop"`, `precision=None` |
| `run_qpfstats` | `prefix`, `pops`, `out=""`, `device=0`, `blgsize=0.05`, `precision=None` |
| `run_f3` | `f2`, `triples` |
| `run_f4ratio` | `f2`, `tuples` |

The `f2` argument on the f2-cache functions is the `F2Handle`; the `prefix`
argument on the genotype-path functions is the shared stem of the three genotype
files. Every `blgsize` default of `0.05` is in Morgans and matches the reference
default. The docstrings restate, for each function, the returned dict columns and
(for the genotype-path functions) that a missing GPU raises.
