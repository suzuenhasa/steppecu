# `cmd_paint.hpp` reference

## 1. Purpose

`src/app/cmd_paint.hpp` declares the single entry point for the `steppe paint`
command — the top-level function the command-line tool calls to run the
Li-Stephens haplotype-copying engine (the ChromoPainter-style "who did each
recipient copy from" model). It is a header in the application layer, so it sits
above the compute library and below `main()`: `main()` parses the command line
into a configuration and then hands that configuration to the one function this
header exposes.

The header itself is tiny — one function declaration. All the substance is in
what that one command *does*, and that is worth spelling out, because `paint` is
where a whole new engine surfaces to the user. The header is deliberately plain
C++20 with no GPU (CUDA) code in it: like the other `steppe` command headers it
reaches the GPU only indirectly, through the `ComputeBackend` seam, so the app
layer still compiles and unit-tests on a machine with no CUDA toolkit installed
(see section 8).

---

## 2. The single entry point: `run_paint_command`

```cpp
int run_paint_command(const config::RunConfig& config);
```

This is the only thing the header declares.

- **Input** — a `RunConfig`, passed by const reference. `RunConfig` is the
  frozen, already-parsed configuration for one run: the resolved device, the
  input prefixes, the output destination and format, and every `paint`-specific
  knob (`--donors`, `--ne`, `--theta`, `--self-copy`, `--recip-batch`,
  `--labels`, `--face`, `--full`, and the bp-fallback / `--yes` flags). It is
  immutable by the time it reaches here; this function reads it and never
  changes it.
- **Return value** — a process exit code as a plain `int`, one of the named
  `CliExitCode` values (section 7). `main()` returns this straight to the
  operating system as the program's exit status.
- **Output ownership** — this function owns everything the command prints. The
  compute library underneath it never writes to `stdout` or `stderr`; the
  coancestry / local-ancestry table and every error message are produced here in
  the application layer, which is what lets the command guarantee one consistent
  format for both results and errors.

---

## 3. What `paint` computes: the Li-Stephens copying model

`paint` treats each **recipient** haplotype as a mosaic copied, piece by piece,
from a panel of `K` **donor** haplotypes. As you walk along the chromosome the
copied template occasionally *switches* to a different donor (recombination), and
at any given site the copied template may *miscopy* — carry a different allele
than the recipient (mutation). Running the forward-backward recurrence over that
model yields, at every SNP `l`, the copying posterior `gamma_l(k)`: the
probability that donor `k` is the template the recipient is copying at `l`. That
posterior is the raw material both output faces are built from.

Three model quantities drive the recurrence, all built host-side (see
`li_stephens.hpp`) before any device work:

- **rho** — the per-SNP recombination ("switch") probability from the cM genetic
  map, `Ne`, and `K`. `rho[0] = 1.0` and every chromosome boundary resets to
  `1.0`, so the copying restarts fresh from the prior at the start of each
  chromosome (you cannot recombine across chromosomes).
- **mu / theta** — the per-site miscopy (emission) rate. `--theta auto` uses the
  Watterson estimate over `K` donors; an explicit `--theta` overrides it.
- **pi** — the copying prior over donors, uniform `1/K` by default. When the
  donor panel *is* the recipient panel (all-vs-all self-painting) and
  `--self-copy` is off, recipient `r`'s own donor column is zeroed and the mass
  is shared over the remaining `K-1` donors: **leave-one-out**, so a haplotype
  can't trivially copy itself.

The engine itself lives behind two `ComputeBackend` virtuals
(`ls_paint_coancestry` and `ls_localanc`) — the exact scalar reference in
`CpuBackend`, the batched kernel in `src/device/cuda`. The command's job is to
canonicalize the inputs, build those three vectors, drive the backend in
recipient batches, and shape the output.

---

## 4. The two faces: coancestry (`paint`) versus `localanc`

`--face` selects which reduction of `gamma` the command emits. Both start from the
same forward-backward; they collapse a different axis.

**Coancestry (the default face).** Collapses the *SNP* axis. For each recipient
it accumulates two per-donor summaries: `chunkcounts` (how many copied chunks
came from each donor) and `chunklengths` (how much genetic length, in Morgans,
was copied from each donor). By default these `N×K` per-donor numbers are further
aggregated by ancestry label into an `N×P` table — one column per distinct donor
label — with columns `expected_chunks` and `expected_length_cM` (lengths are
Morgans scaled to centiMorgans on the way out). Passing `--full` skips the
aggregation and reports every individual donor column instead.

**Local ancestry (`--face localanc`).** Keeps the *SNP* axis and collapses the
*donor* axis instead: at each SNP it sums `gamma` over the donors that carry each
ancestry label, giving an `N×M×P` per-position ancestry posterior — for each
recipient, at each SNP, the probability mass on each ancestry label. The emitted
rows carry the SNP coordinates (`chrom`, `pos_bp`, `genpos_cM`) so the output
lines up with a FLARE/RFMix-style aligner directly, no external join file needed.

Either face can be written as CSV, TSV, or JSON; the format comes from the run
config.

---

## 5. The pipeline, end to end

Calling `run_paint_command` runs a fixed sequence:

1. **Require the two inputs.** `--prefix` (the phased **recipient** haplotypes)
   and `--donors` (the phased **donor** panel) are both mandatory; either one
   missing is an invalid-configuration exit.
2. **Read both genotype triples through a host oracle.** A host `CpuBackend` is
   used *only* as the io / transpose / ploidy-detect oracle to canonicalize each
   `.geno/.snp/.ind` triple into a tile — it does not compute the statistic. Each
   individual is kept (every haploid column is a haplotype).
3. **Validate the request.** `validate_paint_request` runs the host-pure guard:
   phased/haploid input (any diploid sample count is counted here and rejected),
   a monotonic cM map, the self-copy / leave-one-out policy, and the `O(N·K·M)`
   cost guard (which `--yes` acknowledges for a big run).
4. **Insist the panels share a marker set.** Recipients and donors must agree on
   the `.snp`: same `M`, same `chrom`, same genetic position at every index. A
   disagreement is a configuration fault, not a silently mismatched paint.
5. **Decode to haploid allele bytes.** Both panels are decoded to flat
   haplotype-major `{0, 1, missing}` buffers — exactly the layout the
   forward-backward expects.
6. **Build the model vectors.** `rho`, `mu`, `w` (the per-SNP genetic-length
   weight the coancestry lengths integrate against), and the per-recipient `pi`
   (uniform, or leave-one-out for self-painting).
7. **Pick the compute backend.** GPU when a device is visible, otherwise the
   CpuBackend reference oracle. The whole scan and its reductions run in **native
   FP64**.
8. **Run in recipient batches.** The recipients are processed in waves of
   `--recip-batch`. This is the load-bearing memory design: the `O(K·M)` copying
   posterior for a batch **never leaves the device** — only the small
   accumulators return (`N×K` coancestry, or the `N×M×P` local-ancestry
   posterior). A non-`Ok` status from any wave is surfaced as a fault.
9. **Emit.** The default face writes the coancestry table; `--face localanc`
   writes the per-SNP ancestry posterior. Both go to the configured destination
   in the configured format, one long-format block per recipient.

---

## 6. Donor labels — the ancestry partition

Both faces need to know each donor's ancestry label (the coancestry columns, and
the localanc label set). By default the label comes from the donor's population in
the `.ind` file. A `--labels` file overrides that with **one label per donor
haplotype column, in `.ind` order** — for phased diploid donors that is two
identical entries per individual, and a file whose length doesn't match `K` is a
configuration fault with an explicit message. The distinct labels, taken in
first-appearance order, become the `P` ancestry columns. This resolution happens
*before* the compute, because for the localanc face each donor's group index is a
compute input.

---

## 7. Exit-code contract

| Situation | Code | Name |
|---|---|---|
| Normal completion | `0` | `kExitOk` |
| Missing `--prefix` / `--donors`; failed request validation; recipients-vs-donors `.snp` mismatch; a wave returning a non-`Ok` status | `2` | `kExitInvalidConfig` |
| Input read error; device init failure; a `--labels` file that can't be opened; a compute run that threw | `4` | `kExitIoError` |

The two things worth remembering: a bad or degenerate *request* (missing input,
mismatched marker set, an over-budget model the user didn't confirm) is a
configuration fault, and an input/device/run failure is an io fault. There is no
"result was uninteresting" nonzero exit — a completed paint always exits `0`.

---

## 8. Layering: plain C++20, GPU only through the backend

This header, and the command behind it, contain no CUDA code and include no CUDA
headers. `paint` reaches the GPU **only** through the `ComputeBackend` seam
(`make_cuda_backend` when a device is visible, `make_cpu_backend` otherwise, then
the `ls_paint_coancestry` / `ls_localanc` virtuals). Everything CUDA-specific
lives on the far side of that seam in the compute library.

That boundary has the same two payoffs as the other commands: the whole app layer
compiles and unit-tests without a CUDA toolkit, and the separation is checked
mechanically by the build-time grep gate, so GPU code can't quietly leak into the
app layer.

One honest note on the backend fallback: `paint` runs on the `CpuBackend`
reference oracle when no GPU is visible, which is convenient for the tests and the
golden-parity fixtures. But steppe is a GPU product — the CpuBackend is the
dev/test parity oracle, and the GPU forward-backward is the real engine the
command is built to drive.

---

## 9. Contracts and invariants

- **Pre-phased haploid input, always.** steppe builds no phaser. `paint` only
  ever consumes haplotypes some upstream tool already phased; the validator
  counts and rejects any diploid sample before the scan runs.
- **Recipients and donors share one marker set.** Same `M`, same `chrom`, same
  genetic map at every index — enforced up front.
- **Native FP64 throughout.** The scan and its reductions are cancellation-
  sensitive sequential recurrences, and there is **no GEMM** anywhere in the
  Li-Stephens engine — so the emulated-FP64 (Ozaki) default, which is a
  *matmul-only* accelerator, does not apply here. The precision is `fp64()`,
  native, top to bottom.
- **`pi` is a normalized prior.** Uniform `1/K`, or leave-one-out `1/(K-1)` with
  the self column zeroed — either way it sums to 1, so no recipient can copy
  itself in the self-painting case.
- **The copying posterior stays on the device.** Only the `N×K` (coancestry) or
  `N×M×P` (localanc) accumulators cross back to the host; the `O(K·M)` `gamma`
  table is never resident host-side.

---

## 10. Edge cases worth calling out

- **Self-painting (donors prefix == recipients prefix).** Detected by equal
  prefixes; both reads use the identical selection so individual order matches,
  and recipient `r`'s self donor is donor `r`, zeroed out via leave-one-out
  unless `--self-copy` is set.
- **`--theta auto` vs an explicit theta.** With `auto` the emission rate is the
  Watterson estimate over the resolved `K`; an explicit `--theta` is used
  verbatim.
- **A `--labels` file of the wrong length.** Rejected with a message that spells
  out the "one label per donor *haplotype column*" rule (two entries per phased
  diploid individual), rather than silently truncating or mis-aligning.
- **A stray het or missing genotype code.** Decoded to the missing-allele
  sentinel; the emission step reads it as no-information at that site, so a dirty
  site softens the local mosaic instead of corrupting the run.
- **A batch that returns non-`Ok`.** Treated as an invalid-configuration fault
  immediately — the command does not emit a partial table.
