# `qpgraph_opt_constants.hpp` reference

## 1. Purpose

`src/core/qpadm/qpgraph_opt_constants.hpp` is the single place that defines
every fixed number the qpGraph optimizer uses. The qpGraph optimizer is a
deterministic projected-Newton multistart search: it fits a population-graph
model by starting from many different random initial guesses ("restarts") and
refining each one with a Newton-style step until it converges, then keeps the
best score.

The header holds two kinds of numbers, both as named constants:

1. **The random-number-generator seeds** — the fixed magic constants that turn a
   restart index and a parameter index into a reproducible starting guess.
2. **The optimizer hyperparameters** — the finite-difference step size, the
   curvature and step-size thresholds, the line-search settings, and the
   convergence tolerances that drive each restart's refinement loop.

All of these live in the nested namespace `steppe::core::qpadm::qpgraph_opt`.

The header contains *only* constants. There are no functions, no structs, and no
algorithm code — the actual optimizer loop lives in each backend. This file
exists so that the one thing both backends must agree on, exactly, is written
down in exactly one place.

---

## 2. Why every number is single-sourced here

steppe runs the qpGraph optimizer two ways:

- a **CPU reference** implementation (the oracle used for testing), and
- a **GPU implementation** (the production path, a fleet of restarts running in
  parallel on the GPU).

The GPU path is validated against the CPU path by comparing their outputs on
real-data reference cases — for example, a single graph fit whose score is
`80.0674`, and a five-population search. For that comparison to pass, the two
backends cannot merely land on similar answers; they must walk the *same*
restart trajectories step for step. A deterministic projected-Newton multistart
only reproduces exactly if both sides use identical random seeds, identical step
sizes, identical curvature and convergence thresholds — every single number.

These numbers used to be typed out by hand at both sites. That was fragile: if
someone changed a value on one side and not the other, nothing failed to
compile — the two backends just quietly diverged and parity broke. Hoisting all
of them into this one header turns any such drift into a one-line edit in a
single place (or, if a symbol is mistyped, a compile error) instead of a silent
correctness bug.

This mirrors the pattern steppe already uses for the one shared rule that
defines how the genome is partitioned into jackknife blocks: one constant set,
read by both the core library and the GPU kernels.

---

## 3. Host-pure and CUDA-free

The header is deliberately written so it can be included by both the ordinary
host C++ compiler and the GPU compiler, with no CUDA-specific keywords anywhere
in it. It declares only `constexpr` variables of built-in integer and
floating-point types.

This works because of a specific rule in the CUDA language: a `const`-qualified
variable of a built-in integer or floating-point type that is initialized with a
constant expression may be used directly inside GPU (`__device__`) code with no
special memory-space annotation. So each of these constants is usable verbatim
from GPU kernels and from host code alike, without any CUDA types and without
relying on any experimental compiler flag.

### Why the loop itself is not shared, only the numbers

A natural instinct would be to also factor the random-number generator and the
Newton loop into shared functions here, so the two backends literally call the
same code. That is intentionally *not* done. A function callable from both host
and GPU code would need a CUDA-specific annotation (which would break the
CUDA-free rule) or an experimental compiler flag. So instead, each backend keeps
its own copy of the loop — shaped for its world (growable host arrays on the CPU
side, fixed-size on-GPU stack arrays on the GPU side) — but both loops read every
magic number from this header. The only thing that has to match between the two,
the constants, is the only thing single-sourced here.

---

## 4. The multistart seeding constants

Each restart needs a reproducible starting guess for every parameter of the
graph model. steppe generates it with a small, well-known integer hashing scheme
called splitmix64, so that the same (restart index, parameter index) pair always
produces the same starting number on both backends.

The recipe is:

1. Build a 64-bit seed `z` from the restart index (`inst`) and the parameter
   index (`dim`):
   `z = inst * kSplitmixInstMul + dim * kSplitmixDimMul + kSplitmixSeedInc`.
2. Run two splitmix64 "mixing" rounds over `z` using `kSplitmixMix1` and
   `kSplitmixMix2`, with the standard 30 / 27 / 31 right-shift schedule that
   scrambles the bits.
3. Take the low 52 bits of the result (`z & kMantissaMask`) and divide by
   `kMantissaDiv` (= 2⁵²) to land on a uniform value in the range [0, 1).

| Constant | Value | What it's for |
|---|---|---|
| `kSplitmixInstMul` | `0x100000001B3` | The multiplier applied to the restart index. This is the FNV hashing prime; it weights the restart number into the seed. |
| `kSplitmixDimMul` | `0x9E3779B97F4A7C15` | The multiplier applied to the parameter index. This is the golden-ratio constant (2⁶⁴ divided by the golden ratio), the standard choice for spreading successive indices apart. |
| `kSplitmixSeedInc` | `0xD1B54A32D192ED03` | The additive increment that finishes building the seed — the standard splitmix64 increment. |
| `kSplitmixMix1` | `0xBF58476D1CE4E5B9` | The first splitmix64 mixing multiplier. |
| `kSplitmixMix2` | `0x94D049BB133111EB` | The second splitmix64 mixing multiplier. |
| `kMantissaMask` | `0xFFFFFFFFFFFFF` | A mask that keeps the low 52 bits (equals 2⁵² − 1). 52 is the number of fraction bits in a double, so masking to 52 bits gives exactly the bits needed to form a fraction in [0, 1). |
| `kMantissaDiv` | `4503599627370496.0` | The normalizer: 2⁵² as a floating-point value. Dividing the masked 52-bit integer by this maps it into the range [0, 1). |

These are fixed seeds, not tuning knobs. They are the standard splitmix64 and
Fibonacci-hashing constants. Changing any of them does not make the optimizer
better or worse — it just re-rolls every restart's initial guess on both
backends at once, which would invalidate the reference comparisons. Treat them as
frozen.

---

## 5. The projected-Newton step constants

Within one restart, the optimizer repeatedly estimates the local slope
(gradient) and curvature of the score with respect to each parameter, one
parameter at a time, using small finite-difference probes, and takes a step. The
"projected" part means each proposed step is clamped so a parameter can't jump
too far in a single move. These constants control that step.

| Constant | Value | What it's for |
|---|---|---|
| `kFdStep` | `1e-4` | The finite-difference probe distance `h`: how far each parameter is nudged to measure the local slope and curvature numerically. |
| `kCurvGuard` | `1e-30` | A tiny amount added to the curvature denominator so a near-zero curvature can never cause a divide-by-zero. |
| `kCurvHalf` | `0.5` | The one-half factor that appears in the central second-difference formula used to estimate curvature. |
| `kCurvThresh` | `1e-8` | The decision threshold for how to step. If the measured curvature is larger than this, take a proper Newton step (slope divided by curvature); otherwise the curvature is too small to trust, so fall back to a plain gradient step. |
| `kGradStepScale` | `0.5` | The step factor used in that gradient fallback — move by the slope times this. |
| `kTrustClamp` | `0.5` | The trust-region clamp: no single projected step may move a parameter by more than this magnitude, keeping each move from overshooting. |

---

## 6. The backtracking line-search constants

After the optimizer computes a proposed new point, it checks that the point
actually does not make the score worse. If it does, the point is repeatedly moved
halfway back toward the current point until the score stops increasing (a
"backtracking" line search). These two constants bound that inner loop.

| Constant | Value | What it's for |
|---|---|---|
| `kMaxBacktrack` | `8` | The maximum number of halving steps the backtracking loop will take before giving up and accepting the current best. Caps the inner loop so it can't spin. |
| `kBacktrackHalf` | `0.5` | The halving factor: each backtrack replaces the candidate point with the midpoint between it and the current point (candidate ← this × (candidate + current)). |

---

## 7. The per-restart convergence test

A restart stops iterating once it has effectively settled. Two conditions must
*both* hold: the parameters barely moved on the last step, and the score barely
changed. Each condition compares against a base tolerance scaled by one of these
constants.

| Constant | Value | What it's for |
|---|---|---|
| `kTolDxScale` | `1e-2` | Scales the tolerance for the parameter-movement test: the largest single parameter change must fall below the tolerance times this. |
| `kTolDsScale` | `1e-3` | Scales the tolerance for the score-change test: the score change must fall below the tolerance times this. |

Because the score-change scale (`1e-3`) is tighter than the parameter-movement
scale (`1e-2`), a restart is only declared converged when both the point and its
score have genuinely stabilized.
