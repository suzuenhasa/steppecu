# `error.hpp` reference

## 1. Purpose

`include/steppe/error.hpp` defines the small set of outcome codes the steppe
library returns from its early layers. It holds a single strongly-typed C++
enum, `Status`, whose values name every way a call can finish: success, one
resource failure, three expected statistical outcomes, and one setup fault.

The header is deliberately tiny and self-contained. It contains no CUDA code and
does not even use the C++ standard library, so it is cheap to include anywhere —
the core library, the public API, the command-line tool, and the language
bindings — without dragging in the GPU stack or any heavier headers.

This enum is the placeholder for a larger, cross-language C ABI that is planned
but not yet built (a richer C enum plus a separate internal error view). Until
that lands, this header *is* steppe's public status type: the compact, typed set
of outcomes the early layers hand back, without committing to the full ABI yet.

---

## 2. Two categories of outcome: recoverable results versus fail-fast faults

The most important idea in this file is that the values split into two very
different categories, and mixing them up would be a mistake.

**Recoverable outcomes.** Most of the values describe conditions that a caller is
expected to see during normal operation and handle without stopping the run.
These are returned as ordinary status values — never thrown as exceptions and
never turned into a hard abort. This matters because a large model search will
routinely fit models that turn out to be degenerate; hitting one of those is an
expected result, not a crash. This category includes the resource condition
`DeviceOom` and the three statistical "domain outcome" values `RankDeficient`,
`NonSpdCovariance`, and `ChisqUndefined`.

**Fail-fast faults.** The other category is a genuine setup mistake: the caller
built something wrong. The single value here is `InvalidConfig`. Rather than
being quietly surfaced and worked around, this kind of error is meant to fail
fast — it signals a programming or configuration bug that should be fixed, not
tolerated.

### Why the three domain outcomes are statuses, not errors

`RankDeficient`, `NonSpdCovariance`, and `ChisqUndefined` are the three
"domain outcome" values, and they deserve special mention. Each one is a
legitimate statistical result of fitting a particular model, not a defect in the
code. When you fit thousands of candidate models in a search, some of them will
be degenerate or over-parameterized, and the correct behavior is to report that
fact for that model and move on. Treating these as ordinary return values (rather
than exceptions) is what lets a big sweep keep going instead of aborting on the
first bad model.

---

## 3. The `Status` values

`Status` is a scoped enum (`enum class Status`). Every call in the early layers
returns one of these six values.

| Value | Category | Meaning |
|---|---|---|
| `Ok` | Success | The call succeeded. |
| `DeviceOom` | Resource failure (maybe recoverable) | A GPU memory allocation or the VRAM-budget check failed. This is a resource condition, and it is sometimes recoverable — for example by retrying with a smaller chunk size or a smaller memory budget. |
| `RankDeficient` | Recoverable domain outcome | The rank test or the generalized-least-squares solve hit a rank-deficient design matrix `X`, which means the model cannot be uniquely identified. This is the expected result of fitting a degenerate model, not a bug. |
| `NonSpdCovariance` | Recoverable domain outcome | The covariance matrix `Q` is not symmetric positive-definite, so its Cholesky factorization failed. This happens for a degenerate or collinear model, and again is a statistical result, not a bug. |
| `ChisqUndefined` | Recoverable domain outcome | The degrees of freedom are zero or negative (or the chi-squared statistic otherwise cannot be computed) for this model, so its chi-squared tail probability `p` is undefined. Instead of reporting a fabricated value, steppe leaves `p` at its NaN sentinel. This is the expected result of an over-parameterized model — for instance, when the number of sources minus one spans all available columns and leaves zero degrees of freedom — not a bug. |
| `InvalidConfig` | Fail-fast fault | Configuration failed validation in `ConfigBuilder::build()`: a bad architecture list, conflicting flags, an over-budget VRAM request, or a precision mode that cannot be honored. This is a setup mistake and fails fast. |

---

## 4. Relationship to the full C ABI (deferred)

This header is intentionally a trimmed subset. A complete, cross-toolchain C
application binary interface is planned — a richer C enum (`steppe_status_t`)
together with a separate internal `Error` view that can carry more context than a
bare code. That fuller design is deferred to a later milestone and is not part of
this file.

Two consequences follow from that deferral. First, this `Status` enum is the
public status type callers should use today; it is not a temporary internal
detail. Second, the header stays deliberately minimal — CUDA-free and
standard-library-free — precisely so it can serve as the foundation the early
contract layers build against without pulling in the eventual full ABI before it
exists.
