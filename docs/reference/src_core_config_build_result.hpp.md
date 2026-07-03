# `build_result.hpp` reference

## 1. Purpose

`src/core/config/build_result.hpp` defines `BuildResult<T>`, a small result
type that carries **either** a successfully built value of type `T` **or** an
error code (`steppe::Status`) explaining why the build failed. It is the return
type of the configuration `build()` step: a caller writes `if (result)` to check
success, then reads the value, or reads the error to find out what went wrong.

It exists to fill a gap between two C++ standard versions. The type steppe would
naturally use here is `std::expected<T, Status>`, which is the standard library's
own "value-or-error" container. But `std::expected` only arrived in C++23, and
this project compiles as C++20, where it is not available. `BuildResult<T>` is a
hand-written stand-in that reimplements just the slice of the `std::expected`
interface the configuration layer actually uses, with the same spelling, so that
call sites read exactly as they would with the real thing.

Three properties are deliberate:

- **CUDA-free.** The header contains no GPU code and pulls in no GPU headers. It
  depends only on the C++ standard library and on `steppe/error.hpp` (for the
  `Status` type). That keeps it lightweight enough to compile into every layer
  that needs it — the core library, the command-line tool, and the language
  bindings — without forcing any of them to also pull in the device toolkit.
- **Header-only.** Everything is defined inline in the header; there is no
  matching `.cpp` file.
- **Retirable.** Because it mirrors the `std::expected` interface so closely,
  once the toolchain moves to C++23 this file can be deleted and every use
  replaced with `std::expected<T, Status>` without changing any call site.

---

## 2. The error arm: `Unexpected` and `unexpected()`

To return an error, a `build()` failure site does not construct a `BuildResult`
directly. Instead it returns the result of a small free function, so the failure
site reads naturally:

```cpp
return unexpected(Status::InvalidConfig);
```

Two pieces make this work, and both mirror the standard library's
`std::unexpected` mechanism:

| Name | What it is | What it's for |
|---|---|---|
| `struct Unexpected` | A tiny tag struct holding a single `Status status;` field. | It marks a `Status` as "this is the error arm," so it is distinct from a `Status` that might happen to be a value. `BuildResult` has a dedicated constructor that accepts an `Unexpected` and stores its `status` in the error slot. |
| `unexpected(Status s)` | A free function that wraps a `Status` in an `Unexpected` and returns it. | It is the factory that lets a failure site say `unexpected(Status::InvalidConfig)` instead of spelling out `Unexpected{...}`. It is marked `[[nodiscard]]` and `noexcept`. |

Both live in the `steppe::config` namespace alongside `BuildResult`.

---

## 3. The `BuildResult<T>` class

`BuildResult<T>` is a class template. `T` is the type of the success value (in
the configuration layer, the built run configuration). An instance is always in
exactly one of two states, referred to here as its two "arms":

- the **value arm** — it holds a `T`, and the build succeeded;
- the **error arm** — it holds a `Status`, and the build failed.

Internally this is stored as a `std::optional<T> value_` plus a
`Status error_`. When `value_` holds a value the object is on the value arm;
when `value_` is empty the object is on the error arm and `error_` carries the
reason. The `error_` field defaults to `Status::Ok` and is only meaningful when
`value_` is empty.

### Construction — the two arms

There are two constructors, and both are intentionally **implicit** (they are not
marked `explicit`), matching how `std::expected` lets you construct from either a
value or an `unexpected`. That is what makes the `return` statements at call sites
read cleanly.

| Constructor | Arm | Effect |
|---|---|---|
| `BuildResult(T value)` | value | Moves the given value into `value_`. A `build()` that succeeds can simply `return the_value;`. |
| `BuildResult(Unexpected u)` | error | Stores `u.status` in `error_`, leaving `value_` empty. A `build()` that fails can `return unexpected(Status::InvalidConfig);`. |

### Checking which arm

Two accessors report whether the build succeeded:

- `has_value()` — returns `true` on the value arm, `false` on the error arm.
- `explicit operator bool()` — the same test, so `if (result) { ... }` works
  directly. It is `explicit`, so a `BuildResult` will not silently convert to a
  bool in an unintended context.

Both are `noexcept` and `[[nodiscard]]`.

### Reading the value

Three ways to reach the stored `T`, each provided in the usual set of reference
overloads so the value can be read, mutated, or moved out of a temporary:

- `value()` — named accessor. Has `const&`, `&`, and `&&` overloads. The `&&`
  overload moves the value out, which is useful when the result is a temporary.
- `operator*` — the same three overloads, for dereference-style access
  (`*result`).
- `operator->` — member access (`result->field`), in `const` and non-`const`
  forms.

All of these are `noexcept` and `[[nodiscard]]`.

### Reading the error

- `error()` — returns the `Status` from the error arm. `noexcept` and
  `[[nodiscard]]`.

### Invariants and undefined behavior

- **Exactly one arm is populated.** A `BuildResult` is never both and never
  neither; the constructor chosen at creation fixes the arm.
- **Reading the value on the error arm is undefined behavior.** `value()`,
  `operator*`, and `operator->` all assume `has_value()` is true and dereference
  the underlying `std::optional` without checking. Calling them when
  `has_value()` is false is undefined — exactly the same contract as
  `std::expected`'s value accessors. Always check `has_value()` (or `if
  (result)`) first.
- **`error()` is only meaningful on the error arm.** On the value arm `error_`
  still holds its default of `Status::Ok`, which does not mean "there was an OK
  error" — it simply is not a meaningful value there. Only read `error()` when
  `has_value()` is false.

---

## 4. Relationship to `std::expected` and future migration

`BuildResult<T>` is a purpose-built subset of `std::expected<T, Status>`, and the
mapping is one-to-one for the parts it implements:

| `BuildResult` | `std::expected` equivalent |
|---|---|
| `has_value()` / `operator bool` | `has_value()` / `operator bool` |
| `value()` / `operator*` / `operator->` | `value()` / `operator*` / `operator->` |
| `error()` | `error()` |
| `unexpected(Status)` free function | `std::unexpected<Status>` |
| implicit value / `Unexpected` constructors | implicit value / `std::unexpected` constructors |
| value access on the error arm is undefined | value access on the error arm is undefined |

Because the interface and the semantics line up, the intended end state is to
delete this header once the project moves to C++23 and replace `BuildResult<T>`
with `std::expected<T, Status>` and `unexpected(...)` with `std::unexpected(...)`
throughout. No call site should need to change.
