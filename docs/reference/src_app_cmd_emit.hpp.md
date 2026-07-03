# `cmd_emit.hpp` reference

## 1. Purpose

`src/app/cmd_emit.hpp` is the one shared helper that every `steppe <command>` uses
to write its result output — the CSV, TSV, or JSON a command produces — to wherever
that output should go. Instead of each command re-implementing "figure out the
format, open the file or use the screen, write the bytes, and make sure the write
actually succeeded," they all call a single function here and get that behavior
consistently.

There are exactly two functions:

1. `emit_to_destination` — the front door. It reads the requested output format,
   picks the destination (a file named on the command line, or the terminal),
   hands a writable stream to the command's own serializer, and then verifies the
   write finished cleanly.
2. `finish_emit` — the small verification step the front door calls at the end (and
   that a command could call directly). It forces any buffered bytes out and checks
   that the stream is still healthy.

The reason this exists as a shared, verified path — rather than each command just
printing and exiting — is a real correctness concern. steppe's deliverable *is*
files. If a command wrote a partial file (because the disk filled up, a quota was
hit, or a pipe was closed) and then exited with a success code, a downstream script
would treat a truncated, corrupt file as a finished result. This helper turns those
silent partial writes into an explicit I/O-error exit code, so a broken write is
never mistaken for a good one.

The file is plain C++20 and pulls in no GPU code. It is host-side stream I/O only:
opening files, writing to the terminal, and checking stream state. Printing and
process control belong to the command-line layer here — the core library never
prints — which is why this helper lives alongside the commands and not inside the
library.

---

## 2. `emit_to_destination` — routing output to a file or the screen

`emit_to_destination` is a function template. The caller passes a `write` callable —
the command's own serializer — and the helper invokes it as
`write(std::ostream&, OutputFormat)`, letting the command fill the stream however it
likes while the helper owns everything around that write.

It performs four steps in order:

| Step | What happens | If it fails |
|---|---|---|
| 1. Parse the format | Reads the requested format string (from `config.format()`) and maps it to `OutputFormat` — one of `Csv`, `Tsv`, or `Json`. The default, if parsing is needed but nothing forces otherwise, is `Csv`. | An unrecognized token prints `steppe <prefix>: unknown --format '<token>' (csv\|tsv\|json)` and returns the invalid-configuration exit code. |
| 2. Pick the destination | If no output file was named (`config.out_file()` is empty), the destination is standard output — the terminal. Otherwise it opens the named file. | Opening the file can fail (bad path, no permission, read-only location); that prints `steppe <prefix>: cannot open --out file: <path>` and returns the I/O-error exit code. |
| 3. Write | Calls the caller's `write(stream, format)` to serialize the result into the chosen stream. | — |
| 4. Verify | Calls `finish_emit` (see section 3) to flush and confirm the write landed cleanly. | A torn or short write returns the I/O-error exit code. |

The `prefix` argument is the short command name that gets printed in every
diagnostic — the "`steppe <prefix>:`" tag — so error messages name the command that
failed.

### Why the output file is opened in binary + truncate mode

When a file is the destination, it is opened in binary mode and truncating mode.
Binary mode means the bytes the serializer writes are the exact bytes that hit the
file, with no platform-specific newline rewriting — important because the output is
compared byte-for-byte against reference files. Truncating mode means an existing
file at that path is overwritten from empty, not appended to, so re-running a command
replaces the old result rather than piling onto it.

### Re-parsing the format here is deliberate

The format string is already validated earlier, when the run's configuration is
built. This helper parses it again at write time on purpose, so the write path is
self-contained and defensive: it never assumes an upstream check happened, and the
same map from token to format lives in one place the serializer controls.

---

## 3. `finish_emit` — the post-write integrity gate

`finish_emit` is the single check that turns a silent partial write into a real
error. Given the stream that was just written, a command-name prefix, and a
human-readable name for the destination, it does two things:

1. **Flush the stream.** This step is load-bearing. A small write sits in an
   in-memory buffer and has not actually reached the file (or pipe) until it is
   flushed. Checking the stream's health *before* flushing would check the wrong
   thing — it could report success while the real write to the disk or pipe was
   still pending and about to fail.
2. **Check the stream is still good after the flush.** If the flush revealed a
   problem — a full disk, an exceeded quota, a short write, or a closed pipe — the
   stream's failure flags are now set. Those are the same flags a stream's
   destructor would quietly discard, which is exactly how a partial write would
   otherwise go unnoticed.

On a clean write it returns "no error." On a failed write it prints
`steppe <prefix>: write failed (disk full / short write / closed pipe): <dest>` to
standard error, naming the destination, and returns the I/O-error exit code.

Every place that emits output shares this one gate, so the "did the write actually
succeed?" question is answered the same way everywhere, whether the destination was
a file or the terminal.

---

## 4. Exit codes and the caller's return-code contract

Both functions report their outcome the same way: they return an *optional* exit
code. "No value" means the emit succeeded and the caller should carry on. A value
means something went wrong and the caller must return that number as the process's
exit code.

The intended calling shape is a single line at the emit site:

```cpp
if (auto rc = emit_to_destination(config, "qpadm", write)) return *rc;
```

If `emit_to_destination` succeeded, the `if` is skipped and the command continues.
If it failed, the exit code flows straight out of the command unchanged. Both
functions are marked so that ignoring their return value is a compile-time warning —
you cannot accidentally drop the error on the floor.

The two exit codes this helper can produce:

| Situation | Exit code | Meaning |
|---|---|---|
| Unknown `--format` token | `kExitInvalidConfig` (value `2`) | The requested output format was not one of `csv`, `tsv`, or `json`. |
| Cannot open the `--out` file, or the write itself failed | `kExitIoError` (value `4`) | The destination could not be opened, or the bytes did not land cleanly (disk full, quota, short write, closed pipe). |

A clean emit returns no code at all, which the command treats as continue-normally
(and, ultimately, a success exit).

---

## 5. Layering and why this is a header-only helper

This file sits in the command-line layer, above the core library, and it stays there
by construction. It includes only standard C++ headers plus three small app/config
headers (the output-format enum and parser, the exit-code names, and the run
configuration type). It pulls in no GPU code at all, because its whole job is
host-side stream I/O — opening files, writing to the terminal, checking stream state.

Keeping this in the command-line layer follows a firm rule: the core library never
prints and never decides the process exit code. All of the user-facing messages and
the exit codes originate up here, where the command owns the terminal.

`emit_to_destination` is a template (and both functions are header-inline) because
the `write` serializer differs per command — each command has its own result type
and its own way of turning it into text. Making the helper a template lets every
command reuse the exact same open/write/verify/route logic while still plugging in
its own serializer, with no shared base class or virtual dispatch.
