#!/usr/bin/env Rscript
# tests/r/verify_export_rds.R
#
# Box5090-only END-TO-END gate for steppe.export_f2_rds (rds-converter.md §7, gate 2): the
# literal "verified in ADMIXTOOLS 2" proof. Given a directory EXPORTED by steppe.export_f2_rds,
# it asserts admixtools::read_f2() loads it as a valid [P,P,n_block] f2 array AND that
# admixtools::f4 / qpadm on the exported cache reproduce steppe's own fit.
#
# This is NOT part of the CUDA-free CI lane (it needs R + admixtools 2.x + a steppe export); the
# dependency-free round-trip lives in tests/python/test_py_convert_rds.py. It SKIPS cleanly
# (exit 0) when admixtools is not installed, so a harness can invoke it unconditionally.
#
# Usage:
#   Rscript tests/r/verify_export_rds.R <exported_rds_dir> [<steppe_f4_csv>]
#
#   <exported_rds_dir>  a dir written by steppe.export_f2_rds (per-pop subdirs + block_lengths).
#   <steppe_f4_csv>     OPTIONAL steppe-native f4 table over the SAME cache, columns
#                       pop1,pop2,pop3,pop4,est,se[,z,p] (e.g. `steppe f4 ... --format csv`).
#                       When given, AT2's f4 on the exported dir is asserted to MATCH it
#                       (rtol 1e-6, atol 1e-9) — the parity half of the gate.
#
# Example (box5090):
#   /venv/main/bin/python -c "import steppe; \
#     steppe.export_f2_rds(steppe.read_f2('/workspace/data/qpgraph_9pop_stpf2bk1'), '/tmp/exp')"
#   Rscript tests/r/verify_export_rds.R /tmp/exp

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) {
  stop("usage: Rscript verify_export_rds.R <exported_rds_dir> [<steppe_f4_csv>]")
}
rds_dir <- args[[1L]]
f4_csv <- if (length(args) >= 2L) args[[2L]] else NA_character_

if (!requireNamespace("admixtools", quietly = TRUE)) {
  cat("SKIP: admixtools not installed (R end-to-end gate needs it).\n")
  quit(status = 0L)
}
suppressMessages(library(admixtools))

fail <- function(msg) { cat("FAIL:", msg, "\n"); quit(status = 1L) }
close_enough <- function(got, want, rtol = 1e-6, atol = 1e-9) {
  abs(got - want) <= atol + rtol * abs(want)
}

# --- structural checks ----------------------------------------------------------------
fr <- tryCatch(read_f2(rds_dir),
               error = function(e) fail(paste("read_f2 errored:", conditionMessage(e))))
d <- dim(fr)
if (length(d) != 3L || d[[1L]] != d[[2L]]) fail(paste("bad f2 dims:", paste(d, collapse = "x")))
pops <- dimnames(fr)[[1L]]
if (is.null(pops)) fail("read_f2 array has no dimnames (pops)")
if (!identical(pops, sort(pops))) fail("dimnames are not C-locale sorted")
if (any(!is.finite(fr))) fail("f2 array has non-finite values")

# diagonal self-pairs must be exactly 0 (AT2 convention; steppe's export zeros them).
diag_max <- 0
for (i in seq_len(d[[1L]])) diag_max <- max(diag_max, max(abs(fr[i, i, ])))
if (diag_max != 0) fail(paste("diagonal not zero, max |f2[i,i]| =", diag_max))

# symmetry: f2[i,j,] == f2[j,i,].
if (!isTRUE(all.equal(fr, aperm(fr, c(2L, 1L, 3L))))) fail("f2 array is not symmetric")

cat(sprintf("OK read_f2: [%d,%d,%d], %d pops, diagonal==0, symmetric\n",
            d[[1L]], d[[2L]], d[[3L]], length(pops)))

# --- f4 runs on the exported cache (generic quartets from the first pops) --------------
q1 <- pops[c(1L, 2L, 3L, 4L)]
r1 <- f4(fr, q1[[1L]], q1[[2L]], q1[[3L]], q1[[4L]])
cat(sprintf("OK f4(%s): est=%.10g se=%.10g\n",
            paste(q1, collapse = ","), r1$est[[1L]], r1$se[[1L]]))

# --- qpadm smoke on the exported cache (target=pop1, left=pop2, right=rest) ------------
if (length(pops) >= 5L) {
  qa <- tryCatch(
    qpadm(fr, left = pops[2L], right = pops[3:length(pops)], target = pops[1L]),
    error = function(e) { cat("NOTE qpadm skipped:", conditionMessage(e), "\n"); NULL })
  if (!is.null(qa)) cat(sprintf("OK qpadm ran on the exported cache (p=%.4g)\n",
                                qa$rankdrop$p[[1L]]))
}

# --- parity vs steppe native (optional) -----------------------------------------------
if (!is.na(f4_csv)) {
  if (!file.exists(f4_csv)) fail(paste("steppe f4 csv not found:", f4_csv))
  tab <- read.csv(f4_csv, stringsAsFactors = FALSE)
  need <- c("pop1", "pop2", "pop3", "pop4", "est", "se")
  if (!all(need %in% names(tab))) fail(paste("f4 csv missing columns:",
                                             paste(setdiff(need, names(tab)), collapse = ",")))
  nbad <- 0L
  for (k in seq_len(nrow(tab))) {
    rr <- f4(fr, tab$pop1[[k]], tab$pop2[[k]], tab$pop3[[k]], tab$pop4[[k]])
    if (!close_enough(rr$est[[1L]], tab$est[[k]])) {
      nbad <- nbad + 1L
      cat(sprintf("  est mismatch row %d: AT2=%.10g steppe=%.10g\n",
                  k, rr$est[[1L]], tab$est[[k]]))
    }
    if (!close_enough(rr$se[[1L]], tab$se[[k]])) {
      nbad <- nbad + 1L
      cat(sprintf("  se mismatch row %d: AT2=%.10g steppe=%.10g\n",
                  k, rr$se[[1L]], tab$se[[k]]))
    }
  }
  if (nbad > 0L) fail(sprintf("%d f4 est/se mismatches vs steppe native", nbad))
  cat(sprintf("OK parity: AT2 f4 on the exported cache matches steppe native (%d quartets)\n",
              nrow(tab)))
}

cat("PASS: exported dir is a valid ADMIXTOOLS 2 read_f2 cache.\n")
quit(status = 0L)
