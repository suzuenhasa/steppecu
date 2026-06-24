suppressMessages(library(admixtools))
options(width = 220, digits = 12)
sink("/tmp/golden_qpgraph.out.txt", split = TRUE)

cat("================ qpGraph GOLDEN ORACLE (single-graph fit, fixed WELL-IDENTIFIED topology) ================\n")
cat("R.version.string :", R.version.string, "\n")
cat("admixtools ver   :", as.character(packageVersion("admixtools")), "\n")
cat("RNGkind          :", paste(RNGkind(), collapse = ", "), "\n")

f2dir <- "/workspace/data/aadr/f2_fit0_FINAL"
allpops <- c("England_BellBeaker", "Czechia_EBA_CordedWare", "Turkey_N", "Mbuti",
             "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana")
f2b <- read_f2(f2dir, pops = allpops)
cat("\nf2 dims:", paste(dim(f2b), collapse = " x "), "\n")

# ---------------------------------------------------------------------------
# FIXED, WELL-IDENTIFIED TOPOLOGY (nadmix=1).
# Czechia_EBA_CordedWare (aCW) is an admixture of:
#   pSteppe  - a deep EASTERN-Eurasian-related source (off the East-Eurasian clade nEAS),
#   pIran    - the Iran_GanjDareh_N lineage (West-Eurasian).
# The two sources sit on OPPOSITE sides of the deep OOA split, giving the f3 constraints
# strong leverage on the single mixture weight -> a unique INTERIOR optimum (w ~ 0.153),
# NOT a boundary collapse. (Alternatives where both sources were recent sisters drove the
# weight to ~1.0 / 0.0 with disagreeing restarts -> degenerate, rejected.)
# Edge list (parent -> child). Admixture node = aCW (the single 2-parent node).
# ---------------------------------------------------------------------------
edges <- matrix(c(
  "R",      "Mbuti",
  "R",      "nOOA",
  "nOOA",   "Papuan",
  "nOOA",   "nEAS",
  "nEAS",   "Han",
  "nEAS",   "Karitiana",
  "nOOA",   "nWE",
  "nWE",    "Israel_Natufian",
  "nWE",    "nAnat",
  "nAnat",  "Turkey_N",
  "nAnat",  "England_BellBeaker",
  "nWE",    "pIran",
  "pIran",  "Iran_GanjDareh_N",
  "nEAS",   "pSteppe",
  "pSteppe","aCW",
  "pIran",  "aCW",
  "aCW",    "Czechia_EBA_CordedWare"
), ncol = 2, byrow = TRUE)
colnames(edges) <- c("from", "to")
cat("\n--- TOPOLOGY edge list ---\n"); print(edges)
admix_nodes <- names(which(table(edges[, "to"]) >= 2))
cat("\nadmixture nodes (nadmix):", paste(admix_nodes, collapse = ", "),
    " (count =", length(admix_nodes), ")\n")

# ---------------------------------------------------------------------------
# FIT — defaults match the design/spike: diag=1e-4 (fudge), diag_f3=1e-5,
# constrained=TRUE (drift>=0), lsqmode=FALSE, numstart=10, fixed seed.
# ---------------------------------------------------------------------------
set.seed(42)
fit <- qpgraph(f2b, edges, diag = 1e-4, diag_f3 = 1e-5, lsqmode = FALSE,
               numstart = 10, constrained = TRUE, return_fstats = TRUE, verbose = FALSE)

admix_rows <- fit$edges[fit$edges$type == "admix", ]
worst <- head(fit$f3[order(-abs(fit$f3$z)), c("pop1", "pop2", "pop3", "diff", "z")], 5)

cat("\n======== SCORE ========\n");      cat("score:", fit$score, "\n")
cat("\n======== ADMIX WEIGHTS (interior) ========\n"); print(as.data.frame(admix_rows))
cat("\n======== ALL FITTED EDGES ========\n");          print(as.data.frame(fit$edges))
cat("\n======== WORST f-stat residuals (z) ========\n"); print(as.data.frame(worst))

# ---------------------------------------------------------------------------
# IDENTIFIABILITY / REPRODUCIBILITY PROOF.
#  (1) all numstart=10 restarts (single seed) reach the SAME score -> unique optimum.
#  (2) 5 distinct seeds -> score + admix weight agree to tol (reproducible).
#  (3) admix weight is INTERIOR (low/high bracket from restart spread, away from {0,1}).
# ---------------------------------------------------------------------------
cat("\n======== IDENTIFIABILITY CHECK ========\n")
cat("all-10-restart values (seed 42):\n"); print(as.data.frame(fit$opt)[, c("value", "convergence")])
cat(sprintf("restart value spread = %.3e\n", max(fit$opt$value) - min(fit$opt$value)))
cat(sprintf("admix weight low=%.6f high=%.6f (interior iff strictly in (0,1))\n",
            admix_rows$low[1], admix_rows$high[1]))

cat("\n--- reproducibility: 5 seeds ---\n")
scores <- numeric(0); wts <- numeric(0)
for (s in c(1, 7, 42, 123, 2024)) {
  set.seed(s)
  fs <- qpgraph(f2b, edges, diag = 1e-4, diag_f3 = 1e-5, lsqmode = FALSE,
                numstart = 10, constrained = TRUE, verbose = FALSE)
  w1 <- fs$edges$weight[fs$edges$type == "admix"][1]
  cat(sprintf("seed=%-5d  score=%.10f  w1=%.10f\n", s, fs$score, w1))
  scores <- c(scores, fs$score); wts <- c(wts, w1)
}
cat(sprintf("score spread (max-min) = %.3e\n", max(scores) - min(scores)))
cat(sprintf("w1    spread (max-min) = %.3e\n", max(wts) - min(wts)))

# ---------------------------------------------------------------------------
# SAVE GOLDEN
# ---------------------------------------------------------------------------
outdir <- "/workspace/steppe/tests/reference/goldens/at2"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
saveRDS(list(
  fit = fit, edges = edges, score = fit$score,
  admix_weights = admix_rows, edge_lengths = fit$edges, worst_residual = worst,
  restart_spread = max(fit$opt$value) - min(fit$opt$value),
  seed_score_spread = max(scores) - min(scores), seed_w1_spread = max(wts) - min(wts),
  meta = list(R = R.version.string,
              admixtools = as.character(packageVersion("admixtools")),
              f2dir = f2dir, pops = allpops, diag = 1e-4, diag_f3 = 1e-5,
              constrained = TRUE, lsqmode = FALSE, numstart = 10, seed = 42,
              nadmix = 1, dataset = "v66.p1_HO.aadr.patch.PUB")),
  file.path(outdir, "golden_qpgraph_fit.rds"))
write.csv(as.data.frame(admix_rows),  file.path(outdir, "golden_qpgraph_weights.csv"), row.names = FALSE)
write.csv(as.data.frame(fit$edges),   file.path(outdir, "golden_qpgraph_edges.csv"),   row.names = FALSE)
write.csv(data.frame(score = fit$score, restart_spread = max(fit$opt$value) - min(fit$opt$value)),
          file.path(outdir, "golden_qpgraph_score.csv"), row.names = FALSE)
write.csv(as.data.frame(edges), file.path(outdir, "golden_qpgraph_topology.csv"), row.names = FALSE)

cat("\n================ DONE ================\n")
sink()
