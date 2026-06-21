suppressMessages(library(admixtools))
options(width=220, digits=15)
sink("/tmp/golden_qpwave.out.txt", split=TRUE)

# F4 (fit-engine-finish-punchlist): pin a REAL AT2 qpwave() golden for the first-class
# run_qpwave entry (the no-target / left[0]-is-reference semantic). REUSES the
# golden_fit0 REAL-AADR pop set (v66.p1_HO.aadr.patch.PUB, the SAME f2 dir/fixture the
# qpadm golden_fit0 uses) arranged as a qpWave left/right split: the 3 "left" pops
# (England_BellBeaker, Czechia_EBA_CordedWare, Turkey_N — the qpadm target + 2 sources,
# now ALL left rows; left[0]=England_BellBeaker is the reference) vs the 6 outgroups.
# qpwave() takes NO target argument (verified: admixtools qpwave reference) — left[0] is
# the reference row, exactly steppe run_qpwave's no-target-prepend semantic.
cat("================ F4 qpWave GOLDEN ORACLE (REAL AADR, no target) ================\n")
cat("R.version.string :", R.version.string, "\n")
cat("admixtools ver   :", as.character(packageVersion("admixtools")), "\n")
cat("RNGkind          :", paste(RNGkind(), collapse=", "), "\n")

outdir <- "/workspace/data/aadr/f2_fit0_FINAL"   # the SAME REAL f2 dir golden_fit0 used
fudge  <- 1e-4
alpha  <- 0.05
right  <- c("Mbuti", "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana")

# AT2 res$f4rank = smallest tested rank r with p(r) > alpha (the rank-decision rule the
# steppe backend rank_sweep implements). rankdrop rows are f4rank DESCENDING.
f4rank_decision <- function(rd) {
  nonrej <- rd$f4rank[which(rd$p > alpha)]
  if (length(nonrej) > 0) min(nonrej) else max(rd$f4rank)
}

# Two REAL qpWave models (left[0]=reference, NO target argument). Canonical invocation:
# qpwave on the f2 DIRECTORY PATH directly (the same canonical directory-path form
# golden_fit0.json documents as bit-exact; read_f2 then qpwave with non-default read args
# drifts ~1e-5).
#   M1 (3-left, est_rank=1): the qpadm target + 2 sources, now ALL left rows. Validates
#       that steppe's no-target qpWave path produces the SAME f4 matrix as qpadm's
#       internal left=c(target,sources) — the no-target / left[0]-reference semantic.
#   M2 (2-left CLADE, est_rank=0): the canonical qpWave question — are CordedWare and
#       Turkey_N a clade vs the 6 rights? (Two left pops ⇒ a 1×(nr-1) f4 row; ranks 0,1.)
run_one <- function(tag, left) {
  cat("\n--- MODEL", tag, "(qpWave: left[0]=reference, NO target) ---\n")
  cat("left :", paste(left, collapse=", "), "\n")
  cat("right:", paste(right, collapse=", "), "\n")
  set.seed(42)
  qpw <- qpwave(outdir, left=left, right=right, boot=FALSE, fudge=fudge, cpp=TRUE, verbose=TRUE)
  rd  <- as.data.frame(qpw$rankdrop)
  cat("\n======== qpWave RANKDROP (", tag, ") ========\n", sep="")
  print(rd)
  cat("f4rank (smallest non-rejected rank @ alpha=", alpha, ") = ", f4rank_decision(rd), "\n", sep="")
  list(qpw=qpw, rd=rd, left=left, f4rank=f4rank_decision(rd))
}

m1 <- run_one("M1 3-left", c("England_BellBeaker", "Czechia_EBA_CordedWare", "Turkey_N"))
m2 <- run_one("M2 2-left clade", c("Czechia_EBA_CordedWare", "Turkey_N"))

saveRDS(list(m1=m1, m2=m2,
             meta=list(R=R.version.string, admixtools=as.character(packageVersion("admixtools")),
                       right=right, fudge=fudge, boot=FALSE, alpha=alpha,
                       blgsize=0.05, maxmiss=0, seed=42,
                       dataset="v66.p1_HO.aadr.patch.PUB",
                       geno_sha256="7af8c2f5cf0db612e39257e59f20aba87906de90c79f409771c3b3145a253ec3")),
        "/workspace/data/aadr/golden_qpwave.rds")
write.csv(m1$rd, "/workspace/data/aadr/golden_qpwave_m1_rankdrop.csv", row.names=FALSE)
write.csv(m2$rd, "/workspace/data/aadr/golden_qpwave_m2_rankdrop.csv", row.names=FALSE)

cat("\n================ DONE ================\n")
sink()
