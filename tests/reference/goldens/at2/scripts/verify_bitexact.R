suppressMessages(library(admixtools))
outdir <- "/workspace/data/aadr/f2_fit0_FINAL"
target <- "England_BellBeaker"
left   <- c("Czechia_EBA_CordedWare", "Turkey_N")
right  <- c("Mbuti", "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana")
fudge <- 1e-4
set.seed(42)
# EXACT original invocation: directory path string -> qpadm
res <- qpadm(outdir, left=left, right=right, target=target, boot=FALSE, getcov=TRUE, return_f4=TRUE, fudge=fudge, cpp=TRUE, verbose=FALSE)
cat("=== EXACT-PATH weights ===\n")
cat(sprintf("w_CordedWare = %.15g\n", res$weights$weight[1]))
cat(sprintf("se           = %.15g\n", res$weights$se[1]))
cat(sprintf("z_CordedWare = %.15g\n", res$weights$z[1]))
cat(sprintf("z_TurkeyN    = %.15g\n", res$weights$z[2]))
cat(sprintf("rankdrop p   = %.15g\n", res$rankdrop$p[1]))
