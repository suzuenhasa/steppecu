suppressMessages(library(admixtools))
options(width=220, digits=10)
sink("/tmp/golden_final.out.txt", split=TRUE)

cat("================ M(fit-0) GOLDEN ORACLE (FINAL, well-determined feasible 2-way) ================\n")
cat("R.version.string :", R.version.string, "\n")
cat("admixtools ver   :", as.character(packageVersion("admixtools")), "\n")
for (p in c("dplyr","tibble","Rcpp","RcppArmadillo")) cat("   ", p, ":", as.character(packageVersion(p)), "\n")
cat("RNGkind          :", paste(RNGkind(), collapse=", "), "\n")

pref   <- "/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB"
outdir <- "/workspace/data/aadr/f2_fit0_FINAL"
target <- "England_BellBeaker"
left   <- c("Czechia_EBA_CordedWare", "Turkey_N")
right  <- c("Mbuti", "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana")
allpops <- c(target, left, right)
blgsize <- 0.05; maxmiss <- 0; fudge <- 1e-4

cat("\n--- MODEL ---\n")
cat("target:", target, "\nleft  :", paste(left, collapse=", "), "\nright :", paste(right, collapse=", "), "\n")
cat("blgsize:", blgsize, " maxmiss:", maxmiss, " fudge:", fudge, " boot: FALSE  allsnps(maxmiss=0): TRUE\n")

cat("\n--- extract_f2 ---\n")
extract_f2(pref, outdir, pops=allpops, blgsize=blgsize, maxmiss=maxmiss, overwrite=TRUE, n_cores=8, verbose=TRUE)

set.seed(42)
res <- qpadm(outdir, left=left, right=right, target=target,
             boot=FALSE, getcov=TRUE, return_f4=TRUE, fudge=fudge, cpp=TRUE, verbose=TRUE)

cat("\nf4_var_rcond:", tryCatch(res$f4_var_rcond, error=function(e) NA), "\n")
cat("\n======== WEIGHTS (est/se/z) ========\n"); print(as.data.frame(res$weights))
cat("\n======== RANKDROP ========\n"); print(as.data.frame(res$rankdrop))
cat("\n======== POPDROP ========\n"); print(as.data.frame(res$popdrop))
cat("\n======== F4 dataframe ========\n"); print(as.data.frame(res$f4))

cat("\n--- reconstruct X + jackknife Q (qpadm internal pipeline) ---\n")
f2b <- read_f2(outdir, pops=allpops)
fb  <- admixtools:::f2blocks_to_f4blocks(f2b, left=left, right=right)
st  <- admixtools:::f4blocks_to_f4stats(fb, boot=FALSE)
f4_var_fudged <- st$var; diag(f4_var_fudged) <- diag(f4_var_fudged) + fudge*sum(diag(st$var))
cat("\n======== f4 ESTIMATE VECTOR X ========\n"); print(st$est)
cat("\n======== jackknife COVARIANCE Q (UNfudged) ========\n"); print(st$var)
cat("\n======== f4_var FUDGED (matrix qpadm inverts) ========\n"); print(f4_var_fudged)
cat("\nf4 block array dim:", paste(dim(fb), collapse=" x "), "\n")
bl <- attr(fb, "block_lengths"); if(is.null(bl)) bl <- as.numeric(sub("^l","",dimnames(fb)[[3]]))
cat("n_blocks:", dim(fb)[3], " sum(block_lengths):", sum(bl), " head(block_lengths):", paste(head(bl,5),collapse=","), "\n")

saveRDS(list(res=res, X=st$est, Q=st$var, Q_fudged=f4_var_fudged,
             block_lengths=bl, f4dim=dim(fb),
             meta=list(R=R.version.string, admixtools=as.character(packageVersion("admixtools")),
                       target=target, left=left, right=right, blgsize=blgsize, maxmiss=maxmiss,
                       fudge=fudge, boot=FALSE, seed=42,
                       dataset="v66.p1_HO.aadr.patch.PUB",
                       geno_sha256="7af8c2f5cf0db612e39257e59f20aba87906de90c79f409771c3b3145a253ec3")),
        "/workspace/data/aadr/golden_fit0_FINAL.rds")
write.csv(as.data.frame(res$weights),  "/workspace/data/aadr/golden_fit0_FINAL_weights.csv",  row.names=FALSE)
write.csv(as.data.frame(res$rankdrop), "/workspace/data/aadr/golden_fit0_FINAL_rankdrop.csv", row.names=FALSE)
write.csv(as.data.frame(res$popdrop),  "/workspace/data/aadr/golden_fit0_FINAL_popdrop.csv",  row.names=FALSE)
write.csv(as.data.frame(res$f4),       "/workspace/data/aadr/golden_fit0_FINAL_f4.csv",       row.names=FALSE)
write.csv(data.frame(rightpair=names(st$est), f4_est=as.numeric(st$est)),
          "/workspace/data/aadr/golden_fit0_FINAL_X.csv", row.names=FALSE)
write.csv(as.data.frame(st$var),       "/workspace/data/aadr/golden_fit0_FINAL_Q.csv",       row.names=TRUE)

cat("\n================ DONE ================\n")
sink()
