suppressMessages(library(admixtools))
outdir <- "/workspace/data/aadr/f2_fit0_FINAL"
target <- "England_BellBeaker"
left   <- c("Czechia_EBA_CordedWare", "Turkey_N")
right  <- c("Mbuti", "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana")
allpops <- c(target, left, right); fudge <- 1e-4
set.seed(42)
res <- qpadm(outdir, left=left, right=right, target=target, boot=FALSE, getcov=TRUE, return_f4=TRUE, fudge=fudge, cpp=TRUE, verbose=FALSE)
f2b <- read_f2(outdir, pops=allpops)
fb  <- admixtools:::f2blocks_to_f4blocks(f2b, left=left, right=right)
st  <- admixtools:::f4blocks_to_f4stats(fb, boot=FALSE)
Qf  <- st$var; diag(Qf) <- diag(Qf) + fudge*sum(diag(st$var))
# X is a 1x5 matrix: rows=left[-1], cols=right[-1]. Flatten with clear names.
Xmat <- st$est
rightpair <- colnames(Xmat); if (is.null(rightpair)) rightpair <- right[-1]
leftref   <- rownames(Xmat); if (is.null(leftref))   leftref   <- left[-1]
bl <- as.numeric(sub("^l","",dimnames(fb)[[3]]))
saveRDS(list(res=res, X=Xmat, Q=st$var, Q_fudged=Qf, block_lengths=bl, f4dim=dim(fb),
             meta=list(R=R.version.string, admixtools=as.character(packageVersion("admixtools")),
                       target=target, left=left, right=right, blgsize=0.05, maxmiss=0,
                       fudge=fudge, boot=FALSE, seed=42, allsnps=TRUE,
                       dataset="v66.p1_HO.aadr.patch.PUB",
                       geno_sha256="7af8c2f5cf0db612e39257e59f20aba87906de90c79f409771c3b3145a253ec3",
                       snp_sha256="c0d565ee9e6a6edf30c2063430ee4f828e5db45f26d4747fd76f19eee650a5de",
                       ind_sha256="51f99bee3667b5c3a97b663cc12fd3e25486cf64049782657d99db347b65509e")),
        "/workspace/data/aadr/golden_fit0_FINAL.rds")
write.csv(as.data.frame(res$weights),  "/workspace/data/aadr/golden_fit0_FINAL_weights.csv",  row.names=FALSE)
write.csv(as.data.frame(res$rankdrop), "/workspace/data/aadr/golden_fit0_FINAL_rankdrop.csv", row.names=FALSE)
write.csv(as.data.frame(res$popdrop),  "/workspace/data/aadr/golden_fit0_FINAL_popdrop.csv",  row.names=FALSE)
write.csv(as.data.frame(res$f4),       "/workspace/data/aadr/golden_fit0_FINAL_f4.csv",       row.names=FALSE)
Xdf <- data.frame(leftref=leftref, rightref=right[1], rightpair=rightpair, f4_est=as.numeric(Xmat))
write.csv(Xdf, "/workspace/data/aadr/golden_fit0_FINAL_X.csv", row.names=FALSE)
write.csv(as.data.frame(st$var), "/workspace/data/aadr/golden_fit0_FINAL_Q.csv", row.names=TRUE)
cat("X (f4 est) named:\n"); print(Xmat)
cat("\nfiles written. weights:\n"); print(as.data.frame(res$weights))
cat("\n==FIX DONE==\n")
