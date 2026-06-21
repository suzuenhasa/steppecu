suppressMessages(library(admixtools))
suppressMessages(library(dplyr))
options(width=240, digits=12)
sink("/tmp/golden_fitNA.out.txt", split=TRUE)

# F1 (OQ-12) MISSING-BLOCK / NA golden. UNLIKE every existing golden (maxmiss=0, global
# intersection ⇒ NO missing blocks), this is extracted with maxmiss>0 so the real-AADR
# f2 block array contains a PARTIALLY-covered block (a pop pair with Vpair==0). AT2
# read_f2(remove_na=TRUE) DROPS that block before the jackknife; steppe reproduces it.
#
# CANONICAL FORM = the f2-BLOCKS-OBJECT path: qpadm(f2_blocks_object, ...). steppe's
# device-resident fit consumes a precomputed f2 tensor, so it is the analogue of
# qpadm(read_f2(dir), ...), NOT qpadm(dir, ...) (the directory path applies a different
# internal f2-read arg set — the documented golden_fit0 caveat — and on a sparse model
# the two diverge a LOT). The golden values therefore come from qpadm(f2b, ...) +
# f2blocks_to_f4blocks/f4blocks_to_f4stats on the SAME read_f2(dir) object steppe mirrors.
cat("================ F1 / OQ-12 MISSING-BLOCK NA GOLDEN ================\n")
cat("R.version.string :", R.version.string, "\n")
cat("admixtools ver   :", as.character(packageVersion("admixtools")), "\n")
cat("RNGkind          :", paste(RNGkind(), collapse=", "), "\n")

pref   <- "/workspace/data/aadr/raw/v66.p1_HO.aadr.patch.PUB"
blgsize <- 0.05; fudge <- 1e-4

target  <- "England_BellBeaker"
left    <- c("Czechia_EBA_CordedWare", "Turkey_N")
right_base <- c("Mbuti", "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana")

# sparse ancient candidates (low coverage ⇒ a missing block at maxmiss>0). We keep the
# FIRST that (a) extracts, (b) leaves a finite, well-conditioned f2-OBJECT-path qpadm fit,
# (c) the DROP weights differ MEASURABLY from the IMPUTE-0 weights (so the test
# discriminates the correct behavior), (d) |weights| are sane (< 5; not a blown-up solve).
# Afghanistan_DarraiKurCave_MBA = the cleanest case: exactly 1 dropped block, 2 NA pair
# cells (Afghan x Mbuti). The 2-source model is a POOR fit for these outgroups (AT2's own
# f2-object weights are large), but that is AT2's exact answer and steppe reproduces it
# bit-for-bit; the gate is steppe==AT2 WITH the drop + drop!=impute, NOT a pretty fit.
sparse_candidates <- c("Afghanistan_DarraiKurCave_MBA", "Albania_MBA", "Albania_N",
                       "Albania_Cinamak_EBA", "Albania_Dukat_BA_IA",
                       "Armenia_Beniamin_BA_IA")

pick <- NULL
for (sp in sparse_candidates) {
  cat("\n===== TRY sparse right:", sp, "=====\n")
  allpops <- c(target, left, right_base, sp)
  outdir  <- paste0("/workspace/data/aadr/f2_fitNA_", gsub("[^A-Za-z0-9]", "", sp))
  ok <- tryCatch({
    extract_f2(pref, outdir, pops=allpops, blgsize=blgsize, maxmiss=0.99,
               overwrite=TRUE, n_cores=8, verbose=FALSE); TRUE
  }, error=function(e){ cat("  extract_f2 failed:", conditionMessage(e), "\n"); FALSE })
  if (!ok) next

  f2b_raw <- tryCatch(read_f2(outdir, pops=allpops, remove_na=FALSE),
                      error=function(e) NULL)
  if (is.null(f2b_raw)) next
  na_count <- sum(!is.finite(f2b_raw))
  keep_blocks <- apply(f2b_raw, 3, function(x) sum(!is.finite(x)) == 0)
  n_dropped <- sum(!keep_blocks)
  cat("  RAW f2 dim:", paste(dim(f2b_raw), collapse=" x "), " non-finite:", na_count,
      " blocks AT2 drops:", n_dropped, "of", dim(f2b_raw)[3], "\n")
  if (na_count == 0 || n_dropped == 0) { cat("  no missing block — skip\n"); next }

  right <- c(right_base, sp)
  f2b <- read_f2(outdir, pops=allpops)  # remove_na=TRUE -> the survivors qpadm consumes

  # CANONICAL (f2-object path) result — what steppe reproduces.
  res <- tryCatch(qpadm(f2b, left=left, right=right, target=target,
                        boot=FALSE, getcov=TRUE, return_f4=TRUE, fudge=fudge, cpp=TRUE, verbose=FALSE),
                  error=function(e){ cat("  qpadm(f2b) failed:", conditionMessage(e), "\n"); NULL })
  if (is.null(res)) next
  w_drop <- res$weights$weight
  if (any(!is.finite(w_drop))) { cat("  weights not finite — skip\n"); next }
  # No |w| cap: a large weight is AT2's exact answer for a poor 2-source outgroup fit;
  # steppe reproduces it bit-for-bit (the gate is steppe==AT2 + drop!=impute).

  # IMPUTE-0 contrast (the buggy behavior steppe must NOT do): how much does dropping matter?
  f2imp <- f2b_raw; f2imp[!is.finite(f2imp)] <- 0
  attr(f2imp, "block_lengths") <- attr(f2b_raw, "block_lengths")
  w_imp <- tryCatch(qpadm(f2imp, left=left, right=right, target=target,
                          boot=FALSE, fudge=fudge, cpp=TRUE, verbose=FALSE)$weights$weight,
                    error=function(e) rep(NA, length(w_drop)))
  reldiff <- max(abs(w_drop - w_imp) / pmax(abs(w_drop), 1e-12))
  cat("  DROP w:", paste(sprintf("%.6g",w_drop),collapse=","),
      " | IMPUTE w:", paste(sprintf("%.6g",w_imp),collapse=","),
      " | max reldiff:", sprintf("%.3g",reldiff), "\n")
  if (!is.finite(reldiff) || reldiff < 1e-4) { cat("  drop vs impute not discriminating — skip\n"); next }

  pick <- list(sp=sp, outdir=outdir, allpops=allpops, right=right, f2b=f2b, f2b_raw=f2b_raw,
               keep_blocks=keep_blocks, n_dropped=n_dropped, res=res, na_count=na_count,
               w_drop=w_drop, w_imp=w_imp, reldiff=reldiff)
  cat("  >>> PICKED:", sp, " (dropped", n_dropped, "block(s); f4rank", res$rankdrop$f4rank[1],
      "; drop-vs-impute reldiff", sprintf("%.3g",reldiff), ")\n")
  break
}

if (is.null(pick)) stop("no sparse-right model gave a sane, discriminating missing-block fit")

sp <- pick$sp; outdir <- pick$outdir; allpops <- pick$allpops; right <- pick$right
f2b <- pick$f2b; f2b_raw <- pick$f2b_raw; keep_blocks <- pick$keep_blocks; res <- pick$res

cat("\n--- MODEL ---\n")
cat("target:", target, "\nleft  :", paste(left, collapse=", "),
    "\nright :", paste(right, collapse=", "), "\n")
cat("blgsize:", blgsize, " maxmiss:0.99  fudge:", fudge, " boot:FALSE  (f2-OBJECT path)\n")
cat("sparse right (NA source):", sp, "  non-finite f2:", pick$na_count,
    "  AT2-dropped blocks:", pick$n_dropped, "\n")

cat("\n======== WEIGHTS (est/se/z) [DROP, canonical] ========\n"); print(as.data.frame(res$weights))
cat("\n======== RANKDROP ========\n"); print(as.data.frame(res$rankdrop))
cat("\n======== POPDROP ========\n"); print(as.data.frame(res$popdrop))
cat("\nIMPUTE-0 weights (buggy, for the discriminator note):",
    paste(sprintf("%.10g",pick$w_imp),collapse=","), " reldiff:", sprintf("%.4g",pick$reldiff), "\n")

bl_raw <- attr(f2b_raw, "block_lengths")
if (is.null(bl_raw)) bl_raw <- as.numeric(sub("^l", "", dimnames(f2b_raw)[[3]]))
nb_raw <- dim(f2b_raw)[3]
dropped_ids0 <- which(!keep_blocks) - 1L
nfin <- which(!is.finite(f2b_raw[,,dropped_ids0[1]+1]), arr.ind=TRUE)
cat("\nRAW nb:", nb_raw, "  surviving nb:", sum(keep_blocks),
    "  dropped(0-based):", paste(dropped_ids0, collapse=","), "\n")
cat("first dropped block NA pair (1-based i,j):\n"); print(nfin)

cat("\n--- steppe-convention f4 X + jackknife Q (leftref=TARGET; what the device computes) ---\n")
left_st <- c(target, left)  # target prepended (steppe assemble_f4 left_idx)
fb_st <- admixtools:::f2blocks_to_f4blocks(f2b, left=left_st, right=right)
st_st <- admixtools:::f4blocks_to_f4stats(fb_st, boot=FALSE)
cat("steppe f4 dim:", paste(dim(fb_st), collapse=" x "), " rows:", paste(dimnames(fb_st)[[1]],collapse=","), "\n")
cat("steppe X (rows x 6 cols), row-major k=j+nr*i:\n"); print(st_st$est)
bl <- attr(fb_st, "block_lengths"); if(is.null(bl)) bl <- as.numeric(sub("^l","",dimnames(fb_st)[[3]]))
cat("surviving n_blocks:", dim(fb_st)[3], " sum(block_lengths):", sum(bl), "\n")

saveRDS(list(res=res, X_steppe=st_st$est, Q_steppe=st_st$var,
             block_lengths=bl, block_lengths_raw=bl_raw, keep_blocks=keep_blocks,
             dropped_ids0=dropped_ids0, f4dim=dim(fb_st), f2dim_raw=dim(f2b_raw),
             pops=allpops, target=target, left=left, right=right,
             w_drop=pick$w_drop, w_imp=pick$w_imp, reldiff=pick$reldiff,
             n_nonfinite_f2=pick$na_count, n_dropped_blocks=pick$n_dropped,
             meta=list(R=R.version.string, admixtools=as.character(packageVersion("admixtools")),
                       target=target, left=left, right=right, blgsize=blgsize, maxmiss=0.99,
                       fudge=fudge, boot=FALSE, sparse_right=sp, path="f2_object",
                       dataset="v66.p1_HO.aadr.patch.PUB",
                       geno_sha256="7af8c2f5cf0db612e39257e59f20aba87906de90c79f409771c3b3145a253ec3")),
        "/workspace/data/aadr/golden_fitNA.rds")

# RAW (pre-drop) f2 binary with NaN = the NA pair marker (steppe derives Vpair + drops).
P <- dim(f2b_raw)[1]
con <- file("/workspace/data/aadr/f2_fitNA.bin", "wb")
writeBin(as.integer(P), con, size=4L)
writeBin(as.integer(nb_raw), con, size=4L)
writeBin(as.integer(round(bl_raw)), con, size=4L)
writeBin(as.double(as.vector(f2b_raw)), con, size=8L)
close(con)
cat("\nWROTE /workspace/data/aadr/f2_fitNA.bin (RAW P=", P, " nb=", nb_raw, ", NaN = NA marker)\n", sep="")

cat("\nPOP ORDER (index : name):\n")
for (i in seq_along(allpops)) cat("  ", i-1, ":", allpops[i], "\n")
cat("target idx:", match(target, allpops)-1, "\n")
cat("left  idx:", paste(match(left, allpops)-1, collapse=","), "\n")
cat("right idx:", paste(match(right, allpops)-1, collapse=","), "\n")

cat("\n================ DONE ================\n")
sink()
